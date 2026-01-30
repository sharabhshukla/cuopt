/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "feasibility_jump/feasibility_jump.cuh"

#include <mip/mip_constants.hpp>
#include "diversity/diversity_manager.cuh"
#include "local_search/local_search.cuh"
#include "local_search/rounding/simple_rounding.cuh"
#include "solver.cuh"
#include "presolve/trivial_presolve.cuh"

#include <linear_programming/pdlp.cuh>
#include <linear_programming/solve.cuh>

#include <dual_simplex/branch_and_bound.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>

#include <future>
#include <memory>
#include <thread>

namespace cuopt::linear_programming::detail {

// This serves as both a warm up but also a mandatory initial call to setup cuSparse and cuBLAS
static void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

template <typename i_t, typename f_t>
mip_solver_t<i_t, f_t>::mip_solver_t(const problem_t<i_t, f_t>& op_problem,
                                     const mip_solver_settings_t<i_t, f_t>& solver_settings,
                                     pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling,
                                     timer_t timer)
  : op_problem_(op_problem),
    solver_settings_(solver_settings),
    context(op_problem.handle_ptr,
            const_cast<problem_t<i_t, f_t>*>(&op_problem),
            solver_settings,
            scaling),
    timer_(timer)
{
  init_handler(op_problem.handle_ptr);
}

template <typename i_t, typename f_t>
struct branch_and_bound_solution_helper_t {
  branch_and_bound_solution_helper_t(diversity_manager_t<i_t, f_t>* dm,
                                     dual_simplex::simplex_solver_settings_t<i_t, f_t>& settings)
    : dm(dm), settings_(settings) {};

  void solution_callback(std::vector<f_t>& solution, f_t objective)
  {
    dm->population.add_external_solution(solution, objective, solution_origin_t::BRANCH_AND_BOUND);
    dm->rins.new_best_incumbent_callback(solution);
  }

  void set_simplex_solution(std::vector<f_t>& solution,
                            std::vector<f_t>& dual_solution,
                            f_t objective)
  {
    dm->set_simplex_solution(solution, dual_solution, objective);
  }

  void node_processed_callback(const std::vector<f_t>& solution, f_t objective)
  {
    dm->rins.node_callback(solution, objective);
  }

  void preempt_heuristic_solver() { dm->population.preempt_heuristic_solver(); }
  diversity_manager_t<i_t, f_t>* dm;
  dual_simplex::simplex_solver_settings_t<i_t, f_t>& settings_;
};

template <typename i_t, typename f_t>
solution_t<i_t, f_t> mip_solver_t<i_t, f_t>::run_solver()
{
  if (context.settings.get_mip_callbacks().size() > 0) {
    for (auto callback : context.settings.get_mip_callbacks()) {
      callback->template setup<f_t>(context.problem_ptr->original_problem_ptr->get_n_variables());
    }
  }
  //  we need to keep original problem const
  cuopt_assert(context.problem_ptr != nullptr, "invalid problem pointer");
  context.problem_ptr->tolerances = context.settings.get_tolerances();
  cuopt_expects(context.problem_ptr->preprocess_called,
                error_type_t::RuntimeError,
                "preprocess_problem should be called before running the solver");

  if (context.problem_ptr->empty) {
    CUOPT_LOG_INFO("Problem fully reduced in presolve");
    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.set_problem_fully_reduced();
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }

  diversity_manager_t<i_t, f_t> dm(context);
  dm.timer              = timer_;
  bool presolve_success = dm.run_presolve(timer_.remaining_time());
  if (!presolve_success) {
    CUOPT_LOG_INFO("Problem proven infeasible in presolve");
    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.set_problem_fully_reduced();
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }
  if (context.problem_ptr->empty) {
    CUOPT_LOG_INFO("Problem full reduced in presolve");
    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.set_problem_fully_reduced();
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }

  // if the problem was reduced to a LP: run concurrent LP
  if (context.problem_ptr->n_integer_vars == 0) {
    CUOPT_LOG_INFO("Problem reduced to a LP, running concurrent LP");
    pdlp_solver_settings_t<i_t, f_t> settings{};
    settings.time_limit = timer_.remaining_time();
    auto lp_timer       = timer_t(settings.time_limit);
    settings.method     = method_t::Concurrent;

    auto opt_sol = solve_lp_with_method<i_t, f_t>(*context.problem_ptr, settings, lp_timer);

    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.copy_new_assignment(
      host_copy(opt_sol.get_primal_solution(), context.problem_ptr->handle_ptr->get_stream()));
    if (opt_sol.get_termination_status() == pdlp_termination_status_t::Optimal ||
        opt_sol.get_termination_status() == pdlp_termination_status_t::PrimalInfeasible ||
        opt_sol.get_termination_status() == pdlp_termination_status_t::DualInfeasible) {
      sol.set_problem_fully_reduced();
    }
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }

  namespace dual_simplex = cuopt::linear_programming::dual_simplex;
  std::future<dual_simplex::mip_status_t> branch_and_bound_status_future;
  dual_simplex::user_problem_t<i_t, f_t> branch_and_bound_problem(context.problem_ptr->handle_ptr);
  dual_simplex::simplex_solver_settings_t<i_t, f_t> branch_and_bound_settings;
  std::unique_ptr<dual_simplex::branch_and_bound_t<i_t, f_t>> branch_and_bound;
  branch_and_bound_solution_helper_t solution_helper(&dm, branch_and_bound_settings);
  dual_simplex::mip_solution_t<i_t, f_t> branch_and_bound_solution(1);

  if (!context.settings.heuristics_only) {
    // Convert the presolved problem to dual_simplex::user_problem_t
    op_problem_.get_host_user_problem(branch_and_bound_problem);
    // Resize the solution now that we know the number of columns/variables
    branch_and_bound_solution.resize(branch_and_bound_problem.num_cols);

    // Fill in the settings for branch and bound
    branch_and_bound_settings.time_limit           = timer_.remaining_time();
    branch_and_bound_settings.print_presolve_stats = false;
    branch_and_bound_settings.absolute_mip_gap_tol = context.settings.tolerances.absolute_mip_gap;
    branch_and_bound_settings.relative_mip_gap_tol = context.settings.tolerances.relative_mip_gap;
    branch_and_bound_settings.integer_tol = context.settings.tolerances.integrality_tolerance;

    if (context.settings.num_cpu_threads < 0) {
      branch_and_bound_settings.num_threads = omp_get_max_threads() - 1;
    } else {
      branch_and_bound_settings.num_threads = std::max(1, context.settings.num_cpu_threads);
    }

    i_t num_threads                           = branch_and_bound_settings.num_threads;
    i_t num_bfs_workers                       = std::max(1, num_threads / 4);
    i_t num_diving_workers                    = std::max(1, num_threads - num_bfs_workers);
    branch_and_bound_settings.num_bfs_workers = num_bfs_workers;
    branch_and_bound_settings.diving_settings.num_diving_workers = num_diving_workers;

    // Set the branch and bound -> primal heuristics callback
    branch_and_bound_settings.solution_callback =
      std::bind(&branch_and_bound_solution_helper_t<i_t, f_t>::solution_callback,
                &solution_helper,
                std::placeholders::_1,
                std::placeholders::_2);
    branch_and_bound_settings.heuristic_preemption_callback = std::bind(
      &branch_and_bound_solution_helper_t<i_t, f_t>::preempt_heuristic_solver, &solution_helper);

    branch_and_bound_settings.set_simplex_solution_callback =
      std::bind(&branch_and_bound_solution_helper_t<i_t, f_t>::set_simplex_solution,
                &solution_helper,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3);

    branch_and_bound_settings.node_processed_callback =
      std::bind(&branch_and_bound_solution_helper_t<i_t, f_t>::node_processed_callback,
                &solution_helper,
                std::placeholders::_1,
                std::placeholders::_2);

    // Create the branch and bound object
    branch_and_bound = std::make_unique<dual_simplex::branch_and_bound_t<i_t, f_t>>(
      branch_and_bound_problem, branch_and_bound_settings);
    context.branch_and_bound_ptr = branch_and_bound.get();
    // Pass the root LP method to branch_and_bound
    branch_and_bound->set_root_lp_method(static_cast<int>(context.settings.root_lp_method));
    // Enable solve_root_relaxation() path (which waits for diversity_manager) for all methods except pure DualSimplex
    // When method is DualSimplex only, use the simple dual simplex path
    // For PDLP/Barrier/Concurrent, use solve_root_relaxation() which will conditionally launch solvers
    bool use_root_relaxation_path = (context.settings.root_lp_method != static_cast<method_t>(CUOPT_METHOD_DUAL_SIMPLEX));
    branch_and_bound->set_concurrent_lp_root_solve(use_root_relaxation_path);

    // Set the primal heuristics -> branch and bound callback
    context.problem_ptr->branch_and_bound_callback =
      std::bind(&dual_simplex::branch_and_bound_t<i_t, f_t>::set_new_solution,
                branch_and_bound.get(),
                std::placeholders::_1);
    context.problem_ptr->set_root_relaxation_solution_callback =
      std::bind(&dual_simplex::branch_and_bound_t<i_t, f_t>::set_root_relaxation_solution,
                branch_and_bound.get(),
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5,
                std::placeholders::_6);

    // Fork a thread for branch and bound
    // std::async and std::future allow us to get the return value of bb::solve()
    // without having to manually manage the thread
    // std::future.get() performs a join() operation to wait until the return status is available
    branch_and_bound_status_future = std::async(std::launch::async,
                                                &dual_simplex::branch_and_bound_t<i_t, f_t>::solve,
                                                branch_and_bound.get(),
                                                std::ref(branch_and_bound_solution));
  }

  // Start the primal heuristics
  auto sol = dm.run_solver();
  if (!context.settings.heuristics_only) {
    // Wait for the branch and bound to finish
    auto bb_status = branch_and_bound_status_future.get();
    if (branch_and_bound_solution.lower_bound > -std::numeric_limits<f_t>::infinity()) {
      context.stats.solution_bound =
        context.problem_ptr->get_user_obj_from_solver_obj(branch_and_bound_solution.lower_bound);
    }
    if (bb_status == dual_simplex::mip_status_t::INFEASIBLE) { sol.set_problem_fully_reduced(); }
    context.stats.num_nodes              = branch_and_bound_solution.nodes_explored;
    context.stats.num_simplex_iterations = branch_and_bound_solution.simplex_iterations;
  }
  sol.compute_feasibility();
  rmm::device_scalar<i_t> is_feasible(sol.handle_ptr->get_stream());
  sol.test_variable_bounds(true, is_feasible.data());
  // test_variable_bounds clears is_feasible if the test is failed
  if (!is_feasible.value(sol.handle_ptr->get_stream())) {
    CUOPT_LOG_ERROR(
      "Solution is not feasible due to variable bounds, returning infeasible solution!");
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }
  context.problem_ptr->post_process_solution(sol);
  dm.rins.stop_rins();
  return sol;
}

// This is the corrected sequential binary activation implementation
// To be inserted at line 270 in solver.cu

template <typename i_t, typename f_t>
solution_t<i_t, f_t> mip_solver_t<i_t, f_t>::run_solver_with_sequential_binary_activation()
{
  CUOPT_LOG_INFO("Starting sequential binary activation decomposition");

  cuopt_expects(context.problem_ptr->preprocess_called,
                error_type_t::RuntimeError,
                "preprocess_problem should be called before running the solver");

  if (context.problem_ptr->empty) {
    CUOPT_LOG_INFO("Problem fully reduced in presolve");
    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.set_problem_fully_reduced();
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }

  const i_t n_vars = context.problem_ptr->n_variables;
  const i_t n_integer_vars = context.problem_ptr->n_integer_vars;
  const f_t batch_ratio = context.settings.sequential_binary_batch_ratio;
  const i_t batch_size = std::max(static_cast<i_t>(1),
                                   static_cast<i_t>(n_integer_vars * batch_ratio));
  const i_t n_batches = (n_integer_vars + batch_size - 1) / batch_size;

  CUOPT_LOG_INFO("Total variables: %d, Integer variables: %d", n_vars, n_integer_vars);
  CUOPT_LOG_INFO("Batch size: %d (%.1f%%), Number of batches: %d",
                 batch_size, batch_ratio * 100, n_batches);

  // Step 1: Solve root LP relaxation to get initial continuous solution
  CUOPT_LOG_INFO("Solving root LP relaxation for warm start");
  pdlp_solver_settings_t<i_t, f_t> lp_settings{};
  lp_settings.time_limit = std::min(timer_.remaining_time() * 0.15, 600.0);
  lp_settings.method = context.settings.root_lp_method;
  lp_settings.inside_mip = true;
  lp_settings.crossover = context.settings.root_lp_crossover;

  auto lp_timer = timer_t(lp_settings.time_limit);
  auto root_lp = solve_lp_with_method<i_t, f_t>(*context.problem_ptr, lp_settings, lp_timer);

  CUOPT_LOG_INFO("Root LP status: %d", static_cast<int>(root_lp.get_termination_status()));

  // Get root LP solution as device vector
  auto lp_solution_device = cuopt::device_copy(root_lp.get_primal_solution(),
                                                context.problem_ptr->handle_ptr->get_stream());

  // Create initial solution from root LP
  solution_t<i_t, f_t> current_sol(*context.problem_ptr);
  current_sol.copy_new_assignment(lp_solution_device);

  // Copy integer indices to host
  auto h_integer_indices = host_copy(context.problem_ptr->integer_indices,
                                      context.problem_ptr->handle_ptr->get_stream());

  f_t best_objective = std::numeric_limits<f_t>::infinity();
  solution_t<i_t, f_t> best_sol(*context.problem_ptr);
  bool found_feasible = false;

  // Step 2: Sequential batch solving
  for (i_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
    const i_t start_idx = batch_idx * batch_size;
    const i_t end_idx = std::min((batch_idx + 1) * batch_size, n_integer_vars);
    const f_t batch_time_limit = std::min(
      timer_.remaining_time() / (n_batches - batch_idx),
      600.0  // Max 10 minutes per batch
    );

    CUOPT_LOG_INFO("\n=== Batch %d/%d: integer vars [%d, %d) ===",
                   batch_idx + 1, n_batches, start_idx, end_idx);
    CUOPT_LOG_INFO("Time limit for this batch: %.1f seconds", batch_time_limit);

    // Identify variables to fix (all previous batches)
    rmm::device_uvector<i_t> vars_to_fix(start_idx, context.problem_ptr->handle_ptr->get_stream());
    auto h_vars_to_fix = std::vector<i_t>(h_integer_indices.begin(),
                                           h_integer_indices.begin() + start_idx);
    device_copy(h_vars_to_fix, vars_to_fix, context.problem_ptr->handle_ptr->get_stream());

    // Fix those variables in current solution
    auto [batch_problem, batch_assignment, variable_map] = current_sol.fix_variables(vars_to_fix);

    CUOPT_LOG_INFO("Fixed problem: %d vars (original: %d), %d integer vars (original: %d)",
                   batch_problem.n_variables, n_vars,
                   batch_problem.n_integer_vars, n_integer_vars);

    // Now relax future integer variables to continuous
    // Get indices in the NEW problem space
    auto h_batch_integer_indices = host_copy(batch_problem.integer_indices,
                                              batch_problem.handle_ptr->get_stream());

    // Calculate how many to keep as integer (active batch)
    i_t n_active_integers = end_idx - start_idx;

    // Relax the ones beyond active batch
    std::vector<var_t> h_var_types = host_copy(batch_problem.variable_types,
                                                batch_problem.handle_ptr->get_stream());
    i_t n_relaxed = 0;
    for (i_t j = n_active_integers; j < batch_problem.n_integer_vars; ++j) {
      i_t var_idx = h_batch_integer_indices[j];
      h_var_types[var_idx] = var_t::CONTINUOUS;
      n_relaxed++;
    }

    // Update variable types on device
    device_copy(h_var_types, batch_problem.variable_types,
                batch_problem.handle_ptr->get_stream());
    batch_problem.compute_n_integer_vars();

    CUOPT_LOG_INFO("Active batch: %d binary, %d relaxed to continuous",
                   n_active_integers, n_relaxed);

    // Presolve the batch problem
    batch_problem.presolve_data.reset_additional_vars(batch_problem,
                                                       batch_problem.handle_ptr);
    batch_problem.presolve_data.initialize_var_mapping(batch_problem,
                                                         batch_problem.handle_ptr);
    trivial_presolve(batch_problem);

    // Setup batch solver settings
    mip_solver_settings_t<i_t, f_t> batch_settings = context.settings;
    batch_settings.time_limit = batch_time_limit;
    batch_settings.sequential_binary_activation = false;  // Disable recursion

    // Add batch solution as warm start
    auto h_batch_assignment = host_copy(batch_assignment,
                                         batch_problem.handle_ptr->get_stream());
    batch_settings.add_initial_solution(h_batch_assignment.data(),
                                         h_batch_assignment.size(),
                                         batch_problem.handle_ptr->get_stream());

    // Solve batch MIP
    auto batch_timer = timer_t(batch_time_limit);
    mip_solver_t<i_t, f_t> batch_solver(batch_problem,
                                          batch_settings,
                                          context.scaling,
                                          batch_timer);
    auto batch_sol = batch_solver.run_solver();

    if (batch_sol.get_feasible()) {
      f_t batch_obj = batch_sol.get_user_objective();
      CUOPT_LOG_INFO("Batch %d found feasible solution with objective: %f",
                     batch_idx + 1, batch_obj);

      // Unfix variables to get solution in original space
      current_sol.unfix_variables(batch_sol.assignment, variable_map);

      if (batch_obj < best_objective) {
        best_objective = batch_obj;
        best_sol.copy_from(current_sol);
        found_feasible = true;
      }
    } else {
      CUOPT_LOG_WARN("Batch %d failed to find feasible solution - continuing with current solution",
                     batch_idx + 1);
    }

    if (timer_.remaining_time() < 10.0) {
      CUOPT_LOG_WARN("Running out of time, stopping after batch %d/%d",
                     batch_idx + 1, n_batches);
      break;
    }
  }

  if (found_feasible) {
    CUOPT_LOG_INFO("Sequential binary activation completed - best objective: %f", best_objective);
    context.problem_ptr->post_process_solution(best_sol);
    return best_sol;
  } else {
    CUOPT_LOG_WARN("Sequential binary activation failed to find feasible solution");
    context.problem_ptr->post_process_solution(current_sol);
    return current_sol;
  }
}

// Original feasibility jump has only double
#if MIP_INSTANTIATE_FLOAT
template class mip_solver_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class mip_solver_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
