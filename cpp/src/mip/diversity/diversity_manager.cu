/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip/mip_constants.hpp>
#include <mip/presolve/probing_cache.cuh>
#include <mip/presolve/trivial_presolve.cuh>
#include <mip/problem/problem_helpers.cuh>
#include "diversity_manager.cuh"

#include <utilities/scope_guard.hpp>

#include "cuda_profiler_api.h"

#include <random>

namespace cuopt::linear_programming::detail {

size_t fp_recombiner_config_t::max_n_of_vars_from_other =
  fp_recombiner_config_t::initial_n_of_vars_from_other;
size_t ls_recombiner_config_t::max_n_of_vars_from_other =
  ls_recombiner_config_t::initial_n_of_vars_from_other;
size_t bp_recombiner_config_t::max_n_of_vars_from_other =
  bp_recombiner_config_t::initial_n_of_vars_from_other;
size_t sub_mip_recombiner_config_t::max_n_of_vars_from_other =
  sub_mip_recombiner_config_t::initial_n_of_vars_from_other;

template <typename i_t, typename f_t>
std::vector<recombiner_enum_t> recombiner_t<i_t, f_t>::enabled_recombiners;

template <typename i_t, typename f_t>
diversity_manager_t<i_t, f_t>::diversity_manager_t(mip_solver_context_t<i_t, f_t>& context_)
  : context(context_),
    problem_ptr(context.problem_ptr),
    diversity_config(),
    population("population",
               context,
               *this,
               diversity_config.max_var_diff,
               diversity_config.max_solutions,
               diversity_config.initial_infeasibility_weight * context.problem_ptr->n_constraints),
    lp_optimal_solution(context.problem_ptr->n_variables,
                        context.problem_ptr->handle_ptr->get_stream()),
    lp_dual_optimal_solution(context.problem_ptr->n_constraints,
                             context.problem_ptr->handle_ptr->get_stream()),
    ls(context, lp_optimal_solution),
    rins(context, *this),
    timer(context.gpu_heur_loop, diversity_config.default_time_limit),
    bound_prop_recombiner(context,
                          context.problem_ptr->n_variables,
                          ls.constraint_prop,
                          context.problem_ptr->handle_ptr),
    fp_recombiner(context,
                  context.problem_ptr->n_variables,
                  ls.fj,
                  ls.constraint_prop,
                  ls.line_segment_search,
                  lp_optimal_solution,
                  context.problem_ptr->handle_ptr),
    line_segment_recombiner(context,
                            context.problem_ptr->n_variables,
                            ls.line_segment_search,
                            context.problem_ptr->handle_ptr),
    sub_mip_recombiner(
      context, population, context.problem_ptr->n_variables, context.problem_ptr->handle_ptr),
    rng(cuopt::seed_generator::get_seed()),
    stats(context.stats),
    mab_recombiner(0, cuopt::seed_generator::get_seed(), recombiner_alpha, "recombiner"),
    mab_ls(mab_ls_config_t<i_t, f_t>::n_of_arms, cuopt::seed_generator::get_seed(), ls_alpha, "ls"),
    ls_hash_map(*context.problem_ptr)
{
  fp_recombiner_config_t::max_n_of_vars_from_other =
    fp_recombiner_config_t::initial_n_of_vars_from_other;
  ls_recombiner_config_t::max_n_of_vars_from_other =
    ls_recombiner_config_t::initial_n_of_vars_from_other;
  bp_recombiner_config_t::max_n_of_vars_from_other =
    bp_recombiner_config_t::initial_n_of_vars_from_other;

  // Read configuration ID from environment variable
  int max_config = -1;
  // Read max configuration value from environment variable
  const char* env_max_config = std::getenv("CUOPT_MAX_CONFIG");
  if (env_max_config != nullptr) {
    try {
      max_config = std::stoi(env_max_config);
      CUOPT_LOG_INFO("Using maximum configuration value from environment: %d", max_config);
    } catch (const std::exception& e) {
      CUOPT_LOG_WARN("Failed to parse CUOPT_MAX_CONFIG environment variable: %s", e.what());
    }
  }
  if (max_config > 1) {
    [[maybe_unused]] int config_id = -1;  // Default value
    const char* env_config_id      = std::getenv("CUOPT_CONFIG_ID");
    if (env_config_id != nullptr) {
      try {
        config_id = std::stoi(env_config_id);
        CUOPT_LOG_INFO("Using configuration ID from environment: %d", config_id);
      } catch (const std::exception& e) {
        CUOPT_LOG_WARN("Failed to parse CUOPT_CONFIG_ID environment variable: %s", e.what());
      }
    }
  }

  context.gpu_heur_loop.deterministic =
    context.settings.determinism_mode == CUOPT_MODE_DETERMINISTIC;
}

// this function is to specialize the local search with config from diversity manager
template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::run_local_search(solution_t<i_t, f_t>& solution,
                                                     const weight_t<i_t, f_t>& weights,
                                                     work_limit_timer_t& timer,
                                                     ls_config_t<i_t, f_t>& ls_config)
{
  raft::common::nvtx::range fun_scope("run_local_search");
  i_t ls_mab_option = mab_ls.select_mab_option();
  mab_ls_config_t<i_t, f_t>::get_local_search_and_lm_from_config(ls_mab_option, ls_config);
  ls_hash_map.insert(solution);
  constexpr i_t skip_solutions_threshold = 3;
  if (ls_hash_map.check_skip_solution(solution, skip_solutions_threshold)) { return false; }
  ls.run_local_search(solution, weights, timer, ls_config);
  return true;
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::generate_solution(f_t time_limit, bool random_start)
{
  raft::common::nvtx::range fun_scope("generate_solution");
  solution_t<i_t, f_t> sol(*problem_ptr);
  sol.compute_feasibility();
  // if a feasible is found, it is added to the population
  ls.generate_solution(sol, random_start, &population, time_limit);
  population.add_solution(std::move(sol));
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::add_user_given_solutions(
  std::vector<solution_t<i_t, f_t>>& initial_sol_vector)
{
  raft::common::nvtx::range fun_scope("add_user_given_solutions");
  for (const auto& init_sol : context.settings.initial_solutions) {
    solution_t<i_t, f_t> sol(*problem_ptr);
    rmm::device_uvector<f_t> init_sol_assignment(*init_sol, sol.handle_ptr->get_stream());
    if (problem_ptr->pre_process_assignment(init_sol_assignment)) {
      relaxed_lp_settings_t lp_settings;
      lp_settings.time_limit            = std::min(60., timer.remaining_time() / 2);
      lp_settings.work_limit            = lp_settings.time_limit;
      lp_settings.tolerance             = problem_ptr->tolerances.absolute_tolerance;
      lp_settings.save_state            = false;
      lp_settings.return_first_feasible = true;
      run_lp_with_vars_fixed(*problem_ptr,
                             sol,
                             problem_ptr->integer_indices,
                             lp_settings,
                             static_cast<bound_presolve_t<i_t, f_t>*>(nullptr));
      raft::copy(sol.assignment.data(),
                 init_sol_assignment.data(),
                 init_sol_assignment.size(),
                 sol.handle_ptr->get_stream());
      bool is_feasible = sol.compute_feasibility();
      cuopt_func_call(sol.test_variable_bounds(true));
      CUOPT_LOG_INFO("Adding initial solution success! feas %d objective %f excess %f",
                     is_feasible,
                     sol.get_user_objective(),
                     sol.get_total_excess());
      population.run_solution_callbacks(sol);
      initial_sol_vector.emplace_back(std::move(sol));
    } else {
      CUOPT_LOG_ERROR(
        "Error cannot add the provided initial solution! \
    Assignment size %lu \
    initial solution size %lu",
        sol.assignment.size(),
        init_sol_assignment.size());
    }
  }
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::run_presolve(f_t time_limit)
{
  raft::common::nvtx::range fun_scope("run_presolve");
  CUOPT_LOG_INFO("Running presolve!");
  work_limit_timer_t presolve_timer(context.gpu_heur_loop, time_limit);
  auto term_crit = ls.constraint_prop.bounds_update.solve(*problem_ptr);
  if (ls.constraint_prop.bounds_update.infeas_constraints_count > 0) {
    stats.presolve_time = timer.elapsed_time();
    return false;
  }
  if (termination_criterion_t::NO_UPDATE != term_crit) {
    ls.constraint_prop.bounds_update.set_updated_bounds(*problem_ptr);
    trivial_presolve(*problem_ptr);
    if (!problem_ptr->empty && !check_bounds_sanity(*problem_ptr)) { return false; }
  }
  // May overconstrain if Papilo presolve has been run before
  if (!context.settings.presolve) {
    if (!problem_ptr->empty) {
      // do the resizing no-matter what, bounds presolve might not change the bounds but initial
      // trivial presolve might have
      ls.constraint_prop.bounds_update.resize(*problem_ptr);
      ls.constraint_prop.conditional_bounds_update.update_constraint_bounds(
        *problem_ptr, ls.constraint_prop.bounds_update);
      if (!check_bounds_sanity(*problem_ptr)) { return false; }
    }
  }
  stats.presolve_time = presolve_timer.elapsed_time();
  lp_optimal_solution.resize(problem_ptr->n_variables, problem_ptr->handle_ptr->get_stream());
  lp_dual_optimal_solution.resize(problem_ptr->n_constraints,
                                  problem_ptr->handle_ptr->get_stream());
  problem_ptr->handle_ptr->sync_stream();
  CUOPT_LOG_INFO("After trivial presolve: %d constraints, %d variables, objective offset %f.",
                 problem_ptr->n_constraints,
                 problem_ptr->n_variables,
                 problem_ptr->presolve_data.objective_offset);
  return true;
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::generate_quick_feasible_solution()
{
  raft::common::nvtx::range fun_scope("generate_quick_feasible_solution");
  solution_t<i_t, f_t> solution(*problem_ptr);
  // min 1 second, max 10 seconds
  const f_t generate_fast_solution_time =
    std::min(diversity_config.max_fast_sol_time, std::max(1., timer.remaining_time() / 20.));
  work_limit_timer_t sol_timer(context.gpu_heur_loop, generate_fast_solution_time);
  // do very short LP run to get somewhere close to the optimal point
  ls.generate_fast_solution(solution, sol_timer);
  if (solution.get_feasible()) {
    population.run_solution_callbacks(solution);
    initial_sol_vector.emplace_back(std::move(solution));
    problem_ptr->handle_ptr->sync_stream();
    solution_t<i_t, f_t> searched_sol(initial_sol_vector.back());
    ls_config_t<i_t, f_t> ls_config;
    run_local_search(searched_sol, population.weights, sol_timer, ls_config);
    population.run_solution_callbacks(searched_sol);
    initial_sol_vector.emplace_back(std::move(searched_sol));
    auto& feas_sol = initial_sol_vector.back().get_feasible()
                       ? initial_sol_vector.back()
                       : initial_sol_vector[initial_sol_vector.size() - 2];
    CUOPT_LOG_INFO("Generated fast solution with objective %f", feas_sol.get_user_objective());
  }
  problem_ptr->handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::check_b_b_preemption()
{
  if (population.preempt_heuristic_solver_.load()) {
    if (population.current_size() == 0) { population.allocate_solutions(); }
    population.add_external_solutions_to_population();
    return true;
  }
  population.add_external_solutions_to_population();
  return false;
}

// returns the best feasible solution
template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::run_fj_alone(solution_t<i_t, f_t>& solution)
{
  CUOPT_LOG_INFO("Running FJ alone!");
  // Benchmark FJ with 1000 different random starting solutions and varying iteration limits
  CUOPT_LOG_INFO("Starting FJ benchmark: 1000 runs with random starting solutions");

  std::mt19937 rng(cuopt::seed_generator::get_seed());
  std::uniform_int_distribution<i_t> iter_dist(100, 50000);

  for (i_t run = 0; run < 1000; ++run) {
    // Generate random starting solution within bounds
    solution.assign_random_within_bounds(1.0, false);
    solution.round_nearest();

    // Configure FJ settings with random iteration limit
    ls.fj.settings                        = fj_settings_t{};
    ls.fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
    ls.fj.settings.n_of_minimums_for_exit = 20000 * 1000;
    ls.fj.settings.update_weights         = true;
    ls.fj.settings.feasibility_run        = false;
    ls.fj.settings.iteration_limit        = iter_dist(rng);
    ls.fj.settings.time_limit             = std::numeric_limits<double>::infinity();

    if (context.settings.determinism_mode == CUOPT_MODE_DETERMINISTIC) {
      ls.fj.settings.iteration_limit = iter_dist(rng);
    }

    CUOPT_LOG_INFO(
      "FJ benchmark run %d/%d: iteration_limit=%d", run + 1, 1000, ls.fj.settings.iteration_limit);

    ls.fj.solve(solution);
  }

  CUOPT_LOG_INFO("FJ benchmark finished: 1000 runs completed");
  exit(0);
}

// returns the best feasible solution
template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::run_fp_alone()
{
  CUOPT_LOG_DEBUG("Running FP alone!");
  solution_t<i_t, f_t> sol(population.best_feasible());
  ls.run_fp(sol, timer, &population, diversity_config.n_fp_iterations);
  CUOPT_LOG_DEBUG("FP alone finished!");
}

template <typename i_t, typename f_t>
struct ls_cpufj_raii_guard_t {
  ls_cpufj_raii_guard_t(local_search_t<i_t, f_t>& ls) : ls(ls) {}
  ~ls_cpufj_raii_guard_t() { ls.stop_cpufj_scratch_threads(); }
  local_search_t<i_t, f_t>& ls;
};

// returns the best feasible solution
template <typename i_t, typename f_t>
solution_t<i_t, f_t> diversity_manager_t<i_t, f_t>::run_solver()
{
  raft::common::nvtx::range fun_scope("run_solver");

  diversity_config.fj_only_run = false;

  population.timer     = timer;
  const f_t time_limit = timer.remaining_time();
  const f_t lp_time_limit =
    std::min(diversity_config.max_time_on_lp, time_limit * diversity_config.time_ratio_on_init_lp);
  // to automatically compute the solving time on scope exit
  auto timer_raii_guard =
    cuopt::scope_guard([&]() { stats.total_solve_time = timer.timer.elapsed_time(); });
  // after every change to the problem, we should resize all the relevant vars
  // we need to encapsulate that to prevent repetitions
  recombine_stats.reset();
  ls.resize_vectors(*problem_ptr, problem_ptr->handle_ptr);
  ls.constraint_prop.bounds_update.resize(*problem_ptr);
  problem_ptr->check_problem_representation(true);
  // have the structure ready for reusing later
  problem_ptr->compute_integer_fixed_problem();
  recombiner_t<i_t, f_t>::init_enabled_recombiners(context, *problem_ptr);
  mab_recombiner.resize_mab_arm_stats(recombiner_t<i_t, f_t>::enabled_recombiners.size());
  // test problem is not ii
  cuopt_func_call(
    ls.constraint_prop.bounds_update.calculate_activity_on_problem_bounds(*problem_ptr));
  cuopt_assert(
    ls.constraint_prop.bounds_update.calculate_infeasible_redundant_constraints(*problem_ptr),
    "The problem must not be ii");
  population.initialize_population();
  population.allocate_solutions();
  if (check_b_b_preemption()) { return population.best_feasible(); }
  add_user_given_solutions(initial_sol_vector);
  // Run CPUFJ early to find quick initial solutions
  ls_cpufj_raii_guard_t ls_cpufj_raii_guard(ls);  // RAII to stop cpufj threads on solve stop

  if (context.settings.determinism_mode == CUOPT_MODE_OPPORTUNISTIC) {
    ls.start_cpufj_scratch_threads(population);
  }

  // before probing cache or LP, run FJ to generate initial primal feasible solution
  const f_t time_ratio_of_probing_cache = diversity_config.time_ratio_of_probing_cache;
  const f_t max_time_on_probing         = diversity_config.max_time_on_probing;
  f_t time_for_probing_cache =
    std::min(max_time_on_probing, time_limit * time_ratio_of_probing_cache);
  work_limit_timer_t probing_timer{context.gpu_heur_loop, time_for_probing_cache};
  if (check_b_b_preemption()) { return population.best_feasible(); }
  if (!diversity_config.fj_only_run) {
    compute_probing_cache(ls.constraint_prop.bounds_update, *problem_ptr, probing_timer);
  }

  if (check_b_b_preemption()) { return population.best_feasible(); }
  lp_state_t<i_t, f_t>& lp_state = problem_ptr->lp_state;
  // resize because some constructor might be called before the presolve
  lp_state.resize(*problem_ptr, problem_ptr->handle_ptr->get_stream());
  bool bb_thread_solution_exists = simplex_solution_exists.load();
  if (bb_thread_solution_exists) {
    ls.lp_optimal_exists = true;
  } else if (!diversity_config.fj_only_run) {
    relaxed_lp_settings_t lp_settings;
    lp_settings.time_limit            = lp_time_limit;
    lp_settings.work_limit            = lp_time_limit;
    lp_settings.tolerance             = context.settings.tolerances.absolute_tolerance;
    lp_settings.return_first_feasible = false;
    lp_settings.save_state            = true;
    lp_settings.concurrent_halt       = &global_concurrent_halt;
    lp_settings.has_initial_primal    = false;
    rmm::device_uvector<f_t> lp_optimal_solution_copy(lp_optimal_solution.size(),
                                                      problem_ptr->handle_ptr->get_stream());
    auto lp_result =
      get_relaxed_lp_solution(*problem_ptr, lp_optimal_solution_copy, lp_state, lp_settings);
    {
      std::lock_guard<std::mutex> guard(relaxed_solution_mutex);
      if (!simplex_solution_exists.load()) {
        raft::copy(lp_optimal_solution.data(),
                   lp_optimal_solution_copy.data(),
                   lp_optimal_solution.size(),
                   problem_ptr->handle_ptr->get_stream());
      } else {
        // copy the lp state
        raft::copy(lp_state.prev_primal.data(),
                   lp_optimal_solution.data(),
                   lp_optimal_solution.size(),
                   problem_ptr->handle_ptr->get_stream());
        raft::copy(lp_state.prev_dual.data(),
                   lp_dual_optimal_solution.data(),
                   lp_dual_optimal_solution.size(),
                   problem_ptr->handle_ptr->get_stream());
      }
      problem_ptr->handle_ptr->sync_stream();
    }
    ls.lp_optimal_exists = true;
    if (lp_result.get_termination_status() == pdlp_termination_status_t::Optimal) {
      set_new_user_bound(lp_result.get_objective_value());
    } else if (lp_result.get_termination_status() == pdlp_termination_status_t::PrimalInfeasible) {
      CUOPT_LOG_ERROR("Problem is primal infeasible, continuing anyway!");
      ls.lp_optimal_exists = false;
    } else if (lp_result.get_termination_status() == pdlp_termination_status_t::DualInfeasible) {
      CUOPT_LOG_ERROR("PDLP detected dual infeasibility, continuing anyway!");
      ls.lp_optimal_exists = false;
    } else if (lp_result.get_termination_status() == pdlp_termination_status_t::TimeLimit ||
               lp_result.get_termination_status() == pdlp_termination_status_t::IterationLimit) {
      CUOPT_LOG_DEBUG(
        "Initial LP run exceeded time/iteration limit, continuing solver with partial LP result!");
      // note to developer, in debug mode the LP run might be too slow and it might cause PDLP not
      // to bring variables within the bounds
    }
    // in case the pdlp returned var boudns that are out of bounds
    clamp_within_var_bounds(lp_optimal_solution, problem_ptr, problem_ptr->handle_ptr);
  }

  // Run this 100 times with varying iteration limits
  // for (int i = 0; i < 100; i++) {
  //   relaxed_lp_settings_t lp_settings;
  //   lp_settings.time_limit            = lp_time_limit;
  //   lp_settings.tolerance             = context.settings.tolerances.absolute_tolerance;
  //   lp_settings.return_first_feasible = false;
  //   lp_settings.save_state            = true;
  //   lp_settings.concurrent_halt       = &global_concurrent_halt;
  //   lp_settings.has_initial_primal    = false;
  //   lp_settings.iteration_limit       = 100 + i * 100;
  //   rmm::device_uvector<f_t> lp_optimal_solution_copy(lp_optimal_solution.size(),
  //                                                     problem_ptr->handle_ptr->get_stream());
  //   auto lp_result =
  //     get_relaxed_lp_solution(*problem_ptr, lp_optimal_solution_copy, lp_state, lp_settings);
  // }
  // exit(0);

  if (ls.lp_optimal_exists) {
    solution_t<i_t, f_t> lp_rounded_sol(*problem_ptr);
    lp_rounded_sol.copy_new_assignment(lp_optimal_solution);
    lp_rounded_sol.round_nearest();
    lp_rounded_sol.compute_feasibility();
    population.add_solution(std::move(lp_rounded_sol));
    if (context.settings.determinism_mode == CUOPT_MODE_OPPORTUNISTIC) {
      ls.start_cpufj_lptopt_scratch_threads(population);
    }
  }

  population.add_solutions_from_vec(std::move(initial_sol_vector));

  if (check_b_b_preemption()) { return population.best_feasible(); }

  if (context.settings.benchmark_info_ptr != nullptr) {
    context.settings.benchmark_info_ptr->objective_of_initial_population =
      population.best_feasible().get_user_objective();
  }

  if (diversity_config.dry_run) { return population.best_feasible(); }
  if (diversity_config.fj_only_run) {
    solution_t<i_t, f_t> sol(*problem_ptr);
    run_fj_alone(sol);
    return sol;
  }
  if (context.settings.determinism_mode == CUOPT_MODE_OPPORTUNISTIC) { rins.enable(); }

  generate_solution(timer.remaining_time(), false);
  printf("=======================================================\n");
  if (diversity_config.initial_solution_only) { return population.best_feasible(); }
  if (work_limit_reached()) {
    population.add_external_solutions_to_population();
    return population.best_feasible();
  }
  run_fp_alone();
  population.add_external_solutions_to_population();
  return population.best_feasible();
};

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::diversity_step(i_t max_iterations_without_improvement)
{
  bool improved = true;
  while (improved) {
    int k    = max_iterations_without_improvement;
    improved = false;
    while (k-- > 0) {
      if (check_b_b_preemption()) { return; }
      auto new_sol_vector = population.get_external_solutions();
      recombine_and_ls_with_all(new_sol_vector);
      population.adjust_weights_according_to_best_feasible();
      cuopt_assert(population.test_invariant(), "");
      if (population.current_size() < 2) {
        CUOPT_LOG_DEBUG("Population degenerated in diversity step");
        return;
      }
      if (work_limit_reached()) return;
      constexpr bool tournament = true;
      auto [sol1, sol2]         = population.get_two_random(tournament);
      cuopt_assert(population.test_invariant(), "");
      auto [lp_offspring, offspring]        = recombine_and_local_search(sol1, sol2);
      auto [inserted_pos_1, best_updated_1] = population.add_solution(std::move(lp_offspring));
      auto [inserted_pos_2, best_updated_2] = population.add_solution(std::move(offspring));
      if (best_updated_1 || best_updated_2) { recombine_stats.add_best_updated(); }
      cuopt_assert(population.test_invariant(), "");
      if ((inserted_pos_1 != -1 && inserted_pos_1 <= 2) ||
          (inserted_pos_2 != -1 && inserted_pos_2 <= 2)) {
        improved = true;
        recombine_stats.print();
        break;
      }
    }
  }
  recombine_stats.print();
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::set_new_user_bound(f_t new_bound)
{
  stats.solution_bound = new_bound;
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::recombine_and_ls_with_all(solution_t<i_t, f_t>& solution,
                                                              bool add_only_feasible)
{
  raft::common::nvtx::range fun_scope("recombine_and_ls_with_all");
  // if (population.population_hash_map.check_skip_solution(solution, 1)) { return; }
  auto population_vector = population.population_to_vector();
  for (auto& curr_sol : population_vector) {
    if (check_integer_equal_on_indices(problem_ptr->integer_indices,
                                       curr_sol.assignment,
                                       solution.assignment,
                                       problem_ptr->tolerances.integrality_tolerance,
                                       problem_ptr->handle_ptr)) {
      CUOPT_LOG_DEBUG("Skipping solution because it is equal to the given solution");
      continue;
    }
    for (const auto recombiner_type : recombiner_t<i_t, f_t>::enabled_recombiners) {
      if (check_b_b_preemption()) { return; }
      if (curr_sol.get_feasible()) {
        auto [offspring, lp_offspring] =
          recombine_and_local_search(curr_sol, solution, recombiner_type);
        if (!add_only_feasible || lp_offspring.get_feasible()) {
          population.add_solution(std::move(lp_offspring));
        }
        if (!add_only_feasible || offspring.get_feasible()) {
          population.add_solution(std::move(offspring));
        }
        if (work_limit_reached()) { return; }
      }
    }
  }
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::recombine_and_ls_with_all(
  std::vector<solution_t<i_t, f_t>>& solutions, bool add_only_feasible)
{
  raft::common::nvtx::range fun_scope("recombine_and_ls_with_all");
  if (solutions.size() > 0) {
    CUOPT_LOG_DEBUG("Running recombiners on B&B solutions with size %lu", solutions.size());
    // add all solutions because time limit might have been consumed and we might have exited before
    for (auto& sol : solutions) {
      cuopt_func_call(sol.test_feasibility(true));
      population.add_solution(std::move(solution_t<i_t, f_t>(sol)));
    }
    for (auto& sol : solutions) {
      if (work_limit_reached()) { return; }
      solution_t<i_t, f_t> ls_solution(sol);
      ls_config_t<i_t, f_t> ls_config;
      run_local_search(ls_solution, population.weights, timer, ls_config);
      if (work_limit_reached()) { return; }
      // TODO try if running LP with integers fixed makes it feasible
      if (ls_solution.get_feasible()) {
        CUOPT_LOG_DEBUG("LS searched solution feasible, running recombiners!");
        recombine_and_ls_with_all(ls_solution, add_only_feasible);
      } else {
        CUOPT_LOG_DEBUG("Given solution feasible, running recombiners!");
        recombine_and_ls_with_all(sol, add_only_feasible);
      }
    }
  }
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::check_better_than_both(solution_t<i_t, f_t>& offspring,
                                                           solution_t<i_t, f_t>& sol1,
                                                           solution_t<i_t, f_t>& sol2)
{
  bool better_than_both = false;
  if (sol1.get_feasible() && sol2.get_feasible()) {
    better_than_both = offspring.get_objective() <
                       (std::min(sol1.get_objective(), sol2.get_objective()) - OBJECTIVE_EPSILON);
  } else if (sol1.get_feasible()) {
    better_than_both = offspring.get_objective() < (sol1.get_objective() - OBJECTIVE_EPSILON);
  } else if (sol2.get_feasible()) {
    better_than_both = offspring.get_objective() < (sol2.get_objective() - OBJECTIVE_EPSILON);
  } else {
    better_than_both = offspring.get_feasible();
  }
  if (offspring.get_feasible() && better_than_both) {
    context.settings.benchmark_info_ptr->last_improvement_after_recombination =
      timer.elapsed_time();
  }
}

template <typename i_t, typename f_t>
std::pair<solution_t<i_t, f_t>, solution_t<i_t, f_t>>
diversity_manager_t<i_t, f_t>::recombine_and_local_search(solution_t<i_t, f_t>& sol1,
                                                          solution_t<i_t, f_t>& sol2,
                                                          recombiner_enum_t recombiner_type)
{
  raft::common::nvtx::range fun_scope("recombine_and_local_search");
  CUOPT_LOG_DEBUG("Recombining sol cost:feas %f : %d and %f : %d",
                  sol1.get_quality(population.weights),
                  sol1.get_feasible(),
                  sol2.get_quality(population.weights),
                  sol2.get_feasible());
  double best_objective_of_parents  = std::min(sol1.get_objective(), sol2.get_objective());
  bool at_least_one_parent_feasible = sol1.get_feasible() || sol2.get_feasible();
  // randomly choose among 3 recombiners
  auto [offspring, success] = recombine(sol1, sol2, recombiner_type);
  if (!success) {
    // add the attempt
    mab_recombiner.add_mab_reward(mab_recombiner.last_chosen_option,
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::max(),
                                  recombiner_work_normalized_reward_t(0.0));
    return std::make_pair(solution_t<i_t, f_t>(sol1), solution_t<i_t, f_t>(sol2));
  }
  cuopt_assert(population.test_invariant(), "");
  cuopt_func_call(offspring.test_variable_bounds(false));
  CUOPT_LOG_DEBUG("Recombiner offspring sol cost:feas %f : %d",
                  offspring.get_quality(population.weights),
                  offspring.get_feasible());
  cuopt_assert(offspring.test_number_all_integer(), "All must be integers before LS");
  bool feasibility_before = offspring.get_feasible();
  ls_config_t<i_t, f_t> ls_config;
  ls_config.best_objective_of_parents    = best_objective_of_parents;
  ls_config.at_least_one_parent_feasible = at_least_one_parent_feasible;
  success = this->run_local_search(offspring, population.weights, timer, ls_config);
  if (!success) {
    // add the attempt
    mab_recombiner.add_mab_reward(mab_recombiner.last_chosen_option,
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::max(),
                                  recombiner_work_normalized_reward_t(0.0));
    return std::make_pair(solution_t<i_t, f_t>(sol1), solution_t<i_t, f_t>(sol2));
  }
  cuopt_assert(offspring.test_number_all_integer(), "All must be integers after LS");
  cuopt_assert(population.test_invariant(), "");
  offspring.compute_feasibility();
  CUOPT_LOG_DEBUG("After LS offspring sol cost:feas %f : %d",
                  offspring.get_quality(population.weights),
                  offspring.get_feasible());
  cuopt_assert(population.test_invariant(), "");
  // run LP with the vars
  solution_t<i_t, f_t> lp_offspring(offspring);
  cuopt_assert(population.test_invariant(), "");
  cuopt_assert(lp_offspring.test_number_all_integer(), "All must be integers before LP");
  f_t lp_run_time = offspring.get_feasible() ? diversity_config.lp_run_time_if_feasible
                                             : diversity_config.lp_run_time_if_infeasible;
  lp_run_time     = std::min(lp_run_time, timer.remaining_time());
  relaxed_lp_settings_t lp_settings;
  lp_settings.time_limit              = lp_run_time;
  lp_settings.work_limit              = lp_settings.time_limit;
  lp_settings.tolerance               = context.settings.tolerances.absolute_tolerance;
  lp_settings.return_first_feasible   = false;
  lp_settings.save_state              = true;
  lp_settings.per_constraint_residual = true;
  run_lp_with_vars_fixed(*lp_offspring.problem_ptr,
                         lp_offspring,
                         lp_offspring.problem_ptr->integer_indices,
                         lp_settings,
                         &ls.constraint_prop.bounds_update,
                         true /* check fixed assignment is feasible */,
                         true /* use integer fixed problem */);
  cuopt_assert(population.test_invariant(), "");
  cuopt_assert(lp_offspring.test_number_all_integer(), "All must be integers after LP");
  f_t lp_qual = lp_offspring.get_quality(population.weights);
  CUOPT_LOG_DEBUG("After LP offspring sol cost:feas %f : %d", lp_qual, lp_offspring.get_feasible());
  f_t offspring_qual = std::min(offspring.get_quality(population.weights), lp_qual);
  recombine_stats.update_improve_stats(
    offspring_qual, sol1.get_quality(population.weights), sol2.get_quality(population.weights));
  f_t best_quality_of_parents =
    std::min(sol1.get_quality(population.weights), sol2.get_quality(population.weights));
  mab_recombiner.add_mab_reward(
    mab_recombiner.last_chosen_option,
    best_quality_of_parents,
    population.best().get_quality(population.weights),
    offspring_qual,
    recombiner_work_normalized_reward_t(recombine_stats.get_last_recombiner_work()));
  mab_ls.add_mab_reward(mab_ls_config_t<i_t, f_t>::last_ls_mab_option,
                        best_quality_of_parents,
                        population.best_feasible().get_quality(population.weights),
                        offspring_qual,
                        ls_work_normalized_reward_t(mab_ls_config_t<i_t, f_t>::last_lm_config));
  if (context.settings.benchmark_info_ptr != nullptr) {
    check_better_than_both(offspring, sol1, sol2);
    check_better_than_both(lp_offspring, sol1, sol2);
  }
  return std::make_pair(std::move(offspring), std::move(lp_offspring));
}

template <typename i_t, typename f_t>
std::pair<solution_t<i_t, f_t>, bool> diversity_manager_t<i_t, f_t>::recombine(
  solution_t<i_t, f_t>& a, solution_t<i_t, f_t>& b, recombiner_enum_t recombiner_type)
{
  recombiner_enum_t recombiner;
  i_t selected_index = -1;
  if (run_only_ls_recombiner) {
    recombiner = recombiner_enum_t::LINE_SEGMENT;
  } else if (run_only_bp_recombiner) {
    recombiner = recombiner_enum_t::BOUND_PROP;
  } else if (run_only_fp_recombiner) {
    recombiner = recombiner_enum_t::FP;
  } else if (run_only_sub_mip_recombiner) {
    recombiner = recombiner_enum_t::SUB_MIP;
  } else {
    // only run the given recombiner unless it is defult
    if (recombiner_type == recombiner_enum_t::SIZE) {
      selected_index = mab_recombiner.select_mab_option();
      recombiner     = recombiner_t<i_t, f_t>::enabled_recombiners[selected_index];
    } else {
      recombiner = recombiner_type;
      auto it    = std::find(recombiner_t<i_t, f_t>::enabled_recombiners.begin(),
                          recombiner_t<i_t, f_t>::enabled_recombiners.end(),
                          recombiner_type);
      selected_index =
        static_cast<i_t>(std::distance(recombiner_t<i_t, f_t>::enabled_recombiners.begin(), it));
      if (it == recombiner_t<i_t, f_t>::enabled_recombiners.end()) {
        CUOPT_LOG_DEBUG("Recombiner not enabled; falling back to index 0");
        selected_index = 0;
      }
    }
  }
  mab_recombiner.set_last_chosen_option(selected_index);
  recombine_stats.add_attempt((recombiner_enum_t)recombiner);
  CUOPT_LOG_DEBUG("Recombining sol %x and %x with recombiner %d, weights %x",
                  a.get_hash(),
                  b.get_hash(),
                  recombiner,
                  population.weights.get_hash());
  // Refactored code using a switch statement
  switch (recombiner) {
    case recombiner_enum_t::BOUND_PROP: {
      auto [sol, success, work] = bound_prop_recombiner.recombine(a, b, population.weights);
      recombine_stats.set_recombiner_work(work);
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::FP: {
      auto [sol, success, work] = fp_recombiner.recombine(a, b, population.weights);
      recombine_stats.set_recombiner_work(work);
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::LINE_SEGMENT: {
      auto [sol, success, work] = line_segment_recombiner.recombine(a, b, population.weights);
      recombine_stats.set_recombiner_work(work);
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::SUB_MIP: {
      auto [sol, success, work] = sub_mip_recombiner.recombine(a, b, population.weights);
      recombine_stats.set_recombiner_work(work);
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::SIZE: {
      CUOPT_LOG_ERROR("Invalid or unhandled recombiner type: %d", recombiner);
      return std::make_pair(solution_t<i_t, f_t>(a), false);
    }
  }
  CUOPT_LOG_ERROR("Invalid or unhandled recombiner type: %d", recombiner);
  return std::make_pair(solution_t<i_t, f_t>(a), false);
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::set_simplex_solution(const std::vector<f_t>& solution,
                                                         const std::vector<f_t>& dual_solution,
                                                         f_t objective)
{
  CUOPT_LOG_DEBUG("Setting simplex solution with objective %f", objective);
  using sol_t = solution_t<i_t, f_t>;
  RAFT_CUDA_TRY(cudaSetDevice(context.handle_ptr->get_device()));
  context.handle_ptr->sync_stream();
  cuopt_func_call(sol_t new_sol(*problem_ptr));
  cuopt_assert(new_sol.assignment.size() == solution.size(), "Assignment size mismatch");
  cuopt_assert(problem_ptr->n_constraints == dual_solution.size(), "Dual assignment size mismatch");
  cuopt_func_call(new_sol.copy_new_assignment(solution));
  cuopt_func_call(new_sol.compute_feasibility());
  cuopt_assert(integer_equal(new_sol.get_user_objective(), objective, 1e-3), "Objective mismatch");
  std::lock_guard<std::mutex> lock(relaxed_solution_mutex);
  simplex_solution_exists.store(true, std::memory_order_release);
  global_concurrent_halt = 1;
  // global_concurrent_halt.store(1, std::memory_order_release);
  // it is safe to use lp_optimal_solution while executing the copy operation
  // the operations are ordered as long as they are on the same stream
  raft::copy(
    lp_optimal_solution.data(), solution.data(), solution.size(), context.handle_ptr->get_stream());
  raft::copy(lp_dual_optimal_solution.data(),
             dual_solution.data(),
             dual_solution.size(),
             context.handle_ptr->get_stream());
  set_new_user_bound(objective);
  context.handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::work_limit_reached()
{
  return timer.check_time_limit();
}

#if MIP_INSTANTIATE_FLOAT
template class diversity_manager_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class diversity_manager_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
