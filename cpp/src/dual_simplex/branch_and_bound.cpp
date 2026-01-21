/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/branch_and_bound.hpp>

#include <utilities/models/bounds_strengthening_predictor/header.h>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/crossover.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/logger.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/random.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/user_problem.hpp>
#include <raft/common/nvtx.hpp>
#include <utilities/hashing.hpp>
#include <utilities/work_unit_predictor.hpp>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <future>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

namespace {

static constexpr double FEATURE_LOG_INTERVAL = 0.25;  // Log at most every 500ms

template <typename f_t>
bool is_fractional(f_t x, variable_type_t var_type, f_t integer_tol)
{
  if (var_type == variable_type_t::CONTINUOUS) {
    return false;
  } else {
    f_t x_integer = std::round(x);
    return (std::abs(x_integer - x) > integer_tol);
  }
}

template <typename i_t, typename f_t>
i_t fractional_variables(const simplex_solver_settings_t<i_t, f_t>& settings,
                         const std::vector<f_t>& x,
                         const std::vector<variable_type_t>& var_types,
                         std::vector<i_t>& fractional)
{
  const i_t n = x.size();
  assert(x.size() == var_types.size());
  for (i_t j = 0; j < n; ++j) {
    if (is_fractional(x[j], var_types[j], settings.integer_tol)) { fractional.push_back(j); }
  }
  return fractional.size();
}

template <typename i_t, typename f_t>
void full_variable_types(const user_problem_t<i_t, f_t>& original_problem,
                         const lp_problem_t<i_t, f_t>& original_lp,
                         std::vector<variable_type_t>& var_types)
{
  var_types = original_problem.var_types;
  if (original_lp.num_cols > original_problem.num_cols) {
    var_types.resize(original_lp.num_cols);
    for (i_t k = original_problem.num_cols; k < original_lp.num_cols; k++) {
      var_types[k] = variable_type_t::CONTINUOUS;
    }
  }
}

template <typename i_t, typename f_t>
bool check_guess(const lp_problem_t<i_t, f_t>& original_lp,
                 const simplex_solver_settings_t<i_t, f_t>& settings,
                 const std::vector<variable_type_t>& var_types,
                 const std::vector<f_t>& guess,
                 f_t& primal_error,
                 f_t& bound_error,
                 i_t& num_fractional)
{
  bool feasible = false;
  std::vector<f_t> residual(original_lp.num_rows);
  residual = original_lp.rhs;
  matrix_vector_multiply(original_lp.A, 1.0, guess, -1.0, residual);
  primal_error           = vector_norm_inf<i_t, f_t>(residual);
  bound_error            = 0.0;
  constexpr bool verbose = false;
  for (i_t j = 0; j < original_lp.num_cols; j++) {
    // l_j <= x_j  infeas means x_j < l_j or l_j - x_j > 0
    const f_t low_bound_err = std::max(0.0, original_lp.lower[j] - guess[j]);
    // x_j <= u_j infeas means u_j < x_j or x_j - u_j > 0
    const f_t up_bound_err = std::max(0.0, guess[j] - original_lp.upper[j]);

    if (verbose && (low_bound_err > settings.primal_tol || up_bound_err > settings.primal_tol)) {
      settings.log.printf(
        "Bound error %d variable value %e. Low %e Upper %e. Low Error %e Up Error %e\n",
        j,
        guess[j],
        original_lp.lower[j],
        original_lp.upper[j],
        low_bound_err,
        up_bound_err);
    }
    bound_error = std::max(bound_error, std::max(low_bound_err, up_bound_err));
  }
  if (verbose) { settings.log.printf("Bounds infeasibility %e\n", bound_error); }
  std::vector<i_t> fractional;
  num_fractional = fractional_variables(settings, guess, var_types, fractional);
  if (verbose) { settings.log.printf("Fractional in solution %d\n", num_fractional); }
  if (bound_error < settings.primal_tol && primal_error < 2 * settings.primal_tol &&
      num_fractional == 0) {
    if (verbose) { settings.log.printf("Solution is feasible\n"); }
    feasible = true;
  }
  return feasible;
}

template <typename i_t, typename f_t>
void set_uninitialized_steepest_edge_norms(std::vector<f_t>& edge_norms)
{
  for (i_t j = 0; j < edge_norms.size(); ++j) {
    if (edge_norms[j] <= 0.0) { edge_norms[j] = 1e-4; }
  }
}

dual::status_t convert_lp_status_to_dual_status(lp_status_t status)
{
  if (status == lp_status_t::OPTIMAL) {
    return dual::status_t::OPTIMAL;
  } else if (status == lp_status_t::INFEASIBLE) {
    return dual::status_t::DUAL_UNBOUNDED;
  } else if (status == lp_status_t::ITERATION_LIMIT) {
    return dual::status_t::ITERATION_LIMIT;
  } else if (status == lp_status_t::TIME_LIMIT) {
    return dual::status_t::TIME_LIMIT;
  } else if (status == lp_status_t::WORK_LIMIT) {
    return dual::status_t::WORK_LIMIT;
  } else if (status == lp_status_t::NUMERICAL_ISSUES) {
    return dual::status_t::NUMERICAL;
  } else if (status == lp_status_t::CUTOFF) {
    return dual::status_t::CUTOFF;
  } else if (status == lp_status_t::CONCURRENT_LIMIT) {
    return dual::status_t::CONCURRENT_LIMIT;
  } else if (status == lp_status_t::UNSET) {
    return dual::status_t::UNSET;
  } else {
    return dual::status_t::NUMERICAL;
  }
}

template <typename f_t>
f_t sgn(f_t x)
{
  return x < 0 ? -1 : 1;
}

template <typename f_t>
f_t relative_gap(f_t obj_value, f_t lower_bound)
{
  f_t user_mip_gap = obj_value == 0.0
                       ? (lower_bound == 0.0 ? 0.0 : std::numeric_limits<f_t>::infinity())
                       : std::abs(obj_value - lower_bound) / std::abs(obj_value);
  if (std::isnan(user_mip_gap)) { return std::numeric_limits<f_t>::infinity(); }
  return user_mip_gap;
}

template <typename i_t, typename f_t>
f_t user_relative_gap(const lp_problem_t<i_t, f_t>& lp, f_t obj_value, f_t lower_bound)
{
  f_t user_obj         = compute_user_objective(lp, obj_value);
  f_t user_lower_bound = compute_user_objective(lp, lower_bound);
  f_t user_mip_gap     = user_obj == 0.0
                           ? (user_lower_bound == 0.0 ? 0.0 : std::numeric_limits<f_t>::infinity())
                           : std::abs(user_obj - user_lower_bound) / std::abs(user_obj);
  if (std::isnan(user_mip_gap)) { return std::numeric_limits<f_t>::infinity(); }
  return user_mip_gap;
}

template <typename f_t>
std::string user_mip_gap(f_t obj_value, f_t lower_bound)
{
  const f_t user_mip_gap = relative_gap(obj_value, lower_bound);
  if (user_mip_gap == std::numeric_limits<f_t>::infinity()) {
    return "   -  ";
  } else {
    constexpr int BUFFER_LEN = 32;
    char buffer[BUFFER_LEN];
    snprintf(buffer, BUFFER_LEN - 1, "%5.1f%%", user_mip_gap * 100);
    return std::string(buffer);
  }
}

#ifdef SHOW_DIVING_TYPE
inline char feasible_solution_symbol(bnb_worker_type_t type)
{
  switch (type) {
    case bnb_worker_type_t::BEST_FIRST: return 'B';
    case bnb_worker_type_t::COEFFICIENT_DIVING: return 'C';
    case bnb_worker_type_t::LINE_SEARCH_DIVING: return 'L';
    case bnb_worker_type_t::PSEUDOCOST_DIVING: return 'P';
    case bnb_worker_type_t::GUIDED_DIVING: return 'G';
    default: return 'U';
  }
}
#else
inline char feasible_solution_symbol(bnb_worker_type_t type)
{
  switch (type) {
    case bnb_worker_type_t::BEST_FIRST: return 'B';
    case bnb_worker_type_t::COEFFICIENT_DIVING: return 'D';
    case bnb_worker_type_t::LINE_SEARCH_DIVING: return 'D';
    case bnb_worker_type_t::PSEUDOCOST_DIVING: return 'D';
    case bnb_worker_type_t::GUIDED_DIVING: return 'D';
    default: return 'U';
  }
}
#endif

}  // namespace

template <typename i_t, typename f_t>
branch_and_bound_t<i_t, f_t>::branch_and_bound_t(
  const user_problem_t<i_t, f_t>& user_problem,
  const simplex_solver_settings_t<i_t, f_t>& solver_settings)
  : original_problem_(user_problem),
    settings_(solver_settings),
    original_lp_(user_problem.handle_ptr, 1, 1, 1),
    Arow_(1, 1, 0),
    incumbent_(1),
    root_relax_soln_(1, 1),
    root_crossover_soln_(1, 1),
    pc_(1),
    solver_status_(mip_status_t::UNSET),
    bsp_debug_settings_(bsp_debug_settings_t::from_environment())
{
  exploration_stats_.start_time = tic();
  dualize_info_t<i_t, f_t> dualize_info;
  convert_user_problem(original_problem_, settings_, original_lp_, new_slacks_, dualize_info);
  full_variable_types(original_problem_, original_lp_, var_types_);

  upper_bound_ = inf;
}

template <typename i_t, typename f_t>
f_t branch_and_bound_t<i_t, f_t>::get_lower_bound()
{
  f_t lower_bound      = lower_bound_ceiling_.load();
  f_t heap_lower_bound = node_queue_.get_lower_bound();
  lower_bound          = std::min(heap_lower_bound, lower_bound);

  for (i_t i = 0; i < local_lower_bounds_.size(); ++i) {
    lower_bound = std::min(local_lower_bounds_[i].load(), lower_bound);
  }

  return std::isfinite(lower_bound) ? lower_bound : -inf;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::report_heuristic(f_t obj)
{
  if (is_running) {
    f_t user_obj         = compute_user_objective(original_lp_, obj);
    f_t user_lower       = compute_user_objective(original_lp_, get_lower_bound());
    std::string user_gap = user_mip_gap<f_t>(user_obj, user_lower);

    settings_.log.printf(
      "H                            %+13.6e    %+10.6e                        %s %9.2f\n",
      user_obj,
      user_lower,
      user_gap.c_str(),
      toc(exploration_stats_.start_time));
  } else {
    settings_.log.printf("New solution from primal heuristics. Objective %+.6e. Time %.2f\n",
                         compute_user_objective(original_lp_, obj),
                         toc(exploration_stats_.start_time));
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::report(char symbol, f_t obj, f_t lower_bound, i_t node_depth)
{
  i_t nodes_explored   = exploration_stats_.nodes_explored;
  i_t nodes_unexplored = exploration_stats_.nodes_unexplored;
  f_t user_obj         = compute_user_objective(original_lp_, obj);
  f_t user_lower       = compute_user_objective(original_lp_, lower_bound);
  f_t iter_node        = exploration_stats_.total_lp_iters / nodes_explored;
  std::string user_gap = user_mip_gap<f_t>(user_obj, user_lower);
  settings_.log.printf("%c %10d   %10lu    %+13.6e    %+10.6e  %6d    %7.1e     %s %9.2f\n",
                       symbol,
                       nodes_explored,
                       nodes_unexplored,
                       user_obj,
                       user_lower,
                       node_depth,
                       iter_node,
                       user_gap.c_str(),
                       toc(exploration_stats_.start_time));
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::set_new_solution(const std::vector<f_t>& solution)
{
  if (solution.size() != original_problem_.num_cols) {
    settings_.log.printf(
      "Solution size mismatch %ld %d\n", solution.size(), original_problem_.num_cols);
  }
  std::vector<f_t> crushed_solution;
  crush_primal_solution<i_t, f_t>(
    original_problem_, original_lp_, solution, new_slacks_, crushed_solution);
  f_t obj             = compute_objective(original_lp_, crushed_solution);
  bool is_feasible    = false;
  bool attempt_repair = false;
  mutex_upper_.lock();
  if (obj < upper_bound_) {
    f_t primal_err;
    f_t bound_err;
    i_t num_fractional;
    is_feasible = check_guess(
      original_lp_, settings_, var_types_, crushed_solution, primal_err, bound_err, num_fractional);
    if (is_feasible) {
      upper_bound_ = obj;
      incumbent_.set_incumbent_solution(obj, crushed_solution);
    } else {
      attempt_repair         = true;
      constexpr bool verbose = false;
      if (verbose) {
        settings_.log.printf(
          "Injected solution infeasible. Constraint error %e bound error %e integer infeasible "
          "%d\n",
          primal_err,
          bound_err,
          num_fractional);
      }
    }
  }
  mutex_upper_.unlock();

  if (is_feasible) { report_heuristic(obj); }
  if (attempt_repair) {
    mutex_repair_.lock();
    repair_queue_.push_back(crushed_solution);
    mutex_repair_.unlock();
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::set_new_solution_deterministic(const std::vector<f_t>& solution,
                                                                  double work_unit_ts)
{
  // In BSP mode, queue the solution to be processed at the correct work unit timestamp
  // This ensures deterministic ordering of solution events

  if (solution.size() != original_problem_.num_cols) {
    settings_.log.printf(
      "Solution size mismatch %ld %d\n", solution.size(), original_problem_.num_cols);
    return;
  }

  std::vector<f_t> crushed_solution;
  crush_primal_solution<i_t, f_t>(
    original_problem_, original_lp_, solution, new_slacks_, crushed_solution);
  f_t obj = compute_objective(original_lp_, crushed_solution);

  // Validate solution before queueing
  f_t primal_err;
  f_t bound_err;
  i_t num_fractional;
  bool is_feasible = check_guess(
    original_lp_, settings_, var_types_, crushed_solution, primal_err, bound_err, num_fractional);

  if (!is_feasible) {
    // Queue for repair
    mutex_repair_.lock();
    repair_queue_.push_back(crushed_solution);
    mutex_repair_.unlock();
    return;
  }

  // Queue the solution with its work unit timestamp
  mutex_heuristic_queue_.lock();
  heuristic_solution_queue_.push_back({std::move(crushed_solution), obj, work_unit_ts});
  mutex_heuristic_queue_.unlock();
}

template <typename i_t, typename f_t>
bool branch_and_bound_t<i_t, f_t>::repair_solution(const std::vector<f_t>& edge_norms,
                                                   const std::vector<f_t>& potential_solution,
                                                   f_t& repaired_obj,
                                                   std::vector<f_t>& repaired_solution) const
{
  bool feasible = false;
  repaired_obj  = std::numeric_limits<f_t>::quiet_NaN();
  i_t n         = original_lp_.num_cols;
  assert(potential_solution.size() == n);

  lp_problem_t repair_lp = original_lp_;

  // Fix integer variables
  for (i_t j = 0; j < n; ++j) {
    if (var_types_[j] == variable_type_t::INTEGER) {
      const f_t fixed_val = std::round(potential_solution[j]);
      repair_lp.lower[j]  = fixed_val;
      repair_lp.upper[j]  = fixed_val;
    }
  }

  lp_solution_t<i_t, f_t> lp_solution(original_lp_.num_rows, original_lp_.num_cols);

  i_t iter                               = 0;
  f_t lp_start_time                      = tic();
  simplex_solver_settings_t lp_settings  = settings_;
  std::vector<variable_status_t> vstatus = root_vstatus_;
  lp_settings.set_log(false);
  lp_settings.inside_mip           = true;
  std::vector<f_t> leaf_edge_norms = edge_norms;
  // should probably set the cut off here lp_settings.cut_off
  dual::status_t lp_status = dual_phase2(
    2, 0, lp_start_time, repair_lp, lp_settings, vstatus, lp_solution, iter, leaf_edge_norms);
  repaired_solution = lp_solution.x;

  if (lp_status == dual::status_t::OPTIMAL) {
    f_t primal_error;
    f_t bound_error;
    i_t num_fractional;
    feasible               = check_guess(original_lp_,
                           settings_,
                           var_types_,
                           lp_solution.x,
                           primal_error,
                           bound_error,
                           num_fractional);
    repaired_obj           = compute_objective(original_lp_, repaired_solution);
    constexpr bool verbose = false;
    if (verbose) {
      settings_.log.printf(
        "After repair: feasible %d primal error %e bound error %e fractional %d. Objective %e\n",
        feasible,
        primal_error,
        bound_error,
        num_fractional,
        repaired_obj);
    }
  }

  return feasible;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::repair_heuristic_solutions()
{
  raft::common::nvtx::range scope("BB::repair_heuristics");
  // Check if there are any solutions to repair
  std::vector<std::vector<f_t>> to_repair;
  mutex_repair_.lock();
  if (repair_queue_.size() > 0) {
    to_repair = repair_queue_;
    repair_queue_.clear();
  }
  mutex_repair_.unlock();

  if (to_repair.size() > 0) {
    settings_.log.debug("Attempting to repair %ld injected solutions\n", to_repair.size());
    for (const std::vector<f_t>& potential_solution : to_repair) {
      std::vector<f_t> repaired_solution;
      f_t repaired_obj;
      bool is_feasible =
        repair_solution(edge_norms_, potential_solution, repaired_obj, repaired_solution);
      if (is_feasible) {
        mutex_upper_.lock();

        if (repaired_obj < upper_bound_) {
          upper_bound_ = repaired_obj;
          incumbent_.set_incumbent_solution(repaired_obj, repaired_solution);
          report_heuristic(repaired_obj);

          if (settings_.solution_callback != nullptr) {
            std::vector<f_t> original_x;
            uncrush_primal_solution(original_problem_, original_lp_, repaired_solution, original_x);
            settings_.solution_callback(original_x, repaired_obj);
          }
        }

        mutex_upper_.unlock();
      }
    }
  }
}

template <typename i_t, typename f_t>
mip_status_t branch_and_bound_t<i_t, f_t>::set_final_solution(mip_solution_t<i_t, f_t>& solution,
                                                              f_t lower_bound)
{
  if (solver_status_ == mip_status_t::NUMERICAL) {
    settings_.log.printf("Numerical issue encountered. Stopping the solver...\n");
  }

  if (solver_status_ == mip_status_t::TIME_LIMIT) {
    settings_.log.printf("Time limit reached. Stopping the solver...\n");
  }
  if (solver_status_ == mip_status_t::WORK_LIMIT) {
    settings_.log.printf("Work limit reached. Stopping the solver...\n");
  }
  if (solver_status_ == mip_status_t::NODE_LIMIT) {
    settings_.log.printf("Node limit reached. Stopping the solver...\n");
  }

  // Signal heuristic thread to stop for any limit-based termination
  if (solver_status_ == mip_status_t::TIME_LIMIT || solver_status_ == mip_status_t::WORK_LIMIT ||
      solver_status_ == mip_status_t::NODE_LIMIT || solver_status_ == mip_status_t::NUMERICAL) {
    if (settings_.heuristic_preemption_callback != nullptr) {
      settings_.heuristic_preemption_callback();
    }
  }

  f_t gap              = upper_bound_ - lower_bound;
  f_t obj              = compute_user_objective(original_lp_, upper_bound_.load());
  f_t user_bound       = compute_user_objective(original_lp_, lower_bound);
  f_t gap_rel          = user_relative_gap(original_lp_, upper_bound_.load(), lower_bound);
  bool is_maximization = original_lp_.obj_scale < 0.0;

  settings_.log.printf("Explored %d nodes in %.2fs.\n",
                       exploration_stats_.nodes_explored,
                       toc(exploration_stats_.start_time));
  settings_.log.printf("Absolute Gap %e Objective %.16e %s Bound %.16e\n",
                       gap,
                       obj,
                       is_maximization ? "Upper" : "Lower",
                       user_bound);

  if (gap <= settings_.absolute_mip_gap_tol || gap_rel <= settings_.relative_mip_gap_tol) {
    solver_status_ = mip_status_t::OPTIMAL;
    if (gap > 0 && gap <= settings_.absolute_mip_gap_tol) {
      settings_.log.printf("Optimal solution found within absolute MIP gap tolerance (%.1e)\n",
                           settings_.absolute_mip_gap_tol);
    } else if (gap > 0 && gap_rel <= settings_.relative_mip_gap_tol) {
      settings_.log.printf("Optimal solution found within relative MIP gap tolerance (%.1e)\n",
                           settings_.relative_mip_gap_tol);
    } else {
      settings_.log.printf("Optimal solution found.\n");
    }
    if (settings_.heuristic_preemption_callback != nullptr) {
      settings_.heuristic_preemption_callback();
    }
  }

  if (solver_status_ == mip_status_t::UNSET) {
    if (exploration_stats_.nodes_explored > 0 && exploration_stats_.nodes_unexplored == 0 &&
        upper_bound_ == inf) {
      settings_.log.printf("Integer infeasible.\n");
      solver_status_ = mip_status_t::INFEASIBLE;
      if (settings_.heuristic_preemption_callback != nullptr) {
        settings_.heuristic_preemption_callback();
      }
    }
  }

  if (upper_bound_ != inf) {
    assert(incumbent_.has_incumbent);
    uncrush_primal_solution(original_problem_, original_lp_, incumbent_.x, solution.x);
  }
  solution.objective          = incumbent_.objective;
  solution.lower_bound        = lower_bound;
  solution.nodes_explored     = exploration_stats_.nodes_explored;
  solution.simplex_iterations = exploration_stats_.total_lp_iters;
  return solver_status_;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::add_feasible_solution(f_t leaf_objective,
                                                         const std::vector<f_t>& leaf_solution,
                                                         i_t leaf_depth,
                                                         bnb_worker_type_t thread_type)
{
  bool send_solution      = false;
  bool improved_incumbent = false;
  i_t nodes_explored      = exploration_stats_.nodes_explored;
  i_t nodes_unexplored    = exploration_stats_.nodes_unexplored;

  settings_.log.debug("%c found a feasible solution with obj=%.10e.\n",
                      feasible_solution_symbol(thread_type),
                      compute_user_objective(original_lp_, leaf_objective));

  mutex_upper_.lock();
  if (leaf_objective < upper_bound_) {
    incumbent_.set_incumbent_solution(leaf_objective, leaf_solution);
    upper_bound_ = leaf_objective;
    report(feasible_solution_symbol(thread_type), leaf_objective, get_lower_bound(), leaf_depth);
    send_solution      = true;
    improved_incumbent = true;
  }

  if (send_solution && settings_.solution_callback != nullptr) {
    std::vector<f_t> original_x;
    uncrush_primal_solution(original_problem_, original_lp_, incumbent_.x, original_x);
    settings_.solution_callback(original_x, upper_bound_);
  }
  mutex_upper_.unlock();
}

// Martin's criteria for the preferred rounding direction (see [1])
// [1] A. Martin, “Integer Programs with Block Structure,”
// Technische Universit¨at Berlin, Berlin, 1999. Accessed: Aug. 08, 2025.
// [Online]. Available: https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/391
template <typename f_t>
rounding_direction_t martin_criteria(f_t val, f_t root_val)
{
  const f_t down_val  = std::floor(root_val);
  const f_t up_val    = std::ceil(root_val);
  const f_t down_dist = val - down_val;
  const f_t up_dist   = up_val - val;
  constexpr f_t eps   = 1e-6;

  if (down_dist < up_dist + eps) {
    return rounding_direction_t::DOWN;

  } else {
    return rounding_direction_t::UP;
  }
}

template <typename i_t, typename f_t>
branch_variable_t<i_t> branch_and_bound_t<i_t, f_t>::variable_selection(
  mip_node_t<i_t, f_t>* node_ptr,
  const std::vector<i_t>& fractional,
  const std::vector<f_t>& solution,
  bnb_worker_type_t type)
{
  logger_t log;
  log.log                        = false;
  i_t branch_var                 = -1;
  rounding_direction_t round_dir = rounding_direction_t::NONE;
  std::vector<f_t> current_incumbent;

  // If there is no incumbent, use pseudocost diving instead of guided diving
  if (upper_bound_ == inf && type == bnb_worker_type_t::GUIDED_DIVING) {
    type = bnb_worker_type_t::PSEUDOCOST_DIVING;
  }

  switch (type) {
    case bnb_worker_type_t::BEST_FIRST:
      branch_var = pc_.variable_selection(fractional, solution, log);
      round_dir  = martin_criteria(solution[branch_var], root_relax_soln_.x[branch_var]);
      return {branch_var, round_dir};

    case bnb_worker_type_t::COEFFICIENT_DIVING:
      return coefficient_diving(
        original_lp_, fractional, solution, var_up_locks_, var_down_locks_, log);

    case bnb_worker_type_t::LINE_SEARCH_DIVING:
      return line_search_diving(fractional, solution, root_relax_soln_.x, log);

    case bnb_worker_type_t::PSEUDOCOST_DIVING:
      return pseudocost_diving(pc_, fractional, solution, root_relax_soln_.x, log);

    case bnb_worker_type_t::GUIDED_DIVING:
      mutex_upper_.lock();
      current_incumbent = incumbent_.x;
      mutex_upper_.unlock();
      return guided_diving(pc_, fractional, solution, current_incumbent, log);

    default:
      log.debug("Unknown variable selection method: %d\n", type);
      return {-1, rounding_direction_t::NONE};
  }
}

template <typename i_t, typename f_t>
dual::status_t branch_and_bound_t<i_t, f_t>::solve_node_lp(
  mip_node_t<i_t, f_t>* node_ptr,
  lp_problem_t<i_t, f_t>& leaf_problem,
  lp_solution_t<i_t, f_t>& leaf_solution,
  basis_update_mpf_t<i_t, f_t>& basis_factors,
  std::vector<i_t>& basic_list,
  std::vector<i_t>& nonbasic_list,
  bounds_strengthening_t<i_t, f_t>& node_presolver,
  bnb_worker_type_t thread_type,
  bool recompute_bounds_and_basis,
  const std::vector<f_t>& root_lower,
  const std::vector<f_t>& root_upper,
  bnb_stats_t<i_t, f_t>& stats,
  logger_t& log)
{
  raft::common::nvtx::range scope("BB::solve_node");

  std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
  assert(leaf_vstatus.size() == leaf_problem.num_cols);

  simplex_solver_settings_t lp_settings = settings_;
  lp_settings.set_log(false);
  lp_settings.cut_off       = upper_bound_ + settings_.dual_tol;
  lp_settings.inside_mip    = 2;
  lp_settings.time_limit    = settings_.time_limit - toc(exploration_stats_.start_time);
  lp_settings.scale_columns = false;

  if (thread_type != bnb_worker_type_t::BEST_FIRST) {
    i_t bnb_lp_iters            = exploration_stats_.total_lp_iters;
    f_t factor                  = settings_.diving_settings.iteration_limit_factor;
    i_t max_iter                = factor * bnb_lp_iters;
    lp_settings.iteration_limit = max_iter - stats.total_lp_iters;
    if (lp_settings.iteration_limit <= 0) { return dual::status_t::ITERATION_LIMIT; }
  }

#ifdef LOG_NODE_SIMPLEX
  lp_settings.set_log(true);
  std::stringstream ss;
  ss << "simplex-" << std::this_thread::get_id() << ".log";
  std::string logname;
  ss >> logname;
  lp_settings.log.set_log_file(logname, "a");
  lp_settings.log.log_to_console = false;
  lp_settings.log.printf(
    "%scurrent node: id = %d, depth = %d, branch var = %d, branch dir = %s, fractional val = "
    "%f, variable lower bound = %f, variable upper bound = %f, branch vstatus = %d\n\n",
    settings_.log.log_prefix.c_str(),
    node_ptr->node_id,
    node_ptr->depth,
    node_ptr->branch_var,
    node_ptr->branch_dir == rounding_direction_t::DOWN ? "DOWN" : "UP",
    node_ptr->fractional_val,
    node_ptr->branch_var_lower,
    node_ptr->branch_var_upper,
    node_ptr->vstatus[node_ptr->branch_var]);
#endif

  // Reset the bound_changed markers
  std::fill(node_presolver.bounds_changed.begin(), node_presolver.bounds_changed.end(), false);

  // Set the correct bounds for the leaf problem
  if (recompute_bounds_and_basis) {
    leaf_problem.lower = root_lower;
    leaf_problem.upper = root_upper;
    node_ptr->get_variable_bounds(
      leaf_problem.lower, leaf_problem.upper, node_presolver.bounds_changed);

  } else {
    node_ptr->update_branched_variable_bounds(
      leaf_problem.lower, leaf_problem.upper, node_presolver.bounds_changed);
  }

  bool feasible;
  {
    raft::common::nvtx::range scope_bs("BB::bound_strengthening");
    f_t bs_start = tic();
    feasible =
      node_presolver.bounds_strengthening(leaf_problem.lower, leaf_problem.upper, lp_settings);
    f_t bs_runtime = toc(bs_start);

    bs_features_.m             = leaf_problem.num_rows;
    bs_features_.n             = leaf_problem.num_cols;
    bs_features_.nnz           = leaf_problem.A.col_start[leaf_problem.num_cols];
    bs_features_.nnz_processed = node_presolver.last_nnz_processed;
    bs_features_.runtime       = bs_runtime;
    bs_features_.log_single(bs_features_.m, bs_features_.n, bs_features_.nnz);
  }

  dual::status_t lp_status = dual::status_t::DUAL_UNBOUNDED;

  if (feasible) {
    i_t node_iter                    = 0;
    f_t lp_start_time                = tic();
    std::vector<f_t> leaf_edge_norms = edge_norms_;  // = node.steepest_edge_norms;

    {
      raft::common::nvtx::range scope_lp("BB::node_lp_solve");
      lp_status = dual_phase2_with_advanced_basis(2,
                                                  0,
                                                  recompute_bounds_and_basis,
                                                  lp_start_time,
                                                  leaf_problem,
                                                  lp_settings,
                                                  leaf_vstatus,
                                                  basis_factors,
                                                  basic_list,
                                                  nonbasic_list,
                                                  leaf_solution,
                                                  node_iter,
                                                  leaf_edge_norms,
                                                  nullptr);
    }

    if (lp_status == dual::status_t::NUMERICAL) {
      log.printf("Numerical issue node %d. Resolving from scratch.\n", node_ptr->node_id);
      lp_status_t second_status = solve_linear_program_with_advanced_basis(leaf_problem,
                                                                           lp_start_time,
                                                                           lp_settings,
                                                                           leaf_solution,
                                                                           basis_factors,
                                                                           basic_list,
                                                                           nonbasic_list,
                                                                           leaf_vstatus,
                                                                           leaf_edge_norms);

      lp_status = convert_lp_status_to_dual_status(second_status);
    }

    stats.total_lp_solve_time += toc(lp_start_time);
    stats.total_lp_iters += node_iter;
  }

#ifdef LOG_NODE_SIMPLEX
  lp_settings.log.printf("\nLP status: %d\n\n", lp_status);
#endif

  return lp_status;
}

template <typename i_t, typename f_t>
std::pair<node_status_t, rounding_direction_t> branch_and_bound_t<i_t, f_t>::update_tree(
  mip_node_t<i_t, f_t>* node_ptr,
  search_tree_t<i_t, f_t>& search_tree,
  lp_problem_t<i_t, f_t>& leaf_problem,
  lp_solution_t<i_t, f_t>& leaf_solution,
  bnb_worker_type_t thread_type,
  dual::status_t lp_status,
  logger_t& log)
{
  const f_t abs_fathom_tol                     = settings_.absolute_mip_gap_tol / 10;
  std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;

  if (lp_status == dual::status_t::DUAL_UNBOUNDED) {
    // Node was infeasible. Do not branch
    node_ptr->lower_bound = inf;
    search_tree.graphviz_node(log, node_ptr, "infeasible", 0.0);
    search_tree.update(node_ptr, node_status_t::INFEASIBLE);
    return {node_status_t::INFEASIBLE, rounding_direction_t::NONE};

  } else if (lp_status == dual::status_t::CUTOFF) {
    // Node was cut off. Do not branch
    node_ptr->lower_bound = upper_bound_;
    f_t leaf_objective    = compute_objective(leaf_problem, leaf_solution.x);
    search_tree.graphviz_node(log, node_ptr, "cut off", leaf_objective);
    search_tree.update(node_ptr, node_status_t::FATHOMED);
    return {node_status_t::FATHOMED, rounding_direction_t::NONE};

  } else if (lp_status == dual::status_t::OPTIMAL) {
    // LP was feasible
    std::vector<i_t> leaf_fractional;
    i_t leaf_num_fractional =
      fractional_variables(settings_, leaf_solution.x, var_types_, leaf_fractional);

    f_t leaf_objective    = compute_objective(leaf_problem, leaf_solution.x);
    node_ptr->lower_bound = leaf_objective;
    search_tree.graphviz_node(log, node_ptr, "lower bound", leaf_objective);
    pc_.update_pseudo_costs(node_ptr, leaf_objective);

    if (thread_type == bnb_worker_type_t::BEST_FIRST) {
      if (settings_.node_processed_callback != nullptr) {
        std::vector<f_t> original_x;
        uncrush_primal_solution(original_problem_, original_lp_, leaf_solution.x, original_x);
        settings_.node_processed_callback(original_x, leaf_objective);
      }
    }

    if (leaf_num_fractional == 0) {
      // Found a integer feasible solution
      add_feasible_solution(leaf_objective, leaf_solution.x, node_ptr->depth, thread_type);
      search_tree.graphviz_node(log, node_ptr, "integer feasible", leaf_objective);
      search_tree.update(node_ptr, node_status_t::INTEGER_FEASIBLE);
      return {node_status_t::INTEGER_FEASIBLE, rounding_direction_t::NONE};

    } else if (leaf_objective <= upper_bound_ + abs_fathom_tol) {
      // Choose fractional variable to branch on
      auto [branch_var, round_dir] =
        variable_selection(node_ptr, leaf_fractional, leaf_solution.x, thread_type);

      assert(leaf_vstatus.size() == leaf_problem.num_cols);
      assert(branch_var >= 0);
      assert(round_dir != rounding_direction_t::NONE);

      // Note that the exploration thread is the only one that can insert new nodes into the heap,
      // and thus, we only need to calculate the objective estimate here (it is used for
      // sorting the nodes for diving).
      if (thread_type == bnb_worker_type_t::BEST_FIRST) {
        logger_t pc_log;
        pc_log.log = false;
        node_ptr->objective_estimate =
          pc_.obj_estimate(leaf_fractional, leaf_solution.x, node_ptr->lower_bound, pc_log);
      }

      search_tree.branch(
        node_ptr, branch_var, leaf_solution.x[branch_var], leaf_vstatus, leaf_problem, log);
      search_tree.update(node_ptr, node_status_t::HAS_CHILDREN);
      return {node_status_t::HAS_CHILDREN, round_dir};

    } else {
      search_tree.graphviz_node(log, node_ptr, "fathomed", leaf_objective);
      search_tree.update(node_ptr, node_status_t::FATHOMED);
      return {node_status_t::FATHOMED, rounding_direction_t::NONE};
    }
  } else if (lp_status == dual::status_t::TIME_LIMIT) {
    search_tree.graphviz_node(log, node_ptr, "timeout", 0.0);
    return {node_status_t::PENDING, rounding_direction_t::NONE};
  } else if (lp_status == dual::status_t::WORK_LIMIT) {
    search_tree.graphviz_node(log, node_ptr, "work limit", 0.0);
    return {node_status_t::PENDING, rounding_direction_t::NONE};
  } else {
    if (thread_type == bnb_worker_type_t::BEST_FIRST) {
      fetch_min(lower_bound_ceiling_, node_ptr->lower_bound);
      log.printf(
        "LP returned status %d on node %d. This indicates a numerical issue. The best bound is set "
        "to "
        "%+10.6e.\n",
        lp_status,
        node_ptr->node_id,
        compute_user_objective(original_lp_, lower_bound_ceiling_.load()));
    }

    search_tree.graphviz_node(log, node_ptr, "numerical", 0.0);
    search_tree.update(node_ptr, node_status_t::NUMERICAL);
    return {node_status_t::NUMERICAL, rounding_direction_t::NONE};
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::exploration_ramp_up(mip_node_t<i_t, f_t>* node,
                                                       i_t initial_heap_size)
{
  if (solver_status_ != mip_status_t::UNSET) { return; }

  // Note that we do not know which thread will execute the
  // `exploration_ramp_up` task, so we allow to any thread
  // to repair the heuristic solution.
  repair_heuristic_solutions();

  f_t lower_bound = node->lower_bound;
  f_t upper_bound = upper_bound_;
  f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);
  f_t abs_gap     = upper_bound - lower_bound;

  if (lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
    search_tree_.graphviz_node(settings_.log, node, "cutoff", node->lower_bound);
    search_tree_.update(node, node_status_t::FATHOMED);
    --exploration_stats_.nodes_unexplored;
    return;
  }

  f_t now = toc(exploration_stats_.start_time);
  f_t time_since_last_log =
    exploration_stats_.last_log == 0 ? 1.0 : toc(exploration_stats_.last_log);

  if (((exploration_stats_.nodes_since_last_log >= 10 ||
        abs_gap < 10 * settings_.absolute_mip_gap_tol) &&
       (time_since_last_log >= 1)) ||
      (time_since_last_log > 30) || now > settings_.time_limit) {
    bool should_report = should_report_.exchange(false);

    if (should_report) {
      report(' ', upper_bound, root_objective_, node->depth);
      exploration_stats_.nodes_since_last_log = 0;
      exploration_stats_.last_log             = tic();
      should_report_                          = true;
    }
  }

  if (now > settings_.time_limit) {
    solver_status_ = mip_status_t::TIME_LIMIT;
    return;
  }

  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow_, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  lp_solution_t<i_t, f_t> leaf_solution(leaf_problem.num_rows, leaf_problem.num_cols);
  dual::status_t lp_status = solve_node_lp(node,
                                           leaf_problem,
                                           leaf_solution,
                                           basis_factors,
                                           basic_list,
                                           nonbasic_list,
                                           node_presolver,
                                           bnb_worker_type_t::BEST_FIRST,
                                           true,
                                           original_lp_.lower,
                                           original_lp_.upper,
                                           exploration_stats_,
                                           settings_.log);
  if (lp_status == dual::status_t::TIME_LIMIT) {
    solver_status_ = mip_status_t::TIME_LIMIT;
    return;
  }

  ++exploration_stats_.nodes_since_last_log;
  ++exploration_stats_.nodes_explored;
  --exploration_stats_.nodes_unexplored;

  auto [node_status, round_dir] = update_tree(node,
                                              search_tree_,
                                              leaf_problem,
                                              leaf_solution,
                                              bnb_worker_type_t::BEST_FIRST,
                                              lp_status,
                                              settings_.log);

  if (node_status == node_status_t::HAS_CHILDREN) {
    exploration_stats_.nodes_unexplored += 2;

    // If we haven't generated enough nodes to keep the threads busy, continue the ramp up phase
    if (exploration_stats_.nodes_unexplored < initial_heap_size) {
#pragma omp task
      exploration_ramp_up(node->get_down_child(), initial_heap_size);

#pragma omp task
      exploration_ramp_up(node->get_up_child(), initial_heap_size);

    } else {
      // We've generated enough nodes, push further nodes onto the heap
      node_queue_.push(node->get_down_child());
      node_queue_.push(node->get_up_child());
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::plunge_from(i_t task_id,
                                               mip_node_t<i_t, f_t>* start_node,
                                               lp_problem_t<i_t, f_t>& leaf_problem,
                                               bounds_strengthening_t<i_t, f_t>& node_presolver,
                                               basis_update_mpf_t<i_t, f_t>& basis_factors,
                                               std::vector<i_t>& basic_list,
                                               std::vector<i_t>& nonbasic_list)
{
  raft::common::nvtx::range scope("BB::explore_subtree");
  bool recompute_bounds_and_basis = true;
  std::deque<mip_node_t<i_t, f_t>*> stack;
  stack.push_front(start_node);

  while (stack.size() > 0 && solver_status_ == mip_status_t::UNSET && is_running) {
    if (task_id == 0) { repair_heuristic_solutions(); }

    mip_node_t<i_t, f_t>* node_ptr = stack.front();
    stack.pop_front();

    f_t lower_bound = node_ptr->lower_bound;
    f_t upper_bound = upper_bound_;
    f_t abs_gap     = upper_bound - lower_bound;
    f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);

    // This is based on three assumptions:
    // - The stack only contains sibling nodes, i.e., the current node and it sibling (if
    // applicable)
    // - The current node and its siblings uses the lower bound of the parent before solving the LP
    // relaxation
    // - The lower bound of the parent is lower or equal to its children
    assert(task_id < local_lower_bounds_.size());
    local_lower_bounds_[task_id] = lower_bound;

    if (lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
      search_tree_.graphviz_node(settings_.log, node_ptr, "cutoff", node_ptr->lower_bound);
      search_tree_.update(node_ptr, node_status_t::FATHOMED);
      recompute_bounds_and_basis = true;
      --exploration_stats_.nodes_unexplored;
      continue;
    }

    f_t now = toc(exploration_stats_.start_time);

    if (task_id == 0) {
      f_t time_since_last_log =
        exploration_stats_.last_log == 0 ? 1.0 : toc(exploration_stats_.last_log);

      if (((exploration_stats_.nodes_since_last_log >= 1000 ||
            abs_gap < 10 * settings_.absolute_mip_gap_tol) &&
           time_since_last_log >= 1) ||
          (time_since_last_log > 30) || now > settings_.time_limit) {
        report(' ', upper_bound, get_lower_bound(), node_ptr->depth);
        exploration_stats_.last_log             = tic();
        exploration_stats_.nodes_since_last_log = 0;
      }
    }

    if (now > settings_.time_limit) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    }
    if (exploration_stats_.nodes_explored >= settings_.node_limit) {
      solver_status_ = mip_status_t::NODE_LIMIT;
      break;
    }

    lp_solution_t<i_t, f_t> leaf_solution(leaf_problem.num_rows, leaf_problem.num_cols);
    dual::status_t lp_status = solve_node_lp(node_ptr,
                                             leaf_problem,
                                             leaf_solution,
                                             basis_factors,
                                             basic_list,
                                             nonbasic_list,
                                             node_presolver,
                                             bnb_worker_type_t::BEST_FIRST,
                                             recompute_bounds_and_basis,
                                             original_lp_.lower,
                                             original_lp_.upper,
                                             exploration_stats_,
                                             settings_.log);

    if (lp_status == dual::status_t::TIME_LIMIT) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    } else if (lp_status == dual::status_t::ITERATION_LIMIT) {
      break;
    }

    ++exploration_stats_.nodes_since_last_log;
    ++exploration_stats_.nodes_explored;
    --exploration_stats_.nodes_unexplored;

    auto [node_status, round_dir] = update_tree(node_ptr,
                                                search_tree_,
                                                leaf_problem,
                                                leaf_solution,
                                                bnb_worker_type_t::BEST_FIRST,
                                                lp_status,
                                                settings_.log);

    recompute_bounds_and_basis = node_status != node_status_t::HAS_CHILDREN;

    if (node_status == node_status_t::HAS_CHILDREN) {
      // The stack should only contain the children of the current parent.
      // If the stack size is greater than 0,
      // we pop the current node from the stack and place it in the global heap,
      // since we are about to add the two children to the stack
      if (stack.size() > 0) {
        mip_node_t<i_t, f_t>* node = stack.back();
        stack.pop_back();
        node_queue_.push(node);
      }

      exploration_stats_.nodes_unexplored += 2;

      if (round_dir == rounding_direction_t::UP) {
        stack.push_front(node_ptr->get_down_child());
        stack.push_front(node_ptr->get_up_child());
      } else {
        stack.push_front(node_ptr->get_up_child());
        stack.push_front(node_ptr->get_down_child());
      }
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::best_first_thread(i_t task_id)
{
  raft::common::nvtx::range scope("BB::best_first_thread");
  f_t lower_bound = -inf;
  f_t abs_gap     = inf;
  f_t rel_gap     = inf;

  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow_, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  while (solver_status_ == mip_status_t::UNSET && abs_gap > settings_.absolute_mip_gap_tol &&
         rel_gap > settings_.relative_mip_gap_tol &&
         (active_subtrees_ > 0 || node_queue_.best_first_queue_size() > 0)) {
    // In the current implementation, we are use the active number of subtree to decide
    // when to stop the execution. We need to increment the counter at the same
    // time as we pop a node from the queue to avoid some threads exiting
    // the main loop thinking that the solver has already finished.
    // This will be not needed in the master-worker model.
    node_queue_.lock();
    // If there any node left in the heap, we pop the top node and explore it.
    std::optional<mip_node_t<i_t, f_t>*> start_node = node_queue_.pop_best_first();
    if (start_node.has_value()) { active_subtrees_++; };
    node_queue_.unlock();

    if (start_node.has_value()) {
      if (upper_bound_ < start_node.value()->lower_bound) {
        // This node was put on the heap earlier but its lower bound is now greater than the
        // current upper bound
        search_tree_.graphviz_node(
          settings_.log, start_node.value(), "cutoff", start_node.value()->lower_bound);
        search_tree_.update(start_node.value(), node_status_t::FATHOMED);
        active_subtrees_--;
        continue;
      }

      // Best-first search with plunging
      plunge_from(task_id,
                  start_node.value(),
                  leaf_problem,
                  node_presolver,
                  basis_factors,
                  basic_list,
                  nonbasic_list);

      active_subtrees_--;
    }

    lower_bound = get_lower_bound();
    abs_gap     = upper_bound_ - lower_bound;
    rel_gap     = user_relative_gap(original_lp_, upper_bound_.load(), lower_bound);
  }

  is_running = false;

  // Check if it is the last thread that exited the loop and no
  // timeout or numerical error has happen.
  if (solver_status_ == mip_status_t::UNSET) {
    if (active_subtrees_ > 0) { local_lower_bounds_[task_id] = inf; }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::dive_from(mip_node_t<i_t, f_t>& start_node,
                                             const std::vector<f_t>& start_lower,
                                             const std::vector<f_t>& start_upper,
                                             lp_problem_t<i_t, f_t>& leaf_problem,
                                             bounds_strengthening_t<i_t, f_t>& node_presolver,
                                             basis_update_mpf_t<i_t, f_t>& basis_factors,
                                             std::vector<i_t>& basic_list,
                                             std::vector<i_t>& nonbasic_list,
                                             bnb_worker_type_t diving_type)
{
  raft::common::nvtx::range scope("BB::diving_thread");
  logger_t log;
  log.log = false;

  const i_t diving_node_limit      = settings_.diving_settings.node_limit;
  const i_t diving_backtrack_limit = settings_.diving_settings.backtrack_limit;
  bool recompute_bounds_and_basis  = true;
  search_tree_t<i_t, f_t> dive_tree(std::move(start_node));
  std::deque<mip_node_t<i_t, f_t>*> stack;
  stack.push_front(&dive_tree.root);

  bnb_stats_t<i_t, f_t> dive_stats;
  dive_stats.total_lp_iters      = 0;
  dive_stats.total_lp_solve_time = 0;
  dive_stats.nodes_explored      = 0;
  dive_stats.nodes_unexplored    = 1;

  while (stack.size() > 0 && solver_status_ == mip_status_t::UNSET && is_running) {
    mip_node_t<i_t, f_t>* node_ptr = stack.front();
    stack.pop_front();

    f_t lower_bound = node_ptr->lower_bound;
    f_t upper_bound = upper_bound_;
    f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);

    if (node_ptr->lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
      recompute_bounds_and_basis = true;
      continue;
    }

    if (toc(exploration_stats_.start_time) > settings_.time_limit) { break; }
    if (dive_stats.nodes_explored > diving_node_limit) { break; }

    lp_solution_t<i_t, f_t> leaf_solution(leaf_problem.num_rows, leaf_problem.num_cols);
    dual::status_t lp_status = solve_node_lp(node_ptr,
                                             leaf_problem,
                                             leaf_solution,
                                             basis_factors,
                                             basic_list,
                                             nonbasic_list,
                                             node_presolver,
                                             diving_type,
                                             recompute_bounds_and_basis,
                                             start_lower,
                                             start_upper,
                                             dive_stats,
                                             log);

    if (lp_status == dual::status_t::TIME_LIMIT) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    } else if (lp_status == dual::status_t::ITERATION_LIMIT) {
      break;
    }

    ++dive_stats.nodes_explored;

    auto [node_status, round_dir] =
      update_tree(node_ptr, dive_tree, leaf_problem, leaf_solution, diving_type, lp_status, log);
    recompute_bounds_and_basis = node_status != node_status_t::HAS_CHILDREN;

    if (node_status == node_status_t::HAS_CHILDREN) {
      if (round_dir == rounding_direction_t::UP) {
        stack.push_front(node_ptr->get_down_child());
        stack.push_front(node_ptr->get_up_child());
      } else {
        stack.push_front(node_ptr->get_up_child());
        stack.push_front(node_ptr->get_down_child());
      }
    }

    // Remove nodes that we no longer can backtrack to (i.e., from the current node, we can only
    // backtrack to a node that is has a depth of at most 5 levels lower than the current node).
    if (stack.size() > 1 && stack.front()->depth - stack.back()->depth > diving_backtrack_limit) {
      stack.pop_back();
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::diving_thread(bnb_worker_type_t diving_type)
{
  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow_, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  std::vector<f_t> start_lower;
  std::vector<f_t> start_upper;
  bool reset_starting_bounds = true;

  while (solver_status_ == mip_status_t::UNSET && is_running &&
         (active_subtrees_ > 0 || node_queue_.best_first_queue_size() > 0)) {
    if (reset_starting_bounds) {
      start_lower = original_lp_.lower;
      start_upper = original_lp_.upper;
      std::fill(node_presolver.bounds_changed.begin(), node_presolver.bounds_changed.end(), false);
      reset_starting_bounds = false;
    }

    // In the current implementation, multiple threads can pop the nodes
    // from the queue, so we need to initialize the lower and upper bound here
    // to avoid other thread fathoming the node (i.e., deleting) before we can read
    // the variable bounds from the tree.
    // This will be not needed in the master-worker model.
    node_queue_.lock();
    std::optional<mip_node_t<i_t, f_t>*> node_ptr  = node_queue_.pop_diving();
    std::optional<mip_node_t<i_t, f_t>> start_node = std::nullopt;

    if (node_ptr.has_value()) {
      node_ptr.value()->get_variable_bounds(
        start_lower, start_upper, node_presolver.bounds_changed);
      start_node = node_ptr.value()->detach_copy();
    }
    node_queue_.unlock();

    if (start_node.has_value()) {
      reset_starting_bounds = true;

      if (upper_bound_ < start_node->lower_bound) { continue; }
      bool is_feasible = node_presolver.bounds_strengthening(start_lower, start_upper, settings_);
      if (!is_feasible) { continue; }

      dive_from(start_node.value(),
                start_lower,
                start_upper,
                leaf_problem,
                node_presolver,
                basis_factors,
                basic_list,
                nonbasic_list,
                diving_type);
    }
  }
}

template <typename i_t, typename f_t>
lp_status_t branch_and_bound_t<i_t, f_t>::solve_root_relaxation(
  simplex_solver_settings_t<i_t, f_t> const& lp_settings)
{
  // Root node path
  lp_status_t root_status;
  std::future<lp_status_t> root_status_future;
  root_status_future = std::async(std::launch::async,
                                  &solve_linear_program_advanced<i_t, f_t>,
                                  std::ref(original_lp_),
                                  exploration_stats_.start_time,
                                  std::ref(lp_settings),
                                  std::ref(root_relax_soln_),
                                  std::ref(root_vstatus_),
                                  std::ref(edge_norms_));
  // Wait for the root relaxation solution to be sent by the diversity manager or dual simplex
  // to finish
  while (!root_crossover_solution_set_.load(std::memory_order_acquire) &&
         *get_root_concurrent_halt() == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    continue;
  }

  if (root_crossover_solution_set_.load(std::memory_order_acquire)) {
    // Crush the root relaxation solution on converted user problem
    std::vector<f_t> crushed_root_x;
    crush_primal_solution(
      original_problem_, original_lp_, root_crossover_soln_.x, new_slacks_, crushed_root_x);
    std::vector<f_t> crushed_root_y;
    std::vector<f_t> crushed_root_z;

    f_t dual_res_inf = crush_dual_solution(original_problem_,
                                           original_lp_,
                                           new_slacks_,
                                           root_crossover_soln_.y,
                                           root_crossover_soln_.z,
                                           crushed_root_y,
                                           crushed_root_z);

    root_crossover_soln_.x = crushed_root_x;
    root_crossover_soln_.y = crushed_root_y;
    root_crossover_soln_.z = crushed_root_z;

    // Call crossover on the crushed solution
    auto root_crossover_settings            = settings_;
    root_crossover_settings.log.log         = false;
    root_crossover_settings.concurrent_halt = get_root_concurrent_halt();
    crossover_status_t crossover_status     = crossover(original_lp_,
                                                    root_crossover_settings,
                                                    root_crossover_soln_,
                                                    exploration_stats_.start_time,
                                                    root_crossover_soln_,
                                                    crossover_vstatus_);

    if (crossover_status == crossover_status_t::OPTIMAL) {
      settings_.log.printf("Crossover status: %d\n", crossover_status);
    }

    // Check if crossover was stopped by dual simplex
    if (crossover_status == crossover_status_t::OPTIMAL) {
      set_root_concurrent_halt(1);  // Stop dual simplex
      root_status = root_status_future.get();
      // Override the root relaxation solution with the crossover solution
      root_relax_soln_ = root_crossover_soln_;
      root_vstatus_    = crossover_vstatus_;
      root_status      = lp_status_t::OPTIMAL;
    } else {
      root_status = root_status_future.get();
    }
  } else {
    root_status = root_status_future.get();
  }
  return root_status;
}

template <typename i_t, typename f_t>
mip_status_t branch_and_bound_t<i_t, f_t>::solve(mip_solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range scope("BB::solve");

  logger_t log;
  log.log                             = false;
  log.log_prefix                      = settings_.log.log_prefix;
  solver_status_                      = mip_status_t::UNSET;
  is_running                          = false;
  exploration_stats_.nodes_unexplored = 0;
  exploration_stats_.nodes_explored   = 0;
  original_lp_.A.to_compressed_row(Arow_);

  std::vector<bnb_worker_type_t> diving_strategies;
  diving_strategies.reserve(4);

  if (settings_.diving_settings.pseudocost_diving != 0) {
    diving_strategies.push_back(bnb_worker_type_t::PSEUDOCOST_DIVING);
  }

  if (settings_.diving_settings.line_search_diving != 0) {
    diving_strategies.push_back(bnb_worker_type_t::LINE_SEARCH_DIVING);
  }

  if (settings_.diving_settings.guided_diving != 0) {
    diving_strategies.push_back(bnb_worker_type_t::GUIDED_DIVING);
  }

  if (settings_.diving_settings.coefficient_diving != 0) {
    diving_strategies.push_back(bnb_worker_type_t::COEFFICIENT_DIVING);
    calculate_variable_locks(original_lp_, var_up_locks_, var_down_locks_);
  }

  if (diving_strategies.empty()) {
    settings_.log.printf("Warning: All diving heuristics are disabled!\n");
  }

  if (guess_.size() != 0) {
    raft::common::nvtx::range scope_guess("BB::check_initial_guess");
    std::vector<f_t> crushed_guess;
    crush_primal_solution(original_problem_, original_lp_, guess_, new_slacks_, crushed_guess);
    f_t primal_err;
    f_t bound_err;
    i_t num_fractional;
    const bool feasible = check_guess(
      original_lp_, settings_, var_types_, crushed_guess, primal_err, bound_err, num_fractional);
    if (feasible) {
      const f_t computed_obj = compute_objective(original_lp_, crushed_guess);
      mutex_upper_.lock();
      incumbent_.set_incumbent_solution(computed_obj, crushed_guess);
      upper_bound_ = computed_obj;
      mutex_upper_.unlock();
    }
  }

  root_relax_soln_.resize(original_lp_.num_rows, original_lp_.num_cols);

  settings_.log.printf("Solving LP root relaxation\n");

  lp_status_t root_status;
  simplex_solver_settings_t lp_settings = settings_;
  lp_settings.inside_mip                = 1;
  lp_settings.concurrent_halt           = get_root_concurrent_halt();
  // RINS/SUBMIP path
  if (!enable_concurrent_lp_root_solve()) {
    root_status = solve_linear_program_advanced(original_lp_,
                                                exploration_stats_.start_time,
                                                lp_settings,
                                                root_relax_soln_,
                                                root_vstatus_,
                                                edge_norms_);

  } else {
    root_status = solve_root_relaxation(lp_settings);
  }

  exploration_stats_.total_lp_iters      = root_relax_soln_.iterations;
  exploration_stats_.total_lp_solve_time = toc(exploration_stats_.start_time);

  if (root_status == lp_status_t::INFEASIBLE) {
    settings_.log.printf("MIP Infeasible\n");
    // FIXME: rarely dual simplex detects infeasible whereas it is feasible.
    // to add a small safety net, check if there is a primal solution already.
    // Uncomment this if the issue with cost266-UUE is resolved
    // if (settings.heuristic_preemption_callback != nullptr) {
    //   settings.heuristic_preemption_callback();
    // }
    return mip_status_t::INFEASIBLE;
  }
  if (root_status == lp_status_t::UNBOUNDED) {
    settings_.log.printf("MIP Unbounded\n");
    if (settings_.heuristic_preemption_callback != nullptr) {
      settings_.heuristic_preemption_callback();
    }
    return mip_status_t::UNBOUNDED;
  }

  if (root_status == lp_status_t::TIME_LIMIT) {
    solver_status_ = mip_status_t::TIME_LIMIT;
    set_final_solution(solution, -inf);
    return solver_status_;
  }

  if (root_status == lp_status_t::WORK_LIMIT) {
    solver_status_ = mip_status_t::WORK_LIMIT;
    return set_final_solution(solution, -inf);
  }

  assert(root_vstatus_.size() == original_lp_.num_cols);

  {
    const i_t expected_basic_count = original_lp_.num_rows;
    i_t actual_basic_count         = 0;
    for (const auto& status : root_vstatus_) {
      if (status == variable_status_t::BASIC) { actual_basic_count++; }
    }
    assert(actual_basic_count == expected_basic_count &&
           "root_vstatus_ BASIC count mismatch - LP solver returned invalid basis");
  }

  set_uninitialized_steepest_edge_norms<i_t, f_t>(edge_norms_);

  root_objective_ = compute_objective(original_lp_, root_relax_soln_.x);
  local_lower_bounds_.assign(settings_.num_bfs_workers, root_objective_);

  if (settings_.set_simplex_solution_callback != nullptr) {
    std::vector<f_t> original_x;
    uncrush_primal_solution(original_problem_, original_lp_, root_relax_soln_.x, original_x);
    std::vector<f_t> original_dual;
    std::vector<f_t> original_z;
    uncrush_dual_solution(original_problem_,
                          original_lp_,
                          root_relax_soln_.y,
                          root_relax_soln_.z,
                          original_dual,
                          original_z);
    settings_.set_simplex_solution_callback(
      original_x, original_dual, compute_user_objective(original_lp_, root_objective_));
  }

  std::vector<i_t> fractional;
  const i_t num_fractional =
    fractional_variables(settings_, root_relax_soln_.x, var_types_, fractional);

  if (num_fractional == 0) {
    mutex_upper_.lock();
    incumbent_.set_incumbent_solution(root_objective_, root_relax_soln_.x);
    upper_bound_ = root_objective_;
    mutex_upper_.unlock();
    // We should be done here
    uncrush_primal_solution(original_problem_, original_lp_, incumbent_.x, solution.x);
    solution.objective          = incumbent_.objective;
    solution.lower_bound        = root_objective_;
    solution.nodes_explored     = 0;
    solution.simplex_iterations = root_relax_soln_.iterations;
    settings_.log.printf("Optimal solution found at root node. Objective %.16e. Time %.2f.\n",
                         compute_user_objective(original_lp_, root_objective_),
                         toc(exploration_stats_.start_time));

    if (settings_.solution_callback != nullptr) {
      settings_.solution_callback(solution.x, solution.objective);
    }
    if (settings_.heuristic_preemption_callback != nullptr) {
      settings_.heuristic_preemption_callback();
    }
    return mip_status_t::OPTIMAL;
  }

  pc_.resize(original_lp_.num_cols);
  {
    raft::common::nvtx::range scope_sb("BB::strong_branching");
    strong_branching<i_t, f_t>(original_lp_,
                               settings_,
                               exploration_stats_.start_time,
                               var_types_,
                               root_relax_soln_.x,
                               fractional,
                               root_objective_,
                               root_vstatus_,
                               edge_norms_,
                               pc_);
  }

  if (toc(exploration_stats_.start_time) > settings_.time_limit) {
    solver_status_ = mip_status_t::TIME_LIMIT;
    set_final_solution(solution, root_objective_);
    return solver_status_;
  }

  // Choose variable to branch on
  i_t branch_var = pc_.variable_selection(fractional, root_relax_soln_.x, log);

  search_tree_.root      = std::move(mip_node_t<i_t, f_t>(root_objective_, root_vstatus_));
  search_tree_.num_nodes = 0;
  search_tree_.graphviz_node(settings_.log, &search_tree_.root, "lower bound", root_objective_);
  search_tree_.branch(&search_tree_.root,
                      branch_var,
                      root_relax_soln_.x[branch_var],
                      root_vstatus_,
                      original_lp_,
                      log);

  {
    uint32_t lp_hash = detail::compute_hash(original_lp_.objective);
    lp_hash ^= detail::compute_hash(original_lp_.A.x.underlying());
    settings_.log.printf("lp A.x hash: %08x\n",
                         detail::compute_hash(original_lp_.A.x.underlying()));
    lp_hash ^= detail::compute_hash(original_lp_.A.i.underlying());
    settings_.log.printf("lp A.j hash: %08x\n",
                         detail::compute_hash(original_lp_.A.i.underlying()));
    lp_hash ^= detail::compute_hash(original_lp_.A.col_start.underlying());
    settings_.log.printf("lp A.col_start hash: %08x\n",
                         detail::compute_hash(original_lp_.A.col_start.underlying()));
    lp_hash ^= detail::compute_hash(original_lp_.rhs);
    settings_.log.printf("lp rhs hash: %08x\n", detail::compute_hash(original_lp_.rhs));
    lp_hash ^= detail::compute_hash(original_lp_.lower);
    settings_.log.printf("lp lower hash: %08x\n", detail::compute_hash(original_lp_.lower));
    lp_hash ^= detail::compute_hash(original_lp_.upper);
    settings_.log.printf(
      "Exploring the B&B tree using %d threads (best-first = %d, diving = %d) [LP hash: %08x]\n",
      settings_.num_threads,
      settings_.num_bfs_workers,
      settings_.num_threads - settings_.num_bfs_workers,
      lp_hash);
  }

  exploration_stats_.nodes_explored       = 0;
  exploration_stats_.nodes_unexplored     = 2;
  exploration_stats_.nodes_since_last_log = 0;
  exploration_stats_.last_log             = tic();
  active_subtrees_                        = 0;
  is_running                              = true;
  lower_bound_ceiling_                    = inf;
  should_report_                          = true;

  settings_.log.printf(
    "  | Explored | Unexplored |    Objective    |     Bound     | Depth | Iter/Node |   Gap    "
    "|  Time  |\n");

  // Choose between BSP coordinator (deterministic) and opportunistic exploration
  if (settings_.deterministic && settings_.num_bfs_workers > 0) {
    run_bsp_coordinator(Arow_);
  } else {
    // Use traditional opportunistic parallel exploration
#pragma omp parallel num_threads(settings_.num_threads)
    {
      raft::common::nvtx::range scope_tree("BB::tree_exploration");
#pragma omp master
      {
        auto down_child          = search_tree_.root.get_down_child();
        auto up_child            = search_tree_.root.get_up_child();
        i_t initial_size         = 2 * settings_.num_threads;
        const i_t num_strategies = diving_strategies.size();

#pragma omp taskgroup
        {
#pragma omp task
          exploration_ramp_up(down_child, initial_size);

#pragma omp task
          exploration_ramp_up(up_child, initial_size);
        }

        for (i_t i = 0; i < settings_.num_bfs_workers; i++) {
#pragma omp task
          best_first_thread(i);
        }

        if (!diving_strategies.empty()) {
          for (i_t k = 0; k < settings_.diving_settings.num_diving_workers; k++) {
            const bnb_worker_type_t diving_type = diving_strategies[k % num_strategies];
#pragma omp task
            diving_thread(diving_type);
          }
        }
      }
    }
  }
  is_running = false;

  // Compute final lower bound
  f_t lower_bound;
  if (bsp_mode_enabled_) {
    // In BSP mode, compute lower bound from worker queues only (no global heap)
    lower_bound = compute_bsp_lower_bound();
    if (lower_bound == std::numeric_limits<f_t>::infinity() && incumbent_.has_incumbent) {
      lower_bound = upper_bound_.load();
    }
  } else {
    // Non-BSP mode: use node_queue or fall back to root
    lower_bound = node_queue_.best_first_queue_size() > 0 ? node_queue_.get_lower_bound()
                                                          : search_tree_.root.lower_bound;
    // If queue is empty and we have an incumbent, the tree is fully explored
    if (node_queue_.best_first_queue_size() == 0 && incumbent_.has_incumbent) {
      lower_bound = upper_bound_.load();
    }
  }
  return set_final_solution(solution, lower_bound);
}

// ============================================================================
// BSP (Bulk Synchronous Parallel) Deterministic implementation
// ============================================================================

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::run_bsp_coordinator(const csr_matrix_t<i_t, f_t>& Arow)
{
  raft::common::nvtx::range scope("BB::bsp_coordinator");

  bsp_horizon_step_ = 0.05;

  const int num_bfs_workers    = settings_.num_bfs_workers;
  const int num_diving_workers = settings_.diving_settings.num_diving_workers;

  bsp_mode_enabled_    = true;
  bsp_current_horizon_ = bsp_horizon_step_;
  bsp_horizon_number_  = 0;
  bsp_terminated_.store(false);

  // Initialize BFS worker pool
  bsp_workers_ = std::make_unique<bb_worker_pool_t<i_t, f_t>>();
  bsp_workers_->initialize(num_bfs_workers,
                           original_lp_,
                           Arow,
                           var_types_,
                           settings_.refactor_frequency,
                           settings_.deterministic);

  // Initialize diving worker pool if we have diving workers
  if (num_diving_workers > 0) {
    std::vector<bnb_worker_type_t> diving_types = {bnb_worker_type_t::PSEUDOCOST_DIVING,
                                                   bnb_worker_type_t::LINE_SEARCH_DIVING,
                                                   bnb_worker_type_t::GUIDED_DIVING,
                                                   bnb_worker_type_t::COEFFICIENT_DIVING};
    bsp_diving_workers_ = std::make_unique<bsp_diving_worker_pool_t<i_t, f_t>>();
    bsp_diving_workers_->initialize(num_diving_workers,
                                    diving_types,
                                    original_lp_,
                                    Arow,
                                    var_types_,
                                    settings_.refactor_frequency,
                                    settings_.deterministic);

    calculate_variable_locks(original_lp_, var_up_locks_, var_down_locks_);
  }

  // Initialize scheduler for automatic sync at horizon boundaries
  // Workers will block in record_work() when they cross sync points
  bsp_scheduler_ = std::make_unique<work_unit_scheduler_t>(bsp_horizon_step_);

  scoped_context_registrations_t context_registrations(*bsp_scheduler_);
  for (auto& worker : *bsp_workers_) {
    context_registrations.add(worker.work_context);
  }
  if (bsp_diving_workers_) {
    for (auto& worker : *bsp_diving_workers_) {
      context_registrations.add(worker.work_context);
    }
  }

  bsp_debug_logger_.set_settings(bsp_debug_settings_);
  bsp_debug_logger_.set_num_workers(num_bfs_workers);
  bsp_debug_logger_.set_horizon_step(bsp_horizon_step_);

  settings_.log.printf(
    "BSP Mode: %d BFS workers + %d diving workers, horizon step = %.2f work "
    "units\n",
    num_bfs_workers,
    num_diving_workers,
    bsp_horizon_step_);

  // Assign the initial children of the root to worker 0 and worker 1
  search_tree_.root.get_down_child()->origin_worker_id = -1;  // Pre-BSP marker
  search_tree_.root.get_down_child()->creation_seq     = 0;
  search_tree_.root.get_up_child()->origin_worker_id   = -1;
  search_tree_.root.get_up_child()->creation_seq       = 1;

  (*bsp_workers_)[0].enqueue_node_with_identity(search_tree_.root.get_down_child());
  (*bsp_workers_)[0].track_node_assigned();
  (*bsp_workers_)[1 % num_bfs_workers].enqueue_node_with_identity(search_tree_.root.get_up_child());
  (*bsp_workers_)[1 % num_bfs_workers].track_node_assigned();
  BSP_DEBUG_FLUSH_ASSIGN_TRACE(bsp_debug_settings_, bsp_debug_logger_);

  // Set sync callback - executed when all workers arrive at barrier
  // Returns true to stop the scheduler (and all workers exit cleanly together)
  bsp_scheduler_->set_sync_callback([this](double sync_target) -> bool {
    bsp_sync_callback(0);
    return bsp_terminated_.load();
  });

  // initialize global state snapshots
  for (auto& worker : *bsp_workers_) {
    worker.set_snapshots(upper_bound_.load(),
                         pc_.pseudo_cost_sum_up,
                         pc_.pseudo_cost_sum_down,
                         pc_.pseudo_cost_num_up,
                         pc_.pseudo_cost_num_down,
                         0.0,
                         bsp_horizon_step_);
  }

  if (bsp_diving_workers_) {
    std::vector<f_t> incumbent_snapshot;
    if (incumbent_.has_incumbent) { incumbent_snapshot = incumbent_.x; }

    for (auto& worker : *bsp_diving_workers_) {
      worker.set_snapshots(upper_bound_.load(),
                           pc_.pseudo_cost_sum_up,
                           pc_.pseudo_cost_sum_down,
                           pc_.pseudo_cost_num_up,
                           pc_.pseudo_cost_num_down,
                           incumbent_snapshot,
                           &root_relax_soln_.x,
                           0.0,
                           bsp_horizon_step_);
    }
  }

  const int total_thread_count = num_bfs_workers + num_diving_workers;

  // Main BSP execution - workers run in parallel with scheduler-driven sync
#pragma omp parallel num_threads(total_thread_count)
  {
    int thread_id = omp_get_thread_num();

    if (thread_id < num_bfs_workers) {
      // BFS worker
      auto& worker          = (*bsp_workers_)[thread_id];
      f_t worker_start_time = tic();
      run_worker_loop(worker, search_tree_);
      worker.total_runtime += toc(worker_start_time);
    } else {
      // Diving worker
      int diving_id         = thread_id - num_bfs_workers;
      auto& worker          = (*bsp_diving_workers_)[diving_id];
      f_t worker_start_time = tic();
      run_diving_worker_loop(worker);
      worker.total_runtime += toc(worker_start_time);
    }
  }

  // Print per-worker statistics
  settings_.log.printf("\n");
  settings_.log.printf("BSP BFS Worker Statistics:\n");
  settings_.log.printf(
    "  Worker |  Nodes  | Branched | Pruned | Infeas. | IntSol | Assigned |  Clock   | "
    "Sync%% | NoWork\n");
  settings_.log.printf(
    "  "
    "-------+---------+----------+--------+---------+--------+----------+----------+-------+-------"
    "\n");
  for (const auto& worker : *bsp_workers_) {
    double sync_time    = worker.work_context.total_sync_time;
    double total_time   = worker.total_runtime;  // Already includes sync time
    double sync_percent = (total_time > 0) ? (100.0 * sync_time / total_time) : 0.0;
    settings_.log.printf("  %6d | %7d | %8d | %6d | %7d | %6d | %8d | %7.3fs | %4.1f%% | %5.2fs\n",
                         worker.worker_id,
                         worker.total_nodes_processed,
                         worker.total_nodes_branched,
                         worker.total_nodes_pruned,
                         worker.total_nodes_infeasible,
                         worker.total_integer_solutions,
                         worker.total_nodes_assigned,
                         total_time,
                         std::min(99.9, sync_percent),
                         worker.total_nowork_time);
  }

  // Print diving worker statistics
  if (bsp_diving_workers_ && bsp_diving_workers_->size() > 0) {
    settings_.log.printf("\nBSP Diving Worker Statistics:\n");
    settings_.log.printf("  Worker |  Type  |  Dives  | Nodes  | IntSol |  Clock   | NoWork\n");
    settings_.log.printf("  -------+--------+---------+--------+--------+----------+-------\n");
    for (const auto& worker : *bsp_diving_workers_) {
      const char* type_str = "???";
      switch (worker.diving_type) {
        case bnb_worker_type_t::PSEUDOCOST_DIVING: type_str = "PC"; break;
        case bnb_worker_type_t::LINE_SEARCH_DIVING: type_str = "LS"; break;
        case bnb_worker_type_t::GUIDED_DIVING: type_str = "GD"; break;
        case bnb_worker_type_t::COEFFICIENT_DIVING: type_str = "CD"; break;
        default: break;
      }
      settings_.log.printf("  %6d | %6s | %7d | %6d | %6d | %7.3fs | %5.2fs\n",
                           worker.worker_id,
                           type_str,
                           worker.total_dives,
                           worker.total_nodes_explored,
                           worker.total_integer_solutions,
                           worker.total_runtime,
                           worker.total_nowork_time);
    }
  }
  settings_.log.printf("\n");

  // Finalize debug logger
  BSP_DEBUG_FINALIZE(bsp_debug_settings_, bsp_debug_logger_);
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::refill_worker_queues(i_t target_queue_size)
{
  // Distribute nodes from global pool to workers in round-robin fashion
  // This ensures deterministic assignment based on node ordering in the heap

  std::vector<mip_node_t<i_t, f_t>*> nodes_to_assign;

  // Pop nodes from heap while respecting incumbent bound
  mutex_heap_.lock();
  while (!heap_.empty()) {
    mip_node_t<i_t, f_t>* node = heap_.top();

    // Skip pruned nodes
    if (node->lower_bound >= upper_bound_.load()) {
      heap_.pop();
      search_tree_.update(node, node_status_t::FATHOMED);
      --exploration_stats_.nodes_unexplored;
      continue;
    }

    // Check if we have enough nodes
    if (nodes_to_assign.size() >= static_cast<size_t>(target_queue_size * bsp_workers_->size())) {
      break;
    }

    heap_.pop();
    nodes_to_assign.push_back(node);
  }
  mutex_heap_.unlock();

  // Sort by BSP identity for deterministic distribution
  // Uses lexicographic order of (origin_worker_id, creation_seq)
  auto deterministic_less = [](const mip_node_t<i_t, f_t>* a, const mip_node_t<i_t, f_t>* b) {
    // Lexicographic comparison of BSP identity tuple
    if (a->origin_worker_id != b->origin_worker_id) {
      return a->origin_worker_id < b->origin_worker_id;
    }
    return a->creation_seq < b->creation_seq;
  };
  std::sort(nodes_to_assign.begin(), nodes_to_assign.end(), deterministic_less);

  for (size_t i = 0; i < nodes_to_assign.size(); ++i) {
    int worker_id = i % bsp_workers_->size();
    auto* node    = nodes_to_assign[i];
    // Use enqueue_node_with_identity since these nodes already have BSP identity from root setup
    (*bsp_workers_)[worker_id].enqueue_node_with_identity(node);
    (*bsp_workers_)[worker_id].track_node_assigned();

    // Debug: Log node assignment
    double wut = bsp_current_horizon_ - bsp_horizon_step_;  // Start of current horizon
    BSP_DEBUG_LOG_NODE_ASSIGNED(bsp_debug_settings_,
                                bsp_debug_logger_,
                                wut,
                                worker_id,
                                node->node_id,
                                node->origin_worker_id,
                                node->lower_bound);
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::run_worker_loop(bb_worker_state_t<i_t, f_t>& worker,
                                                   search_tree_t<i_t, f_t>& search_tree)
{
  raft::common::nvtx::range scope("BB::worker_loop");

  // Workers run continuously until scheduler signals stop (via sync callback)
  // The scheduler handles synchronization at horizon boundaries via record_work()
  while (!bsp_terminated_.load() && !bsp_scheduler_->is_stopped() &&
         solver_status_ == mip_status_t::UNSET) {
    // Check time limit directly - don't wait for sync if time is up
    if (toc(exploration_stats_.start_time) > settings_.time_limit) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      bsp_terminated_.store(true);
      bsp_scheduler_->stop();  // Wake up workers waiting at barrier
      break;
    }

    if (worker.has_work()) {
      mip_node_t<i_t, f_t>* node = worker.dequeue_node();
      if (node == nullptr) { continue; }

      // Track that this node is being actively processed
      worker.current_node = node;

      // Check if node should be pruned (use worker's snapshot for determinism)
      f_t upper_bound = worker.local_upper_bound;
      if (node->lower_bound >= upper_bound) {
        worker.current_node = nullptr;
        worker.record_fathomed(node, node->lower_bound);
        worker.track_node_pruned();
        search_tree.update(node, node_status_t::FATHOMED);
        --exploration_stats_.nodes_unexplored;
        continue;
      }

      // Check if we can warm-start from the previous solve's basis
      bool is_child                     = (node->parent == worker.last_solved_node);
      worker.recompute_bounds_and_basis = !is_child;

      // Solve the node - record_work() inside may block at sync points
      // The scheduler's sync callback will execute during barrier waits
      node_solve_info_t status = solve_node_bsp(worker, node, search_tree, worker.horizon_end);

      // Track last solved node for warm-start detection
      worker.last_solved_node = node;

      // Handle result
      worker.current_node = nullptr;
      if (status == node_solve_info_t::TIME_LIMIT || status == node_solve_info_t::WORK_LIMIT) {
        // Time/work limit hit - the loop head will detect this and terminate properly
        continue;
      }
      // Node completed successfully - loop back to process children
      continue;
    }

    // No work available - advance to next sync point to participate in barrier
    // This ensures all workers reach the sync point even if some have no work
    f_t nowork_start            = tic();
    cuopt::sync_result_t result = bsp_scheduler_->wait_for_next_sync(worker.work_context);
    worker.total_nowork_time += toc(nowork_start);
    if (result == cuopt::sync_result_t::STOPPED) { break; }
    // After sync, bsp_sync_callback may have redistributed nodes to us
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::bsp_sync_callback(int worker_id)
{
  raft::common::nvtx::range scope("BB::bsp_sync_callback");

  ++bsp_horizon_number_;
  double horizon_start = bsp_current_horizon_ - bsp_horizon_step_;
  double horizon_end   = bsp_current_horizon_;

  // Wait for external producers (CPUFJ) to reach horizon_start before processing
  // This ensures we don't process B&B events before producers have caught up
  producer_sync_.wait_for_producers(horizon_start);

  work_unit_context_.global_work_units_elapsed = horizon_end;

  BSP_DEBUG_LOG_HORIZON_START(
    bsp_debug_settings_, bsp_debug_logger_, bsp_horizon_number_, horizon_start, horizon_end);

  bb_event_batch_t<i_t, f_t> all_events = bsp_workers_->collect_and_sort_events();

  BSP_DEBUG_LOG_SYNC_PHASE_START(
    bsp_debug_settings_, bsp_debug_logger_, horizon_end, all_events.size());

  process_history_and_sync(all_events);

  BSP_DEBUG_LOG_SYNC_PHASE_END(bsp_debug_settings_, bsp_debug_logger_, horizon_end);

  prune_worker_nodes_vs_incumbent();

  merge_diving_solutions();

  populate_diving_heap_at_sync();

  assign_diving_nodes();

  balance_worker_loads();
  BSP_DEBUG_FLUSH_ASSIGN_TRACE(bsp_debug_settings_, bsp_debug_logger_);

  BSP_DEBUG_LOG_HORIZON_END(
    bsp_debug_settings_, bsp_debug_logger_, bsp_horizon_number_, horizon_end);

  uint32_t state_hash = 0;
  {
    std::vector<uint64_t> state_data;
    state_data.push_back(static_cast<uint64_t>(exploration_stats_.nodes_explored));
    state_data.push_back(static_cast<uint64_t>(exploration_stats_.nodes_unexplored));
    f_t ub = upper_bound_.load();
    f_t lb = compute_bsp_lower_bound();
    state_data.push_back(static_cast<uint64_t>(ub * 1000000));
    state_data.push_back(static_cast<uint64_t>(lb * 1000000));

    for (auto& worker : *bsp_workers_) {
      if (worker.current_node != nullptr) {
        state_data.push_back(worker.current_node->get_id_packed());
      }
      for (auto* node : worker.plunge_stack) {
        state_data.push_back(node->get_id_packed());
      }
      for (auto* node : worker.backlog) {
        state_data.push_back(node->get_id_packed());
      }
    }

    state_hash = detail::compute_hash(state_data);
    state_hash ^= pc_.compute_state_hash();
    BSP_DEBUG_LOG_HORIZON_HASH(
      bsp_debug_settings_, bsp_debug_logger_, bsp_horizon_number_, horizon_end, state_hash);
  }

  BSP_DEBUG_EMIT_TREE_STATE(bsp_debug_settings_,
                            bsp_debug_logger_,
                            bsp_horizon_number_,
                            search_tree_.root,
                            upper_bound_.load());

  std::vector<mip_node_t<i_t, f_t>*> heap_snapshot;
  BSP_DEBUG_EMIT_STATE_JSON(bsp_debug_settings_,
                            bsp_debug_logger_,
                            bsp_horizon_number_,
                            horizon_start,
                            horizon_end,
                            0,
                            upper_bound_.load(),
                            compute_bsp_lower_bound(),
                            exploration_stats_.nodes_explored,
                            exploration_stats_.nodes_unexplored,
                            *bsp_workers_,
                            heap_snapshot,
                            all_events);

  // Advance the horizon for next sync
  bsp_current_horizon_ += bsp_horizon_step_;

  // Update worker snapshots for next horizon
  for (auto& worker : *bsp_workers_) {
    worker.set_snapshots(upper_bound_.load(),
                         pc_.pseudo_cost_sum_up,
                         pc_.pseudo_cost_sum_down,
                         pc_.pseudo_cost_num_up,
                         pc_.pseudo_cost_num_down,
                         horizon_end,
                         bsp_current_horizon_);
  }

  // Update diving worker snapshots for next horizon
  if (bsp_diving_workers_) {
    std::vector<f_t> incumbent_snapshot;
    if (incumbent_.has_incumbent) { incumbent_snapshot = incumbent_.x; }

    for (auto& worker : *bsp_diving_workers_) {
      worker.set_snapshots(upper_bound_.load(),
                           pc_.pseudo_cost_sum_up,
                           pc_.pseudo_cost_sum_down,
                           pc_.pseudo_cost_num_up,
                           pc_.pseudo_cost_num_down,
                           incumbent_snapshot,
                           &root_relax_soln_.x,
                           horizon_end,
                           bsp_current_horizon_);
    }
  }

  // Check termination conditions
  f_t lower_bound = compute_bsp_lower_bound();
  f_t upper_bound = upper_bound_.load();
  f_t abs_gap     = upper_bound - lower_bound;
  f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);

  bool should_terminate = false;

  // Gap tolerance reached
  if (abs_gap <= settings_.absolute_mip_gap_tol || rel_gap <= settings_.relative_mip_gap_tol) {
    should_terminate = true;
  }

  bool diving_has_work = bsp_diving_workers_ && bsp_diving_workers_->any_has_work();
  if (!bsp_workers_->any_has_work() && !diving_has_work) { should_terminate = true; }

  if (toc(exploration_stats_.start_time) > settings_.time_limit) {
    solver_status_   = mip_status_t::TIME_LIMIT;
    should_terminate = true;
  }

  // Check if the next horizon would exceed work limit. If so, terminate now rather than
  // letting workers continue past the limit. This is conservative (stops slightly early)
  // but prevents workers from processing nodes beyond the work budget.
  // bsp_current_horizon_ now holds the NEXT horizon's end value after the increment above.
  if (bsp_current_horizon_ > settings_.work_limit) {
    solver_status_   = mip_status_t::WORK_LIMIT;
    should_terminate = true;
  }

  if (should_terminate) { bsp_terminated_.store(true); }

  // Progress logging with horizon number and state hash
  f_t obj              = compute_user_objective(original_lp_, upper_bound);
  f_t user_lower       = compute_user_objective(original_lp_, lower_bound);
  std::string gap_user = user_mip_gap<f_t>(obj, user_lower);

  // Build list of workers that reached sync with no work
  std::string idle_workers;
  for (const auto& w : *bsp_workers_) {
    if (!w.has_work() && w.current_node == nullptr) {
      if (!idle_workers.empty()) idle_workers += ",";
      idle_workers += "W" + std::to_string(w.worker_id);
    }
  }

  settings_.log.printf("S%-4d %8d   %8lu    %+13.6e    %+10.6e    %s %8.2f  [%08x]%s%s\n",
                       bsp_horizon_number_,
                       exploration_stats_.nodes_explored,
                       exploration_stats_.nodes_unexplored,
                       obj,
                       user_lower,
                       gap_user.c_str(),
                       toc(exploration_stats_.start_time),
                       state_hash,
                       idle_workers.empty() ? "" : " ",
                       idle_workers.c_str());
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::run_worker_until_horizon(bb_worker_state_t<i_t, f_t>& worker,
                                                            search_tree_t<i_t, f_t>& search_tree,
                                                            double current_horizon)
{
  raft::common::nvtx::range scope("BB::worker_run");

  while (worker.clock < current_horizon && worker.has_work() &&
         solver_status_ == mip_status_t::UNSET) {
    mip_node_t<i_t, f_t>* node = worker.dequeue_node();
    if (node == nullptr) break;

    // Check if node should be pruned
    f_t upper_bound = upper_bound_.load();
    if (node->lower_bound >= upper_bound) {
      worker.record_fathomed(node, node->lower_bound);
      worker.track_node_pruned();
      search_tree.update(node, node_status_t::FATHOMED);
      --exploration_stats_.nodes_unexplored;
      continue;
    }

    // basis warm-start detection
    bool is_child                     = (node->parent == worker.last_solved_node);
    worker.recompute_bounds_and_basis = !is_child;

    node_solve_info_t status = solve_node_bsp(worker, node, search_tree, current_horizon);
    worker.last_solved_node  = node;

    if (status == node_solve_info_t::TIME_LIMIT) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    } else if (status == node_solve_info_t::WORK_LIMIT) {
      solver_status_ = mip_status_t::WORK_LIMIT;
      break;
    }
  }
}

template <typename i_t, typename f_t>
node_solve_info_t branch_and_bound_t<i_t, f_t>::solve_node_bsp(bb_worker_state_t<i_t, f_t>& worker,
                                                               mip_node_t<i_t, f_t>* node_ptr,
                                                               search_tree_t<i_t, f_t>& search_tree,
                                                               double current_horizon)
{
  raft::common::nvtx::range scope("BB::solve_node_bsp");

  {
    const i_t expected_basic_count = original_lp_.num_rows;
    i_t actual_basic_count         = 0;
    for (const auto& status : node_ptr->vstatus) {
      if (status == variable_status_t::BASIC) { actual_basic_count++; }
    }
    if (actual_basic_count != expected_basic_count) {
      settings_.log.printf(
        "ERROR: Node %d (worker %d, seq %d) vstatus has %d BASIC entries, expected %d (num_rows)\n",
        node_ptr->node_id,
        node_ptr->origin_worker_id,
        node_ptr->creation_seq,
        actual_basic_count,
        expected_basic_count);
      settings_.log.printf("       vstatus.size() = %zu, num_cols = %d\n",
                           node_ptr->vstatus.size(),
                           original_lp_.num_cols);
      assert(actual_basic_count == expected_basic_count &&
             "vstatus BASIC count mismatch - this indicates vstatus corruption");
    }
  }

  double work_units_at_start = worker.work_context.global_work_units_elapsed;
  double clock_at_start      = worker.clock;

  double work_limit = worker.horizon_end - worker.clock;
  BSP_DEBUG_LOG_SOLVE_START(bsp_debug_settings_,
                            bsp_debug_logger_,
                            worker.clock,
                            worker.worker_id,
                            node_ptr->node_id,
                            node_ptr->origin_worker_id,
                            work_limit,
                            false);

  std::fill(worker.node_presolver->bounds_changed.begin(),
            worker.node_presolver->bounds_changed.end(),
            false);

  if (worker.recompute_bounds_and_basis) {
    worker.leaf_problem->lower = original_lp_.lower;
    worker.leaf_problem->upper = original_lp_.upper;
    node_ptr->get_variable_bounds(worker.leaf_problem->lower,
                                  worker.leaf_problem->upper,
                                  worker.node_presolver->bounds_changed);
  } else {
    node_ptr->update_branched_variable_bounds(worker.leaf_problem->lower,
                                              worker.leaf_problem->upper,
                                              worker.node_presolver->bounds_changed);
  }

  double remaining_time = settings_.time_limit - toc(exploration_stats_.start_time);
  if (remaining_time <= 0) { return node_solve_info_t::TIME_LIMIT; }

  // Bounds strengthening
  simplex_solver_settings_t<i_t, f_t> lp_settings = settings_;
  lp_settings.set_log(false);

  lp_settings.cut_off       = worker.local_upper_bound + settings_.dual_tol;
  lp_settings.inside_mip    = 2;
  lp_settings.time_limit    = remaining_time;
  lp_settings.scale_columns = false;

  bool feasible = true;
  if (false) {
    raft::common::nvtx::range scope_bs("BB::bound_strengthening");
    feasible = worker.node_presolver->bounds_strengthening(
      worker.leaf_problem->lower, worker.leaf_problem->upper, lp_settings);

    if (settings_.deterministic) {
      static cuopt::work_unit_predictor_t<bounds_strengthening_predictor, cpu_work_unit_scaler_t>
        bs_predictor;

      const i_t m   = worker.leaf_problem->num_rows;
      const i_t n   = worker.leaf_problem->num_cols;
      const i_t nnz = worker.leaf_problem->A.col_start[n];

      i_t num_bounds_changed = 0;
      for (bool changed : worker.node_presolver->bounds_changed) {
        if (changed) ++num_bounds_changed;
      }

      std::map<std::string, float> features;
      features["m"]              = static_cast<float>(m);
      features["n"]              = static_cast<float>(n);
      features["nnz"]            = static_cast<float>(nnz);
      features["nnz_processed"]  = static_cast<float>(worker.node_presolver->last_nnz_processed);
      features["bounds_changed"] = static_cast<float>(num_bounds_changed);

      // predicts milliseconds
      f_t prediction =
        std::max(f_t(0), static_cast<f_t>(bs_predictor.predict_scalar(features))) / 1000;
      worker.work_context.record_work(prediction);
    }
  }

  if (!feasible) {
    node_ptr->lower_bound = std::numeric_limits<f_t>::infinity();
    search_tree.update(node_ptr, node_status_t::INFEASIBLE);
    worker.record_infeasible(node_ptr);
    worker.track_node_infeasible();
    worker.track_node_processed();
    --exploration_stats_.nodes_unexplored;
    ++exploration_stats_.nodes_explored;
    worker.recompute_bounds_and_basis = true;
    return node_solve_info_t::NO_CHILDREN;
  }

  // Solve LP relaxation
  lp_solution_t<i_t, f_t> leaf_solution(worker.leaf_problem->num_rows,
                                        worker.leaf_problem->num_cols);
  std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
  i_t node_iter                                = 0;
  f_t lp_start_time                            = tic();
  std::vector<f_t> leaf_edge_norms             = edge_norms_;

  // Debug: Log LP input for determinism analysis
  if (bsp_debug_settings_.any_enabled()) {
    uint64_t path_hash    = node_ptr->compute_path_hash();
    uint64_t vstatus_hash = detail::compute_hash(leaf_vstatus);
    uint64_t bounds_hash  = detail::compute_hash(worker.leaf_problem->lower) ^
                           detail::compute_hash(worker.leaf_problem->upper);
    BSP_DEBUG_LOG_LP_INPUT(bsp_debug_settings_,
                           bsp_debug_logger_,
                           worker.worker_id,
                           node_ptr->node_id,
                           path_hash,
                           node_ptr->depth,
                           vstatus_hash,
                           bounds_hash);
  }

  dual::status_t lp_status = dual_phase2_with_advanced_basis(2,
                                                             0,
                                                             worker.recompute_bounds_and_basis,
                                                             lp_start_time,
                                                             *worker.leaf_problem,
                                                             lp_settings,
                                                             leaf_vstatus,
                                                             *worker.basis_factors,
                                                             worker.basic_list,
                                                             worker.nonbasic_list,
                                                             leaf_solution,
                                                             node_iter,
                                                             leaf_edge_norms,
                                                             &worker.work_context);

  if (bsp_debug_settings_.any_enabled()) {
    uint64_t path_hash = node_ptr->compute_path_hash();
    uint64_t sol_hash  = detail::compute_hash(leaf_solution.x);
    f_t obj            = (lp_status == dual::status_t::OPTIMAL)
                           ? compute_objective(*worker.leaf_problem, leaf_solution.x)
                           : std::numeric_limits<f_t>::infinity();
    uint64_t obj_hash  = detail::compute_hash(obj);
    BSP_DEBUG_LOG_LP_OUTPUT(bsp_debug_settings_,
                            bsp_debug_logger_,
                            worker.worker_id,
                            node_ptr->node_id,
                            path_hash,
                            static_cast<int>(lp_status),
                            node_iter,
                            obj_hash,
                            sol_hash);
  }

  // Validate vstatus after LP solve - check for corruption during simplex
  {
    const i_t expected_basic_count = original_lp_.num_rows;
    i_t actual_basic_count         = 0;
    for (const auto& status : leaf_vstatus) {
      if (status == variable_status_t::BASIC) { actual_basic_count++; }
    }
    if (actual_basic_count != expected_basic_count) {
      settings_.log.printf(
        "ERROR: After LP solve, node %d vstatus has %d BASIC entries, expected %d\n",
        node_ptr->node_id,
        actual_basic_count,
        expected_basic_count);
      settings_.log.printf("       lp_status = %d, recompute_basis = %d\n",
                           static_cast<int>(lp_status),
                           worker.recompute_bounds_and_basis ? 1 : 0);
      assert(actual_basic_count == expected_basic_count && "vstatus corrupted during LP solve");
    }
  }

  double work_performed = worker.work_context.global_work_units_elapsed - work_units_at_start;
  worker.clock += work_performed;
  worker.work_units_this_horizon += work_performed;

  exploration_stats_.total_lp_solve_time += toc(lp_start_time);
  exploration_stats_.total_lp_iters += node_iter;
  ++exploration_stats_.nodes_explored;
  --exploration_stats_.nodes_unexplored;

  // Process LP result
  if (lp_status == dual::status_t::DUAL_UNBOUNDED) {
    node_ptr->lower_bound = std::numeric_limits<f_t>::infinity();

    worker.record_infeasible(node_ptr);
    worker.track_node_infeasible();
    worker.track_node_processed();
    worker.recompute_bounds_and_basis = true;

    BSP_DEBUG_LOG_SOLVE_END(bsp_debug_settings_,
                            bsp_debug_logger_,
                            worker.clock,
                            worker.worker_id,
                            node_ptr->node_id,
                            node_ptr->origin_worker_id,
                            "INFEASIBLE",
                            node_ptr->lower_bound);
    BSP_DEBUG_LOG_INFEASIBLE(
      bsp_debug_settings_, bsp_debug_logger_, worker.clock, worker.worker_id, node_ptr->node_id);

    search_tree.update(node_ptr, node_status_t::INFEASIBLE);
    return node_solve_info_t::NO_CHILDREN;

  } else if (lp_status == dual::status_t::CUTOFF) {
    node_ptr->lower_bound = worker.local_upper_bound;

    worker.record_fathomed(node_ptr, node_ptr->lower_bound);
    worker.track_node_pruned();
    worker.track_node_processed();
    worker.recompute_bounds_and_basis = true;

    BSP_DEBUG_LOG_SOLVE_END(bsp_debug_settings_,
                            bsp_debug_logger_,
                            worker.clock,
                            worker.worker_id,
                            node_ptr->node_id,
                            node_ptr->origin_worker_id,
                            "FATHOMED",
                            node_ptr->lower_bound);
    BSP_DEBUG_LOG_FATHOMED(bsp_debug_settings_,
                           bsp_debug_logger_,
                           worker.clock,
                           worker.worker_id,
                           node_ptr->node_id,
                           node_ptr->lower_bound);

    search_tree.update(node_ptr, node_status_t::FATHOMED);
    return node_solve_info_t::NO_CHILDREN;

  } else if (lp_status == dual::status_t::OPTIMAL) {
    std::vector<i_t> leaf_fractional;
    i_t leaf_num_fractional =
      fractional_variables(settings_, leaf_solution.x, var_types_, leaf_fractional);

    f_t leaf_objective    = compute_objective(*worker.leaf_problem, leaf_solution.x);
    node_ptr->lower_bound = leaf_objective;

    // Queue pseudo-cost update for deterministic application at sync
    if (node_ptr->branch_var >= 0) {
      const f_t change_in_obj = leaf_objective - node_ptr->lower_bound;
      const f_t frac          = node_ptr->branch_dir == rounding_direction_t::DOWN
                                  ? node_ptr->fractional_val - std::floor(node_ptr->fractional_val)
                                  : std::ceil(node_ptr->fractional_val) - node_ptr->fractional_val;
      if (frac > 1e-10) {
        worker.queue_pseudo_cost_update(
          node_ptr->branch_var, node_ptr->branch_dir, change_in_obj / frac);
      }
    }

    if (leaf_num_fractional == 0) {
      // Integer feasible - queue for deterministic processing at sync
      if (leaf_objective < worker.local_upper_bound) {
        worker.local_upper_bound = leaf_objective;
        worker.integer_solutions.push_back({leaf_objective, leaf_solution.x, node_ptr->depth});
      }

      worker.record_integer_solution(node_ptr, leaf_objective);
      worker.track_integer_solution();
      worker.track_node_processed();
      worker.recompute_bounds_and_basis = true;

      BSP_DEBUG_LOG_SOLVE_END(bsp_debug_settings_,
                              bsp_debug_logger_,
                              worker.clock,
                              worker.worker_id,
                              node_ptr->node_id,
                              node_ptr->origin_worker_id,
                              "INTEGER",
                              leaf_objective);
      BSP_DEBUG_LOG_INTEGER(bsp_debug_settings_,
                            bsp_debug_logger_,
                            worker.clock,
                            worker.worker_id,
                            node_ptr->node_id,
                            leaf_objective);

      search_tree.update(node_ptr, node_status_t::INTEGER_FEASIBLE);
      return node_solve_info_t::NO_CHILDREN;

    } else if (leaf_objective <= worker.local_upper_bound + settings_.absolute_mip_gap_tol / 10) {
      // Branch - use worker-local upper bound for deterministic pruning decision
      // Use pseudo-cost snapshot for variable selection
      const i_t branch_var =
        worker.variable_selection_from_snapshot(leaf_fractional, leaf_solution.x);

      logger_t log;
      log.log = false;

      search_tree.branch(
        node_ptr, branch_var, leaf_solution.x[branch_var], leaf_vstatus, *worker.leaf_problem, log);
      search_tree.update(node_ptr, node_status_t::HAS_CHILDREN);

      i_t down_child_id = node_ptr->get_down_child()->node_id;
      i_t up_child_id   = node_ptr->get_up_child()->node_id;
      worker.record_branched(
        node_ptr, down_child_id, up_child_id, branch_var, leaf_solution.x[branch_var]);
      worker.track_node_branched();
      worker.track_node_processed();

      BSP_DEBUG_LOG_SOLVE_END(bsp_debug_settings_,
                              bsp_debug_logger_,
                              worker.clock,
                              worker.worker_id,
                              node_ptr->node_id,
                              node_ptr->origin_worker_id,
                              "BRANCH",
                              leaf_objective);
      BSP_DEBUG_LOG_BRANCHED(bsp_debug_settings_,
                             bsp_debug_logger_,
                             worker.clock,
                             worker.worker_id,
                             node_ptr->node_id,
                             node_ptr->origin_worker_id,
                             down_child_id,
                             up_child_id);

      exploration_stats_.nodes_unexplored += 2;

      rounding_direction_t preferred =
        martin_criteria(leaf_solution.x[branch_var], root_relax_soln_.x[branch_var]);
      worker.enqueue_children_for_plunge(
        node_ptr->get_down_child(), node_ptr->get_up_child(), preferred);

      return preferred == rounding_direction_t::DOWN ? node_solve_info_t::DOWN_CHILD_FIRST
                                                     : node_solve_info_t::UP_CHILD_FIRST;

    } else {
      // Record event and debug logs BEFORE search_tree.update() which may delete the node
      worker.record_fathomed(node_ptr, leaf_objective);
      worker.track_node_pruned();
      worker.track_node_processed();
      worker.recompute_bounds_and_basis = true;

      BSP_DEBUG_LOG_SOLVE_END(bsp_debug_settings_,
                              bsp_debug_logger_,
                              worker.clock,
                              worker.worker_id,
                              node_ptr->node_id,
                              node_ptr->origin_worker_id,
                              "FATHOMED",
                              leaf_objective);
      BSP_DEBUG_LOG_FATHOMED(bsp_debug_settings_,
                             bsp_debug_logger_,
                             worker.clock,
                             worker.worker_id,
                             node_ptr->node_id,
                             leaf_objective);

      search_tree.update(node_ptr, node_status_t::FATHOMED);
      return node_solve_info_t::NO_CHILDREN;
    }

  } else if (lp_status == dual::status_t::TIME_LIMIT) {
    return node_solve_info_t::TIME_LIMIT;

  } else {
    worker.record_numerical(node_ptr);
    worker.recompute_bounds_and_basis = true;
    search_tree.update(node_ptr, node_status_t::NUMERICAL);
    return node_solve_info_t::NUMERICAL;
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::process_history_and_sync(
  const bb_event_batch_t<i_t, f_t>& events)
{
  // Infeasible solutions from GPU heuristics are queued for repair; process them now
  {
    std::vector<std::vector<f_t>> to_repair;
    mutex_repair_.lock();
    if (repair_queue_.size() > 0) {
      to_repair = repair_queue_;
      repair_queue_.clear();
    }
    mutex_repair_.unlock();

    if (to_repair.size() > 0) {
      settings_.log.debug("BSP sync: Attempting to repair %ld injected solutions\n",
                          to_repair.size());
      for (const std::vector<f_t>& potential_solution : to_repair) {
        std::vector<f_t> repaired_solution;
        f_t repaired_obj;
        bool success =
          repair_solution(edge_norms_, potential_solution, repaired_obj, repaired_solution);
        if (success) {
          // Queue repaired solution with work unit timestamp (...workstamp?)
          mutex_heuristic_queue_.lock();
          heuristic_solution_queue_.push_back(
            {std::move(repaired_solution), repaired_obj, bsp_current_horizon_});
          mutex_heuristic_queue_.unlock();
        }
      }
    }
  }

  // Extract heuristic solutions, keeping future solutions for next horizon
  // Use bsp_current_horizon_ as the upper bound (horizon_end)
  std::vector<queued_heuristic_solution_t> heuristic_solutions;
  mutex_heuristic_queue_.lock();
  {
    std::vector<queued_heuristic_solution_t> future_solutions;
    for (auto& sol : heuristic_solution_queue_) {
      if (sol.wut < bsp_current_horizon_) {
        heuristic_solutions.push_back(std::move(sol));
      } else {
        future_solutions.push_back(std::move(sol));
      }
    }
    heuristic_solution_queue_ = std::move(future_solutions);
  }
  mutex_heuristic_queue_.unlock();

  // sort by work unit timestamp, with objective and solution values as tie-breakers
  std::sort(heuristic_solutions.begin(),
            heuristic_solutions.end(),
            [](const queued_heuristic_solution_t& a, const queued_heuristic_solution_t& b) {
              if (a.wut != b.wut) { return a.wut < b.wut; }
              if (a.objective != b.objective) { return a.objective < b.objective; }
              return a.solution < b.solution;  // edge-case - lexicographical comparison
            });

  // Merge B&B events and heuristic solutions for unified timeline replay
  size_t event_idx     = 0;
  size_t heuristic_idx = 0;

  while (event_idx < events.events.size() || heuristic_idx < heuristic_solutions.size()) {
    bool process_event     = false;
    bool process_heuristic = false;

    if (event_idx >= events.events.size()) {
      process_heuristic = true;
    } else if (heuristic_idx >= heuristic_solutions.size()) {
      process_event = true;
    } else {
      // Both have items - pick the one with smaller WUT
      if (events.events[event_idx].wut <= heuristic_solutions[heuristic_idx].wut) {
        process_event = true;
      } else {
        process_heuristic = true;
      }
    }

    if (process_event) {
      const auto& event = events.events[event_idx++];
      switch (event.type) {
        case bb_event_type_t::NODE_INTEGER:
        case bb_event_type_t::NODE_BRANCHED:
        case bb_event_type_t::NODE_FATHOMED:
        case bb_event_type_t::NODE_INFEASIBLE:
        case bb_event_type_t::NODE_NUMERICAL:
        case bb_event_type_t::HEURISTIC_SOLUTION: break;
      }
    }

    if (process_heuristic) {
      const auto& hsol = heuristic_solutions[heuristic_idx++];

      // Debug: Log heuristic received
      BSP_DEBUG_LOG_HEURISTIC_RECEIVED(
        bsp_debug_settings_, bsp_debug_logger_, hsol.wut, hsol.objective);

      // Process heuristic solution at its correct work unit timestamp position
      f_t new_upper = std::numeric_limits<f_t>::infinity();

      mutex_upper_.lock();
      if (hsol.objective < upper_bound_) {
        upper_bound_ = hsol.objective;
        incumbent_.set_incumbent_solution(hsol.objective, hsol.solution);
        new_upper = hsol.objective;

        // Debug: Log incumbent update
        BSP_DEBUG_LOG_INCUMBENT_UPDATE(
          bsp_debug_settings_, bsp_debug_logger_, hsol.wut, hsol.objective, "heuristic");
      }
      mutex_upper_.unlock();

      if (new_upper < std::numeric_limits<f_t>::infinity()) {
        f_t user_obj    = compute_user_objective(original_lp_, new_upper);
        f_t user_lower  = compute_user_objective(original_lp_, get_lower_bound());
        std::string gap = user_mip_gap<f_t>(user_obj, user_lower);

        settings_.log.printf(
          "H                           %+13.6e    %+10.6e                        %s %9.2f\n",
          user_obj,
          user_lower,
          gap.c_str(),
          toc(exploration_stats_.start_time));

        if (settings_.solution_callback != nullptr) {
          std::vector<f_t> original_x;
          uncrush_primal_solution(original_problem_, original_lp_, hsol.solution, original_x);
          settings_.solution_callback(original_x, hsol.objective);
        }
      }
    }
  }

  // Merge integer solutions from all workers and update global incumbent
  // Sort by (objective, worker_id) for deterministic winner selection
  // lexicographical sort as a fallback
  struct worker_solution_t {
    f_t objective;
    const std::vector<f_t>* solution;
    i_t depth;
    int worker_id;
  };
  std::vector<worker_solution_t> all_integer_solutions;
  for (auto& worker : *bsp_workers_) {
    for (auto& sol : worker.integer_solutions) {
      all_integer_solutions.push_back({sol.objective, &sol.solution, sol.depth, worker.worker_id});
    }
  }

  f_t bsp_lower     = compute_bsp_lower_bound();
  f_t current_upper = upper_bound_.load();

  for (const auto& sol : all_integer_solutions) {
    // improving solution found, log it
    if (sol.objective < current_upper) {
      f_t user_obj         = compute_user_objective(original_lp_, sol.objective);
      f_t user_lower       = compute_user_objective(original_lp_, bsp_lower);
      i_t nodes_explored   = exploration_stats_.nodes_explored.load();
      i_t nodes_unexplored = exploration_stats_.nodes_unexplored.load();
      settings_.log.printf(
        "%c %10d   %10lu    %+13.6e    %+10.6e   %6d   %7.1e     %s %9.2f\n",
        feasible_solution_symbol(bnb_worker_type_t::BEST_FIRST),
        nodes_explored,
        nodes_unexplored,
        user_obj,
        user_lower,
        sol.depth,
        nodes_explored > 0 ? exploration_stats_.total_lp_iters.load() / nodes_explored : 0.0,
        user_mip_gap<f_t>(user_obj, user_lower).c_str(),
        toc(exploration_stats_.start_time));

      // Update incumbent
      bool improved = false;
      mutex_upper_.lock();
      if (sol.objective < upper_bound_) {
        upper_bound_ = sol.objective;
        incumbent_.set_incumbent_solution(sol.objective, *sol.solution);
        current_upper = sol.objective;
        improved      = true;
      }
      mutex_upper_.unlock();

      // Notify diversity manager of new incumbent
      if (improved && settings_.solution_callback != nullptr) {
        std::vector<f_t> original_x;
        uncrush_primal_solution(original_problem_, original_lp_, *sol.solution, original_x);
        settings_.solution_callback(original_x, sol.objective);
      }
    }
  }

  // Merge and apply pseudo-cost updates from all workers in deterministic order
  std::vector<pseudo_cost_update_t<i_t, f_t>> all_pc_updates;
  for (auto& worker : *bsp_workers_) {
    for (auto& upd : worker.pseudo_cost_updates) {
      all_pc_updates.push_back(upd);
    }
  }

  for (const auto& upd : all_pc_updates) {
    if (upd.direction == rounding_direction_t::DOWN) {
      pc_.pseudo_cost_sum_down[upd.variable] += upd.delta;
      pc_.pseudo_cost_num_down[upd.variable]++;
    } else {
      pc_.pseudo_cost_sum_up[upd.variable] += upd.delta;
      pc_.pseudo_cost_num_up[upd.variable]++;
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::prune_worker_nodes_vs_incumbent()
{
  f_t upper_bound = upper_bound_.load();

  for (auto& worker : *bsp_workers_) {
    // Check nodes in plunge stack - filter in place
    {
      std::deque<mip_node_t<i_t, f_t>*> surviving;
      for (auto* node : worker.plunge_stack) {
        if (node->lower_bound >= upper_bound) {
          search_tree_.update(node, node_status_t::FATHOMED);
          --exploration_stats_.nodes_unexplored;
        } else {
          surviving.push_back(node);
        }
      }
      worker.plunge_stack = std::move(surviving);
    }

    // Check nodes in backlog - filter in place
    {
      std::vector<mip_node_t<i_t, f_t>*> surviving;
      for (auto* node : worker.backlog) {
        if (node->lower_bound >= upper_bound) {
          search_tree_.update(node, node_status_t::FATHOMED);
          --exploration_stats_.nodes_unexplored;
        } else {
          surviving.push_back(node);
        }
      }
      worker.backlog = std::move(surviving);
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::balance_worker_loads()
{
  const size_t num_workers = bsp_workers_->size();
  if (num_workers <= 1) return;

  constexpr bool force_rebalance_every_sync = true;

  // Count work for each worker: current_node (if any) + plunge_stack + backlog
  std::vector<size_t> work_counts(num_workers);
  size_t total_work = 0;
  size_t max_work   = 0;
  size_t min_work   = std::numeric_limits<size_t>::max();

  for (size_t w = 0; w < num_workers; ++w) {
    auto& worker   = (*bsp_workers_)[w];
    work_counts[w] = worker.queue_size();
    total_work += work_counts[w];
    max_work = std::max(max_work, work_counts[w]);
    min_work = std::min(min_work, work_counts[w]);
  }
  if (total_work == 0) return;

  bool needs_balance;
  if (force_rebalance_every_sync) {
    // Always rebalance if there's more than one node to distribute
    needs_balance = (total_work > 1);
  } else {
    needs_balance = (min_work == 0 && max_work >= 2) || (min_work > 0 && max_work > 4 * min_work);
  }

  if (!needs_balance) return;

  // Collect all redistributable nodes from worker queues
  std::vector<mip_node_t<i_t, f_t>*> all_nodes;
  for (auto& worker : *bsp_workers_) {
    // Extract backlog nodes
    for (auto* node : worker.backlog) {
      all_nodes.push_back(node);
    }
    worker.backlog.clear();

    // Extract plunge stack nodes
    for (auto* node : worker.plunge_stack) {
      all_nodes.push_back(node);
    }
    worker.plunge_stack.clear();
  }

  if (all_nodes.empty()) return;

  // Sort by BSP identity for deterministic distribution
  // Uses lexicographic order of (origin_worker_id, creation_seq)
  auto deterministic_less = [](const mip_node_t<i_t, f_t>* a, const mip_node_t<i_t, f_t>* b) {
    if (a->origin_worker_id != b->origin_worker_id) {
      return a->origin_worker_id < b->origin_worker_id;
    }
    return a->creation_seq < b->creation_seq;
  };
  std::sort(all_nodes.begin(), all_nodes.end(), deterministic_less);

  // Redistribute round-robin
  std::vector<size_t> worker_order;
  for (size_t w = 0; w < num_workers; ++w) {
    worker_order.push_back(w);
  }

  // Distribute nodes - use enqueue_node_with_identity to preserve existing identity
  for (size_t i = 0; i < all_nodes.size(); ++i) {
    size_t worker_idx = worker_order[i % num_workers];
    (*bsp_workers_)[worker_idx].enqueue_node_with_identity(all_nodes[i]);
    (*bsp_workers_)[worker_idx].track_node_assigned();

    double wut = bsp_current_horizon_;
    BSP_DEBUG_LOG_NODE_ASSIGNED(bsp_debug_settings_,
                                bsp_debug_logger_,
                                wut,
                                static_cast<int>(worker_idx),
                                all_nodes[i]->node_id,
                                all_nodes[i]->origin_worker_id,
                                all_nodes[i]->lower_bound);
  }
}

template <typename i_t, typename f_t>
f_t branch_and_bound_t<i_t, f_t>::compute_bsp_lower_bound()
{
  // Compute lower bound from BFS worker local structures only
  const f_t inf   = std::numeric_limits<f_t>::infinity();
  f_t lower_bound = inf;

  // Check all BFS worker queues
  for (const auto& worker : *bsp_workers_) {
    // Check paused node (current_node)
    if (worker.current_node != nullptr) {
      lower_bound = std::min(worker.current_node->lower_bound, lower_bound);
    }

    // Check plunge stack nodes
    for (auto* node : worker.plunge_stack) {
      lower_bound = std::min(node->lower_bound, lower_bound);
    }

    // Check backlog nodes
    for (auto* node : worker.backlog) {
      lower_bound = std::min(node->lower_bound, lower_bound);
    }
  }

  return lower_bound;
}

// ============================================================================
// BSP Diving
// ============================================================================

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::populate_diving_heap_at_sync()
{
  // Clear diving heap from previous horizon
  diving_heap_.clear();

  if (!bsp_diving_workers_ || bsp_diving_workers_->size() == 0) return;

  const int num_diving                  = bsp_diving_workers_->size();
  constexpr int target_nodes_per_worker = 10;
  const int target_total                = num_diving * target_nodes_per_worker;
  f_t upper_bound                       = upper_bound_.load();

  // Collect candidate nodes from BFS worker backlogs
  std::vector<std::pair<mip_node_t<i_t, f_t>*, f_t>> candidates;

  for (auto& worker : *bsp_workers_) {
    for (auto* node : worker.backlog) {
      if (node->lower_bound < upper_bound) {
        f_t score = node->objective_estimate;
        if (!std::isfinite(score)) { score = node->lower_bound; }
        candidates.push_back({node, score});
      }
    }
  }

  if (candidates.empty()) return;

  // Sort candidates by score (lower is better for diving - closer to optimum)
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.second < b.second;
  });

  int nodes_to_take = std::min(target_total, (int)candidates.size());

  for (int i = 0; i < nodes_to_take; ++i) {
    diving_heap_.push({candidates[i].first, candidates[i].second});
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::assign_diving_nodes()
{
  if (!bsp_diving_workers_ || bsp_diving_workers_->size() == 0) {
    diving_heap_.clear();
    return;
  }

  constexpr int target_nodes_per_worker = 10;

  // Round-robin assignment to balance load across workers
  int worker_idx        = 0;
  const int num_workers = bsp_diving_workers_->size();

  while (!diving_heap_.empty()) {
    auto& worker = (*bsp_diving_workers_)[worker_idx];

    // Skip workers that already have enough nodes
    if ((int)worker.dive_queue_size() >= target_nodes_per_worker) {
      worker_idx = (worker_idx + 1) % num_workers;
      // Check if all workers are full
      bool all_full = true;
      for (auto& w : *bsp_diving_workers_) {
        if ((int)w.dive_queue_size() < target_nodes_per_worker) {
          all_full = false;
          break;
        }
      }
      if (all_full) break;
      continue;
    }

    auto entry = diving_heap_.pop();
    if (entry.has_value()) { worker.enqueue_dive_node(entry.value().node); }

    worker_idx = (worker_idx + 1) % num_workers;
  }

  diving_heap_.clear();
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::merge_diving_solutions()
{
  if (!bsp_diving_workers_) return;

  // Collect all integer solutions from diving workers
  std::vector<queued_integer_solution_t<i_t, f_t>*> all_solutions;

  for (auto& worker : *bsp_diving_workers_) {
    for (auto& sol : worker.integer_solutions) {
      all_solutions.push_back(&sol);
    }
  }

  // Apply improving solutions to incumbent
  f_t current_upper = upper_bound_.load();
  for (const auto* sol : all_solutions) {
    if (sol->objective < current_upper) {
      f_t user_obj         = compute_user_objective(original_lp_, sol->objective);
      f_t bsp_lower        = compute_bsp_lower_bound();
      f_t user_lower       = compute_user_objective(original_lp_, bsp_lower);
      i_t nodes_explored   = exploration_stats_.nodes_explored.load();
      i_t nodes_unexplored = exploration_stats_.nodes_unexplored.load();

      settings_.log.printf("D %10d   %10d    %+13.6e    %+10.6e   %6d   %7.1e     %s %9.2f\n",
                           nodes_explored,
                           nodes_unexplored,
                           user_obj,
                           user_lower,
                           sol->depth,
                           nodes_explored > 0
                             ? (double)exploration_stats_.total_lp_iters.load() / nodes_explored
                             : 0.0,
                           user_mip_gap<f_t>(user_obj, user_lower).c_str(),
                           toc(exploration_stats_.start_time));

      bool improved = false;
      mutex_upper_.lock();
      if (sol->objective < upper_bound_) {
        upper_bound_ = sol->objective;
        incumbent_.set_incumbent_solution(sol->objective, sol->solution);
        current_upper = sol->objective;
        improved      = true;
      }
      mutex_upper_.unlock();

      // Notify diversity manager of new incumbent
      if (improved && settings_.solution_callback != nullptr) {
        std::vector<f_t> original_x;
        uncrush_primal_solution(original_problem_, original_lp_, sol->solution, original_x);
        settings_.solution_callback(original_x, sol->objective);
      }
    }
  }

  // Clear solution queues
  for (auto& worker : *bsp_diving_workers_) {
    worker.integer_solutions.clear();
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::run_diving_worker_loop(
  bsp_diving_worker_state_t<i_t, f_t>& worker)
{
  raft::common::nvtx::range scope("BB::diving_worker_loop");

  while (!bsp_terminated_.load() && !bsp_scheduler_->is_stopped() &&
         solver_status_ == mip_status_t::UNSET) {
    // Check time limit directly - don't wait for sync if time is up
    if (toc(exploration_stats_.start_time) > settings_.time_limit) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      bsp_terminated_.store(true);
      bsp_scheduler_->stop();  // Wake up workers waiting at barrier
      break;
    }

    // Process dives from queue until empty or horizon exhausted
    auto node_opt = worker.dequeue_dive_node();
    if (node_opt.has_value()) {
      dive_from_bsp(worker, std::move(node_opt.value()));
      continue;
    }

    // Queue empty - wait for next sync point where we'll be assigned new nodes
    f_t nowork_start            = tic();
    cuopt::sync_result_t result = bsp_scheduler_->wait_for_next_sync(worker.work_context);
    worker.total_nowork_time += toc(nowork_start);
    if (result == cuopt::sync_result_t::STOPPED) { break; }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::dive_from_bsp(bsp_diving_worker_state_t<i_t, f_t>& worker,
                                                 mip_node_t<i_t, f_t> starting_node)
{
  raft::common::nvtx::range scope("BB::dive_from_bsp");

  // Create local search tree for the dive
  search_tree_t<i_t, f_t> dive_tree(std::move(starting_node));
  std::deque<mip_node_t<i_t, f_t>*> stack;
  stack.push_front(&dive_tree.root);

  // Initialize bounds from root node
  worker.dive_lower = original_lp_.lower;
  worker.dive_upper = original_lp_.upper;
  std::fill(worker.node_presolver->bounds_changed.begin(),
            worker.node_presolver->bounds_changed.end(),
            false);
  dive_tree.root.get_variable_bounds(
    worker.dive_lower, worker.dive_upper, worker.node_presolver->bounds_changed);

  const i_t max_nodes_per_dive      = 100;
  const i_t max_backtrack_depth     = 5;
  i_t nodes_this_dive               = 0;
  worker.recompute_bounds_and_basis = true;

  while (!stack.empty() && solver_status_ == mip_status_t::UNSET && !bsp_terminated_.load() &&
         nodes_this_dive < max_nodes_per_dive) {
    // Check time limit directly
    if (toc(exploration_stats_.start_time) > settings_.time_limit) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      bsp_terminated_.store(true);
      bsp_scheduler_->stop();  // Wake up workers waiting at barrier
      break;
    }

    // Check horizon budget
    if (worker.work_context.global_work_units_elapsed >= worker.horizon_end) {
      bsp_scheduler_->wait_for_next_sync(worker.work_context);
      if (bsp_terminated_.load()) break;
    }

    mip_node_t<i_t, f_t>* node_ptr = stack.front();
    stack.pop_front();

    // Prune check using snapshot upper bound
    if (node_ptr->lower_bound >= worker.local_upper_bound) {
      worker.recompute_bounds_and_basis = true;
      continue;
    }

    // Setup bounds for this node
    std::fill(worker.node_presolver->bounds_changed.begin(),
              worker.node_presolver->bounds_changed.end(),
              false);

    if (worker.recompute_bounds_and_basis) {
      worker.leaf_problem->lower = worker.dive_lower;
      worker.leaf_problem->upper = worker.dive_upper;
      node_ptr->get_variable_bounds(worker.leaf_problem->lower,
                                    worker.leaf_problem->upper,
                                    worker.node_presolver->bounds_changed);
    } else {
      node_ptr->update_branched_variable_bounds(worker.leaf_problem->lower,
                                                worker.leaf_problem->upper,
                                                worker.node_presolver->bounds_changed);
    }

    double remaining_time = settings_.time_limit - toc(exploration_stats_.start_time);
    if (remaining_time <= 0) { break; }

    // Setup LP settings
    simplex_solver_settings_t<i_t, f_t> lp_settings = settings_;
    lp_settings.set_log(false);
    lp_settings.cut_off       = worker.local_upper_bound + settings_.dual_tol;
    lp_settings.inside_mip    = 2;
    lp_settings.time_limit    = remaining_time;
    lp_settings.scale_columns = false;

    // Solve LP relaxation
    lp_solution_t<i_t, f_t> leaf_solution(worker.leaf_problem->num_rows,
                                          worker.leaf_problem->num_cols);
    std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
    i_t node_iter                                = 0;
    f_t lp_start_time                            = tic();
    std::vector<f_t> leaf_edge_norms             = edge_norms_;

    dual::status_t lp_status = dual_phase2_with_advanced_basis(2,
                                                               0,
                                                               worker.recompute_bounds_and_basis,
                                                               lp_start_time,
                                                               *worker.leaf_problem,
                                                               lp_settings,
                                                               leaf_vstatus,
                                                               *worker.basis_factors,
                                                               worker.basic_list,
                                                               worker.nonbasic_list,
                                                               leaf_solution,
                                                               node_iter,
                                                               leaf_edge_norms,
                                                               &worker.work_context);

    ++nodes_this_dive;
    ++worker.nodes_explored_this_horizon;
    ++worker.total_nodes_explored;

    worker.clock = worker.work_context.global_work_units_elapsed;

    if (lp_status == dual::status_t::TIME_LIMIT || lp_status == dual::status_t::WORK_LIMIT) {
      break;
    }

    if (lp_status == dual::status_t::DUAL_UNBOUNDED || lp_status == dual::status_t::CUTOFF) {
      worker.recompute_bounds_and_basis = true;
      continue;
    }

    if (lp_status == dual::status_t::OPTIMAL) {
      std::vector<i_t> leaf_fractional;
      fractional_variables(settings_, leaf_solution.x, var_types_, leaf_fractional);

      f_t leaf_objective    = compute_objective(*worker.leaf_problem, leaf_solution.x);
      node_ptr->lower_bound = leaf_objective;

      if (leaf_fractional.empty()) {
        // Integer feasible solution found!
        if (leaf_objective < worker.local_upper_bound) {
          worker.queue_integer_solution(leaf_objective, leaf_solution.x, node_ptr->depth);
        }
        worker.recompute_bounds_and_basis = true;
        continue;
      }

      if (leaf_objective <= worker.local_upper_bound + settings_.absolute_mip_gap_tol / 10) {
        // Branch - select variable using diving-type-specific strategy
        branch_variable_t<i_t> branch_result;

        switch (worker.diving_type) {
          case bnb_worker_type_t::PSEUDOCOST_DIVING:
            branch_result =
              worker.variable_selection_from_snapshot(leaf_fractional, leaf_solution.x);
            break;

          case bnb_worker_type_t::LINE_SEARCH_DIVING:
            if (worker.root_solution) {
              logger_t log;
              log.log       = false;
              branch_result = line_search_diving<i_t, f_t>(
                leaf_fractional, leaf_solution.x, *worker.root_solution, log);
            } else {
              branch_result =
                worker.variable_selection_from_snapshot(leaf_fractional, leaf_solution.x);
            }
            break;

          case bnb_worker_type_t::GUIDED_DIVING:
            branch_result = worker.guided_variable_selection(leaf_fractional, leaf_solution.x);
            break;

          case bnb_worker_type_t::COEFFICIENT_DIVING: {
            logger_t log;
            log.log       = false;
            branch_result = coefficient_diving<i_t, f_t>(*worker.leaf_problem,
                                                         leaf_fractional,
                                                         leaf_solution.x,
                                                         var_up_locks_,
                                                         var_down_locks_,
                                                         log);
          } break;

          default:
            branch_result =
              worker.variable_selection_from_snapshot(leaf_fractional, leaf_solution.x);
            break;
        }

        i_t branch_var                 = branch_result.variable;
        rounding_direction_t round_dir = branch_result.direction;

        if (branch_var < 0) {
          worker.recompute_bounds_and_basis = true;
          continue;
        }

        // Create children
        logger_t log;
        log.log = false;
        dive_tree.branch(node_ptr,
                         branch_var,
                         leaf_solution.x[branch_var],
                         leaf_vstatus,
                         *worker.leaf_problem,
                         log);

        // Add children to stack (preferred direction first)
        if (round_dir == rounding_direction_t::UP) {
          stack.push_front(node_ptr->get_down_child());
          stack.push_front(node_ptr->get_up_child());
        } else {
          stack.push_front(node_ptr->get_up_child());
          stack.push_front(node_ptr->get_down_child());
        }

        // Limit backtracking depth
        if (stack.size() > 1 && stack.front()->depth - stack.back()->depth > max_backtrack_depth) {
          stack.pop_back();
        }

        worker.recompute_bounds_and_basis = false;
      } else {
        // Fathomed by bound
        worker.recompute_bounds_and_basis = true;
      }
    } else {
      // Numerical or other error
      worker.recompute_bounds_and_basis = true;
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class branch_and_bound_t<int, double>;

#endif

}  // namespace cuopt::linear_programming::dual_simplex
