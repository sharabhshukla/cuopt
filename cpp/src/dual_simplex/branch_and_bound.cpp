/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <omp.h>
#include <algorithm>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/branch_and_bound.hpp>
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
#include <utilities/hashing.hpp>

#include <raft/common/nvtx.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <future>
#include <limits>
#include <optional>
#include <string>
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

inline const char* feasible_solution_symbol(thread_type_t type)
{
  switch (type) {
    case thread_type_t::EXPLORATION: return "B ";
    case thread_type_t::DIVING: return "D ";
    default: return "U ";
  }
}

inline bool has_children(node_solve_info_t status)
{
  return status == node_solve_info_t::UP_CHILD_FIRST ||
         status == node_solve_info_t::DOWN_CHILD_FIRST;
}

}  // namespace

template <typename i_t, typename f_t>
branch_and_bound_t<i_t, f_t>::branch_and_bound_t(
  const user_problem_t<i_t, f_t>& user_problem,
  const simplex_solver_settings_t<i_t, f_t>& solver_settings)
  : original_problem_(user_problem),
    settings_(solver_settings),
    original_lp_(user_problem.handle_ptr, 1, 1, 1),
    incumbent_(1),
    root_relax_soln_(1, 1),
    root_crossover_soln_(1, 1),
    pc_(1),
    solver_status_(mip_exploration_status_t::UNSET),
    bsp_debug_settings_(bsp_debug_settings_t::from_environment())
{
  bsp_debug_settings_.enable_all();
  bsp_debug_settings_.output_dir          = "/home/scratch.yboucher_gpu_1/bsp_debug/";
  bsp_debug_settings_.flush_every_horizon = false;
  bsp_debug_settings_.disable_all();

  exploration_stats_.start_time = tic();
  dualize_info_t<i_t, f_t> dualize_info;
  convert_user_problem(original_problem_, settings_, original_lp_, new_slacks_, dualize_info);
  full_variable_types(original_problem_, original_lp_, var_types_);

  mutex_upper_.lock();
  upper_bound_ = inf;
  mutex_upper_.unlock();

  // Compute static problem features for regression model
  compute_static_features();
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::compute_static_features()
{
  const auto& A               = original_lp_.A;
  static_features_.n_rows     = A.m;
  static_features_.n_cols     = A.n;
  static_features_.n_nonzeros = A.col_start[A.n];
  static_features_.density    = (f_t)static_features_.n_nonzeros / ((f_t)A.m * A.n);

  // Count variable types
  static_features_.n_binary     = 0;
  static_features_.n_integer    = 0;
  static_features_.n_continuous = 0;
  for (const auto& vt : var_types_) {
    if (vt == variable_type_t::BINARY) {
      static_features_.n_binary++;
    } else if (vt == variable_type_t::INTEGER) {
      static_features_.n_integer++;
    } else {
      static_features_.n_continuous++;
    }
  }
  static_features_.integrality_ratio =
    (f_t)(static_features_.n_binary + static_features_.n_integer) / A.n;

  // Compute row statistics (constraint sizes)
  std::vector<i_t> row_nnz(A.m, 0);
  for (i_t j = 0; j < A.n; j++) {
    for (i_t k = A.col_start[j]; k < A.col_start[j + 1]; k++) {
      row_nnz[A.i[k]]++;
    }
  }

  static_features_.max_row_nnz = 0;
  f_t sum_row_nnz              = 0;
  for (i_t i = 0; i < A.m; i++) {
    static_features_.max_row_nnz = std::max(static_features_.max_row_nnz, row_nnz[i]);
    sum_row_nnz += row_nnz[i];
  }
  static_features_.avg_row_nnz = sum_row_nnz / A.m;

  // Compute row coefficient of variation
  f_t row_variance = 0;
  for (i_t i = 0; i < A.m; i++) {
    f_t diff = row_nnz[i] - static_features_.avg_row_nnz;
    row_variance += diff * diff;
  }
  row_variance /= A.m;
  f_t row_std = std::sqrt(row_variance);
  static_features_.row_nnz_cv =
    static_features_.avg_row_nnz > 0 ? row_std / static_features_.avg_row_nnz : 0.0;

  // Compute column statistics (variable degrees)
  static_features_.max_col_nnz = 0;
  f_t sum_col_nnz              = 0;
  for (i_t j = 0; j < A.n; j++) {
    i_t col_nnz                  = A.col_start[j + 1] - A.col_start[j];
    static_features_.max_col_nnz = std::max(static_features_.max_col_nnz, col_nnz);
    sum_col_nnz += col_nnz;
  }
  static_features_.avg_col_nnz = sum_col_nnz / A.n;

  // Compute column coefficient of variation
  f_t col_variance = 0;
  for (i_t j = 0; j < A.n; j++) {
    i_t col_nnz = A.col_start[j + 1] - A.col_start[j];
    f_t diff    = col_nnz - static_features_.avg_col_nnz;
    col_variance += diff * diff;
  }
  col_variance /= A.n;
  f_t col_std = std::sqrt(col_variance);
  static_features_.col_nnz_cv =
    static_features_.avg_col_nnz > 0 ? col_std / static_features_.avg_col_nnz : 0.0;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::flush_pending_features()
{
  // Must be called with mutex_feature_log_ already locked
  if (!has_pending_features_) return;

  constexpr int LINE_BUFFER_SIZE = 512;
  char line_buffer[LINE_BUFFER_SIZE];

  snprintf(line_buffer,
           LINE_BUFFER_SIZE,
           "BB_NODE_FEATURES "
           "node_id=%d depth=%d time=%.6f "
           "n_rows=%d n_cols=%d n_nnz=%d density=%.6f "
           "n_bin=%d n_int=%d n_cont=%d int_ratio=%.4f "
           "avg_row_nnz=%.2f max_row_nnz=%d row_nnz_cv=%.4f "
           "avg_col_nnz=%.2f max_col_nnz=%d col_nnz_cv=%.4f "
           "n_bounds_chg=%d cutoff_gap=%.4f basis_from_parent=%d "
           "simplex_iters=%d n_refact=%d lp_time=%.6f bound_str_time=%.6f var_sel_time=%.6f "
           "n_frac=%d strong_branch=%d n_sb_cand=%d sb_time=%.6f "
           "lp_status=%d node_status=%d\n",
           last_features_.node_id,
           last_features_.node_depth,
           last_features_.total_node_time,
           last_features_.n_rows,
           last_features_.n_cols,
           last_features_.n_nonzeros,
           last_features_.density,
           last_features_.n_binary,
           last_features_.n_integer,
           last_features_.n_continuous,
           last_features_.integrality_ratio,
           last_features_.avg_row_nnz,
           last_features_.max_row_nnz,
           last_features_.row_nnz_cv,
           last_features_.avg_col_nnz,
           last_features_.max_col_nnz,
           last_features_.col_nnz_cv,
           last_features_.n_bounds_changed,
           last_features_.cutoff_gap_ratio,
           last_features_.basis_from_parent ? 1 : 0,
           last_features_.simplex_iterations,
           last_features_.n_refactorizations,
           last_features_.lp_solve_time,
           last_features_.bound_str_time,
           last_features_.variable_sel_time,
           last_features_.n_fractional,
           last_features_.strong_branch_performed ? 1 : 0,
           last_features_.n_strong_branch_candidates,
           last_features_.strong_branch_time,
           last_features_.lp_status,
           last_features_.node_status);

  // Single printf call
  settings_.log.printf("%s", line_buffer);

  has_pending_features_ = false;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::log_node_features(
  const node_solve_features_t<i_t, f_t>& features)
{
  mutex_feature_log_.lock();

  f_t current_time        = toc(exploration_stats_.start_time);
  f_t time_since_last_log = current_time - last_feature_log_time_;

  // Always store the latest features
  last_features_        = features;
  has_pending_features_ = true;

  // Log if enough time has passed (500ms)
  if (time_since_last_log >= FEATURE_LOG_INTERVAL) {
    flush_pending_features();
    last_feature_log_time_ = current_time;
  }

  mutex_feature_log_.unlock();
}

template <typename i_t, typename f_t>
f_t branch_and_bound_t<i_t, f_t>::get_upper_bound()
{
  mutex_upper_.lock();
  const f_t upper_bound = upper_bound_;
  mutex_upper_.unlock();
  return upper_bound;
}

template <typename i_t, typename f_t>
f_t branch_and_bound_t<i_t, f_t>::get_lower_bound()
{
  f_t lower_bound = lower_bound_ceiling_.load();
  mutex_heap_.lock();
  if (heap_.size() > 0) { lower_bound = std::min(heap_.top()->lower_bound, lower_bound); }
  mutex_heap_.unlock();

  for (i_t i = 0; i < local_lower_bounds_.size(); ++i) {
    lower_bound = std::min(local_lower_bounds_[i].load(), lower_bound);
  }

  return lower_bound;
}

template <typename i_t, typename f_t>
i_t branch_and_bound_t<i_t, f_t>::get_heap_size()
{
  mutex_heap_.lock();
  i_t size = heap_.size();
  mutex_heap_.unlock();
  return size;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::report_heuristic(f_t obj)
{
  if (solver_status_ == mip_exploration_status_t::RUNNING) {
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
void branch_and_bound_t<i_t, f_t>::report(std::string symbol,
                                          f_t obj,
                                          f_t lower_bound,
                                          i_t node_depth)
{
  i_t nodes_explored   = exploration_stats_.nodes_explored;
  i_t nodes_unexplored = exploration_stats_.nodes_unexplored;
  f_t user_obj         = compute_user_objective(original_lp_, obj);
  f_t user_lower       = compute_user_objective(original_lp_, lower_bound);
  f_t iter_node        = exploration_stats_.total_lp_iters / nodes_explored;
  std::string user_gap = user_mip_gap<f_t>(user_obj, user_lower);
  settings_.log.printf("%s%10d   %10lu    %+13.6e    %+10.6e  %6d    %7.1e     %s %9.2f\n",
                       symbol.c_str(),
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
                                                                  double vt_timestamp)
{
  // In BSP mode, queue the solution to be processed at the correct virtual time
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

  // Queue the solution with its VT timestamp
  mutex_heuristic_queue_.lock();
  heuristic_solution_queue_.push_back({std::move(crushed_solution), obj, vt_timestamp});
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
  mip_status_t mip_status = mip_status_t::UNSET;

  if (solver_status_ == mip_exploration_status_t::NUMERICAL) {
    settings_.log.printf("Numerical issue encountered. Stopping the solver...\n");
    mip_status = mip_status_t::NUMERICAL;
  }

  if (solver_status_ == mip_exploration_status_t::TIME_LIMIT) {
    settings_.log.printf("Time limit reached. Stopping the solver...\n");
    mip_status = mip_status_t::TIME_LIMIT;
  }
  if (solver_status_ == mip_exploration_status_t::WORK_LIMIT) {
    settings_.log.printf("Work limit reached. Stopping the solver...\n");
    mip_status = mip_status_t::WORK_LIMIT;
  }
  if (solver_status_ == mip_exploration_status_t::NODE_LIMIT) {
    settings_.log.printf("Node limit reached. Stopping the solver...\n");
    mip_status = mip_status_t::NODE_LIMIT;
  }

  // Signal heuristic thread to stop for any limit-based termination
  if (mip_status == mip_status_t::TIME_LIMIT || mip_status == mip_status_t::WORK_LIMIT ||
      mip_status == mip_status_t::NODE_LIMIT || mip_status == mip_status_t::NUMERICAL) {
    if (settings_.heuristic_preemption_callback != nullptr) {
      settings_.heuristic_preemption_callback();
    }
  }

  f_t upper_bound      = get_upper_bound();
  f_t gap              = upper_bound - lower_bound;
  f_t obj              = compute_user_objective(original_lp_, upper_bound);
  f_t user_bound       = compute_user_objective(original_lp_, lower_bound);
  f_t gap_rel          = user_relative_gap(original_lp_, upper_bound, lower_bound);
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
    mip_status = mip_status_t::OPTIMAL;
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

  if (solver_status_ == mip_exploration_status_t::COMPLETED) {
    if (exploration_stats_.nodes_explored > 0 && exploration_stats_.nodes_unexplored == 0 &&
        upper_bound == inf) {
      settings_.log.printf("Integer infeasible.\n");
      mip_status = mip_status_t::INFEASIBLE;
      if (settings_.heuristic_preemption_callback != nullptr) {
        settings_.heuristic_preemption_callback();
      }
    }
  }

  if (upper_bound != inf) {
    assert(incumbent_.has_incumbent);
    uncrush_primal_solution(original_problem_, original_lp_, incumbent_.x, solution.x);
  }
  solution.objective          = incumbent_.objective;
  solution.lower_bound        = lower_bound;
  solution.nodes_explored     = exploration_stats_.nodes_explored;
  solution.simplex_iterations = exploration_stats_.total_lp_iters;

  return mip_status;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::add_feasible_solution(f_t leaf_objective,
                                                         const std::vector<f_t>& leaf_solution,
                                                         i_t leaf_depth,
                                                         thread_type_t thread_type)
{
  bool send_solution      = false;
  bool improved_incumbent = false;
  i_t nodes_explored      = exploration_stats_.nodes_explored;
  i_t nodes_unexplored    = exploration_stats_.nodes_unexplored;

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

  // Debug: Log incumbent update (after releasing mutex to avoid potential issues)
  if (improved_incumbent && bsp_mode_enabled_) {
    std::string source = (thread_type == thread_type_t::EXPLORATION) ? "bb_integer" : "diving";
    BSP_DEBUG_LOG_INCUMBENT_UPDATE(
      bsp_debug_settings_, bsp_debug_logger_, bsp_current_horizon_, leaf_objective, source);
  }
}

template <typename i_t, typename f_t>
rounding_direction_t branch_and_bound_t<i_t, f_t>::child_selection(mip_node_t<i_t, f_t>* node_ptr)
{
  const i_t branch_var     = node_ptr->get_down_child()->branch_var;
  const f_t branch_var_val = node_ptr->get_down_child()->fractional_val;
  const f_t down_val       = std::floor(root_relax_soln_.x[branch_var]);
  const f_t up_val         = std::ceil(root_relax_soln_.x[branch_var]);
  const f_t down_dist      = branch_var_val - down_val;
  const f_t up_dist        = up_val - branch_var_val;
  constexpr f_t eps        = 1e-6;

  if (down_dist < up_dist + eps) {
    return rounding_direction_t::DOWN;

  } else {
    return rounding_direction_t::UP;
  }
}

template <typename i_t, typename f_t>
node_solve_info_t branch_and_bound_t<i_t, f_t>::solve_node(
  mip_node_t<i_t, f_t>* node_ptr,
  search_tree_t<i_t, f_t>& search_tree,
  lp_problem_t<i_t, f_t>& leaf_problem,
  basis_update_mpf_t<i_t, f_t>& basis_factors,
  std::vector<i_t>& basic_list,
  std::vector<i_t>& nonbasic_list,
  bounds_strengthening_t<i_t, f_t>& node_presolver,
  thread_type_t thread_type,
  bool recompute_bounds_and_basis,
  const std::vector<f_t>& root_lower,
  const std::vector<f_t>& root_upper,
  logger_t& log)
{
  raft::common::nvtx::range scope("BB::solve_node");

  // Initialize feature tracking for this node
  node_solve_features_t<i_t, f_t> features = static_features_;
  f_t node_start_time                      = tic();
  features.node_id                         = node_ptr->node_id;
  features.node_depth                      = node_ptr->depth;
  const f_t abs_fathom_tol                 = settings_.absolute_mip_gap_tol / 10;
  const f_t upper_bound                    = get_upper_bound();

  lp_solution_t<i_t, f_t> leaf_solution(leaf_problem.num_rows, leaf_problem.num_cols);
  std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
  assert(leaf_vstatus.size() == leaf_problem.num_cols);

  // Track cutoff gap ratio
  if (upper_bound < inf) {
    features.cutoff_gap_ratio =
      (upper_bound - node_ptr->lower_bound) / std::max(std::abs(upper_bound), f_t(1.0));
  }

  // Track if we have parent's basis
  features.basis_from_parent = !leaf_vstatus.empty();

  simplex_solver_settings_t lp_settings = settings_;
  lp_settings.set_log(false);
  lp_settings.cut_off       = upper_bound + settings_.dual_tol;
  lp_settings.inside_mip    = 2;
  lp_settings.time_limit    = settings_.time_limit - toc(exploration_stats_.start_time);
  lp_settings.work_limit    = settings_.work_limit;
  lp_settings.scale_columns = false;

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
    f_t bs_start_time = tic();
    feasible =
      node_presolver.bounds_strengthening(leaf_problem.lower, leaf_problem.upper, lp_settings);
    features.bound_str_time = toc(bs_start_time);
  }

  dual::status_t lp_status = dual::status_t::DUAL_UNBOUNDED;

  if (feasible) {
    i_t node_iter                    = 0;
    f_t lp_start_time                = tic();
    std::vector<f_t> leaf_edge_norms = edge_norms_;  // = node.steepest_edge_norms;

    {
      raft::common::nvtx::range scope_lp("BB::node_lp_solve");
      lp_status =
        dual_phase2_with_advanced_basis(2,
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
                                        settings_.deterministic ? &work_unit_context_ : nullptr);
    }
    if (settings_.deterministic &&
        work_unit_context_.global_work_units_elapsed >= settings_.work_limit) {
      lp_status = dual::status_t::WORK_LIMIT;
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

    f_t lp_time = toc(lp_start_time);
    if (thread_type == thread_type_t::EXPLORATION) {
      exploration_stats_.total_lp_solve_time += lp_time;
      exploration_stats_.total_lp_iters += node_iter;

      // Track LP solve metrics
      features.lp_solve_time      = lp_time;
      features.simplex_iterations = node_iter;
      // Note: We don't directly track refactorizations here, would need instrumentation in
      // dual_phase2
    }
  }

#ifdef LOG_NODE_SIMPLEX
  lp_settings.log.printf("\nLP status: %d\n\n", lp_status);
#endif

  if (lp_status == dual::status_t::DUAL_UNBOUNDED) {
    // Node was infeasible. Do not branch
    node_ptr->lower_bound = inf;
    search_tree.graphviz_node(log, node_ptr, "infeasible", 0.0);
    search_tree.update(node_ptr, node_status_t::INFEASIBLE);

    // Log features before return
    features.lp_status       = static_cast<i_t>(lp_status);
    features.node_status     = static_cast<i_t>(node_status_t::INFEASIBLE);
    features.total_node_time = toc(node_start_time);
    log_node_features(features);

    return node_solve_info_t::NO_CHILDREN;

  } else if (lp_status == dual::status_t::CUTOFF) {
    // Node was cut off. Do not branch
    node_ptr->lower_bound = upper_bound;
    f_t leaf_objective    = compute_objective(leaf_problem, leaf_solution.x);
    search_tree.graphviz_node(log, node_ptr, "cut off", leaf_objective);
    search_tree.update(node_ptr, node_status_t::FATHOMED);

    // Log features before return
    features.lp_status       = static_cast<i_t>(lp_status);
    features.node_status     = static_cast<i_t>(node_status_t::FATHOMED);
    features.total_node_time = toc(node_start_time);
    log_node_features(features);

    return node_solve_info_t::NO_CHILDREN;

  } else if (lp_status == dual::status_t::OPTIMAL) {
    // LP was feasible
    std::vector<i_t> leaf_fractional;
    i_t leaf_num_fractional =
      fractional_variables(settings_, leaf_solution.x, var_types_, leaf_fractional);

    features.n_fractional = leaf_num_fractional;

    f_t leaf_objective    = compute_objective(leaf_problem, leaf_solution.x);
    node_ptr->lower_bound = leaf_objective;
    search_tree.graphviz_node(log, node_ptr, "lower bound", leaf_objective);
    pc_.update_pseudo_costs(node_ptr, leaf_objective);

    if (settings_.node_processed_callback != nullptr) {
      std::vector<f_t> original_x;
      uncrush_primal_solution(original_problem_, original_lp_, leaf_solution.x, original_x);
      settings_.node_processed_callback(original_x, leaf_objective);
    }

    if (leaf_num_fractional == 0) {
      // Found a integer feasible solution
      add_feasible_solution(leaf_objective, leaf_solution.x, node_ptr->depth, thread_type);
      search_tree.graphviz_node(log, node_ptr, "integer feasible", leaf_objective);
      search_tree.update(node_ptr, node_status_t::INTEGER_FEASIBLE);

      // Log features before return
      features.lp_status       = static_cast<i_t>(lp_status);
      features.node_status     = static_cast<i_t>(node_status_t::INTEGER_FEASIBLE);
      features.total_node_time = toc(node_start_time);
      log_node_features(features);

      return node_solve_info_t::NO_CHILDREN;

    } else if (leaf_objective <= upper_bound + abs_fathom_tol) {
      // Choose fractional variable to branch on
      f_t var_sel_start = tic();
      const i_t branch_var =
        pc_.variable_selection(leaf_fractional, leaf_solution.x, lp_settings.log);
      features.variable_sel_time = toc(var_sel_start);

      assert(leaf_vstatus.size() == leaf_problem.num_cols);
      search_tree.branch(
        node_ptr, branch_var, leaf_solution.x[branch_var], leaf_vstatus, leaf_problem, log);
      search_tree.update(node_ptr, node_status_t::HAS_CHILDREN);

      // Log features before return
      features.lp_status       = static_cast<i_t>(lp_status);
      features.node_status     = static_cast<i_t>(node_status_t::HAS_CHILDREN);
      features.total_node_time = toc(node_start_time);
      log_node_features(features);

      rounding_direction_t round_dir = child_selection(node_ptr);

      if (round_dir == rounding_direction_t::UP) {
        return node_solve_info_t::UP_CHILD_FIRST;
      } else {
        return node_solve_info_t::DOWN_CHILD_FIRST;
      }

    } else {
      search_tree.graphviz_node(log, node_ptr, "fathomed", leaf_objective);
      search_tree.update(node_ptr, node_status_t::FATHOMED);

      // Log features before return
      features.lp_status       = static_cast<i_t>(lp_status);
      features.node_status     = static_cast<i_t>(node_status_t::FATHOMED);
      features.total_node_time = toc(node_start_time);
      log_node_features(features);

      return node_solve_info_t::NO_CHILDREN;
    }
  } else if (lp_status == dual::status_t::TIME_LIMIT) {
    search_tree.graphviz_node(log, node_ptr, "timeout", 0.0);

    return node_solve_info_t::TIME_LIMIT;
  } else if (lp_status == dual::status_t::WORK_LIMIT) {
    search_tree.graphviz_node(log, node_ptr, "work limit", 0.0);
    return node_solve_info_t::WORK_LIMIT;
  } else {
    if (thread_type == thread_type_t::EXPLORATION) {
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

    // Log features before return
    features.lp_status       = static_cast<i_t>(lp_status);
    features.node_status     = static_cast<i_t>(node_status_t::NUMERICAL);
    features.total_node_time = toc(node_start_time);
    log_node_features(features);

    return node_solve_info_t::NUMERICAL;
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::exploration_ramp_up(mip_node_t<i_t, f_t>* node,
                                                       search_tree_t<i_t, f_t>* search_tree,
                                                       const csr_matrix_t<i_t, f_t>& Arow,
                                                       i_t initial_heap_size)
{
  if (solver_status_ != mip_exploration_status_t::RUNNING) { return; }

  // Note that we do not know which thread will execute the
  // `exploration_ramp_up` task, so we allow to any thread
  // to repair the heuristic solution.
  repair_heuristic_solutions();

  f_t lower_bound = node->lower_bound;
  f_t upper_bound = get_upper_bound();
  f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);
  f_t abs_gap     = upper_bound - lower_bound;

  if (lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
    search_tree->graphviz_node(settings_.log, node, "cutoff", node->lower_bound);
    search_tree->update(node, node_status_t::FATHOMED);
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
      report("  ", upper_bound, root_objective_, node->depth);
      exploration_stats_.nodes_since_last_log = 0;
      exploration_stats_.last_log             = tic();
      should_report_                          = true;
    }
  }

  if (now > settings_.time_limit) {
    solver_status_ = mip_exploration_status_t::TIME_LIMIT;
    return;
  }

  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  node_solve_info_t status = solve_node(node,
                                        *search_tree,
                                        leaf_problem,
                                        basis_factors,
                                        basic_list,
                                        nonbasic_list,
                                        node_presolver,
                                        thread_type_t::EXPLORATION,
                                        true,
                                        original_lp_.lower,
                                        original_lp_.upper,
                                        settings_.log);

  ++exploration_stats_.nodes_since_last_log;
  ++exploration_stats_.nodes_explored;
  --exploration_stats_.nodes_unexplored;

  if (status == node_solve_info_t::TIME_LIMIT) {
    solver_status_ = mip_exploration_status_t::TIME_LIMIT;
    return;

  } else if (status == node_solve_info_t::WORK_LIMIT) {
    solver_status_ = mip_exploration_status_t::WORK_LIMIT;
    return;

  } else if (has_children(status)) {
    exploration_stats_.nodes_unexplored += 2;

    // If we haven't generated enough nodes to keep the threads busy, continue the ramp up phase
    if (exploration_stats_.nodes_unexplored < initial_heap_size) {
#pragma omp task
      exploration_ramp_up(node->get_down_child(), search_tree, Arow, initial_heap_size);

#pragma omp task
      exploration_ramp_up(node->get_up_child(), search_tree, Arow, initial_heap_size);

    } else {
      // We've generated enough nodes, push further nodes onto the heap
      mutex_heap_.lock();
      heap_.push(node->get_down_child());
      heap_.push(node->get_up_child());
      mutex_heap_.unlock();
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::explore_subtree(i_t task_id,
                                                   mip_node_t<i_t, f_t>* start_node,
                                                   search_tree_t<i_t, f_t>& search_tree,
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

  while (stack.size() > 0 && solver_status_ == mip_exploration_status_t::RUNNING) {
    if (task_id == 0) { repair_heuristic_solutions(); }

    mip_node_t<i_t, f_t>* node_ptr = stack.front();
    stack.pop_front();

    f_t lower_bound = node_ptr->lower_bound;
    f_t upper_bound = get_upper_bound();
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
      search_tree.graphviz_node(settings_.log, node_ptr, "cutoff", node_ptr->lower_bound);
      search_tree.update(node_ptr, node_status_t::FATHOMED);
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
        report("  ", upper_bound, get_lower_bound(), node_ptr->depth);
        exploration_stats_.last_log             = tic();
        exploration_stats_.nodes_since_last_log = 0;
      }
    }

    if (now > settings_.time_limit) {
      solver_status_ = mip_exploration_status_t::TIME_LIMIT;
      return;
    }
    if (exploration_stats_.nodes_explored >= settings_.node_limit) {
      solver_status_ = mip_exploration_status_t::NODE_LIMIT;
      return;
    }

    node_solve_info_t status = solve_node(node_ptr,
                                          search_tree,
                                          leaf_problem,
                                          basis_factors,
                                          basic_list,
                                          nonbasic_list,
                                          node_presolver,
                                          thread_type_t::EXPLORATION,
                                          recompute_bounds_and_basis,
                                          original_lp_.lower,
                                          original_lp_.upper,
                                          settings_.log);

    recompute_bounds_and_basis = !has_children(status);

    ++exploration_stats_.nodes_since_last_log;
    ++exploration_stats_.nodes_explored;
    --exploration_stats_.nodes_unexplored;

    if (status == node_solve_info_t::TIME_LIMIT) {
      solver_status_ = mip_exploration_status_t::TIME_LIMIT;
      return;

    } else if (status == node_solve_info_t::WORK_LIMIT) {
      solver_status_ = mip_exploration_status_t::WORK_LIMIT;
      return;

    } else if (has_children(status)) {
      // The stack should only contain the children of the current parent.
      // If the stack size is greater than 0,
      // we pop the current node from the stack and place it in the global heap,
      // since we are about to add the two children to the stack
      if (stack.size() > 0) {
        mip_node_t<i_t, f_t>* node = stack.back();
        stack.pop_back();

        // The order here matters. We want to create a copy of the node
        // before adding to the global heap. Otherwise,
        // some thread may consume the node (possibly fathoming it)
        // before we had the chance to add to the diving queue.
        // This lead to a SIGSEGV. Although, in this case, it
        // would be better if we discard the node instead.
        if (get_heap_size() > settings_.num_bfs_threads) {
          std::vector<f_t> lower = original_lp_.lower;
          std::vector<f_t> upper = original_lp_.upper;
          std::fill(
            node_presolver.bounds_changed.begin(), node_presolver.bounds_changed.end(), false);
          node->get_variable_bounds(lower, upper, node_presolver.bounds_changed);

          mutex_dive_queue_.lock();
          diving_queue_.emplace(node->detach_copy(), std::move(lower), std::move(upper));
          mutex_dive_queue_.unlock();
        }

        mutex_heap_.lock();
        heap_.push(node);
        mutex_heap_.unlock();
      }

      exploration_stats_.nodes_unexplored += 2;

      if (status == node_solve_info_t::UP_CHILD_FIRST) {
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
void branch_and_bound_t<i_t, f_t>::best_first_thread(i_t task_id,
                                                     search_tree_t<i_t, f_t>& search_tree,
                                                     const csr_matrix_t<i_t, f_t>& Arow)
{
  raft::common::nvtx::range scope("BB::best_first_thread");
  f_t lower_bound = -inf;
  f_t upper_bound = inf;
  f_t abs_gap     = inf;
  f_t rel_gap     = inf;

  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  while (solver_status_ == mip_exploration_status_t::RUNNING &&
         abs_gap > settings_.absolute_mip_gap_tol && rel_gap > settings_.relative_mip_gap_tol &&
         (active_subtrees_ > 0 || get_heap_size() > 0)) {
    mip_node_t<i_t, f_t>* start_node = nullptr;

    // If there any node left in the heap, we pop the top node and explore it.
    mutex_heap_.lock();
    if (heap_.size() > 0) {
      start_node = heap_.top();
      heap_.pop();
      active_subtrees_++;
    }
    mutex_heap_.unlock();

    if (start_node != nullptr) {
      if (get_upper_bound() < start_node->lower_bound) {
        // This node was put on the heap earlier but its lower bound is now greater than the
        // current upper bound
        search_tree.graphviz_node(settings_.log, start_node, "cutoff", start_node->lower_bound);
        search_tree.update(start_node, node_status_t::FATHOMED);
        active_subtrees_--;
        continue;
      }

      // Best-first search with plunging
      explore_subtree(task_id,
                      start_node,
                      search_tree,
                      leaf_problem,
                      node_presolver,
                      basis_factors,
                      basic_list,
                      nonbasic_list);

      active_subtrees_--;
    }

    lower_bound = get_lower_bound();
    upper_bound = get_upper_bound();
    abs_gap     = upper_bound - lower_bound;
    rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);
  }

  // Check if it is the last thread that exited the loop and no
  // timeout or numerical error has happen.
  if (solver_status_ == mip_exploration_status_t::RUNNING) {
    if (active_subtrees_ == 0) {
      solver_status_ = mip_exploration_status_t::COMPLETED;
    } else {
      local_lower_bounds_[task_id] = inf;
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::diving_thread(const csr_matrix_t<i_t, f_t>& Arow)
{
  raft::common::nvtx::range scope("BB::diving_thread");
  logger_t log;
  log.log = false;
  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  while (solver_status_ == mip_exploration_status_t::RUNNING &&
         (active_subtrees_ > 0 || get_heap_size() > 0)) {
    std::optional<diving_root_t<i_t, f_t>> start_node;

    mutex_dive_queue_.lock();
    if (diving_queue_.size() > 0) { start_node = diving_queue_.pop(); }
    mutex_dive_queue_.unlock();

    if (start_node.has_value()) {
      if (get_upper_bound() < start_node->node.lower_bound) { continue; }

      bool recompute_bounds_and_basis = true;
      i_t nodes_explored              = 0;
      search_tree_t<i_t, f_t> subtree(std::move(start_node->node));
      std::deque<mip_node_t<i_t, f_t>*> stack;
      stack.push_front(&subtree.root);

      while (stack.size() > 0 && solver_status_ == mip_exploration_status_t::RUNNING) {
        mip_node_t<i_t, f_t>* node_ptr = stack.front();
        stack.pop_front();
        f_t upper_bound = get_upper_bound();
        f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, node_ptr->lower_bound);

        if (node_ptr->lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
          recompute_bounds_and_basis = true;
          continue;
        }

        if (toc(exploration_stats_.start_time) > settings_.time_limit) { return; }

        if (nodes_explored >= 1000) { break; }

        node_solve_info_t status = solve_node(node_ptr,
                                              subtree,
                                              leaf_problem,
                                              basis_factors,
                                              basic_list,
                                              nonbasic_list,
                                              node_presolver,
                                              thread_type_t::DIVING,
                                              recompute_bounds_and_basis,
                                              start_node->lower,
                                              start_node->upper,
                                              log);

        nodes_explored++;

        recompute_bounds_and_basis = !has_children(status);

        if (status == node_solve_info_t::TIME_LIMIT) {
          solver_status_ = mip_exploration_status_t::TIME_LIMIT;
          return;
        } else if (status == node_solve_info_t::WORK_LIMIT) {
          solver_status_ = mip_exploration_status_t::WORK_LIMIT;
          return;

        } else if (has_children(status)) {
          if (status == node_solve_info_t::UP_CHILD_FIRST) {
            stack.push_front(node_ptr->get_down_child());
            stack.push_front(node_ptr->get_up_child());
          } else {
            stack.push_front(node_ptr->get_up_child());
            stack.push_front(node_ptr->get_down_child());
          }
        }

        if (stack.size() > 1) {
          // If the diving thread is consuming the nodes faster than the
          // best first search, then we split the current subtree at the
          // lowest possible point and move to the queue, so it can
          // be picked by another thread.
          if (std::lock_guard<omp_mutex_t> lock(mutex_dive_queue_);
              diving_queue_.size() < min_diving_queue_size_) {
            mip_node_t<i_t, f_t>* new_node = stack.back();
            stack.pop_back();

            std::vector<f_t> lower = start_node->lower;
            std::vector<f_t> upper = start_node->upper;
            std::fill(
              node_presolver.bounds_changed.begin(), node_presolver.bounds_changed.end(), false);
            new_node->get_variable_bounds(lower, upper, node_presolver.bounds_changed);

            diving_queue_.emplace(new_node->detach_copy(), std::move(lower), std::move(upper));
          }
        }
      }
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
  solver_status_                      = mip_exploration_status_t::UNSET;
  exploration_stats_.nodes_unexplored = 0;
  exploration_stats_.nodes_explored   = 0;

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
    solver_status_ = mip_exploration_status_t::TIME_LIMIT;
    return set_final_solution(solution, -inf);
  }

  if (root_status == lp_status_t::WORK_LIMIT) {
    solver_status_ = mip_exploration_status_t::WORK_LIMIT;
    return set_final_solution(solution, -inf);
  }

  assert(root_vstatus_.size() == original_lp_.num_cols);

  // Validate root_vstatus_ has correct BASIC count
  // This catches bugs where the root LP solve produces an invalid vstatus
  {
    const i_t expected_basic_count = original_lp_.num_rows;
    i_t actual_basic_count         = 0;
    for (const auto& status : root_vstatus_) {
      if (status == variable_status_t::BASIC) { actual_basic_count++; }
    }
    if (actual_basic_count != expected_basic_count) {
      settings_.log.printf("ERROR: root_vstatus_ has %d BASIC entries, expected %d (num_rows)\n",
                           actual_basic_count,
                           expected_basic_count);
      assert(actual_basic_count == expected_basic_count &&
             "root_vstatus_ BASIC count mismatch - LP solver returned invalid basis");
    }
  }

  set_uninitialized_steepest_edge_norms<i_t, f_t>(edge_norms_);

  root_objective_ = compute_objective(original_lp_, root_relax_soln_.x);
  local_lower_bounds_.assign(settings_.num_bfs_threads, root_objective_);

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

  // Log strong branching results for determinism debugging
  {
    uint32_t sb_hash = pc_.compute_strong_branch_hash();
    uint32_t pc_hash = pc_.compute_state_hash();
    CUOPT_LOG_DEBUG("Strong branching completed: %zu variables, SB hash=0x%08x, PC hash=0x%08x",
                    fractional.size(),
                    sb_hash,
                    pc_hash);

    // Detailed logging for divergence diagnosis (enabled via environment variable)
    const char* log_sb_detail = std::getenv("CUOPT_LOG_STRONG_BRANCHING");
    if (log_sb_detail != nullptr && std::string(log_sb_detail) == "1") {
      settings_.log.printf("Strong branching detailed results:\n");
      for (size_t k = 0; k < fractional.size(); ++k) {
        i_t var = fractional[k];
        settings_.log.printf("  var[%zu]=%d: down=%+.10e, up=%+.10e\n",
                             k,
                             var,
                             pc_.strong_branch_down[k],
                             pc_.strong_branch_up[k]);
      }
      settings_.log.printf("Pseudo-cost state after strong branching:\n");
      i_t non_zero_count = 0;
      for (i_t j = 0; j < original_lp_.num_cols; ++j) {
        if (pc_.pseudo_cost_num_down[j] > 0 || pc_.pseudo_cost_num_up[j] > 0) {
          settings_.log.printf("  pc[%d]: sum_down=%+.10e, num_down=%d, sum_up=%+.10e, num_up=%d\n",
                               j,
                               pc_.pseudo_cost_sum_down[j],
                               pc_.pseudo_cost_num_down[j],
                               pc_.pseudo_cost_sum_up[j],
                               pc_.pseudo_cost_num_up[j]);
          ++non_zero_count;
        }
      }
      settings_.log.printf("Total %d variables with pseudo-cost data\n", non_zero_count);
    }
  }

  if (toc(exploration_stats_.start_time) > settings_.time_limit) {
    solver_status_ = mip_exploration_status_t::TIME_LIMIT;
    return set_final_solution(solution, root_objective_);
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

  csr_matrix_t<i_t, f_t> Arow(1, 1, 0);
  original_lp_.A.to_compressed_row(Arow);

  settings_.log.printf("Exploring the B&B tree using %d threads (best-first = %d, diving = %d)\n",
                       settings_.num_threads,
                       settings_.num_bfs_threads,
                       settings_.num_diving_threads);

  settings_.log.printf(
    "  | Explored | Unexplored |    Objective    |     Bound     | Depth | Iter/Node |   Gap    "
    "|  Time  |\n");

  exploration_stats_.nodes_explored       = 1;
  exploration_stats_.nodes_unexplored     = 2;
  exploration_stats_.nodes_since_last_log = 0;
  exploration_stats_.last_log             = tic();
  active_subtrees_                        = 0;
  min_diving_queue_size_                  = 4 * settings_.num_diving_threads;
  solver_status_                          = mip_exploration_status_t::RUNNING;
  lower_bound_ceiling_                    = inf;
  work_unit_context_.deterministic        = settings_.deterministic;
  should_report_                          = true;

  // Choose between BSP coordinator (deterministic) and opportunistic exploration
  if (settings_.deterministic && settings_.num_bfs_threads > 0) {
    // Use deterministic BSP coordinator for parallel execution
    settings_.log.printf("Using BSP coordinator for deterministic parallel B&B\n");
    run_bsp_coordinator(Arow);
  } else {
    // Use traditional opportunistic parallel exploration
#pragma omp parallel num_threads(settings_.num_threads)
    {
      raft::common::nvtx::range scope_tree("BB::tree_exploration");
#pragma omp master
      {
        auto down_child  = search_tree_.root.get_down_child();
        auto up_child    = search_tree_.root.get_up_child();
        i_t initial_size = 2 * settings_.num_threads;

#pragma omp task
        exploration_ramp_up(down_child, &search_tree_, Arow, initial_size);

#pragma omp task
        exploration_ramp_up(up_child, &search_tree_, Arow, initial_size);
      }

#pragma omp barrier

#pragma omp master
      {
        for (i_t i = 0; i < settings_.num_bfs_threads; i++) {
#pragma omp task
          best_first_thread(i, search_tree_, Arow);
        }

        for (i_t i = 0; i < settings_.num_diving_threads; i++) {
#pragma omp task
          diving_thread(Arow);
        }
      }
    }
  }

  // Flush any pending features
  mutex_feature_log_.lock();
  if (has_pending_features_) { flush_pending_features(); }
  mutex_feature_log_.unlock();

  f_t lower_bound = heap_.size() > 0 ? heap_.top()->lower_bound : search_tree_.root.lower_bound;
  return set_final_solution(solution, lower_bound);
}

// ============================================================================
// BSP (Bulk Synchronous Parallel) Implementation
// ============================================================================

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::run_bsp_coordinator(const csr_matrix_t<i_t, f_t>& Arow)
{
  raft::common::nvtx::range scope("BB::bsp_coordinator");

  bsp_horizon_step_ = 0.05;

  const int num_workers = settings_.num_bfs_threads;
  bsp_mode_enabled_     = true;
  bsp_current_horizon_  = bsp_horizon_step_;
  bsp_horizon_number_   = 0;

  // Initialize worker pool
  bsp_workers_ = std::make_unique<bb_worker_pool_t<i_t, f_t>>();
  bsp_workers_->initialize(num_workers,
                           original_lp_,
                           Arow,
                           var_types_,
                           settings_.refactor_frequency,
                           settings_.deterministic);

  // Initialize debug logger
  bsp_debug_logger_.set_settings(bsp_debug_settings_);
  bsp_debug_logger_.set_num_workers(num_workers);
  bsp_debug_logger_.set_horizon_step(bsp_horizon_step_);

  settings_.log.printf(
    "BSP Mode: %d workers, horizon step = %.2f work units\n", num_workers, bsp_horizon_step_);

  // Push initial children to the global heap
  // Set deterministic BSP identity for root children (pre-BSP origin with seq 0 and 1)
  search_tree_.root.get_down_child()->origin_worker_id = -1;  // Pre-BSP marker
  search_tree_.root.get_down_child()->creation_seq     = 0;
  search_tree_.root.get_up_child()->origin_worker_id   = -1;
  search_tree_.root.get_up_child()->creation_seq       = 1;

  heap_.push(search_tree_.root.get_down_child());
  heap_.push(search_tree_.root.get_up_child());

  const f_t inf   = std::numeric_limits<f_t>::infinity();
  f_t lower_bound = get_lower_bound();
  f_t upper_bound = get_upper_bound();
  f_t abs_gap     = upper_bound - lower_bound;
  f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);

  constexpr i_t target_queue_size = 5;  // Target nodes per worker

  // Initial distribution: fill worker queues once at the start
  refill_worker_queues(target_queue_size);
  BSP_DEBUG_FLUSH_ASSIGN_TRACE(bsp_debug_settings_, bsp_debug_logger_);

  // Main BSP coordinator loop
  while (solver_status_ == mip_exploration_status_t::RUNNING &&
         abs_gap > settings_.absolute_mip_gap_tol && rel_gap > settings_.relative_mip_gap_tol &&
         (heap_.size() > 0 || bsp_workers_->any_has_work())) {
    ++bsp_horizon_number_;
    double horizon_start = bsp_current_horizon_ - bsp_horizon_step_;
    double horizon_end   = bsp_current_horizon_;

    // Debug: Log horizon start
    BSP_DEBUG_LOG_HORIZON_START(
      bsp_debug_settings_, bsp_debug_logger_, bsp_horizon_number_, horizon_start, horizon_end);

    // Reset workers for new horizon with current global upper bound
    // Each worker gets a snapshot of the upper bound for deterministic pruning
    bsp_workers_->reset_for_horizon(horizon_start, horizon_end, get_upper_bound());

    // Snapshot pseudo-costs for deterministic variable selection
    for (auto& worker : *bsp_workers_) {
      worker.pc_sum_up_snapshot   = pc_.pseudo_cost_sum_up;
      worker.pc_sum_down_snapshot = pc_.pseudo_cost_sum_down;
      worker.pc_num_up_snapshot   = pc_.pseudo_cost_num_up;
      worker.pc_num_down_snapshot = pc_.pseudo_cost_num_down;
    }

    // PHASE 2: PARALLEL EXECUTION - Workers run until horizon
#pragma omp parallel num_threads(num_workers)
    {
      int worker_id = omp_get_thread_num();
      auto& worker  = (*bsp_workers_)[worker_id];

      f_t worker_start_time = tic();

      // Run worker until horizon
      run_worker_until_horizon(worker, search_tree_, bsp_current_horizon_);

      // Record when this worker finished (for barrier wait calculation)
      // Store raw timestamp - barrier wait = barrier_end - finish_time
      worker.horizon_finish_time = tic();
      worker.total_runtime += toc(worker_start_time);
    }
    // Implicit OMP barrier here - all workers have finished
    f_t barrier_end_time = tic();

    // Calculate barrier wait time for each worker
    // barrier_wait = time from worker finish to barrier completion
    for (auto& worker : *bsp_workers_) {
      double wait_time = barrier_end_time - worker.horizon_finish_time;
      if (wait_time > 0) { worker.total_barrier_wait += wait_time; }
    }

    // Aggregate worker work into global context for work limit tracking
    // The global work is the horizon boundary (all workers synchronized to this point)
    work_unit_context_.global_work_units_elapsed = horizon_end;

    raft::common::nvtx::range scope("BB::bsp_coordinator::sync_phase");

    // PHASE 3: SYNCHRONIZATION - The Barrier
    // Collect and sort all events deterministically
    bb_event_batch_t<i_t, f_t> all_events;
    {
      raft::common::nvtx::range scope("BB::bsp_coordinator::collect_and_sort_events");
      all_events = bsp_workers_->collect_and_sort_events();
    }

    // Debug: Log sync phase
    BSP_DEBUG_LOG_SYNC_PHASE_START(
      bsp_debug_settings_, bsp_debug_logger_, horizon_end, all_events.size());

    // Process history and sync
    {
      raft::common::nvtx::range scope("BB::bsp_coordinator::process_history_and_sync");
      process_history_and_sync(all_events);
    }

    // Debug: Log sync end (no final_id assignment needed with BSP identity tuples)
    BSP_DEBUG_LOG_SYNC_PHASE_END(bsp_debug_settings_, bsp_debug_logger_, horizon_end);

    // Prune paused nodes that are now dominated by new incumbent
    {
      raft::common::nvtx::range scope("BB::bsp_coordinator::prune_worker_nodes_vs_incumbent");
      prune_worker_nodes_vs_incumbent();
    }

    // Balance worker loads if significant imbalance detected
    {
      raft::common::nvtx::range scope("BB::bsp_coordinator::balance_worker_loads");
      balance_worker_loads();
    }
    BSP_DEBUG_FLUSH_ASSIGN_TRACE(bsp_debug_settings_, bsp_debug_logger_);

    // Debug: Log horizon end, emit tree state and JSON state
    BSP_DEBUG_LOG_HORIZON_END(
      bsp_debug_settings_, bsp_debug_logger_, bsp_horizon_number_, horizon_end);

    // Compute and log determinism fingerprint hash
    // This hash captures all state that should be identical across deterministic runs
    if (bsp_debug_settings_.any_enabled() || true) {
      // Collect all determinism-critical state into a vector for hashing
      std::vector<uint64_t> state_data;

      // Global state
      state_data.push_back(static_cast<uint64_t>(exploration_stats_.nodes_explored));
      state_data.push_back(static_cast<uint64_t>(exploration_stats_.nodes_unexplored));

      // Upper/lower bounds (convert to fixed-point for exact comparison)
      // Use compute_bsp_lower_bound() for accurate LB from all worker queues
      f_t ub = get_upper_bound();
      f_t lb = compute_bsp_lower_bound();
      state_data.push_back(static_cast<uint64_t>(ub * 1000000));  // 6 decimal places
      state_data.push_back(static_cast<uint64_t>(lb * 1000000));

      // Worker queue contents using BSP identity tuple (origin_worker_id, creation_seq)
      // Each worker's queue is a priority queue - we extract nodes in priority order
      // Note: BSP identity is always set for nodes in BSP mode
      int nodes_without_identity = 0;
      for (auto& worker : *bsp_workers_) {
        // Hash paused node if any
        if (worker.current_node != nullptr) {
          if (!worker.current_node->has_bsp_identity()) {
            ++nodes_without_identity;
            CUOPT_LOG_WARN(
              "BSP Hash: Worker %d current_node has no BSP identity (node_id=%d, depth=%d)",
              worker.worker_id,
              worker.current_node->node_id,
              worker.current_node->depth);
          }
          state_data.push_back(worker.current_node->get_bsp_identity_hash());
        }

        // Extract queue contents for hashing (preserves priority order)
        // We need to temporarily extract to iterate, then restore
        std::vector<mip_node_t<i_t, f_t>*> queue_nodes;
        auto queue_copy = worker.local_queue;  // Copy the priority queue
        while (!queue_copy.empty()) {
          auto* node = queue_copy.top();
          queue_copy.pop();
          if (!node->has_bsp_identity()) {
            ++nodes_without_identity;
            CUOPT_LOG_WARN(
              "BSP Hash: Worker %d queue node has no BSP identity (node_id=%d, depth=%d)",
              worker.worker_id,
              node->node_id,
              node->depth);
          }
          state_data.push_back(node->get_bsp_identity_hash());
        }
      }
      if (nodes_without_identity > 0) {
        CUOPT_LOG_WARN(
          "BSP Hash at horizon %d: %d nodes without BSP identity - HASH MAY BE "
          "NON-DETERMINISTIC!",
          bsp_horizon_number_,
          nodes_without_identity);
      }

      // Compute hash from state data
      uint32_t hash = 0x811c9dc5u;  // FNV-1a initial value
      for (uint64_t val : state_data) {
        hash ^= static_cast<uint32_t>(val & 0xFFFFFFFF);
        hash *= 0x01000193u;
        hash ^= static_cast<uint32_t>(val >> 32);
        hash *= 0x01000193u;
      }
      CUOPT_LOG_DEBUG("BSP Hash at horizon %d: 0x%x", bsp_horizon_number_, hash);
      BSP_DEBUG_LOG_HORIZON_HASH(
        bsp_debug_settings_, bsp_debug_logger_, bsp_horizon_number_, horizon_end, hash);
    }

    BSP_DEBUG_EMIT_TREE_STATE(bsp_debug_settings_,
                              bsp_debug_logger_,
                              bsp_horizon_number_,
                              search_tree_.root,
                              get_upper_bound());
    // Collect heap nodes for JSON state (Note: We can't easily iterate the heap, so just log the
    // size)
    std::vector<mip_node_t<i_t, f_t>*> heap_snapshot;
    BSP_DEBUG_EMIT_STATE_JSON(bsp_debug_settings_,
                              bsp_debug_logger_,
                              bsp_horizon_number_,
                              horizon_start,
                              horizon_end,
                              0,  // No longer tracking next_final_id with BSP identity tuples
                              get_upper_bound(),
                              compute_bsp_lower_bound(),
                              exploration_stats_.nodes_explored,
                              exploration_stats_.nodes_unexplored,
                              *bsp_workers_,
                              heap_snapshot,
                              all_events);

    // Advance the horizon
    bsp_current_horizon_ += bsp_horizon_step_;

    // Update gap info using accurate BSP lower bound from all worker queues
    lower_bound = compute_bsp_lower_bound();
    upper_bound = get_upper_bound();
    abs_gap     = upper_bound - lower_bound;
    rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);

    // Check time/work limits
    if (toc(exploration_stats_.start_time) > settings_.time_limit) {
      solver_status_ = mip_exploration_status_t::TIME_LIMIT;
    }
    if (settings_.deterministic &&
        work_unit_context_.global_work_units_elapsed >= settings_.work_limit) {
      solver_status_ = mip_exploration_status_t::WORK_LIMIT;
    }

    // Progress logging
    f_t obj              = compute_user_objective(original_lp_, upper_bound);
    f_t user_lower       = compute_user_objective(original_lp_, lower_bound);
    std::string gap_user = user_mip_gap<f_t>(obj, user_lower);
    i_t nodes_explored   = exploration_stats_.nodes_explored;
    i_t nodes_unexplored = exploration_stats_.nodes_unexplored;

    settings_.log.printf(" %10d   %10lu    %+13.6e    %+10.6e                           %s %9.2f\n",
                         nodes_explored,
                         nodes_unexplored,
                         obj,
                         user_lower,
                         gap_user.c_str(),
                         toc(exploration_stats_.start_time));
  }

  // Print per-worker statistics
  settings_.log.printf("\n");
  settings_.log.printf("BSP Worker Statistics:\n");
  settings_.log.printf(
    "  Worker | Processed | Branched | Pruned | Infeasible | IntSol | Assigned |  Runtime  |  "
    "Wait\n");
  settings_.log.printf(
    "  "
    "-------+-----------+----------+--------+------------+--------+----------+-----------+-------"
    "\n");
  for (const auto& worker : *bsp_workers_) {
    settings_.log.printf("  %6d | %9d | %8d | %6d | %10d | %6d | %8d | %8.3fs | %5.3fs\n",
                         worker.worker_id,
                         worker.total_nodes_processed,
                         worker.total_nodes_branched,
                         worker.total_nodes_pruned,
                         worker.total_nodes_infeasible,
                         worker.total_integer_solutions,
                         worker.total_nodes_assigned,
                         worker.total_runtime,
                         worker.total_barrier_wait);
  }
  settings_.log.printf("\n");

  // Finalize debug logger
  BSP_DEBUG_FINALIZE(bsp_debug_settings_, bsp_debug_logger_);

  // Mark completed if we finished exploring
  if (solver_status_ == mip_exploration_status_t::RUNNING) {
    solver_status_ = mip_exploration_status_t::COMPLETED;
  }
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
    if (node->lower_bound >= get_upper_bound()) {
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
    double vt = bsp_current_horizon_ - bsp_horizon_step_;  // Start of current horizon
    BSP_DEBUG_LOG_NODE_ASSIGNED(bsp_debug_settings_,
                                bsp_debug_logger_,
                                vt,
                                worker_id,
                                node->node_id,
                                node->origin_worker_id,
                                node->lower_bound);
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::run_worker_until_horizon(bb_worker_state_t<i_t, f_t>& worker,
                                                            search_tree_t<i_t, f_t>& search_tree,
                                                            double current_horizon)
{
  raft::common::nvtx::range scope("BB::worker_run");

  while (worker.clock < current_horizon && worker.has_work() &&
         solver_status_ == mip_exploration_status_t::RUNNING) {
    mip_node_t<i_t, f_t>* node = worker.dequeue_node();
    if (node == nullptr) break;

    // Check if node should be pruned
    f_t upper_bound = get_upper_bound();
    if (node->lower_bound >= upper_bound) {
      worker.record_fathomed(node, node->lower_bound);
      worker.track_node_pruned();
      search_tree.update(node, node_status_t::FATHOMED);
      --exploration_stats_.nodes_unexplored;
      // Don't update last_solved_node - pruning doesn't change the basis
      continue;
    }

    // Check if we can warm-start from the previous solve's basis.
    // Two cases where we can reuse the basis:
    // 1. Resumed node: continuing the same solve (basis already set up)
    // 2. Child of last solved: child vstatus is a copy of parent's final vstatus
    bool is_resumed                   = (node->bsp_state == bsp_node_state_t::PAUSED);
    bool is_child                     = (node->parent == worker.last_solved_node);
    bool can_warm_start               = is_resumed || is_child;
    worker.recompute_bounds_and_basis = !can_warm_start;

    // Solve the node (this records events)
    node_solve_info_t status = solve_node_bsp(worker, node, search_tree, current_horizon);

    // Track last solved node for warm-start detection
    worker.last_solved_node = node;

    // Handle result
    if (status == node_solve_info_t::TIME_LIMIT) {
      solver_status_ = mip_exploration_status_t::TIME_LIMIT;
      break;
    } else if (status == node_solve_info_t::WORK_LIMIT) {
      // Node paused at horizon - already handled in solve_node_bsp
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

  // Validate vstatus has correct BASIC count before processing
  // This helps diagnose heap overflow bugs where vstatus has too many BASIC entries
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

  // Track work units at start (from work_context, which simplex solver updates)
  double work_units_at_start = worker.work_context.global_work_units_elapsed;
  double clock_at_start      = worker.clock;
  bool is_resumed            = (node_ptr->bsp_state == bsp_node_state_t::PAUSED);

  // Debug: Log solve start (pass origin_worker_id as identifier)
  double work_limit = current_horizon - worker.clock;
  BSP_DEBUG_LOG_SOLVE_START(bsp_debug_settings_,
                            bsp_debug_logger_,
                            worker.clock,
                            worker.worker_id,
                            node_ptr->node_id,
                            node_ptr->origin_worker_id,
                            work_limit,
                            is_resumed);

  // Setup leaf problem bounds
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

  // Bounds strengthening
  simplex_solver_settings_t<i_t, f_t> lp_settings = settings_;
  lp_settings.set_log(false);
  // Use worker-local upper bound for LP cutoff (deterministic)
  lp_settings.cut_off    = worker.local_upper_bound + settings_.dual_tol;
  lp_settings.inside_mip = 2;
  lp_settings.time_limit = settings_.time_limit - toc(exploration_stats_.start_time);
  // Work limit is the ABSOLUTE VT at which to pause (LP solver compares against absolute elapsed)
  lp_settings.work_limit    = current_horizon;
  lp_settings.scale_columns = false;

  bool feasible = true;
  // TODO: incorporate into work unit estimation
  // feasible = worker.node_presolver->bounds_strengthening(
  //   worker.leaf_problem->lower, worker.leaf_problem->upper, lp_settings);

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

  // Debug: Log LP input for determinism analysis (enabled via CUOPT_BSP_DEBUG_TRACE=1,
  // log_level>=2)
  if (bsp_debug_settings_.any_enabled()) {
    uint64_t path_hash = node_ptr->compute_path_hash();
    // Compute vstatus hash
    uint64_t vstatus_hash = leaf_vstatus.size();
    for (size_t i = 0; i < leaf_vstatus.size(); ++i) {
      vstatus_hash ^= (static_cast<uint64_t>(leaf_vstatus[i]) << (i % 56));
      vstatus_hash *= 0x100000001b3ULL;
    }
    // Compute bounds hash
    uint64_t bounds_hash = 0;
    for (i_t j = 0; j < worker.leaf_problem->num_cols; ++j) {
      union {
        f_t f;
        uint64_t u;
      } lb_bits, ub_bits;
      lb_bits.f = worker.leaf_problem->lower[j];
      ub_bits.f = worker.leaf_problem->upper[j];
      bounds_hash ^= lb_bits.u + ub_bits.u;
      bounds_hash *= 0x100000001b3ULL;
    }
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

  // Debug: Log LP output for determinism analysis (enabled via CUOPT_BSP_DEBUG_TRACE=1,
  // log_level>=2)
  if (bsp_debug_settings_.any_enabled()) {
    uint64_t path_hash = node_ptr->compute_path_hash();
    // Compute solution hash
    uint64_t sol_hash = 0;
    for (i_t j = 0;
         j < worker.leaf_problem->num_cols && j < static_cast<i_t>(leaf_solution.x.size());
         ++j) {
      union {
        f_t f;
        uint64_t u;
      } val_bits;
      val_bits.f = leaf_solution.x[j];
      sol_hash ^= val_bits.u;
      sol_hash *= 0x100000001b3ULL;
    }
    f_t obj = (lp_status == dual::status_t::OPTIMAL)
                ? compute_objective(*worker.leaf_problem, leaf_solution.x)
                : std::numeric_limits<f_t>::infinity();
    union {
      f_t f;
      uint64_t u;
    } obj_bits;
    obj_bits.f = obj;
    BSP_DEBUG_LOG_LP_OUTPUT(bsp_debug_settings_,
                            bsp_debug_logger_,
                            worker.worker_id,
                            node_ptr->node_id,
                            path_hash,
                            static_cast<int>(lp_status),
                            node_iter,
                            obj_bits.u,
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

  // Update worker clock with work performed
  // The simplex solver recorded work to work_context.global_work_units_elapsed
  // Compute the delta and advance the worker clock (but don't double-record to work_context)
  double work_performed = worker.work_context.global_work_units_elapsed - work_units_at_start;
  worker.clock += work_performed;
  worker.work_units_this_horizon += work_performed;
  // Note: don't call advance_clock() as work_context was already updated by simplex solver

  // Check if we hit the horizon mid-solve
  if (lp_status == dual::status_t::WORK_LIMIT || worker.clock >= current_horizon) {
    // Pause this node - accumulated_vt is the total work spent on this node
    double accumulated_vt = (worker.clock - clock_at_start) + node_ptr->accumulated_vt;
    worker.pause_current_node(node_ptr, accumulated_vt);

    // Debug: Log solve end (paused)
    BSP_DEBUG_LOG_SOLVE_END(bsp_debug_settings_,
                            bsp_debug_logger_,
                            worker.clock,
                            worker.worker_id,
                            node_ptr->node_id,
                            node_ptr->origin_worker_id,
                            "PAUSED",
                            node_ptr->lower_bound);
    BSP_DEBUG_LOG_PAUSED(bsp_debug_settings_,
                         bsp_debug_logger_,
                         worker.clock,
                         worker.worker_id,
                         node_ptr->node_id,
                         node_ptr->origin_worker_id,
                         static_cast<f_t>(accumulated_vt));
    return node_solve_info_t::WORK_LIMIT;
  }

  exploration_stats_.total_lp_solve_time += toc(lp_start_time);
  exploration_stats_.total_lp_iters += node_iter;
  ++exploration_stats_.nodes_explored;
  --exploration_stats_.nodes_unexplored;

  // Process LP result
  if (lp_status == dual::status_t::DUAL_UNBOUNDED) {
    node_ptr->lower_bound = std::numeric_limits<f_t>::infinity();
    search_tree.update(node_ptr, node_status_t::INFEASIBLE);
    worker.record_infeasible(node_ptr);
    worker.track_node_infeasible();
    worker.track_node_processed();
    worker.recompute_bounds_and_basis = true;

    // Debug: Log solve end (infeasible)
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
    return node_solve_info_t::NO_CHILDREN;

  } else if (lp_status == dual::status_t::CUTOFF) {
    // Use worker-local upper bound for determinism
    node_ptr->lower_bound = worker.local_upper_bound;
    search_tree.update(node_ptr, node_status_t::FATHOMED);
    worker.record_fathomed(node_ptr, node_ptr->lower_bound);
    worker.track_node_pruned();
    worker.track_node_processed();
    worker.recompute_bounds_and_basis = true;

    // Debug: Log solve end (fathomed - cutoff)
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
    return node_solve_info_t::NO_CHILDREN;

  } else if (lp_status == dual::status_t::OPTIMAL) {
    std::vector<i_t> leaf_fractional;
    i_t leaf_num_fractional =
      fractional_variables(settings_, leaf_solution.x, var_types_, leaf_fractional);

    f_t leaf_objective    = compute_objective(*worker.leaf_problem, leaf_solution.x);
    node_ptr->lower_bound = leaf_objective;

    // Queue pseudo-cost update for deterministic application at sync
    // (Replicates pc_.update_pseudo_costs logic but defers application to sync phase)
    // Note: Original code also sets lower_bound before this, so change_in_obj = 0.
    // This matches the original behavior exactly.
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
        // Note: Logging deferred to sync phase for deterministic output
      }
      search_tree.update(node_ptr, node_status_t::INTEGER_FEASIBLE);
      worker.record_integer_solution(node_ptr, leaf_objective);
      worker.track_integer_solution();
      worker.track_node_processed();
      worker.recompute_bounds_and_basis = true;

      // Debug: Log solve end (integer)
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
      return node_solve_info_t::NO_CHILDREN;

    } else if (leaf_objective <= worker.local_upper_bound + settings_.absolute_mip_gap_tol / 10) {
      // Branch - use worker-local upper bound for deterministic pruning decision
      // Use pseudo-cost snapshot for deterministic variable selection
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

      // Debug: Log solve end (branched) and branched event
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

      // Add children to local queue - they get BSP identity on enqueue
      // Note: recompute_bounds_and_basis is set in run_worker_until_horizon based on
      // whether we branched (has_children), matching opportunistic mode behavior.
      worker.enqueue_node(node_ptr->get_down_child());
      worker.enqueue_node(node_ptr->get_up_child());

      return rounding_direction_t::DOWN == child_selection(node_ptr)
               ? node_solve_info_t::DOWN_CHILD_FIRST
               : node_solve_info_t::UP_CHILD_FIRST;

    } else {
      search_tree.update(node_ptr, node_status_t::FATHOMED);
      worker.record_fathomed(node_ptr, leaf_objective);
      worker.track_node_pruned();
      worker.track_node_processed();
      worker.recompute_bounds_and_basis = true;

      // Debug: Log solve end (fathomed by bound)
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
      return node_solve_info_t::NO_CHILDREN;
    }

  } else if (lp_status == dual::status_t::TIME_LIMIT) {
    return node_solve_info_t::TIME_LIMIT;

  } else {
    // Numerical issue
    search_tree.update(node_ptr, node_status_t::NUMERICAL);
    worker.record_numerical(node_ptr);
    worker.recompute_bounds_and_basis = true;
    return node_solve_info_t::NUMERICAL;
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::process_history_and_sync(
  const bb_event_batch_t<i_t, f_t>& events)
{
  // With BSP identity tuples (origin_worker_id, creation_seq), we no longer need to assign
  // final_ids during sync. Each node gets its identity when created, and it never changes.
  // This function now only processes heuristic solutions to update the incumbent.

  // Process repair queue first (for BSP mode)
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
          // Queue repaired solution with VT = current horizon for deterministic processing
          mutex_heuristic_queue_.lock();
          heuristic_solution_queue_.push_back(
            {std::move(repaired_solution), repaired_obj, bsp_current_horizon_});
          mutex_heuristic_queue_.unlock();
        }
      }
    }
  }

  // Collect queued heuristic solutions (including any newly repaired ones)
  std::vector<queued_heuristic_solution_t> heuristic_solutions;
  mutex_heuristic_queue_.lock();
  heuristic_solutions = std::move(heuristic_solution_queue_);
  heuristic_solution_queue_.clear();
  mutex_heuristic_queue_.unlock();

  // Sort heuristic solutions by VT for deterministic processing order
  std::sort(heuristic_solutions.begin(),
            heuristic_solutions.end(),
            [](const queued_heuristic_solution_t& a, const queued_heuristic_solution_t& b) {
              return a.vt_timestamp < b.vt_timestamp;
            });

  // Merge B&B events and heuristic solutions for unified timeline replay
  // Both are sorted by VT, so we can do a merge-style iteration
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
      // Both have items - pick the one with smaller VT
      if (events.events[event_idx].vt_timestamp <=
          heuristic_solutions[heuristic_idx].vt_timestamp) {
        process_event = true;
      } else {
        process_heuristic = true;
      }
    }

    if (process_event) {
      const auto& event = events.events[event_idx++];
      switch (event.type) {
        case bb_event_type_t::NODE_INTEGER: {
          // Check if this solution beats the incumbent KNOWN AT THAT VIRTUAL TIME
          f_t obj = event.payload.integer_solution.objective_value;
          mutex_upper_.lock();
          if (obj < upper_bound_) {
            // Note: The actual solution was already stored during solve_node_bsp
            // Here we just acknowledge the event order
          }
          mutex_upper_.unlock();
          break;
        }

        case bb_event_type_t::NODE_BRANCHED:
        case bb_event_type_t::NODE_FATHOMED:
        case bb_event_type_t::NODE_INFEASIBLE:
        case bb_event_type_t::NODE_NUMERICAL:
        case bb_event_type_t::NODE_PAUSED:
        case bb_event_type_t::HEURISTIC_SOLUTION:
          // These events don't need additional processing during replay
          // (BSP identity is already assigned at node creation time)
          break;
      }
    }

    if (process_heuristic) {
      const auto& hsol = heuristic_solutions[heuristic_idx++];

      // Debug: Log heuristic received
      BSP_DEBUG_LOG_HEURISTIC_RECEIVED(
        bsp_debug_settings_, bsp_debug_logger_, hsol.vt_timestamp, hsol.objective);

      // Process heuristic solution at its correct VT position
      f_t new_upper = std::numeric_limits<f_t>::infinity();

      mutex_upper_.lock();
      if (hsol.objective < upper_bound_) {
        upper_bound_ = hsol.objective;
        incumbent_.set_incumbent_solution(hsol.objective, hsol.solution);
        new_upper = hsol.objective;

        // Debug: Log incumbent update
        BSP_DEBUG_LOG_INCUMBENT_UPDATE(
          bsp_debug_settings_, bsp_debug_logger_, hsol.vt_timestamp, hsol.objective, "heuristic");
      }
      mutex_upper_.unlock();

      // Log after releasing mutex to avoid holding mutex while calling get_lower_bound()
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
      }
    }
  }

  // Merge integer solutions from all workers and update global incumbent
  // Sort by (objective, worker_id) for deterministic winner selection
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

  // Sort by objective, then worker_id for deterministic tie-breaking
  std::sort(all_integer_solutions.begin(),
            all_integer_solutions.end(),
            [](const worker_solution_t& a, const worker_solution_t& b) {
              if (a.objective != b.objective) return a.objective < b.objective;
              return a.worker_id < b.worker_id;
            });

  // Apply the best solution to global incumbent and log all improving solutions
  // Use compute_bsp_lower_bound() for accurate lower bound in logs
  f_t bsp_lower     = compute_bsp_lower_bound();
  f_t current_upper = get_upper_bound();

  for (const auto& sol : all_integer_solutions) {
    if (sol.objective < current_upper) {
      // Log this improving solution (deterministic: sorted order)
      f_t user_obj         = compute_user_objective(original_lp_, sol.objective);
      f_t user_lower       = compute_user_objective(original_lp_, bsp_lower);
      i_t nodes_explored   = exploration_stats_.nodes_explored.load();
      i_t nodes_unexplored = exploration_stats_.nodes_unexplored.load();
      settings_.log.printf(
        "%s%10d   %10lu    %+13.6e    %+10.6e   %6d   %7.1e     %s %9.2f\n",
        feasible_solution_symbol(thread_type_t::EXPLORATION),
        nodes_explored,
        nodes_unexplored,
        user_obj,
        user_lower,
        sol.depth,
        nodes_explored > 0 ? exploration_stats_.total_lp_iters.load() / nodes_explored : 0.0,
        user_mip_gap<f_t>(user_obj, user_lower).c_str(),
        toc(exploration_stats_.start_time));

      // Update incumbent
      mutex_upper_.lock();
      if (sol.objective < upper_bound_) {
        upper_bound_ = sol.objective;
        incumbent_.set_incumbent_solution(sol.objective, *sol.solution);
        current_upper = sol.objective;
      }
      mutex_upper_.unlock();
    }
  }

  // Merge and apply pseudo-cost updates from all workers in deterministic order
  std::vector<pseudo_cost_update_t<i_t, f_t>> all_pc_updates;
  for (auto& worker : *bsp_workers_) {
    for (auto& upd : worker.pseudo_cost_updates) {
      all_pc_updates.push_back(upd);
    }
  }

  // Sort by (vt, worker_id) for deterministic order
  std::sort(all_pc_updates.begin(),
            all_pc_updates.end(),
            [](const pseudo_cost_update_t<i_t, f_t>& a, const pseudo_cost_update_t<i_t, f_t>& b) {
              if (a.vt != b.vt) return a.vt < b.vt;
              return a.worker_id < b.worker_id;
            });

  // Apply updates in deterministic order
  for (const auto& upd : all_pc_updates) {
    if (upd.direction == rounding_direction_t::DOWN) {
      pc_.pseudo_cost_sum_down[upd.variable] += upd.delta;
      pc_.pseudo_cost_num_down[upd.variable]++;
    } else {
      pc_.pseudo_cost_sum_up[upd.variable] += upd.delta;
      pc_.pseudo_cost_num_up[upd.variable]++;
    }
  }

  // No final_id application needed - BSP identity is assigned at node creation time
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::prune_worker_nodes_vs_incumbent()
{
  f_t upper_bound = get_upper_bound();

  for (auto& worker : *bsp_workers_) {
    // Check paused node
    if (worker.current_node != nullptr) {
      if (worker.current_node->lower_bound >= upper_bound) {
        // Prune the paused node
        search_tree_.update(worker.current_node, node_status_t::FATHOMED);
        --exploration_stats_.nodes_unexplored;
        worker.current_node = nullptr;
      }
    }

    // Check nodes in local queue - need to extract, filter, and rebuild
    // since priority_queue doesn't support iteration
    std::vector<mip_node_t<i_t, f_t>*> surviving_nodes;
    while (!worker.local_queue.empty()) {
      auto* node = worker.local_queue.top();
      worker.local_queue.pop();
      if (node->lower_bound >= upper_bound) {
        search_tree_.update(node, node_status_t::FATHOMED);
        --exploration_stats_.nodes_unexplored;
      } else {
        surviving_nodes.push_back(node);
      }
    }
    // Rebuild the queue with surviving nodes
    for (auto* node : surviving_nodes) {
      worker.local_queue.push(node);
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::balance_worker_loads()
{
  const size_t num_workers = bsp_workers_->size();
  if (num_workers <= 1) return;

  // Count work for each worker: current_node (if any) + local_queue size
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

  // Check if we need to balance: significant imbalance = some worker has 0 work while others have
  // 2+ Or max/min ratio is very high
  bool needs_balance =
    (min_work == 0 && max_work >= 2) || (min_work > 0 && max_work > 4 * min_work);

  if (!needs_balance) return;

  // Collect all redistributable nodes from worker queues (excluding paused current_node)
  std::vector<mip_node_t<i_t, f_t>*> all_nodes;
  for (auto& worker : *bsp_workers_) {
    // Extract all nodes from this worker's priority queue (not current_node)
    while (!worker.local_queue.empty()) {
      all_nodes.push_back(worker.local_queue.top());
      worker.local_queue.pop();
    }
  }

  // Also pull nodes from global heap if workers need work
  mutex_heap_.lock();
  f_t upper_bound = get_upper_bound();
  while (!heap_.empty() && all_nodes.size() < num_workers * 5) {
    mip_node_t<i_t, f_t>* node = heap_.top();
    heap_.pop();
    if (node->lower_bound < upper_bound) {
      all_nodes.push_back(node);
    } else {
      search_tree_.update(node, node_status_t::FATHOMED);
      --exploration_stats_.nodes_unexplored;
    }
  }
  mutex_heap_.unlock();

  if (all_nodes.empty()) return;

  // Sort by BSP identity for deterministic distribution
  // Uses lexicographic order of (origin_worker_id, creation_seq)
  auto deterministic_less = [](const mip_node_t<i_t, f_t>* a, const mip_node_t<i_t, f_t>* b) {
    // Lexicographic comparison of BSP identity tuple
    if (a->origin_worker_id != b->origin_worker_id) {
      return a->origin_worker_id < b->origin_worker_id;
    }
    return a->creation_seq < b->creation_seq;
  };
  std::sort(all_nodes.begin(), all_nodes.end(), deterministic_less);

  // Redistribute round-robin, but skip workers that have a paused current_node
  // (they already have work and will resume that node first)
  std::vector<size_t> worker_order;
  for (size_t w = 0; w < num_workers; ++w) {
    // Prioritize workers without a paused node
    if ((*bsp_workers_)[w].current_node == nullptr) { worker_order.push_back(w); }
  }
  for (size_t w = 0; w < num_workers; ++w) {
    if ((*bsp_workers_)[w].current_node != nullptr) { worker_order.push_back(w); }
  }

  // Distribute nodes - use enqueue_node_with_identity to preserve existing identity
  for (size_t i = 0; i < all_nodes.size(); ++i) {
    size_t worker_idx = worker_order[i % num_workers];
    (*bsp_workers_)[worker_idx].enqueue_node_with_identity(all_nodes[i]);
    (*bsp_workers_)[worker_idx].track_node_assigned();

    // Debug: Log redistribution (happens at horizon END, at the sync point)
    double vt = bsp_current_horizon_;
    BSP_DEBUG_LOG_NODE_ASSIGNED(bsp_debug_settings_,
                                bsp_debug_logger_,
                                vt,
                                static_cast<int>(worker_idx),
                                all_nodes[i]->node_id,
                                all_nodes[i]->origin_worker_id,
                                all_nodes[i]->lower_bound);
  }
}

template <typename i_t, typename f_t>
f_t branch_and_bound_t<i_t, f_t>::compute_bsp_lower_bound()
{
  // Compute accurate lower bound from all BSP sources
  // Called during sync phase (single-threaded), so no locking needed for worker queues
  const f_t inf   = std::numeric_limits<f_t>::infinity();
  f_t lower_bound = inf;

  // Check global heap (may have nodes not yet distributed)
  mutex_heap_.lock();
  if (heap_.size() > 0) { lower_bound = std::min(heap_.top()->lower_bound, lower_bound); }
  mutex_heap_.unlock();

  // Check all worker queues
  for (const auto& worker : *bsp_workers_) {
    // Check paused node (current_node)
    if (worker.current_node != nullptr) {
      lower_bound = std::min(worker.current_node->lower_bound, lower_bound);
    }

    // Check queue top (min lower bound due to priority queue ordering)
    if (!worker.local_queue.empty()) {
      lower_bound = std::min(worker.local_queue.top()->lower_bound, lower_bound);
    }
  }

  return lower_bound;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class branch_and_bound_t<int, double>;

#endif

}  // namespace cuopt::linear_programming::dual_simplex
