/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/bb_worker_state.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/bsp_debug.hpp>
#include <dual_simplex/diving_heuristics.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/omp_helpers.hpp>
#include <utilities/work_limit_timer.hpp>

#include <functional>

#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// Import types from parent namespace
using cuopt::omp_atomic_t;

template <typename i_t, typename f_t>
struct bnb_stats_t;

// Martin's criteria for the preferred rounding direction
// Reference: A. Martin, "Integer Programs with Block Structure,"
// Technische Universitat Berlin, Berlin, 1999.
// https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/391
template <typename f_t>
inline rounding_direction_t martin_criteria(f_t val, f_t root_val)
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

// Result of solving a node LP and processing the outcome
template <typename i_t, typename f_t>
struct node_solve_result_t {
  node_status_t status;
  rounding_direction_t preferred_direction;
  bool needs_recompute_basis;
};

// =============================================================================
// Standard (non-deterministic) policy implementation
// =============================================================================

template <typename i_t, typename f_t>
class standard_solve_policy_t {
 public:
  // Callback types for optional hooks
  using node_processed_callback_t = std::function<void(const std::vector<f_t>&, f_t)>;
  using objective_estimate_fn_t =
    std::function<f_t(const std::vector<i_t>&, const std::vector<f_t>&, f_t)>;

  standard_solve_policy_t(
    omp_atomic_t<f_t>& upper_bound,
    f_t fathom_tolerance,
    const std::vector<f_t>& root_lower,
    const std::vector<f_t>& root_upper,
    const std::vector<f_t>& edge_norms,
    const std::vector<f_t>& root_solution,
    pseudo_costs_t<i_t, f_t>& pc,
    bnb_stats_t<i_t, f_t>& stats,
    logger_t& log,
    std::function<void(f_t, const std::vector<f_t>&, i_t, bnb_worker_type_t)> add_solution_fn,
    node_processed_callback_t node_processed_callback = nullptr,
    objective_estimate_fn_t objective_estimate_fn     = nullptr)
    : upper_bound_(upper_bound),
      fathom_tolerance_(fathom_tolerance),
      root_lower_(root_lower),
      root_upper_(root_upper),
      edge_norms_(edge_norms),
      root_solution_(root_solution),
      pc_(pc),
      stats_(stats),
      log_(log),
      add_solution_fn_(add_solution_fn),
      node_processed_callback_(node_processed_callback),
      objective_estimate_fn_(objective_estimate_fn)
  {
  }

  f_t get_upper_bound() const { return upper_bound_.load(); }
  f_t get_fathom_tolerance() const { return fathom_tolerance_; }
  bool should_run_bounds_strengthening() const { return true; }
  work_limit_context_t* get_work_context() { return nullptr; }
  const std::vector<f_t>& get_root_lower() const { return root_lower_; }
  const std::vector<f_t>& get_root_upper() const { return root_upper_; }
  const std::vector<f_t>& get_edge_norms() const { return edge_norms_; }
  const std::vector<f_t>& get_root_solution() const { return root_solution_; }

  void on_solve_start(mip_node_t<i_t, f_t>* node) {}

  void on_lp_input(mip_node_t<i_t, f_t>* node,
                   const lp_problem_t<i_t, f_t>& problem,
                   const std::vector<variable_status_t>& vstatus)
  {
  }

  void on_lp_output(mip_node_t<i_t, f_t>* node,
                    dual::status_t status,
                    i_t iterations,
                    const lp_solution_t<i_t, f_t>& solution,
                    const lp_problem_t<i_t, f_t>& problem)
  {
    if (status == dual::status_t::OPTIMAL) {
      f_t objective = compute_objective(problem, solution.x);
      log_.graphviz_node(node, "lower bound", objective);
      if (node_processed_callback_) { node_processed_callback_(solution.x, objective); }
    }
  }

  void on_lp_solve_complete(mip_node_t<i_t, f_t>* node,
                            i_t iterations,
                            f_t solve_time,
                            dual::status_t status)
  {
    stats_.total_lp_solve_time += solve_time;
    stats_.total_lp_iters += iterations;
  }

  void on_infeasible(mip_node_t<i_t, f_t>* node, search_tree_t<i_t, f_t>& tree)
  {
    node->lower_bound = std::numeric_limits<f_t>::infinity();
    tree.graphviz_node(log_, node, "infeasible", 0.0);
    tree.update(node, node_status_t::INFEASIBLE);
  }

  void on_fathomed(mip_node_t<i_t, f_t>* node, f_t objective, search_tree_t<i_t, f_t>& tree)
  {
    tree.graphviz_node(log_, node, "fathomed", objective);
    tree.update(node, node_status_t::FATHOMED);
  }

  void on_integer_solution(mip_node_t<i_t, f_t>* node,
                           f_t objective,
                           const std::vector<f_t>& solution,
                           i_t depth,
                           search_tree_t<i_t, f_t>& tree)
  {
    add_solution_fn_(objective, solution, depth, bnb_worker_type_t::BEST_FIRST);
    tree.graphviz_node(log_, node, "integer feasible", objective);
    tree.update(node, node_status_t::INTEGER_FEASIBLE);
  }

  void on_branched(mip_node_t<i_t, f_t>* node,
                   i_t branch_var,
                   f_t branch_val,
                   i_t down_child_id,
                   i_t up_child_id,
                   rounding_direction_t preferred,
                   search_tree_t<i_t, f_t>& tree,
                   const std::vector<i_t>& fractional,
                   const std::vector<f_t>& solution)
  {
    if (objective_estimate_fn_) {
      node->objective_estimate = objective_estimate_fn_(fractional, solution, node->lower_bound);
    }
    tree.update(node, node_status_t::HAS_CHILDREN);
  }

  void on_numerical(mip_node_t<i_t, f_t>* node, search_tree_t<i_t, f_t>& tree)
  {
    tree.graphviz_node(log_, node, "numerical", 0.0);
    tree.update(node, node_status_t::NUMERICAL);
  }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node, f_t objective)
  {
    pc_.update_pseudo_costs(node, objective);
  }

  i_t select_branch_variable(const std::vector<i_t>& fractional, const std::vector<f_t>& solution)
  {
    logger_t pc_log;
    pc_log.log = false;
    return pc_.variable_selection(fractional, solution, pc_log);
  }

  rounding_direction_t get_preferred_direction(i_t branch_var, f_t branch_val)
  {
    f_t root_val =
      branch_var < (i_t)root_solution_.size() ? root_solution_[branch_var] : branch_val;
    return martin_criteria(branch_val, root_val);
  }

  void track_node_explored() { ++stats_.nodes_explored; }

  void track_node_unexplored_delta(i_t delta) { stats_.nodes_unexplored += delta; }

 private:
  omp_atomic_t<f_t>& upper_bound_;
  f_t fathom_tolerance_;
  const std::vector<f_t>& root_lower_;
  const std::vector<f_t>& root_upper_;
  const std::vector<f_t>& edge_norms_;
  const std::vector<f_t>& root_solution_;
  pseudo_costs_t<i_t, f_t>& pc_;
  bnb_stats_t<i_t, f_t>& stats_;
  logger_t& log_;
  std::function<void(f_t, const std::vector<f_t>&, i_t, bnb_worker_type_t)> add_solution_fn_;
  node_processed_callback_t node_processed_callback_;
  objective_estimate_fn_t objective_estimate_fn_;
};

// =============================================================================
// BSP (deterministic) policy implementation
// =============================================================================

template <typename i_t, typename f_t>
class bsp_solve_policy_t {
 public:
  bsp_solve_policy_t(bb_worker_state_t<i_t, f_t>& worker,
                     f_t fathom_tolerance,
                     const std::vector<f_t>& root_lower,
                     const std::vector<f_t>& root_upper,
                     const std::vector<f_t>& edge_norms,
                     const std::vector<f_t>& root_solution,
                     bnb_stats_t<i_t, f_t>& stats,
                     bsp_debug_settings_t& debug_settings,
                     bsp_debug_logger_t<i_t, f_t>& debug_logger)
    : worker_(worker),
      fathom_tolerance_(fathom_tolerance),
      root_lower_(root_lower),
      root_upper_(root_upper),
      edge_norms_(edge_norms),
      root_solution_(root_solution),
      stats_(stats),
      debug_settings_(debug_settings),
      debug_logger_(debug_logger)
  {
  }

  f_t get_upper_bound() const { return worker_.local_upper_bound; }
  f_t get_fathom_tolerance() const { return fathom_tolerance_; }
  bool should_run_bounds_strengthening() const { return false; }
  work_limit_context_t* get_work_context() { return &worker_.work_context; }
  const std::vector<f_t>& get_root_lower() const { return root_lower_; }
  const std::vector<f_t>& get_root_upper() const { return root_upper_; }
  const std::vector<f_t>& get_edge_norms() const { return edge_norms_; }
  const std::vector<f_t>& get_root_solution() const { return root_solution_; }

  void on_solve_start(mip_node_t<i_t, f_t>* node)
  {
    work_units_at_start_ = worker_.work_context.global_work_units_elapsed;
    double work_limit    = worker_.horizon_end - worker_.clock;
    BSP_DEBUG_LOG_SOLVE_START(debug_settings_,
                              debug_logger_,
                              worker_.clock,
                              worker_.worker_id,
                              node->node_id,
                              node->origin_worker_id,
                              work_limit,
                              false);
  }

  void on_lp_input(mip_node_t<i_t, f_t>* node,
                   const lp_problem_t<i_t, f_t>& problem,
                   const std::vector<variable_status_t>& vstatus)
  {
    if (debug_settings_.any_enabled()) {
      uint64_t path_hash    = node->compute_path_hash();
      uint64_t vstatus_hash = detail::compute_hash(vstatus);
      uint64_t bounds_hash =
        detail::compute_hash(problem.lower) ^ detail::compute_hash(problem.upper);
      BSP_DEBUG_LOG_LP_INPUT(debug_settings_,
                             debug_logger_,
                             worker_.worker_id,
                             node->node_id,
                             path_hash,
                             node->depth,
                             vstatus_hash,
                             bounds_hash);
    }
  }

  void on_lp_output(mip_node_t<i_t, f_t>* node,
                    dual::status_t status,
                    i_t iterations,
                    const lp_solution_t<i_t, f_t>& solution,
                    const lp_problem_t<i_t, f_t>& problem)
  {
    if (debug_settings_.any_enabled()) {
      uint64_t path_hash = node->compute_path_hash();
      uint64_t sol_hash  = detail::compute_hash(solution.x);
      f_t obj = (status == dual::status_t::OPTIMAL) ? compute_objective(problem, solution.x)
                                                    : std::numeric_limits<f_t>::infinity();
      uint64_t obj_hash = detail::compute_hash(obj);
      BSP_DEBUG_LOG_LP_OUTPUT(debug_settings_,
                              debug_logger_,
                              worker_.worker_id,
                              node->node_id,
                              path_hash,
                              static_cast<int>(status),
                              iterations,
                              obj_hash,
                              sol_hash);
    }
  }

  void on_lp_solve_complete(mip_node_t<i_t, f_t>* node,
                            i_t iterations,
                            f_t solve_time,
                            dual::status_t status)
  {
    stats_.total_lp_solve_time += solve_time;
    stats_.total_lp_iters += iterations;

    double work_performed = worker_.work_context.global_work_units_elapsed - work_units_at_start_;
    worker_.clock += work_performed;
    worker_.work_units_this_horizon += work_performed;
  }

  void on_infeasible(mip_node_t<i_t, f_t>* node, search_tree_t<i_t, f_t>& tree)
  {
    node->lower_bound = std::numeric_limits<f_t>::infinity();
    worker_.record_infeasible(node);
    worker_.track_node_infeasible();
    worker_.track_node_processed();

    BSP_DEBUG_LOG_SOLVE_END(debug_settings_,
                            debug_logger_,
                            worker_.clock,
                            worker_.worker_id,
                            node->node_id,
                            node->origin_worker_id,
                            "INFEASIBLE",
                            node->lower_bound);
    BSP_DEBUG_LOG_INFEASIBLE(
      debug_settings_, debug_logger_, worker_.clock, worker_.worker_id, node->node_id);

    tree.update(node, node_status_t::INFEASIBLE);
  }

  void on_fathomed(mip_node_t<i_t, f_t>* node, f_t objective, search_tree_t<i_t, f_t>& tree)
  {
    worker_.record_fathomed(node, objective);
    worker_.track_node_pruned();
    worker_.track_node_processed();
    worker_.recompute_bounds_and_basis = true;

    BSP_DEBUG_LOG_SOLVE_END(debug_settings_,
                            debug_logger_,
                            worker_.clock,
                            worker_.worker_id,
                            node->node_id,
                            node->origin_worker_id,
                            "FATHOMED",
                            objective);
    BSP_DEBUG_LOG_FATHOMED(
      debug_settings_, debug_logger_, worker_.clock, worker_.worker_id, node->node_id, objective);

    tree.update(node, node_status_t::FATHOMED);
  }

  void on_integer_solution(mip_node_t<i_t, f_t>* node,
                           f_t objective,
                           const std::vector<f_t>& solution,
                           i_t depth,
                           search_tree_t<i_t, f_t>& tree)
  {
    if (objective < worker_.local_upper_bound) {
      worker_.local_upper_bound = objective;
      worker_.integer_solutions.push_back({objective, solution, depth});
    }

    worker_.record_integer_solution(node, objective);
    worker_.track_integer_solution();
    worker_.track_node_processed();
    worker_.recompute_bounds_and_basis = true;

    BSP_DEBUG_LOG_SOLVE_END(debug_settings_,
                            debug_logger_,
                            worker_.clock,
                            worker_.worker_id,
                            node->node_id,
                            node->origin_worker_id,
                            "INTEGER",
                            objective);
    BSP_DEBUG_LOG_INTEGER(
      debug_settings_, debug_logger_, worker_.clock, worker_.worker_id, node->node_id, objective);

    tree.update(node, node_status_t::INTEGER_FEASIBLE);
  }

  void on_branched(mip_node_t<i_t, f_t>* node,
                   i_t branch_var,
                   f_t branch_val,
                   i_t down_child_id,
                   i_t up_child_id,
                   rounding_direction_t preferred,
                   search_tree_t<i_t, f_t>& tree,
                   const std::vector<i_t>& /*fractional*/,
                   const std::vector<f_t>& /*solution*/)
  {
    tree.update(node, node_status_t::HAS_CHILDREN);

    worker_.record_branched(node, down_child_id, up_child_id, branch_var, branch_val);
    worker_.track_node_branched();
    worker_.track_node_processed();

    BSP_DEBUG_LOG_SOLVE_END(debug_settings_,
                            debug_logger_,
                            worker_.clock,
                            worker_.worker_id,
                            node->node_id,
                            node->origin_worker_id,
                            "BRANCH",
                            node->lower_bound);
    BSP_DEBUG_LOG_BRANCHED(debug_settings_,
                           debug_logger_,
                           worker_.clock,
                           worker_.worker_id,
                           node->node_id,
                           node->origin_worker_id,
                           down_child_id,
                           up_child_id);

    worker_.enqueue_children_for_plunge(node->get_down_child(), node->get_up_child(), preferred);
  }

  void on_numerical(mip_node_t<i_t, f_t>* node, search_tree_t<i_t, f_t>& tree)
  {
    worker_.record_numerical(node);
    worker_.recompute_bounds_and_basis = true;
    tree.update(node, node_status_t::NUMERICAL);
  }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node, f_t objective)
  {
    if (node->branch_var >= 0) {
      const f_t change_in_obj = objective - node->lower_bound;
      const f_t frac          = node->branch_dir == rounding_direction_t::DOWN
                                  ? node->fractional_val - std::floor(node->fractional_val)
                                  : std::ceil(node->fractional_val) - node->fractional_val;
      if (frac > 1e-10) {
        worker_.queue_pseudo_cost_update(node->branch_var, node->branch_dir, change_in_obj / frac);
      }
    }
  }

  i_t select_branch_variable(const std::vector<i_t>& fractional, const std::vector<f_t>& solution)
  {
    return worker_.variable_selection_from_snapshot(fractional, solution);
  }

  rounding_direction_t get_preferred_direction(i_t branch_var, f_t branch_val)
  {
    f_t root_val =
      branch_var < (i_t)root_solution_.size() ? root_solution_[branch_var] : branch_val;
    f_t frac = branch_val - std::floor(branch_val);
    if (branch_val < root_val - 0.4 || frac < 0.3) {
      return rounding_direction_t::DOWN;
    } else if (branch_val > root_val + 0.4 || frac > 0.7) {
      return rounding_direction_t::UP;
    }
    return rounding_direction_t::DOWN;
  }

  void track_node_explored() { ++stats_.nodes_explored; }

  void track_node_unexplored_delta(i_t delta) { stats_.nodes_unexplored += delta; }

  void set_work_units_at_start(double work_units) { work_units_at_start_ = work_units; }

 private:
  bb_worker_state_t<i_t, f_t>& worker_;
  f_t fathom_tolerance_;
  const std::vector<f_t>& root_lower_;
  const std::vector<f_t>& root_upper_;
  const std::vector<f_t>& edge_norms_;
  const std::vector<f_t>& root_solution_;
  bnb_stats_t<i_t, f_t>& stats_;
  bsp_debug_settings_t& debug_settings_;
  bsp_debug_logger_t<i_t, f_t>& debug_logger_;
  double work_units_at_start_{0.0};
};

// =============================================================================
// Standard diving policy implementation (non-deterministic)
// =============================================================================

template <typename i_t, typename f_t>
class standard_diving_solve_policy_t {
 public:
  standard_diving_solve_policy_t(
    omp_atomic_t<f_t>& upper_bound,
    f_t fathom_tolerance,
    const std::vector<f_t>& dive_lower,
    const std::vector<f_t>& dive_upper,
    const std::vector<f_t>& edge_norms,
    const std::vector<f_t>& root_solution,
    pseudo_costs_t<i_t, f_t>& pc,
    bnb_stats_t<i_t, f_t>& stats,
    std::function<void(f_t, const std::vector<f_t>&, i_t, bnb_worker_type_t)> add_solution_fn,
    bnb_worker_type_t diving_type,
    const lp_problem_t<i_t, f_t>* lp_problem,
    const std::vector<i_t>* up_locks,
    const std::vector<i_t>* down_locks)
    : upper_bound_(upper_bound),
      fathom_tolerance_(fathom_tolerance),
      dive_lower_(dive_lower),
      dive_upper_(dive_upper),
      edge_norms_(edge_norms),
      root_solution_(root_solution),
      pc_(pc),
      stats_(stats),
      add_solution_fn_(add_solution_fn),
      diving_type_(diving_type),
      lp_problem_(lp_problem),
      up_locks_(up_locks),
      down_locks_(down_locks)
  {
    log_.log = false;
  }

  f_t get_upper_bound() const { return upper_bound_.load(); }
  f_t get_fathom_tolerance() const { return fathom_tolerance_; }
  bool should_run_bounds_strengthening() const { return true; }
  work_limit_context_t* get_work_context() { return nullptr; }
  const std::vector<f_t>& get_root_lower() const { return dive_lower_; }
  const std::vector<f_t>& get_root_upper() const { return dive_upper_; }
  const std::vector<f_t>& get_edge_norms() const { return edge_norms_; }
  const std::vector<f_t>& get_root_solution() const { return root_solution_; }

  void on_solve_start(mip_node_t<i_t, f_t>* node) {}

  void on_lp_input(mip_node_t<i_t, f_t>* node,
                   const lp_problem_t<i_t, f_t>& problem,
                   const std::vector<variable_status_t>& vstatus)
  {
  }

  void on_lp_output(mip_node_t<i_t, f_t>* node,
                    dual::status_t status,
                    i_t iterations,
                    const lp_solution_t<i_t, f_t>& solution,
                    const lp_problem_t<i_t, f_t>& problem)
  {
  }

  void on_lp_solve_complete(mip_node_t<i_t, f_t>* node,
                            i_t iterations,
                            f_t solve_time,
                            dual::status_t status)
  {
    stats_.total_lp_solve_time += solve_time;
    stats_.total_lp_iters += iterations;
  }

  void on_infeasible(mip_node_t<i_t, f_t>* node, search_tree_t<i_t, f_t>& tree)
  {
    node->lower_bound = std::numeric_limits<f_t>::infinity();
    tree.update(node, node_status_t::INFEASIBLE);
  }

  void on_fathomed(mip_node_t<i_t, f_t>* node, f_t objective, search_tree_t<i_t, f_t>& tree)
  {
    tree.update(node, node_status_t::FATHOMED);
  }

  void on_integer_solution(mip_node_t<i_t, f_t>* node,
                           f_t objective,
                           const std::vector<f_t>& solution,
                           i_t depth,
                           search_tree_t<i_t, f_t>& tree)
  {
    add_solution_fn_(objective, solution, depth, diving_type_);
    tree.update(node, node_status_t::INTEGER_FEASIBLE);
  }

  void on_branched(mip_node_t<i_t, f_t>* node,
                   i_t branch_var,
                   f_t branch_val,
                   i_t down_child_id,
                   i_t up_child_id,
                   rounding_direction_t preferred,
                   search_tree_t<i_t, f_t>& tree,
                   const std::vector<i_t>& /*fractional*/,
                   const std::vector<f_t>& /*solution*/)
  {
    tree.update(node, node_status_t::HAS_CHILDREN);
  }

  void on_numerical(mip_node_t<i_t, f_t>* node, search_tree_t<i_t, f_t>& tree)
  {
    tree.update(node, node_status_t::NUMERICAL);
  }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node, f_t objective)
  {
    pc_.update_pseudo_costs(node, objective);
  }

  i_t select_branch_variable(const std::vector<i_t>& fractional, const std::vector<f_t>& solution)
  {
    branch_variable_t<i_t> result;
    switch (diving_type_) {
      case bnb_worker_type_t::PSEUDOCOST_DIVING:
        result = pseudocost_diving(pc_, fractional, solution, root_solution_, log_);
        break;
      case bnb_worker_type_t::GUIDED_DIVING:
        if (!incumbent_.empty()) {
          result = guided_diving(pc_, fractional, solution, incumbent_, log_);
        } else {
          result = pseudocost_diving(pc_, fractional, solution, root_solution_, log_);
        }
        break;
      case bnb_worker_type_t::LINE_SEARCH_DIVING:
        result = line_search_diving<i_t, f_t>(fractional, solution, root_solution_, log_);
        break;
      case bnb_worker_type_t::COEFFICIENT_DIVING:
        if (lp_problem_ && up_locks_ && down_locks_) {
          result =
            coefficient_diving(*lp_problem_, fractional, solution, *up_locks_, *down_locks_, log_);
        } else {
          result = pseudocost_diving(pc_, fractional, solution, root_solution_, log_);
        }
        break;
      default: result = pseudocost_diving(pc_, fractional, solution, root_solution_, log_); break;
    }
    selected_direction_ = result.direction;
    return result.variable;
  }

  rounding_direction_t get_preferred_direction(i_t branch_var, f_t branch_val)
  {
    return selected_direction_;
  }

  void track_node_explored() { ++stats_.nodes_explored; }

  void track_node_unexplored_delta(i_t delta) {}

  void set_incumbent(const std::vector<f_t>& incumbent) { incumbent_ = incumbent; }

 private:
  omp_atomic_t<f_t>& upper_bound_;
  f_t fathom_tolerance_;
  const std::vector<f_t>& dive_lower_;
  const std::vector<f_t>& dive_upper_;
  const std::vector<f_t>& edge_norms_;
  const std::vector<f_t>& root_solution_;
  pseudo_costs_t<i_t, f_t>& pc_;
  bnb_stats_t<i_t, f_t>& stats_;
  std::function<void(f_t, const std::vector<f_t>&, i_t, bnb_worker_type_t)> add_solution_fn_;
  bnb_worker_type_t diving_type_;
  const lp_problem_t<i_t, f_t>* lp_problem_;
  const std::vector<i_t>* up_locks_;
  const std::vector<i_t>* down_locks_;
  logger_t log_;
  std::vector<f_t> incumbent_;
  rounding_direction_t selected_direction_{rounding_direction_t::DOWN};
};

// =============================================================================
// BSP diving policy implementation (deterministic)
// =============================================================================

template <typename i_t, typename f_t>
class bsp_diving_solve_policy_t {
 public:
  bsp_diving_solve_policy_t(bsp_diving_worker_state_t<i_t, f_t>& worker,
                            f_t fathom_tolerance,
                            const std::vector<f_t>& dive_lower,
                            const std::vector<f_t>& dive_upper,
                            const std::vector<f_t>& edge_norms,
                            const std::vector<f_t>& root_solution,
                            bnb_stats_t<i_t, f_t>& stats,
                            const lp_problem_t<i_t, f_t>* lp_problem,
                            const std::vector<i_t>* up_locks,
                            const std::vector<i_t>* down_locks)
    : worker_(worker),
      fathom_tolerance_(fathom_tolerance),
      dive_lower_(dive_lower),
      dive_upper_(dive_upper),
      edge_norms_(edge_norms),
      root_solution_(root_solution),
      stats_(stats),
      lp_problem_(lp_problem),
      up_locks_(up_locks),
      down_locks_(down_locks)
  {
    log_.log = false;
  }

  f_t get_upper_bound() const { return worker_.local_upper_bound; }
  f_t get_fathom_tolerance() const { return fathom_tolerance_; }
  bool should_run_bounds_strengthening() const { return true; }
  work_limit_context_t* get_work_context() { return &worker_.work_context; }
  const std::vector<f_t>& get_root_lower() const { return dive_lower_; }
  const std::vector<f_t>& get_root_upper() const { return dive_upper_; }
  const std::vector<f_t>& get_edge_norms() const { return edge_norms_; }
  const std::vector<f_t>& get_root_solution() const { return root_solution_; }

  void on_solve_start(mip_node_t<i_t, f_t>* node) {}

  void on_lp_input(mip_node_t<i_t, f_t>* node,
                   const lp_problem_t<i_t, f_t>& problem,
                   const std::vector<variable_status_t>& vstatus)
  {
  }

  void on_lp_output(mip_node_t<i_t, f_t>* node,
                    dual::status_t status,
                    i_t iterations,
                    const lp_solution_t<i_t, f_t>& solution,
                    const lp_problem_t<i_t, f_t>& problem)
  {
  }

  void on_lp_solve_complete(mip_node_t<i_t, f_t>* node,
                            i_t iterations,
                            f_t solve_time,
                            dual::status_t status)
  {
    stats_.total_lp_solve_time += solve_time;
    stats_.total_lp_iters += iterations;
    worker_.clock = worker_.work_context.global_work_units_elapsed;
  }

  void on_infeasible(mip_node_t<i_t, f_t>* node, search_tree_t<i_t, f_t>& tree)
  {
    node->lower_bound = std::numeric_limits<f_t>::infinity();
    tree.update(node, node_status_t::INFEASIBLE);
  }

  void on_fathomed(mip_node_t<i_t, f_t>* node, f_t objective, search_tree_t<i_t, f_t>& tree)
  {
    tree.update(node, node_status_t::FATHOMED);
  }

  void on_integer_solution(mip_node_t<i_t, f_t>* node,
                           f_t objective,
                           const std::vector<f_t>& solution,
                           i_t depth,
                           search_tree_t<i_t, f_t>& tree)
  {
    if (objective < worker_.local_upper_bound) {
      worker_.queue_integer_solution(objective, solution, depth);
    }
    tree.update(node, node_status_t::INTEGER_FEASIBLE);
  }

  void on_branched(mip_node_t<i_t, f_t>* node,
                   i_t branch_var,
                   f_t branch_val,
                   i_t down_child_id,
                   i_t up_child_id,
                   rounding_direction_t preferred,
                   search_tree_t<i_t, f_t>& tree,
                   const std::vector<i_t>& /*fractional*/,
                   const std::vector<f_t>& /*solution*/)
  {
    tree.update(node, node_status_t::HAS_CHILDREN);
  }

  void on_numerical(mip_node_t<i_t, f_t>* node, search_tree_t<i_t, f_t>& tree)
  {
    tree.update(node, node_status_t::NUMERICAL);
  }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node, f_t objective) {}

  i_t select_branch_variable(const std::vector<i_t>& fractional, const std::vector<f_t>& solution)
  {
    branch_variable_t<i_t> result;
    switch (worker_.diving_type) {
      case bnb_worker_type_t::PSEUDOCOST_DIVING:
        result = worker_.variable_selection_from_snapshot(fractional, solution);
        break;
      case bnb_worker_type_t::GUIDED_DIVING:
        result = worker_.guided_variable_selection(fractional, solution);
        break;
      case bnb_worker_type_t::LINE_SEARCH_DIVING:
        if (worker_.root_solution) {
          result = line_search_diving<i_t, f_t>(fractional, solution, *worker_.root_solution, log_);
        } else {
          result = worker_.variable_selection_from_snapshot(fractional, solution);
        }
        break;
      case bnb_worker_type_t::COEFFICIENT_DIVING:
        if (lp_problem_ && up_locks_ && down_locks_) {
          result =
            coefficient_diving(*lp_problem_, fractional, solution, *up_locks_, *down_locks_, log_);
        } else {
          result = worker_.variable_selection_from_snapshot(fractional, solution);
        }
        break;
      default: result = worker_.variable_selection_from_snapshot(fractional, solution); break;
    }
    selected_direction_ = result.direction;
    return result.variable;
  }

  rounding_direction_t get_preferred_direction(i_t branch_var, f_t branch_val)
  {
    return selected_direction_;
  }

  void track_node_explored()
  {
    ++worker_.nodes_explored_this_horizon;
    ++worker_.total_nodes_explored;
  }

  void track_node_unexplored_delta(i_t delta) {}

 private:
  bsp_diving_worker_state_t<i_t, f_t>& worker_;
  f_t fathom_tolerance_;
  const std::vector<f_t>& dive_lower_;
  const std::vector<f_t>& dive_upper_;
  const std::vector<f_t>& edge_norms_;
  const std::vector<f_t>& root_solution_;
  bnb_stats_t<i_t, f_t>& stats_;
  const lp_problem_t<i_t, f_t>* lp_problem_;
  const std::vector<i_t>* up_locks_;
  const std::vector<i_t>* down_locks_;
  logger_t log_;
  rounding_direction_t selected_direction_{rounding_direction_t::DOWN};
};

}  // namespace cuopt::linear_programming::dual_simplex
