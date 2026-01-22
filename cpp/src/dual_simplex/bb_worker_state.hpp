/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bb_event.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/diving_heuristics.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/work_limit_timer.hpp>

#include <optional>

#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <queue>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// Indicate the search and variable selection algorithms used by each thread
// in B&B (See [1]).
//
// [1] T. Achterberg, “Constraint Integer Programming,” PhD, Technischen Universität Berlin,
// Berlin, 2007. doi: 10.14279/depositonce-1634.
enum class bnb_worker_type_t {
  BEST_FIRST         = 0,  // Best-First + Plunging.
  PSEUDOCOST_DIVING  = 1,  // Pseudocost diving (9.2.5)
  LINE_SEARCH_DIVING = 2,  // Line search diving (9.2.4)
  GUIDED_DIVING = 3,  // Guided diving (9.2.3). If no incumbent is found yet, use pseudocost diving.
  COEFFICIENT_DIVING = 4  // Coefficient diving (9.2.1)
};

// Queued pseudo-cost update for BSP determinism
// Updates are collected during horizon, then applied in deterministic order at sync
template <typename i_t, typename f_t>
struct pseudo_cost_update_t {
  i_t variable;
  rounding_direction_t direction;
  f_t delta;      // change_in_obj / frac
  double wut;     // work unit timestamp when update occurred (for deterministic ordering)
  int worker_id;  // for tie-breaking in sort
};

// Queued integer solution found during a horizon (merged at sync)
template <typename i_t, typename f_t>
struct queued_integer_solution_t {
  f_t objective;
  std::vector<f_t> solution;
  i_t depth;
  int worker_id;
  int sequence_id;

  bool operator<(const queued_integer_solution_t& other) const
  {
    if (objective != other.objective) return objective < other.objective;
    if (worker_id != other.worker_id) return worker_id < other.worker_id;
    return sequence_id < other.sequence_id;
  }
};

// Per-worker state for BSP (Bulk Synchronous Parallel) branch-and-bound
template <typename i_t, typename f_t>
struct bb_worker_state_t {
  int worker_id{0};

  // ==========================================================================
  // Plunging data structures (matching explore_subtree strategy)
  // ==========================================================================

  // Plunge stack: depth-first path through the tree
  // - Front = next node to process (LIFO)
  // - Max size = 2 (current node's sibling only)
  // - When branching with sibling on stack, sibling is moved to backlog
  std::deque<mip_node_t<i_t, f_t>*> plunge_stack;

  // Backlog: nodes "plugged" when branching - candidates for load balancing
  // When branching with a sibling on the plunge stack, that sibling moves here.
  // At horizon sync, backlog nodes participate in redistribution.
  std::vector<mip_node_t<i_t, f_t>*> backlog;

  // Current node being processed (may be paused at horizon boundary)
  mip_node_t<i_t, f_t>* current_node{nullptr};

  // Last node that was solved (for basis warm-start detection)
  // If next node's parent == last_solved_node, we can reuse basis
  mip_node_t<i_t, f_t>* last_solved_node{nullptr};

  // Worker's work unit clock (cumulative)
  double clock{0.0};

  // Current horizon boundaries (for BSP sync)
  double horizon_start{0.0};
  double horizon_end{0.0};

  // Creation sequence counter - cumulative across horizons for unique identity
  // Each node created by this worker gets (worker_id, next_creation_seq++)
  int32_t next_creation_seq{0};

  // Events generated during this horizon
  bb_event_batch_t<i_t, f_t> events;

  // Event sequence counter for deterministic tie-breaking
  int event_sequence{0};

  // LP problem copy for this worker (bounds modified per node)
  std::unique_ptr<lp_problem_t<i_t, f_t>> leaf_problem;

  // Basis factorization state
  std::unique_ptr<basis_update_mpf_t<i_t, f_t>> basis_factors;

  // Bounds strengthening (node presolver)
  std::unique_ptr<bounds_strengthening_t<i_t, f_t>> node_presolver;

  // Working vectors for basis
  std::vector<i_t> basic_list;
  std::vector<i_t> nonbasic_list;

  // Work unit context for this worker
  work_limit_context_t work_context;

  // Whether basis needs recomputation for next node
  bool recompute_bounds_and_basis{true};

  // Per-horizon statistics (reset each horizon)
  i_t nodes_processed_this_horizon{0};
  double work_units_this_horizon{0.0};

  // Cumulative statistics (across all horizons)
  i_t total_nodes_processed{0};
  i_t total_nodes_pruned{0};
  i_t total_nodes_branched{0};
  i_t total_nodes_infeasible{0};
  i_t total_integer_solutions{0};
  i_t total_nodes_assigned{0};  // via load balancing
  double total_work_units{0.0};

  // Timing statistics (in seconds)
  double total_runtime{0.0};      // Total time spent doing actual work
  double total_nowork_time{0.0};  // Total time spent with no nodes to work on

  // Worker-local upper bound for BSP determinism (prevents cross-worker pruning races)
  f_t local_upper_bound{std::numeric_limits<f_t>::infinity()};

  // Queued integer solutions found during this horizon (merged at sync)
  std::vector<queued_integer_solution_t<i_t, f_t>> integer_solutions;

  // Solution sequence counter for deterministic tie-breaking (cumulative across horizons)
  int next_solution_seq{0};

  // Queued pseudo-cost updates (applied to global pseudo-costs at sync)
  std::vector<pseudo_cost_update_t<i_t, f_t>> pseudo_cost_updates;

  // Pseudo-cost snapshot for deterministic variable selection
  // Initialized from global pseudo-costs at horizon start, then updated locally
  // during the horizon as the worker processes nodes. This allows within-horizon
  // learning while maintaining determinism (each worker's updates are sequential).
  // At sync, all workers' updates are merged into global pseudo-costs, and new
  // snapshots are taken at the next horizon start.
  std::vector<f_t> pc_sum_up_snapshot;
  std::vector<f_t> pc_sum_down_snapshot;
  std::vector<i_t> pc_num_up_snapshot;
  std::vector<i_t> pc_num_down_snapshot;

  // Constructor
  explicit bb_worker_state_t(int id)
    : worker_id(id), work_context("BB_Worker_" + std::to_string(id))
  {
  }

  // Initialize worker with problem data
  void initialize(const lp_problem_t<i_t, f_t>& original_lp,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  const std::vector<variable_type_t>& var_types,
                  i_t refactor_frequency,
                  bool deterministic)
  {
    // Create copy of LP problem for this worker
    leaf_problem = std::make_unique<lp_problem_t<i_t, f_t>>(original_lp);

    // Initialize basis factors
    const i_t m   = leaf_problem->num_rows;
    basis_factors = std::make_unique<basis_update_mpf_t<i_t, f_t>>(m, refactor_frequency);

    // Initialize bounds strengthening
    std::vector<char> row_sense;
    node_presolver =
      std::make_unique<bounds_strengthening_t<i_t, f_t>>(*leaf_problem, Arow, row_sense, var_types);

    // Initialize working vectors
    basic_list.resize(m);
    nonbasic_list.clear();

    // Configure work context
    work_context.deterministic = deterministic;
  }

  // Reset for new horizon
  void reset_for_horizon(double horizon_start, double horizon_end, f_t global_upper_bound)
  {
    clock = horizon_start;
    events.clear();
    events.horizon_start         = horizon_start;
    events.horizon_end           = horizon_end;
    event_sequence               = 0;
    nodes_processed_this_horizon = 0;
    work_units_this_horizon      = 0.0;
    // Also sync work_context to match clock for consistent tracking
    work_context.global_work_units_elapsed = horizon_start;
    // Note: next_creation_seq is NOT reset - it's cumulative for unique identity

    // Initialize worker-local upper bound from global (for BSP determinism)
    local_upper_bound = global_upper_bound;

    // Clear queued updates from previous horizon
    integer_solutions.clear();
    pseudo_cost_updates.clear();
  }

  // Update snapshots from global state at horizon boundary
  void set_snapshots(f_t global_upper_bound,
                     const std::vector<f_t>& pc_sum_up,
                     const std::vector<f_t>& pc_sum_down,
                     const std::vector<i_t>& pc_num_up,
                     const std::vector<i_t>& pc_num_down,
                     double new_horizon_start,
                     double new_horizon_end)
  {
    local_upper_bound    = global_upper_bound;
    pc_sum_up_snapshot   = pc_sum_up;
    pc_sum_down_snapshot = pc_sum_down;
    pc_num_up_snapshot   = pc_num_up;
    pc_num_down_snapshot = pc_num_down;
    horizon_start        = new_horizon_start;
    horizon_end          = new_horizon_end;
  }

  // Queue a pseudo-cost update for global sync AND apply it to local snapshot immediately.
  // Local snapshot updates are sequential within each worker (deterministic).
  // Global updates are merged at sync in sorted (wut, worker_id) order (deterministic).
  void queue_pseudo_cost_update(i_t variable, rounding_direction_t direction, f_t delta)
  {
    // Queue for global sync at horizon end
    pseudo_cost_updates.push_back({variable, direction, delta, clock, worker_id});

    // Also apply to local snapshot immediately for better variable selection
    if (direction == rounding_direction_t::DOWN) {
      pc_sum_down_snapshot[variable] += delta;
      pc_num_down_snapshot[variable]++;
    } else {
      pc_sum_up_snapshot[variable] += delta;
      pc_num_up_snapshot[variable]++;
    }
  }

  // Variable selection using snapshot (for BSP determinism)
  // Returns the best variable to branch on based on pseudo-cost scores
  i_t variable_selection_from_snapshot(const std::vector<i_t>& fractional,
                                       const std::vector<f_t>& solution) const
  {
    return variable_selection_from_pseudo_costs(pc_sum_down_snapshot.data(),
                                                pc_sum_up_snapshot.data(),
                                                pc_num_down_snapshot.data(),
                                                pc_num_up_snapshot.data(),
                                                (i_t)pc_sum_down_snapshot.size(),
                                                fractional,
                                                solution);
  }

  // ==========================================================================
  // Node enqueueing methods
  // ==========================================================================

  // Add a node that already has BSP identity (from load balancing or initial distribution)
  void enqueue_node_with_identity(mip_node_t<i_t, f_t>* node) { plunge_stack.push_front(node); }

  // Add children after branching with proper plunging behavior:
  // 1. If plunge stack has a sibling, move it to backlog (plugging)
  // 2. Push both children to plunge stack with preferred child on top
  // Returns the child that was placed on top (to be explored first)
  mip_node_t<i_t, f_t>* enqueue_children_for_plunge(mip_node_t<i_t, f_t>* down_child,
                                                    mip_node_t<i_t, f_t>* up_child,
                                                    rounding_direction_t preferred_direction)
  {
    // PLUGGING: If plunge stack has a sibling from previous branch, move it to backlog
    if (!plunge_stack.empty()) {
      mip_node_t<i_t, f_t>* sibling = plunge_stack.back();
      plunge_stack.pop_back();
      backlog.push_back(sibling);
    }

    // Assign BSP identity to children
    down_child->origin_worker_id = worker_id;
    down_child->creation_seq     = next_creation_seq++;
    up_child->origin_worker_id   = worker_id;
    up_child->creation_seq       = next_creation_seq++;

    // Push children - preferred child on top (front) for immediate exploration
    mip_node_t<i_t, f_t>* first_child;
    if (preferred_direction == rounding_direction_t::UP) {
      plunge_stack.push_front(down_child);  // Second to explore
      plunge_stack.push_front(up_child);    // First to explore (on top)
      first_child = up_child;
    } else {
      plunge_stack.push_front(up_child);    // Second to explore
      plunge_stack.push_front(down_child);  // First to explore (on top)
      first_child = down_child;
    }

    return first_child;
  }

  // ==========================================================================
  // Node dequeueing methods
  // ==========================================================================

  // Get next node to process using plunging strategy:
  // 1. Resume paused node if any
  // 2. Pop from plunge stack (depth-first continuation)
  // 3. Fall back to backlog (best-first from plugged nodes)
  mip_node_t<i_t, f_t>* dequeue_node()
  {
    // 1. Resume paused node if any
    if (current_node != nullptr) {
      mip_node_t<i_t, f_t>* node = current_node;
      current_node               = nullptr;
      return node;
    }

    // 2. Prefer plunge stack (depth-first continuation)
    if (!plunge_stack.empty()) {
      mip_node_t<i_t, f_t>* node = plunge_stack.front();
      plunge_stack.pop_front();
      return node;
    }

    // 3. Fall back to backlog - select best node (lowest lower_bound)
    if (!backlog.empty()) {
      auto best_it =
        std::min_element(backlog.begin(),
                         backlog.end(),
                         [](const mip_node_t<i_t, f_t>* a, const mip_node_t<i_t, f_t>* b) {
                           // Best-first: prefer lower bound
                           if (a->lower_bound != b->lower_bound) {
                             return a->lower_bound < b->lower_bound;
                           }
                           // Deterministic tie-breaking by BSP identity
                           if (a->origin_worker_id != b->origin_worker_id) {
                             return a->origin_worker_id < b->origin_worker_id;
                           }
                           return a->creation_seq < b->creation_seq;
                         });
      mip_node_t<i_t, f_t>* node = *best_it;
      backlog.erase(best_it);
      return node;
    }

    return nullptr;
  }

  // ==========================================================================
  // Queue state queries
  // ==========================================================================

  // Check if worker has work available
  bool has_work() const
  {
    return current_node != nullptr || !plunge_stack.empty() || !backlog.empty();
  }

  // Get number of nodes in worker's queues (including paused node)
  size_t queue_size() const
  {
    return plunge_stack.size() + backlog.size() + (current_node != nullptr ? 1 : 0);
  }

  // Extract only backlog nodes (for redistribution at horizon sync)
  // Plunge stack nodes stay with worker for locality
  std::vector<mip_node_t<i_t, f_t>*> extract_backlog_nodes()
  {
    std::vector<mip_node_t<i_t, f_t>*> nodes = std::move(backlog);
    backlog.clear();
    return nodes;
  }

  // Record an event
  void record_event(bb_event_t<i_t, f_t> event)
  {
    event.event_sequence = event_sequence++;
    events.add(std::move(event));
  }

  // Record node branching event
  void record_branched(
    mip_node_t<i_t, f_t>* node, i_t down_child_id, i_t up_child_id, i_t branch_var, f_t branch_val)
  {
    record_event(bb_event_t<i_t, f_t>::make_branched(clock,
                                                     worker_id,
                                                     node->node_id,
                                                     0,
                                                     down_child_id,
                                                     up_child_id,
                                                     node->lower_bound,
                                                     branch_var,
                                                     branch_val));
  }

  // Record integer solution found
  void record_integer_solution(mip_node_t<i_t, f_t>* node, f_t objective)
  {
    record_event(
      bb_event_t<i_t, f_t>::make_integer_solution(clock, worker_id, node->node_id, 0, objective));
  }

  // Record node fathomed
  void record_fathomed(mip_node_t<i_t, f_t>* node, f_t lower_bound)
  {
    record_event(
      bb_event_t<i_t, f_t>::make_fathomed(clock, worker_id, node->node_id, 0, lower_bound));
  }

  // Record node infeasible
  void record_infeasible(mip_node_t<i_t, f_t>* node)
  {
    record_event(bb_event_t<i_t, f_t>::make_infeasible(clock, worker_id, node->node_id, 0));
  }

  // Record numerical error
  void record_numerical(mip_node_t<i_t, f_t>* node)
  {
    record_event(bb_event_t<i_t, f_t>::make_numerical(clock, worker_id, node->node_id, 0));
  }

  // Update clock with work units
  void advance_clock(double work_units)
  {
    clock += work_units;
    work_units_this_horizon += work_units;
    total_work_units += work_units;
    work_context.record_work(work_units);
  }

  // Track node processed (called when a node LP solve completes)
  void track_node_processed()
  {
    ++nodes_processed_this_horizon;
    ++total_nodes_processed;
  }

  // Track node branched
  void track_node_branched() { ++total_nodes_branched; }

  // Track node pruned (fathomed due to bound)
  void track_node_pruned() { ++total_nodes_pruned; }

  // Track node infeasible
  void track_node_infeasible() { ++total_nodes_infeasible; }

  // Track integer solution found
  void track_integer_solution() { ++total_integer_solutions; }

  // Track node assigned via load balancing
  void track_node_assigned() { ++total_nodes_assigned; }
};

// =============================================================================
// BSP Diving Worker State
// =============================================================================

// Per-worker state for BSP deterministic diving
// Diving workers operate on detached copies of nodes and don't modify the main tree
template <typename i_t, typename f_t>
struct bsp_diving_worker_state_t {
  int worker_id{0};
  bnb_worker_type_t diving_type{bnb_worker_type_t::PSEUDOCOST_DIVING};

  // Worker's work unit clock (cumulative)
  double clock{0.0};

  // Current horizon boundaries
  double horizon_start{0.0};
  double horizon_end{0.0};

  // Work context for horizon sync
  work_limit_context_t work_context;

  // LP problem copy for this worker
  std::unique_ptr<lp_problem_t<i_t, f_t>> leaf_problem;

  // Basis factorization state
  std::unique_ptr<basis_update_mpf_t<i_t, f_t>> basis_factors;

  // Bounds strengthening
  std::unique_ptr<bounds_strengthening_t<i_t, f_t>> node_presolver;

  // Working vectors for basis
  std::vector<i_t> basic_list;
  std::vector<i_t> nonbasic_list;

  // Whether basis needs recomputation for next node
  bool recompute_bounds_and_basis{true};

  // ==========================================================================
  // Snapshots for determinism (taken at horizon start)
  // ==========================================================================

  f_t local_upper_bound{std::numeric_limits<f_t>::infinity()};

  // Incumbent snapshot for guided diving
  std::vector<f_t> incumbent_snapshot;

  // Pseudo-cost snapshots
  std::vector<f_t> pc_sum_up_snapshot;
  std::vector<f_t> pc_sum_down_snapshot;
  std::vector<i_t> pc_num_up_snapshot;
  std::vector<i_t> pc_num_down_snapshot;

  // Root relaxation solution (for line search diving)
  const std::vector<f_t>* root_solution{nullptr};

  // ==========================================================================
  // Diving-specific state
  // ==========================================================================

  // Queue of starting nodes for dives (detached copies assigned at sync)
  // Worker processes these until queue empty or horizon exhausted
  std::deque<mip_node_t<i_t, f_t>> dive_queue;

  // Current lower/upper bounds for the dive (initialized from starting node)
  std::vector<f_t> dive_lower;
  std::vector<f_t> dive_upper;

  // Queued integer solutions found during this horizon (merged at sync)
  std::vector<queued_integer_solution_t<i_t, f_t>> integer_solutions;

  // Solution sequence counter for deterministic tie-breaking (cumulative across horizons)
  int next_solution_seq{0};

  // ==========================================================================
  // Statistics
  // ==========================================================================

  i_t nodes_explored_this_horizon{0};
  i_t total_nodes_explored{0};
  i_t total_integer_solutions{0};
  i_t total_dives{0};
  double total_runtime{0.0};
  double total_nowork_time{0.0};

  // ==========================================================================
  // Constructor and initialization
  // ==========================================================================

  explicit bsp_diving_worker_state_t(int id, bnb_worker_type_t type)
    : worker_id(id), diving_type(type), work_context("Diving_Worker_" + std::to_string(id))
  {
  }

  void initialize(const lp_problem_t<i_t, f_t>& original_lp,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  const std::vector<variable_type_t>& var_types,
                  i_t refactor_frequency,
                  bool deterministic)
  {
    leaf_problem = std::make_unique<lp_problem_t<i_t, f_t>>(original_lp);

    const i_t m   = leaf_problem->num_rows;
    basis_factors = std::make_unique<basis_update_mpf_t<i_t, f_t>>(m, refactor_frequency);

    std::vector<char> row_sense;
    node_presolver =
      std::make_unique<bounds_strengthening_t<i_t, f_t>>(*leaf_problem, Arow, row_sense, var_types);

    basic_list.resize(m);
    nonbasic_list.clear();

    dive_lower = original_lp.lower;
    dive_upper = original_lp.upper;

    work_context.deterministic = deterministic;
  }

  void reset_for_horizon(double start, double end, f_t upper_bound)
  {
    clock                                  = start;
    horizon_start                          = start;
    horizon_end                            = end;
    work_context.global_work_units_elapsed = start;

    local_upper_bound           = upper_bound;
    nodes_explored_this_horizon = 0;
    // Note: Don't clear dive_queue here - workers may still have nodes to process
    integer_solutions.clear();
    recompute_bounds_and_basis = true;
  }

  void set_snapshots(f_t global_upper_bound,
                     const std::vector<f_t>& pc_sum_up,
                     const std::vector<f_t>& pc_sum_down,
                     const std::vector<i_t>& pc_num_up,
                     const std::vector<i_t>& pc_num_down,
                     const std::vector<f_t>& incumbent,
                     const std::vector<f_t>* root_sol,
                     double new_horizon_start,
                     double new_horizon_end)
  {
    local_upper_bound    = global_upper_bound;
    pc_sum_up_snapshot   = pc_sum_up;
    pc_sum_down_snapshot = pc_sum_down;
    pc_num_up_snapshot   = pc_num_up;
    pc_num_down_snapshot = pc_num_down;
    incumbent_snapshot   = incumbent;
    root_solution        = root_sol;
    horizon_start        = new_horizon_start;
    horizon_end          = new_horizon_end;
  }

  void enqueue_dive_node(mip_node_t<i_t, f_t>* node) { dive_queue.push_back(node->detach_copy()); }

  std::optional<mip_node_t<i_t, f_t>> dequeue_dive_node()
  {
    if (dive_queue.empty()) return std::nullopt;
    auto node = std::move(dive_queue.front());
    dive_queue.pop_front();
    ++total_dives;
    return node;
  }

  bool has_work() const { return !dive_queue.empty(); }

  size_t dive_queue_size() const { return dive_queue.size(); }

  void queue_integer_solution(f_t objective, const std::vector<f_t>& solution, i_t depth)
  {
    integer_solutions.push_back({objective, solution, depth, worker_id, next_solution_seq++});
    ++total_integer_solutions;
  }

  // Variable selection using snapshot pseudo-costs (for pseudocost diving)
  branch_variable_t<i_t> variable_selection_from_snapshot(const std::vector<i_t>& fractional,
                                                          const std::vector<f_t>& solution) const
  {
    // Use root_solution if available, otherwise use solution as fallback
    const std::vector<f_t>& root_sol = (root_solution != nullptr) ? *root_solution : solution;
    return pseudocost_diving_from_arrays(pc_sum_down_snapshot.data(),
                                         pc_sum_up_snapshot.data(),
                                         pc_num_down_snapshot.data(),
                                         pc_num_up_snapshot.data(),
                                         (i_t)pc_sum_down_snapshot.size(),
                                         fractional,
                                         solution,
                                         root_sol);
  }

  // Guided diving variable selection using incumbent snapshot
  branch_variable_t<i_t> guided_variable_selection(const std::vector<i_t>& fractional,
                                                   const std::vector<f_t>& solution) const
  {
    if (incumbent_snapshot.empty()) {
      return variable_selection_from_snapshot(fractional, solution);
    }

    return guided_diving_from_arrays(pc_sum_down_snapshot.data(),
                                     pc_sum_up_snapshot.data(),
                                     pc_num_down_snapshot.data(),
                                     pc_num_up_snapshot.data(),
                                     (i_t)pc_sum_down_snapshot.size(),
                                     fractional,
                                     solution,
                                     incumbent_snapshot);
  }
};

// Container for all diving worker states
template <typename i_t, typename f_t>
class bsp_diving_worker_pool_t {
 public:
  bsp_diving_worker_pool_t() = default;

  void initialize(int num_workers,
                  const std::vector<bnb_worker_type_t>& diving_types,
                  const lp_problem_t<i_t, f_t>& original_lp,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  const std::vector<variable_type_t>& var_types,
                  i_t refactor_frequency,
                  bool deterministic)
  {
    workers_.clear();
    workers_.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      bnb_worker_type_t type = diving_types[i % diving_types.size()];
      workers_.emplace_back(i, type);
      workers_.back().initialize(original_lp, Arow, var_types, refactor_frequency, deterministic);
    }
  }

  bsp_diving_worker_state_t<i_t, f_t>& operator[](int worker_id) { return workers_[worker_id]; }
  const bsp_diving_worker_state_t<i_t, f_t>& operator[](int worker_id) const
  {
    return workers_[worker_id];
  }

  int size() const { return static_cast<int>(workers_.size()); }

  void reset_for_horizon(double horizon_start, double horizon_end, f_t global_upper_bound)
  {
    for (auto& worker : workers_) {
      worker.reset_for_horizon(horizon_start, horizon_end, global_upper_bound);
    }
  }

  bool any_has_work() const
  {
    for (const auto& worker : workers_) {
      if (worker.has_work()) return true;
    }
    return false;
  }

  auto begin() { return workers_.begin(); }
  auto end() { return workers_.end(); }
  auto begin() const { return workers_.begin(); }
  auto end() const { return workers_.end(); }

 private:
  std::vector<bsp_diving_worker_state_t<i_t, f_t>> workers_;
};

// Container for all worker states in BSP B&B
template <typename i_t, typename f_t>
class bb_worker_pool_t {
 public:
  bb_worker_pool_t() = default;

  // Initialize pool with specified number of workers
  void initialize(int num_workers,
                  const lp_problem_t<i_t, f_t>& original_lp,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  const std::vector<variable_type_t>& var_types,
                  i_t refactor_frequency,
                  bool deterministic)
  {
    workers_.clear();
    workers_.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      workers_.emplace_back(i);
      workers_.back().initialize(original_lp, Arow, var_types, refactor_frequency, deterministic);
    }
  }

  // Get worker by ID
  bb_worker_state_t<i_t, f_t>& operator[](int worker_id) { return workers_[worker_id]; }

  const bb_worker_state_t<i_t, f_t>& operator[](int worker_id) const { return workers_[worker_id]; }

  // Get number of workers
  int size() const { return static_cast<int>(workers_.size()); }

  // Reset all workers for new horizon
  void reset_for_horizon(double horizon_start, double horizon_end, f_t global_upper_bound)
  {
    for (auto& worker : workers_) {
      worker.reset_for_horizon(horizon_start, horizon_end, global_upper_bound);
    }
  }

  // Collect all events from all workers into a single sorted batch
  bb_event_batch_t<i_t, f_t> collect_and_sort_events()
  {
    bb_event_batch_t<i_t, f_t> all_events;
    for (auto& worker : workers_) {
      for (auto& event : worker.events.events) {
        all_events.add(std::move(event));
      }
      worker.events.clear();
    }
    all_events.sort_for_replay();
    return all_events;
  }

  // Check if any worker has work
  bool any_has_work() const
  {
    for (const auto& worker : workers_) {
      if (worker.has_work()) return true;
    }
    return false;
  }

  // Get total queue size across all workers
  size_t total_queue_size() const
  {
    size_t total = 0;
    for (const auto& worker : workers_) {
      total += worker.queue_size();
    }
    return total;
  }

  // Iterator support
  auto begin() { return workers_.begin(); }
  auto end() { return workers_.end(); }
  auto begin() const { return workers_.begin(); }
  auto end() const { return workers_.end(); }

 private:
  std::vector<bb_worker_state_t<i_t, f_t>> workers_;
};

}  // namespace cuopt::linear_programming::dual_simplex
