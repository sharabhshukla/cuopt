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
#include <dual_simplex/node_queue.hpp>
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

// Comparator for backlog heap: best-first by lower_bound with deterministic BSP identity tie-break
// Returns true if 'a' has lower priority than 'b' (for max-heap behavior in std::push_heap)
template <typename i_t, typename f_t>
struct backlog_node_compare_t {
  bool operator()(const mip_node_t<i_t, f_t>* a, const mip_node_t<i_t, f_t>* b) const
  {
    // Primary: prefer smaller lower_bound (best-first search)
    if (a->lower_bound != b->lower_bound) { return a->lower_bound > b->lower_bound; }
    // Deterministic tie-breaking by BSP identity tuple
    if (a->origin_worker_id != b->origin_worker_id) {
      return a->origin_worker_id > b->origin_worker_id;
    }
    return a->creation_seq > b->creation_seq;
  }
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

  bool operator<(const pseudo_cost_update_t& other) const
  {
    if (wut != other.wut) return wut < other.wut;
    if (variable != other.variable) return variable < other.variable;
    if (delta != other.delta) return delta < other.delta;
    return worker_id < other.worker_id;
  }
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

// Per-worker state for BSP branch-and-bound
template <typename i_t, typename f_t>
struct bb_worker_state_t {
  int worker_id{0};

  // Plunge stack: LIFO queue for depth-first exploration
  std::deque<mip_node_t<i_t, f_t>*> plunge_stack;

  // Backlog heap: nodes plugged when branching, ordered by lower_bound for best-first selection
  heap_t<mip_node_t<i_t, f_t>*, backlog_node_compare_t<i_t, f_t>> backlog;

  mip_node_t<i_t, f_t>* current_node{nullptr};
  mip_node_t<i_t, f_t>* last_solved_node{nullptr};  // For basis warm-start detection

  double clock{0.0};
  double horizon_start{0.0};
  double horizon_end{0.0};

  // Cumulative counter for unique node identity: (worker_id, next_creation_seq++)
  int32_t next_creation_seq{0};

  bb_event_batch_t<i_t, f_t> events;
  int event_sequence{0};

  std::unique_ptr<lp_problem_t<i_t, f_t>> leaf_problem;
  std::unique_ptr<basis_update_mpf_t<i_t, f_t>> basis_factors;
  std::unique_ptr<bounds_strengthening_t<i_t, f_t>> node_presolver;
  std::vector<i_t> basic_list;
  std::vector<i_t> nonbasic_list;
  work_limit_context_t work_context;
  bool recompute_bounds_and_basis{true};

  i_t nodes_processed_this_horizon{0};
  i_t total_nodes_processed{0};
  i_t total_nodes_pruned{0};
  i_t total_nodes_branched{0};
  i_t total_nodes_infeasible{0};
  i_t total_integer_solutions{0};
  i_t total_nodes_assigned{0};
  double total_runtime{0.0};
  double total_nowork_time{0.0};

  f_t local_upper_bound{std::numeric_limits<f_t>::infinity()};
  f_t local_lower_bound_ceiling{std::numeric_limits<f_t>::infinity()};

  std::vector<queued_integer_solution_t<i_t, f_t>> integer_solutions;
  int next_solution_seq{0};
  std::vector<pseudo_cost_update_t<i_t, f_t>> pseudo_cost_updates;

  // Pseudo-cost snapshots: local copies updated within horizon, merged at sync
  std::vector<f_t> pc_sum_up_snapshot;
  std::vector<f_t> pc_sum_down_snapshot;
  std::vector<i_t> pc_num_up_snapshot;
  std::vector<i_t> pc_num_down_snapshot;

  explicit bb_worker_state_t(int id)
    : worker_id(id), work_context("BB_Worker_" + std::to_string(id))
  {
  }

  void initialize(const lp_problem_t<i_t, f_t>& original_lp,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  const std::vector<variable_type_t>& var_types,
                  i_t refactor_frequency,
                  bool deterministic)
  {
    leaf_problem  = std::make_unique<lp_problem_t<i_t, f_t>>(original_lp);
    const i_t m   = leaf_problem->num_rows;
    basis_factors = std::make_unique<basis_update_mpf_t<i_t, f_t>>(m, refactor_frequency);
    std::vector<char> row_sense;
    node_presolver =
      std::make_unique<bounds_strengthening_t<i_t, f_t>>(*leaf_problem, Arow, row_sense, var_types);
    basic_list.resize(m);
    nonbasic_list.clear();
    work_context.deterministic = deterministic;
  }

  void reset_for_horizon(double horizon_start, double horizon_end, f_t global_upper_bound)
  {
    clock                                  = horizon_start;
    work_context.global_work_units_elapsed = horizon_start;
    events.clear();
    events.horizon_start         = horizon_start;
    events.horizon_end           = horizon_end;
    event_sequence               = 0;
    nodes_processed_this_horizon = 0;
    local_upper_bound            = global_upper_bound;
    local_lower_bound_ceiling    = std::numeric_limits<f_t>::infinity();
    integer_solutions.clear();
    pseudo_cost_updates.clear();
  }

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

  // Queue pseudo-cost update for global sync and apply to local snapshot
  void queue_pseudo_cost_update(i_t variable, rounding_direction_t direction, f_t delta)
  {
    pseudo_cost_updates.push_back({variable, direction, delta, clock, worker_id});
    if (direction == rounding_direction_t::DOWN) {
      pc_sum_down_snapshot[variable] += delta;
      pc_num_down_snapshot[variable]++;
    } else {
      pc_sum_up_snapshot[variable] += delta;
      pc_num_up_snapshot[variable]++;
    }
  }

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

  void enqueue_node(mip_node_t<i_t, f_t>* node) { plunge_stack.push_front(node); }

  // Enqueue children with plunging: move any existing sibling to backlog, push both children
  mip_node_t<i_t, f_t>* enqueue_children_for_plunge(mip_node_t<i_t, f_t>* down_child,
                                                    mip_node_t<i_t, f_t>* up_child,
                                                    rounding_direction_t preferred_direction)
  {
    if (!plunge_stack.empty()) {
      backlog.push(plunge_stack.back());
      plunge_stack.pop_back();
    }

    down_child->origin_worker_id = worker_id;
    down_child->creation_seq     = next_creation_seq++;
    up_child->origin_worker_id   = worker_id;
    up_child->creation_seq       = next_creation_seq++;

    mip_node_t<i_t, f_t>* first_child;
    if (preferred_direction == rounding_direction_t::UP) {
      plunge_stack.push_front(down_child);
      plunge_stack.push_front(up_child);
      first_child = up_child;
    } else {
      plunge_stack.push_front(up_child);
      plunge_stack.push_front(down_child);
      first_child = down_child;
    }
    return first_child;
  }

  // Dequeue: current_node first, then plunge_stack, then backlog heap
  mip_node_t<i_t, f_t>* dequeue_node()
  {
    if (current_node != nullptr) {
      mip_node_t<i_t, f_t>* node = current_node;
      current_node               = nullptr;
      return node;
    }
    if (!plunge_stack.empty()) {
      mip_node_t<i_t, f_t>* node = plunge_stack.front();
      plunge_stack.pop_front();
      return node;
    }
    auto node_opt = backlog.pop();
    return node_opt.has_value() ? node_opt.value() : nullptr;
  }

  bool has_work() const
  {
    return current_node != nullptr || !plunge_stack.empty() || !backlog.empty();
  }

  size_t queue_size() const
  {
    return plunge_stack.size() + backlog.size() + (current_node != nullptr ? 1 : 0);
  }

  void record_event(bb_event_t<i_t, f_t> event)
  {
    event.event_sequence = event_sequence++;
    events.add(std::move(event));
  }

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

  void record_integer_solution(mip_node_t<i_t, f_t>* node, f_t objective)
  {
    record_event(
      bb_event_t<i_t, f_t>::make_integer_solution(clock, worker_id, node->node_id, 0, objective));
  }

  void record_fathomed(mip_node_t<i_t, f_t>* node, f_t lower_bound)
  {
    record_event(
      bb_event_t<i_t, f_t>::make_fathomed(clock, worker_id, node->node_id, 0, lower_bound));
  }

  void record_infeasible(mip_node_t<i_t, f_t>* node)
  {
    record_event(bb_event_t<i_t, f_t>::make_infeasible(clock, worker_id, node->node_id, 0));
  }

  void record_numerical(mip_node_t<i_t, f_t>* node)
  {
    record_event(bb_event_t<i_t, f_t>::make_numerical(clock, worker_id, node->node_id, 0));
  }

  void track_node_processed()
  {
    ++nodes_processed_this_horizon;
    ++total_nodes_processed;
  }

  void track_node_branched() { ++total_nodes_branched; }
  void track_node_pruned() { ++total_nodes_pruned; }
  void track_node_infeasible() { ++total_nodes_infeasible; }
  void track_integer_solution() { ++total_integer_solutions; }
  void track_node_assigned() { ++total_nodes_assigned; }
};

// Per-worker state for BSP diving (operates on detached node copies)
template <typename i_t, typename f_t>
struct bsp_diving_worker_state_t {
  int worker_id{0};
  bnb_worker_type_t diving_type{bnb_worker_type_t::PSEUDOCOST_DIVING};

  double clock{0.0};
  double horizon_start{0.0};
  double horizon_end{0.0};
  work_limit_context_t work_context;

  std::unique_ptr<lp_problem_t<i_t, f_t>> leaf_problem;
  std::unique_ptr<basis_update_mpf_t<i_t, f_t>> basis_factors;
  std::unique_ptr<bounds_strengthening_t<i_t, f_t>> node_presolver;
  std::vector<i_t> basic_list;
  std::vector<i_t> nonbasic_list;
  bool recompute_bounds_and_basis{true};

  // Snapshots taken at horizon start
  f_t local_upper_bound{std::numeric_limits<f_t>::infinity()};
  i_t total_lp_iters_snapshot{0};
  std::vector<f_t> incumbent_snapshot;
  std::vector<f_t> pc_sum_up_snapshot;
  std::vector<f_t> pc_sum_down_snapshot;
  std::vector<i_t> pc_num_up_snapshot;
  std::vector<i_t> pc_num_down_snapshot;
  const std::vector<f_t>* root_solution{nullptr};

  std::deque<mip_node_t<i_t, f_t>> dive_queue;
  std::vector<f_t> dive_lower;
  std::vector<f_t> dive_upper;

  std::vector<queued_integer_solution_t<i_t, f_t>> integer_solutions;
  int next_solution_seq{0};
  std::vector<pseudo_cost_update_t<i_t, f_t>> pseudo_cost_updates;

  i_t total_nodes_explored{0};
  i_t total_integer_solutions{0};
  i_t total_dives{0};
  i_t lp_iters_this_dive{0};
  double total_runtime{0.0};
  double total_nowork_time{0.0};

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
    leaf_problem  = std::make_unique<lp_problem_t<i_t, f_t>>(original_lp);
    const i_t m   = leaf_problem->num_rows;
    basis_factors = std::make_unique<basis_update_mpf_t<i_t, f_t>>(m, refactor_frequency);
    std::vector<char> row_sense;
    node_presolver =
      std::make_unique<bounds_strengthening_t<i_t, f_t>>(*leaf_problem, Arow, row_sense, var_types);
    basic_list.resize(m);
    nonbasic_list.clear();
    dive_lower                 = original_lp.lower;
    dive_upper                 = original_lp.upper;
    work_context.deterministic = deterministic;
  }

  void reset_for_horizon(double start, double end, f_t upper_bound)
  {
    clock                                  = start;
    horizon_start                          = start;
    horizon_end                            = end;
    work_context.global_work_units_elapsed = start;

    local_upper_bound = upper_bound;
    integer_solutions.clear();
    pseudo_cost_updates.clear();
    recompute_bounds_and_basis = true;
  }

  void set_snapshots(f_t global_upper_bound,
                     i_t total_lp_iters,
                     const std::vector<f_t>& pc_sum_up,
                     const std::vector<f_t>& pc_sum_down,
                     const std::vector<i_t>& pc_num_up,
                     const std::vector<i_t>& pc_num_down,
                     const std::vector<f_t>& incumbent,
                     const std::vector<f_t>* root_sol,
                     double new_horizon_start,
                     double new_horizon_end)
  {
    local_upper_bound       = global_upper_bound;
    total_lp_iters_snapshot = total_lp_iters;
    pc_sum_up_snapshot      = pc_sum_up;
    pc_sum_down_snapshot    = pc_sum_down;
    pc_num_up_snapshot      = pc_num_up;
    pc_num_down_snapshot    = pc_num_down;
    incumbent_snapshot      = incumbent;
    root_solution           = root_sol;
    horizon_start           = new_horizon_start;
    horizon_end             = new_horizon_end;
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

  void queue_pseudo_cost_update(i_t variable, rounding_direction_t direction, f_t delta)
  {
    pseudo_cost_updates.push_back({variable, direction, delta, clock, worker_id});
    if (direction == rounding_direction_t::DOWN) {
      pc_sum_down_snapshot[variable] += delta;
      pc_num_down_snapshot[variable]++;
    } else {
      pc_sum_up_snapshot[variable] += delta;
      pc_num_up_snapshot[variable]++;
    }
  }

  branch_variable_t<i_t> variable_selection_from_snapshot(const std::vector<i_t>& fractional,
                                                          const std::vector<f_t>& solution) const
  {
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

template <typename i_t, typename f_t>
class bb_worker_pool_t {
 public:
  bb_worker_pool_t() = default;

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

  bb_worker_state_t<i_t, f_t>& operator[](int worker_id) { return workers_[worker_id]; }
  const bb_worker_state_t<i_t, f_t>& operator[](int worker_id) const { return workers_[worker_id]; }
  int size() const { return static_cast<int>(workers_.size()); }

  void reset_for_horizon(double horizon_start, double horizon_end, f_t global_upper_bound)
  {
    for (auto& worker : workers_) {
      worker.reset_for_horizon(horizon_start, horizon_end, global_upper_bound);
    }
  }

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

  bool any_has_work() const
  {
    for (const auto& worker : workers_) {
      if (worker.has_work()) return true;
    }
    return false;
  }

  size_t total_queue_size() const
  {
    size_t total = 0;
    for (const auto& worker : workers_) {
      total += worker.queue_size();
    }
    return total;
  }

  auto begin() { return workers_.begin(); }
  auto end() { return workers_.end(); }
  auto begin() const { return workers_.begin(); }
  auto end() const { return workers_.end(); }

 private:
  std::vector<bb_worker_state_t<i_t, f_t>> workers_;
};

}  // namespace cuopt::linear_programming::dual_simplex
