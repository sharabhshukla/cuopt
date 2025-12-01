/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/diving_queue.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/types.hpp>

#include <utilities/omp_helpers.hpp>
#include <utilities/work_limit_timer.hpp>
#include <utilities/work_unit_predictor.hpp>

#include <omp.h>
#include <queue>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

enum class mip_status_t {
  OPTIMAL    = 0,  // The optimal integer solution was found
  UNBOUNDED  = 1,  // The problem is unbounded
  INFEASIBLE = 2,  // The problem is infeasible
  TIME_LIMIT = 3,  // The solver reached a time limit
  NODE_LIMIT = 4,  // The maximum number of nodes was reached (not implemented)
  NUMERICAL  = 5,  // The solver encountered a numerical error
  UNSET      = 6,  // The status is not set
  WORK_LIMIT = 7,  // The solver reached a deterministic work limit
};

enum class mip_exploration_status_t {
  UNSET      = 0,  // The status is not set
  TIME_LIMIT = 1,  // The solver reached a time limit
  NODE_LIMIT = 2,  // The maximum number of nodes was reached (not implemented)
  NUMERICAL  = 3,  // The solver encountered a numerical error
  RUNNING    = 4,  // The solver is currently exploring the tree
  COMPLETED  = 5,  // The solver finished exploring the tree
  WORK_LIMIT = 6,  // The solver reached a deterministic work limit
};

enum class node_solve_info_t {
  NO_CHILDREN      = 0,  // The node does not produced children
  UP_CHILD_FIRST   = 1,  // The up child should be explored first
  DOWN_CHILD_FIRST = 2,  // The down child should be explored first
  TIME_LIMIT       = 3,  // The solver reached a time limit
  ITERATION_LIMIT  = 4,  // The solver reached a iteration limit
  NUMERICAL        = 5,  // The solver encounter a numerical error when solving the node
  WORK_LIMIT       = 6,  // The solver reached a deterministic work limit
};

// Indicate the search and variable selection algorithms used by the thread (See [1]).
//
// [1] T. Achterberg, “Constraint Integer Programming,” PhD, Technischen Universität Berlin,
// Berlin, 2007. doi: 10.14279/depositonce-1634.
enum class thread_type_t {
  EXPLORATION = 0,  // Best-First + Plunging. Pseudocost branching + Martin's criteria.
  DIVING      = 1,
};

template <typename i_t, typename f_t>
class bounds_strengthening_t;

template <typename i_t, typename f_t>
void upper_bound_callback(f_t upper_bound);

// Feature tracking for solve_node regression model
template <typename i_t, typename f_t>
struct node_solve_features_t {
  // Static problem features (compute once)
  i_t n_rows{0};
  i_t n_cols{0};
  i_t n_nonzeros{0};
  f_t density{0.0};
  i_t n_binary{0};
  i_t n_integer{0};
  i_t n_continuous{0};
  f_t integrality_ratio{0.0};
  f_t avg_row_nnz{0.0};
  i_t max_row_nnz{0};
  f_t avg_col_nnz{0.0};
  i_t max_col_nnz{0};
  f_t row_nnz_cv{0.0};
  f_t col_nnz_cv{0.0};

  // Dynamic node state
  i_t node_id{0};
  i_t node_depth{0};
  i_t n_bounds_changed{0};
  f_t cutoff_gap_ratio{0.0};
  bool basis_from_parent{false};

  // LP solve metrics
  i_t simplex_iterations{0};
  i_t n_refactorizations{0};
  f_t lp_solve_time{0.0};
  f_t bound_str_time{0.0};
  f_t variable_sel_time{0.0};

  // Outcome metrics
  i_t n_fractional{0};
  bool strong_branch_performed{false};
  i_t n_strong_branch_candidates{0};
  f_t strong_branch_time{0.0};
  i_t lp_status{0};    // Convert dual::status_t to int
  i_t node_status{0};  // Convert node_status_t to int

  // Computed at node end
  f_t total_node_time{0.0};
};

template <typename i_t, typename f_t>
class branch_and_bound_t {
 public:
  template <typename T>
  using mip_node_heap_t = std::priority_queue<T, std::vector<T>, node_compare_t<i_t, f_t>>;

  branch_and_bound_t(const user_problem_t<i_t, f_t>& user_problem,
                     const simplex_solver_settings_t<i_t, f_t>& solver_settings);

  // Set an initial guess based on the user_problem. This should be called before solve.
  void set_initial_guess(const std::vector<f_t>& user_guess) { guess_ = user_guess; }

  // Set a solution based on the user problem during the course of the solve
  void set_new_solution(const std::vector<f_t>& solution);

  // Repair a low-quality solution from the heuristics.
  bool repair_solution(const std::vector<f_t>& leaf_edge_norms,
                       const std::vector<f_t>& potential_solution,
                       f_t& repaired_obj,
                       std::vector<f_t>& repaired_solution) const;

  f_t get_upper_bound();
  f_t get_lower_bound();
  i_t get_heap_size();

  // The main entry routine. Returns the solver status and populates solution with the incumbent.
  mip_status_t solve(mip_solution_t<i_t, f_t>& solution);

  work_limit_context_t& get_work_unit_context() { return work_unit_context_; }

 private:
  const user_problem_t<i_t, f_t>& original_problem_;
  const simplex_solver_settings_t<i_t, f_t> settings_;

  // Work unit contexts for each worker
  // TODO: only one for now, sequential B&B for now
  work_limit_context_t work_unit_context_{"B&B"};

  // Initial guess.
  std::vector<f_t> guess_;

  // LP relaxation
  lp_problem_t<i_t, f_t> original_lp_;
  std::vector<i_t> new_slacks_;
  std::vector<variable_type_t> var_types_;

  // Local lower bounds for each thread
  std::vector<omp_atomic_t<f_t>> local_lower_bounds_;

  // Mutex for upper bound
  omp_mutex_t mutex_upper_;

  // Global variable for upper bound
  f_t upper_bound_;

  // Global variable for incumbent. The incumbent should be updated with the upper bound
  mip_solution_t<i_t, f_t> incumbent_;

  // Structure with the general info of the solver.
  struct stats_t {
    f_t start_time                        = 0.0;
    omp_atomic_t<f_t> total_lp_solve_time = 0.0;
    omp_atomic_t<i_t> nodes_explored      = 0;
    omp_atomic_t<i_t> nodes_unexplored    = 0;
    omp_atomic_t<f_t> total_lp_iters      = 0;

    // This should only be used by the main thread
    omp_atomic_t<f_t> last_log             = 0.0;
    omp_atomic_t<i_t> nodes_since_last_log = 0;
  } exploration_stats_;

  // Mutex for repair
  omp_mutex_t mutex_repair_;
  std::vector<std::vector<f_t>> repair_queue_;

  // Variables for the root node in the search tree.
  std::vector<variable_status_t> root_vstatus_;
  f_t root_objective_;
  lp_solution_t<i_t, f_t> root_relax_soln_;
  std::vector<f_t> edge_norms_;

  // Pseudocosts
  pseudo_costs_t<i_t, f_t> pc_;

  // Heap storing the nodes to be explored.
  omp_mutex_t mutex_heap_;
  mip_node_heap_t<mip_node_t<i_t, f_t>*> heap_;

  // Count the number of subtrees that are currently being explored.
  omp_atomic_t<i_t> active_subtrees_;

  // Queue for storing the promising node for performing dives.
  omp_mutex_t mutex_dive_queue_;
  diving_queue_t<i_t, f_t> diving_queue_;
  i_t min_diving_queue_size_;

  // Global status of the solver.
  omp_atomic_t<mip_exploration_status_t> solver_status_;

  omp_atomic_t<bool> should_report_;

  // In case, a best-first thread encounters a numerical issue when solving a node,
  // its blocks the progression of the lower bound.
  omp_atomic_t<f_t> lower_bound_ceiling_;

  // Feature tracking for solve_node regression model
  node_solve_features_t<i_t, f_t> static_features_;  // Static problem features, computed once
  omp_mutex_t mutex_feature_log_;                    // Protect feature logging
  node_solve_features_t<i_t, f_t> last_features_;    // Last captured features
  f_t last_feature_log_time_{0.0};                   // Time of last feature log
  bool has_pending_features_{false};                 // Whether we have features to log

  // Helper to compute static features once
  void compute_static_features();

  // Helper to log node solve features with time-based throttling
  void log_node_features(const node_solve_features_t<i_t, f_t>& features);

  // Helper to flush any pending features at end of solve
  void flush_pending_features();

  // Set the final solution.
  mip_status_t set_final_solution(mip_solution_t<i_t, f_t>& solution, f_t lower_bound);

  // Update the incumbent solution with the new feasible solution
  // found during branch and bound.
  void add_feasible_solution(f_t leaf_objective,
                             const std::vector<f_t>& leaf_solution,
                             i_t leaf_depth,
                             thread_type_t thread_type);

  // Repairs low-quality solutions from the heuristics, if it is applicable.
  void repair_heuristic_solutions();

  // Ramp-up phase of the solver, where we greedily expand the tree until
  // there is enough unexplored nodes. This is done recursively using OpenMP tasks.
  void exploration_ramp_up(mip_node_t<i_t, f_t>* node,
                           search_tree_t<i_t, f_t>* search_tree,
                           const csr_matrix_t<i_t, f_t>& Arow,
                           i_t initial_heap_size);

  // Explore the search tree using the best-first search with plunging strategy.
  void explore_subtree(i_t task_id,
                       mip_node_t<i_t, f_t>* start_node,
                       search_tree_t<i_t, f_t>& search_tree,
                       lp_problem_t<i_t, f_t>& leaf_problem,
                       bounds_strengthening_t<i_t, f_t>& node_presolver,
                       basis_update_mpf_t<i_t, f_t>& basis_update,
                       std::vector<i_t>& basic_list,
                       std::vector<i_t>& nonbasic_list);

  // Each "main" thread pops a node from the global heap and then performs a plunge
  // (i.e., a shallow dive) into the subtree determined by the node.
  void best_first_thread(i_t task_id,
                         search_tree_t<i_t, f_t>& search_tree,
                         const csr_matrix_t<i_t, f_t>& Arow);

  // Each diving thread pops the first node from the dive queue and then performs
  // a deep dive into the subtree determined by the node.
  void diving_thread(const csr_matrix_t<i_t, f_t>& Arow);

  // Solve the LP relaxation of a leaf node and update the tree.
  node_solve_info_t solve_node(mip_node_t<i_t, f_t>* node_ptr,
                               search_tree_t<i_t, f_t>& search_tree,
                               lp_problem_t<i_t, f_t>& leaf_problem,
                               basis_update_mpf_t<i_t, f_t>& basis_factors,
                               std::vector<i_t>& basic_list,
                               std::vector<i_t>& nonbasic_list,
                               bounds_strengthening_t<i_t, f_t>& node_presolver,
                               thread_type_t thread_type,
                               bool recompute_basis_and_bounds,
                               const std::vector<f_t>& root_lower,
                               const std::vector<f_t>& root_upper,
                               logger_t& log);

  // Sort the children based on the Martin's criteria.
  rounding_direction_t child_selection(mip_node_t<i_t, f_t>* node_ptr);
};

}  // namespace cuopt::linear_programming::dual_simplex
