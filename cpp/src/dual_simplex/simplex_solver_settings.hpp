/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/logger.hpp>
#include <dual_simplex/types.hpp>

#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct diving_heuristics_settings_t {
  // -1 automatic, 0 disabled, 1 enabled
  i_t line_search_diving = -1;
  i_t pseudocost_diving  = -1;
  i_t guided_diving      = -1;
  i_t coefficient_diving = -1;

  // The minimum depth to start diving from.
  i_t min_node_depth = 10;

  // The maximum number of nodes when performing a dive.
  i_t node_limit = 500;

  // The maximum number of dual simplex iteration allowed
  // in a single dive. This set in terms of the total number of
  // iterations in the best-first threads.
  f_t iteration_limit_factor = 0.05;

  // The maximum backtracking allowed.
  i_t backtrack_limit = 5;
};

template <typename i_t, typename f_t>
struct reliability_branching_settings_t {
  // Enable or disable reliability branching
  bool enable = false;

  // Lower bound for the maximum number of LP iterations for a single trial branching
  i_t lower_max_lp_iter = 10;

  // Upper bound for the maximum number of LP iterations for a single trial branching
  i_t upper_max_lp_iter = 500;

  // Priority of the tasks created when running the trial branching in parallel.
  // Set to 1 to have the same priority as the other tasks.
  i_t task_priority = 5;

  // The number of tasks spawned for performing strong branching.
  i_t num_tasks = -1;

  // The maximum number of candidates initialized by strong branching in a single
  // node
  i_t max_num_candidates = 100;

  // The maximum number of candidates evaluated that does not improve the best score.
  i_t max_lookahead = 10;

  // Define the maximum number of iteration spent in strong branching.
  // Let `bnb_lp_iter` = total number of iterations in B&B, then
  // `max iter in strong branching = bnb_lp_factor * bnb_lp_iter + bnb_lp_offset`.
  // This is used for determining the `reliable_threshold`.
  i_t bnb_lp_factor = 0.5;
  i_t bnb_lp_offset = 100000;

  // Threshold for determining for the number of pseudocost updates. Used for
  // determining if the pseudocost is reliable or not.
  // - <0: automatic
  // - 0: disable (use pseudocost branching instead)
  // - >0: will use the value for the threshold.
  i_t reliable_threshold = -1;

  // Maximum and minimum values for `reliable_threshold`. If strong branching is
  // cheap, then the value of the `reliable_threshold` can be greater
  // than the `max_reliable_threshold`.
  // Only used when `reliable_threshold` is negative
  i_t max_reliable_threshold = 5;
  i_t min_reliable_threshold = 1;
};

template <typename i_t, typename f_t>
struct simplex_solver_settings_t {
 public:
  simplex_solver_settings_t()
    : iteration_limit(std::numeric_limits<i_t>::max()),
      node_limit(std::numeric_limits<i_t>::max()),
      time_limit(std::numeric_limits<f_t>::infinity()),
      absolute_mip_gap_tol(0.0),
      relative_mip_gap_tol(1e-3),
      integer_tol(1e-5),
      primal_tol(1e-6),
      dual_tol(1e-6),
      pivot_tol(1e-7),
      tight_tol(1e-10),
      fixed_tol(1e-10),
      zero_tol(1e-12),
      barrier_relative_feasibility_tol(1e-8),
      barrier_relative_optimality_tol(1e-8),
      barrier_relative_complementarity_tol(1e-8),
      barrier_relaxed_feasibility_tol(1e-4),
      barrier_relaxed_optimality_tol(1e-4),
      barrier_relaxed_complementarity_tol(1e-4),
      cut_off(std::numeric_limits<f_t>::infinity()),
      steepest_edge_ratio(0.5),
      steepest_edge_primal_tol(1e-9),
      hypersparse_threshold(0.05),
      threshold_partial_pivoting_tol(1.0 / 10.0),
      use_steepest_edge_pricing(true),
      use_harris_ratio(false),
      use_bound_flip_ratio(true),
      scale_columns(true),
      relaxation(false),
      use_left_looking_lu(false),
      eliminate_singletons(true),
      print_presolve_stats(true),
      barrier_presolve(false),
      cudss_deterministic(false),
      barrier(false),
      eliminate_dense_columns(true),
      num_gpus(1),
      folding(-1),
      augmented(0),
      dualize(-1),
      ordering(-1),
      barrier_dual_initial_point(-1),
      check_Q(false),
      crossover(false),
      refactor_frequency(100),
      iteration_log_frequency(1000),
      first_iteration_log(2),
      num_threads(omp_get_max_threads() - 1),
      random_seed(0),
      inside_mip(0),
      solution_callback(nullptr),
      heuristic_preemption_callback(nullptr),
      concurrent_halt(nullptr)
  {
  }

  void set_log(bool logging) const { log.log = logging; }
  void enable_log_to_file() { log.enable_log_to_file(); }
  void set_log_filename(const std::string& log_filename) { log.set_log_file(log_filename); }
  void close_log_file() { log.close_log_file(); }
  i_t iteration_limit;
  i_t node_limit;
  f_t time_limit;
  f_t absolute_mip_gap_tol;  // Tolerance on mip gap to declare optimal
  f_t relative_mip_gap_tol;  // Tolerance on mip gap to declare optimal
  f_t integer_tol;           // Tolerance on integralitiy violation
  f_t primal_tol;            // Absolute primal infeasibility tolerance
  f_t dual_tol;              // Absolute dual infeasibility tolerance
  f_t pivot_tol;             // Simplex pivot tolerance
  f_t tight_tol;             // A tight tolerance used to check for infeasibility
  f_t fixed_tol;             // If l <= x <= u with u - l < fixed_tol a variable is consider fixed
  f_t zero_tol;              // Values below this tolerance are considered numerically zero
  f_t barrier_relative_feasibility_tol;  // Relative feasibility tolerance for barrier method
  f_t barrier_relative_optimality_tol;   // Relative optimality tolerance for barrier method
  f_t
    barrier_relative_complementarity_tol;   // Relative complementarity tolerance for barrier method
  f_t barrier_relaxed_feasibility_tol;      // Relative feasibility tolerance for barrier method
  f_t barrier_relaxed_optimality_tol;       // Relative optimality tolerance for barrier method
  f_t barrier_relaxed_complementarity_tol;  // Relative complementarity tolerance for barrier method
  f_t cut_off;  // If the dual objective is greater than the cutoff we stop
  f_t
    steepest_edge_ratio;  // the ratio of computed steepest edge mismatch from updated steepest edge
  f_t steepest_edge_primal_tol;  // Primal tolerance divided by steepest edge norm
  f_t hypersparse_threshold;
  mutable f_t threshold_partial_pivoting_tol;
  bool use_steepest_edge_pricing;  // true if using steepest edge pricing, false if using max
                                   // infeasibility pricing
  bool use_harris_ratio;           // true if using the harris ratio test
  bool use_bound_flip_ratio;       // true if using the bound flip ratio test
  bool scale_columns;              // true to scale the columns of A
  bool relaxation;                 // true to only solve the LP relaxation of a MIP
  bool
    use_left_looking_lu;  // true to use left looking LU factorization, false to use right looking
  bool eliminate_singletons;  // true to eliminate singletons from the basis
  bool print_presolve_stats;  // true to print presolve stats
  bool barrier_presolve;      // true to use barrier presolve
  bool cudss_deterministic;   // true to use cuDSS deterministic mode, false for non-deterministic
  bool barrier;               // true to use barrier method, false to use dual simplex method
  bool eliminate_dense_columns;  // true to eliminate dense columns from A*D*A^T
  int num_gpus;   // Number of GPUs to use (maximum of 2 gpus are supported at the moment)
  i_t folding;    // -1 automatic, 0 don't fold, 1 fold
  i_t augmented;  // -1 automatic, 0 to solve with ADAT, 1 to solve with augmented system
  i_t dualize;    // -1 automatic, 0 to not dualize, 1 to dualize
  i_t ordering;   // -1 automatic, 0 to use nested dissection, 1 to use AMD
  i_t barrier_dual_initial_point;  // -1 automatic, 0 to use Lustig, Marsten, and Shanno initial
                                   // point, 1 to use initial point form dual least squares problem
  bool check_Q;                    // true to check if Q is positive semidefinite
  bool crossover;                  // true to do crossover, false to not
  i_t refactor_frequency;          // number of basis updates before refactorization
  i_t iteration_log_frequency;     // number of iterations between log updates
  i_t first_iteration_log;         // number of iterations to log at beginning of solve
  i_t num_threads;                 // number of threads to use
  i_t random_seed;                 // random seed

  diving_heuristics_settings_t<i_t, f_t> diving_settings;  // Settings for the diving heuristics
  reliability_branching_settings_t<i_t, f_t>
    reliability_branching_settings;  // Settings for reliability branching

  i_t inside_mip;  // 0 if outside MIP, 1 if inside MIP at root node, 2 if inside MIP at leaf node
  std::function<void(std::vector<f_t>&, f_t)> solution_callback;
  std::function<void(const std::vector<f_t>&, f_t)> node_processed_callback;
  std::function<void()> heuristic_preemption_callback;
  std::function<void(std::vector<f_t>&, std::vector<f_t>&, f_t)> set_simplex_solution_callback;
  mutable logger_t log;
  std::atomic<int>* concurrent_halt;  // if nullptr ignored, if !nullptr, 0 if solver should
                                      // continue, 1 if solver should halt
};

}  // namespace cuopt::linear_programming::dual_simplex
