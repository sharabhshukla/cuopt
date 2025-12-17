/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Indicate the search and variable selection algorithms used by each task
// in B&B (See [1]).
//
// [1] T. Achterberg, “Constraint Integer Programming,” PhD, Technischen Universität Berlin,
// Berlin, 2007. doi: 10.14279/depositonce-1634.
enum bnb_task_type_t {
  EXPLORATION        = 0,  // Best-First + Plunging.
  PSEUDOCOST_DIVING  = 1,  // Pseudocost diving (9.2.5)
  LINE_SEARCH_DIVING = 2,  // Line search diving (9.2.4)
  GUIDED_DIVING = 3,  // Guided diving (9.2.3). If no incumbent is found yet, use pseudocost diving.
  COEFFICIENT_DIVING = 4  // Coefficient diving (9.2.1)
};

// Settings for each task type in B&B.
template <typename i_t, typename f_t>
struct bnb_task_settings_t {
  // Type of the task.
  bnb_task_type_t type;

  // Is this type of task enabled?
  // This will be ignored if `type == EXPLORATION`.
  bool is_enabled;

  // Number of tasks of this type.
  i_t num_tasks;

  // Minimum node depth to start this task
  // This will be ignored if `type == EXPLORATION`.
  i_t min_node_depth;

  // Maximum number of nodes explored in this task.
  i_t node_limit;

  // Maximum fraction of the number of simplex iterations for this task
  // compared to the number of simplex iterations for normal exploration.
  f_t iteration_limit_factor;

  // Number of nodes that it allows to backtrack when
  // reaching the bottom of a given branch of the tree.
  i_t backtrack;
};

template <typename i_t, typename f_t>
bnb_task_settings_t<i_t, f_t> get_default_diving_settings(bnb_task_type_t type);

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
      random_seed(0),
      inside_mip(0),
      solution_callback(nullptr),
      heuristic_preemption_callback(nullptr),
      concurrent_halt(nullptr)
  {
    bnb_task_settings[EXPLORATION] =
      bnb_task_settings_t<i_t, f_t>{.type                   = EXPLORATION,
                                    .is_enabled             = true,
                                    .num_tasks              = -1,
                                    .min_node_depth         = 0,
                                    .node_limit             = std::numeric_limits<i_t>::max(),
                                    .iteration_limit_factor = std::numeric_limits<f_t>::max(),
                                    .backtrack              = 1};

    bnb_task_settings[PSEUDOCOST_DIVING] = get_default_diving_settings<i_t, f_t>(PSEUDOCOST_DIVING);

    bnb_task_settings[LINE_SEARCH_DIVING] =
      get_default_diving_settings<i_t, f_t>(LINE_SEARCH_DIVING);

    bnb_task_settings[GUIDED_DIVING] = get_default_diving_settings<i_t, f_t>(GUIDED_DIVING);

    bnb_task_settings[COEFFICIENT_DIVING] =
      get_default_diving_settings<i_t, f_t>(COEFFICIENT_DIVING);

    set_bnb_tasks(omp_get_max_threads() - 1);
  }

  void set_bnb_tasks(i_t num_threads)
  {
    this->num_threads                        = num_threads;
    bnb_task_settings[EXPLORATION].num_tasks = std::max(1, num_threads / 4);

    i_t diving_tasks = num_threads - bnb_task_settings[EXPLORATION].num_tasks;
    i_t num_enabled  = 0;

    for (size_t i = 1; i < bnb_task_settings.size(); ++i) {
      num_enabled += static_cast<i_t>(bnb_task_settings[i].is_enabled);
    }

    for (size_t i = 1, k = 0; i < bnb_task_settings.size(); ++i) {
      i_t start = (double)k * diving_tasks / num_enabled;
      i_t end   = (double)(k + 1) * diving_tasks / num_enabled;

      if (bnb_task_settings[i].is_enabled) {
        bnb_task_settings[i].num_tasks = end - start;
        ++k;

      } else {
        bnb_task_settings[i].num_tasks = 0;
      }
    }
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

  // Indicate the settings used by each task
  // The position in the array is indicated by the `bnb_task_type_t`.
  std::array<bnb_task_settings_t<i_t, f_t>, 5> bnb_task_settings;

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
