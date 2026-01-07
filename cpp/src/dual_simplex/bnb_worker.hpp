/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/phase2.hpp>

#include <deque>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

constexpr int bnb_num_worker_types = 5;

// Indicate the search and variable selection algorithms used by each thread
// in B&B (See [1]).
//
// [1] T. Achterberg, “Constraint Integer Programming,” PhD, Technischen Universität Berlin,
// Berlin, 2007. doi: 10.14279/depositonce-1634.
enum bnb_worker_type_t : int {
  EXPLORATION        = 0,  // Best-First + Plunging.
  PSEUDOCOST_DIVING  = 1,  // Pseudocost diving (9.2.5)
  LINE_SEARCH_DIVING = 2,  // Line search diving (9.2.4)
  GUIDED_DIVING      = 3,  // Guided diving (9.2.3).
  COEFFICIENT_DIVING = 4   // Coefficient diving (9.2.1)
};

template <typename i_t, typename f_t>
struct bnb_stats_t {
  f_t start_time                        = 0.0;
  omp_atomic_t<f_t> total_lp_solve_time = 0.0;
  omp_atomic_t<i_t> nodes_explored      = 0;
  omp_atomic_t<i_t> nodes_unexplored    = 0;
  omp_atomic_t<f_t> total_lp_iters      = 0;
};

template <typename i_t, typename f_t>
class bnb_worker_t {
 public:
  const i_t worker_id;
  omp_atomic_t<bnb_worker_type_t> worker_type;
  omp_atomic_t<bool> is_active;
  omp_atomic_t<f_t> lower_bound;

  lp_problem_t<i_t, f_t> leaf_problem;

  basis_update_mpf_t<i_t, f_t> basis_factors;
  std::vector<i_t> basic_list;
  std::vector<i_t> nonbasic_list;

  bounds_strengthening_t<i_t, f_t> node_presolver;
  std::vector<bool> bounds_changed;

  std::vector<f_t> start_lower;
  std::vector<f_t> start_upper;
  mip_node_t<i_t, f_t>* start_node;

  bool recompute_basis  = true;
  bool recompute_bounds = true;

  bnb_worker_t(i_t worker_id,
               const lp_problem_t<i_t, f_t>& original_lp,
               const csr_matrix_t<i_t, f_t>& Arow,
               const std::vector<variable_type_t>& var_type,
               const simplex_solver_settings_t<i_t, f_t>& settings);

  // Set the `start_node` for best-first search.
  void init_best_first(mip_node_t<i_t, f_t>* node, const lp_problem_t<i_t, f_t>& original_lp)
  {
    start_node  = node;
    start_lower = original_lp.lower;
    start_upper = original_lp.upper;
    worker_type = EXPLORATION;
    lower_bound = node->lower_bound;
    is_active   = true;
  }

  // Initialize the worker for diving, setting the `start_node`, `start_lower` and
  // `start_upper`. Returns `true` if the starting node is feasible via
  // bounds propagation.
  bool init_diving(mip_node_t<i_t, f_t>* node,
                   bnb_worker_type_t type,
                   const lp_problem_t<i_t, f_t>& original_lp,
                   const simplex_solver_settings_t<i_t, f_t>& settings);

  // Set the variables bounds for the LP relaxation of the current node.
  bool set_lp_variable_bounds_for(mip_node_t<i_t, f_t>* node_ptr,
                                  const simplex_solver_settings_t<i_t, f_t>& settings);

 private:
  // For diving, we need to store the full node instead of
  // of just a pointer, since it is not store in the tree anymore.
  // To keep the same interface across all worker types,
  // this will be used as a temporary storage and
  // will be pointed by `start_node`.
  // For exploration, this will not be used.
  mip_node_t<i_t, f_t> internal_node;
};

template <typename i_t, typename f_t>
class bnb_worker_pool_t {
 public:
  void init(i_t num_workers,
            const lp_problem_t<i_t, f_t>& original_lp,
            const csr_matrix_t<i_t, f_t>& Arow,
            const std::vector<variable_type_t>& var_type,
            const simplex_solver_settings_t<i_t, f_t>& settings)
  {
    workers_.resize(num_workers);
    for (i_t i = 0; i < num_workers; ++i) {
      workers_[i] =
        std::make_unique<bnb_worker_t<i_t, f_t>>(i, original_lp, Arow, var_type, settings);
      available_workers_.push_front(i);
    }
  }

  bnb_worker_t<i_t, f_t>* get_worker()
  {
    std::lock_guard<omp_mutex_t> lock(mutex_);

    if (available_workers_.empty()) {
      return nullptr;
    } else {
      i_t idx = available_workers_.front();
      available_workers_.pop_front();
      return workers_[idx].get();
    }
  }

  void return_worker_to_pool(bnb_worker_t<i_t, f_t>* worker)
  {
    worker->is_active = false;
    std::lock_guard<omp_mutex_t> lock(mutex_);
    available_workers_.push_back(worker->worker_id);
  }

  f_t get_lower_bounds()
  {
    f_t lower_bound = std::numeric_limits<f_t>::infinity();

    for (i_t i = 0; i < workers_.size(); ++i) {
      if (workers_[i]->worker_type == EXPLORATION && workers_[i]->is_active) {
        lower_bound = std::min(workers_[i]->lower_bound.load(), lower_bound);
      }
    }

    return lower_bound;
  }

 private:
  // Worker pool
  std::vector<std::unique_ptr<bnb_worker_t<i_t, f_t>>> workers_;

  // FIXME: Implement a lock-free queue
  omp_mutex_t mutex_;
  std::deque<i_t> available_workers_;
};

template <typename f_t, typename i_t>
std::vector<bnb_worker_type_t> bnb_get_worker_types(diving_heuristics_settings_t<i_t, f_t> settings)
{
  std::vector<bnb_worker_type_t> types;
  types.reserve(bnb_num_worker_types);
  types.push_back(EXPLORATION);
  if (!settings.disable_pseudocost_diving) { types.push_back(PSEUDOCOST_DIVING); }
  if (!settings.disable_line_search_diving) { types.push_back(LINE_SEARCH_DIVING); }
  if (!settings.disable_guided_diving) { types.push_back(GUIDED_DIVING); }
  if (!settings.disable_coefficient_diving) { types.push_back(COEFFICIENT_DIVING); }
  return types;
}

template <typename f_t, typename i_t>
std::array<i_t, bnb_num_worker_types> bnb_get_num_workers_round_robin(
  i_t num_threads, diving_heuristics_settings_t<i_t, f_t> settings)
{
  std::array<i_t, bnb_num_worker_types> max_num_workers;
  auto worker_types = bnb_get_worker_types(settings);

  max_num_workers.fill(0);
  max_num_workers[EXPLORATION] = std::max(1, num_threads / 2);

  i_t diving_workers = 2 * settings.num_diving_workers;
  i_t m              = worker_types.size() - 1;
  for (size_t i = 1, k = 0; i < bnb_num_worker_types; ++i) {
    i_t start          = (double)k * diving_workers / m;
    i_t end            = (double)(k + 1) * diving_workers / m;
    max_num_workers[i] = end - start;
    ++k;
  }

  return max_num_workers;
}

}  // namespace cuopt::linear_programming::dual_simplex
