/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/bnb_worker.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
bnb_worker_t<i_t, f_t>::bnb_worker_t(i_t worker_id,
                                     const lp_problem_t<i_t, f_t>& original_lp,
                                     const csr_matrix_t<i_t, f_t>& Arow,
                                     const std::vector<variable_type_t>& var_type,
                                     const simplex_solver_settings_t<i_t, f_t>& settings)
  : worker_id(worker_id),
    worker_type(EXPLORATION),
    is_active(false),
    lower_bound(-std::numeric_limits<f_t>::infinity()),
    leaf_problem(original_lp),
    basis_factors(original_lp.num_rows, settings.refactor_frequency),
    basic_list(original_lp.num_rows),
    nonbasic_list(),
    node_presolver(leaf_problem, Arow, {}, var_type),
    bounds_changed(original_lp.num_cols, false)
{
}

template <typename i_t, typename f_t>
bool bnb_worker_t<i_t, f_t>::init_diving(mip_node_t<i_t, f_t>* node,
                                         bnb_worker_type_t type,
                                         const lp_problem_t<i_t, f_t>& original_lp,
                                         const simplex_solver_settings_t<i_t, f_t>& settings)
{
  internal_node = node->detach_copy();
  start_node    = &internal_node;

  start_lower = original_lp.lower;
  start_upper = original_lp.upper;
  worker_type = type;
  lower_bound = node->lower_bound;
  is_active   = true;

  std::fill(bounds_changed.begin(), bounds_changed.end(), false);
  node->get_variable_bounds(start_lower, start_upper, bounds_changed);

  return node_presolver.bounds_strengthening(start_lower, start_upper, bounds_changed, settings);
}

template <typename i_t, typename f_t>
bool bnb_worker_t<i_t, f_t>::set_lp_variable_bounds_for(
  mip_node_t<i_t, f_t>* node_ptr, const simplex_solver_settings_t<i_t, f_t>& settings)
{
  // Reset the bound_changed markers
  std::fill(bounds_changed.begin(), bounds_changed.end(), false);

  // Set the correct bounds for the leaf problem
  if (recompute_bounds) {
    leaf_problem.lower = start_lower;
    leaf_problem.upper = start_upper;
    node_ptr->get_variable_bounds(leaf_problem.lower, leaf_problem.upper, bounds_changed);

  } else {
    node_ptr->update_branched_variable_bounds(
      leaf_problem.lower, leaf_problem.upper, bounds_changed);
  }

  return node_presolver.bounds_strengthening(
    leaf_problem.lower, leaf_problem.upper, bounds_changed, settings);
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class bnb_worker_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::dual_simplex
