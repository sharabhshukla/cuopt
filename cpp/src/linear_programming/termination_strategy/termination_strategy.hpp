/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <linear_programming/pdhg.hpp>
#include <linear_programming/termination_strategy/convergence_information.hpp>
#include <linear_programming/termination_strategy/infeasibility_information.hpp>
#include <linear_programming/pdlp_climber_strategy.hpp>

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <mip/problem/problem.cuh>

#include <utilities/unique_pinned_ptr.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/universal_vector.h>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
class pdlp_termination_strategy_t {
 public:
  pdlp_termination_strategy_t(raft::handle_t const* handle_ptr,
                              problem_t<i_t, f_t>& op_problem,
                              cusparse_view_t<i_t, f_t>& cusparse_view,
                              const i_t primal_size,
                              const i_t dual_size,
                              const pdlp_solver_settings_t<i_t, f_t>& settings,
                              const std::vector<pdlp_climber_strategy_t>& climber_strategies);

  void evaluate_termination_criteria(
    pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
    rmm::device_uvector<f_t>& primal_iterate,
    rmm::device_uvector<f_t>& dual_iterate,
    [[maybe_unused]] const rmm::device_uvector<f_t>& dual_slack,
    const rmm::device_uvector<f_t>& combined_bounds,  // Only useful if per_constraint_residual
    const rmm::device_uvector<f_t>&
      objective_coefficients  // Only useful if per_constraint_residual
  );

  void print_termination_criteria(i_t iteration, f_t elapsed, i_t best_id = 0) const;

  void set_relative_dual_tolerance_factor(f_t dual_tolerance_factor);
  void set_relative_primal_tolerance_factor(f_t primal_tolerance_factor);
  f_t get_relative_dual_tolerance_factor() const;
  f_t get_relative_primal_tolerance_factor() const;

  pdlp_termination_status_t get_termination_status(i_t id) const;
  std::vector<pdlp_termination_status_t> get_terminations_status();
  bool all_optimal_status();
  bool has_optimal_status(int custom_climber_log = -1) const;
  i_t nb_optimal_solutions() const;
  i_t get_optimal_solution_id() const;

  const convergence_information_t<i_t, f_t>& get_convergence_information() const;

  // Deep copy is used when save best primal so far is toggled
  optimization_problem_solution_t<i_t, f_t> fill_return_problem_solution(
    i_t number_of_iterations,
    pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
    rmm::device_uvector<f_t>& primal_iterate,
    rmm::device_uvector<f_t>& dual_iterate,
    pdlp_warm_start_data_t<i_t, f_t> warm_start_data,
    std::vector<pdlp_termination_status_t>&& termination_status,
    bool deep_copy = false);

  // This verions simply calls the above with an empty pdlp_warm_start_data
  // It is used when we return without an optimal solution (infeasible, time limit...)
  optimization_problem_solution_t<i_t, f_t> fill_return_problem_solution(
    i_t number_of_iterations,
    pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
    rmm::device_uvector<f_t>& primal_iterate,
    rmm::device_uvector<f_t>& dual_iterate,
    std::vector<pdlp_termination_status_t>&& termination_status,
    bool deep_copy = false);

 private:
  void check_termination_criteria();

  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  problem_t<i_t, f_t>* problem_ptr;

  convergence_information_t<i_t, f_t> convergence_information_;
  infeasibility_information_t<i_t, f_t> infeasibility_information_;

  thrust::universal_host_pinned_vector<i_t> termination_status_;
  const pdlp_solver_settings_t<i_t, f_t>& settings_;

  const std::vector<pdlp_climber_strategy_t>& climber_strategies_;

};
}  // namespace cuopt::linear_programming::detail
