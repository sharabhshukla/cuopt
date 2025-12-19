/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#include <linear_programming/pdhg.hpp>
#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/utilities/ping_pong_graph.cuh>
#include <linear_programming/pdlp_climber_strategy.hpp>
#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>

#include <cuopt/error.hpp>

#ifdef CUPDLP_DEBUG_MODE
#include <utilities/copy_helpers.hpp>
#endif

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/ternary_op.cuh>

#include <cub/cub.cuh>

#include <cusparse_v2.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
pdhg_solver_t<i_t, f_t>::pdhg_solver_t(raft::handle_t const* handle_ptr,
                                       problem_t<i_t, f_t>& op_problem_scaled,
                                       bool is_legacy_batch_mode, // Batch mode with streams
                                       const std::vector<pdlp_climber_strategy_t>& climber_strategies,
                                       const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params,
                                       const std::vector<std::tuple<i_t, f_t, f_t>>& new_bounds) 
  : batch_mode_(climber_strategies.size() > 1),
    handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    problem_ptr(&op_problem_scaled),
    primal_size_h_(problem_ptr->n_variables),
    dual_size_h_(problem_ptr->n_constraints),
    current_saddle_point_state_{handle_ptr_, problem_ptr->n_variables, problem_ptr->n_constraints, climber_strategies.size()},
    tmp_primal_{(climber_strategies.size() * problem_ptr->n_variables), stream_view_},
    tmp_dual_{(climber_strategies.size() * problem_ptr->n_constraints), stream_view_},
    potential_next_primal_solution_{(climber_strategies.size() * problem_ptr->n_variables), stream_view_},
    potential_next_dual_solution_{(climber_strategies.size() * problem_ptr->n_constraints), stream_view_},
    total_pdhg_iterations_{0},
    dual_slack_{static_cast<size_t>(
                  (hyper_params.use_reflected_primal_dual) ? problem_ptr->n_variables * climber_strategies.size() : 0),
                stream_view_},
    reflected_primal_{
      static_cast<size_t>((hyper_params.use_reflected_primal_dual) ? problem_ptr->n_variables * climber_strategies.size()
                                                                         : 0),
      stream_view_},
    reflected_dual_{static_cast<size_t>((hyper_params.use_reflected_primal_dual)
                                          ? problem_ptr->n_constraints * climber_strategies.size()
                                          : 0),
                    stream_view_},
    cusparse_view_{handle_ptr_,
                   op_problem_scaled,
                   current_saddle_point_state_,
                   tmp_primal_,
                   tmp_dual_,
                   potential_next_dual_solution_,
                   reflected_primal_,
                   climber_strategies,
                   hyper_params},
    reusable_device_scalar_value_1_{1.0, stream_view_},
    reusable_device_scalar_value_0_{0.0, stream_view_},
    reusable_device_scalar_value_neg_1_{f_t(-1.0), stream_view_},
    reusable_device_scalar_1_{stream_view_},
    graph_all{stream_view_, is_legacy_batch_mode},
    graph_prim_proj_gradient_dual{stream_view_, is_legacy_batch_mode},
    d_total_pdhg_iterations_{0, stream_view_},
    climber_strategies_(climber_strategies),
    hyper_params_(hyper_params),
    new_bounds_idx_{new_bounds.size(), stream_view_},
    new_bounds_lower_{new_bounds.size(), stream_view_},
    new_bounds_upper_{new_bounds.size(), stream_view_}
{
  if (!new_bounds.empty()) {
    std::vector<i_t> idx(new_bounds.size());
    std::vector<f_t> lower(new_bounds.size());
    std::vector<f_t> upper(new_bounds.size());
    for (size_t i = 0; i < new_bounds.size(); ++i) {
      idx[i]   = std::get<0>(new_bounds[i]);
      lower[i] = std::get<1>(new_bounds[i]);
      upper[i] = std::get<2>(new_bounds[i]);
    }
    raft::copy(new_bounds_idx_.data(), idx.data(), idx.size(), stream_view_);
    raft::copy(new_bounds_lower_.data(), lower.data(), lower.size(), stream_view_);
    raft::copy(new_bounds_upper_.data(), upper.data(), upper.size(), stream_view_);
  }
  thrust::fill(handle_ptr->get_thrust_policy(), tmp_primal_.data(), tmp_primal_.end(), f_t(0));
  thrust::fill(handle_ptr->get_thrust_policy(), tmp_dual_.data(), tmp_dual_.end(), f_t(0));
  thrust::fill(handle_ptr->get_thrust_policy(),
               potential_next_primal_solution_.data(),
               potential_next_primal_solution_.end(),
               f_t(0));
  thrust::fill(handle_ptr->get_thrust_policy(),
               potential_next_dual_solution_.data(),
               potential_next_dual_solution_.end(),
               f_t(0));
  thrust::fill(
    handle_ptr->get_thrust_policy(), reflected_primal_.data(), reflected_primal_.end(), f_t(0));
  thrust::fill(
    handle_ptr->get_thrust_policy(), reflected_dual_.data(), reflected_dual_.end(), f_t(0));
  thrust::fill(handle_ptr->get_thrust_policy(), dual_slack_.data(), dual_slack_.end(), f_t(0));
}

template <typename i_t, typename f_t>
rmm::device_scalar<i_t>& pdhg_solver_t<i_t, f_t>::get_d_total_pdhg_iterations()
{
  return d_total_pdhg_iterations_;
}

template <typename i_t, typename f_t>
i_t pdhg_solver_t<i_t, f_t>::get_primal_size() const
{
  return primal_size_h_;
}

template <typename i_t, typename f_t>
i_t pdhg_solver_t<i_t, f_t>::get_dual_size() const
{
  return dual_size_h_;
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::compute_next_dual_solution(rmm::device_uvector<f_t>& dual_step_size)
{
  raft::common::nvtx::range fun_scope("compute_next_dual_solution");
  // proj(y+sigma(b-K(2x'-x)))
  // rewritten as proj(y+sigma(b-K(x'+delta_x)))
  // with the introduction of constraint lower and upper bounds the b
  // term no longer exists, but instead becomes
  // max(min(0, sigma*constraint_upper+primal_product),sigma*constraint_lower+primal_product)
  // where primal_product = y-sigma(K(x'+delta_x))

  // x+delta_x
  // Done in previous function

  // K(x'+delta_x)
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       reusable_device_scalar_value_1_.data(),  // 1
                                       cusparse_view_.A,
                                       cusparse_view_.tmp_primal,
                                       reusable_device_scalar_value_0_.data(),  // 1
                                       cusparse_view_.dual_gradient,
                                       CUSPARSE_SPMV_CSR_ALG2,
                                       (f_t*)cusparse_view_.buffer_non_transpose.data(),
                                       stream_view_));

  // y - (sigma*dual_gradient)
  // max(min(0, sigma*constraint_upper+primal_product), sigma*constraint_lower+primal_product)
  // Each element of y - (sigma*dual_gradient) of the min is the critical point
  // of the respective 1D minimization problem if it's negative.
  // Likewise the argument to the max is the critical point if
  // positive.

  // All is fused in a single call to limit number of read / write in memory
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(current_saddle_point_state_.get_dual_solution().data(),
                          current_saddle_point_state_.get_dual_gradient().data(),
                          problem_ptr->constraint_lower_bounds.data(),
                          problem_ptr->constraint_upper_bounds.data()),
    thrust::make_zip_iterator(potential_next_dual_solution_.data(),
                              current_saddle_point_state_.get_delta_dual().data()),
    dual_size_h_,
    dual_projection<f_t>(dual_step_size.data()),
    stream_view_);
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::compute_At_y()
{
  // A_t @ y

  if (!batch_mode_)
  {
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       reusable_device_scalar_value_1_.data(),
                                                       cusparse_view_.A_T,
                                                       cusparse_view_.dual_solution,
                                                       reusable_device_scalar_value_0_.data(),
                                                       cusparse_view_.current_AtY,
                                                       CUSPARSE_SPMV_CSR_ALG2,
                                                       (f_t*)cusparse_view_.buffer_transpose.data(),
                                                       stream_view_));
  }
  else
  {
    if (!deterministic_batch_pdlp)
    {
      // When using row row we need to transpose the input / output before / after the SpMM
      if (use_row_row)
      {
        RAFT_CUBLAS_TRY( cublasDgeam(handle_ptr_->get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                  climber_strategies_.size(), // B_NUM_ROWS <=> climber_strategies_.size()
                                  problem_ptr->n_constraints, // B_ROWS <=> A_NUM_COLS (A_NUM_ROW if A transposed, here yes) <=> n_constraints
                                  reusable_device_scalar_value_1_.data(), // alpha <=> 1.0
                                  current_saddle_point_state_.get_dual_solution().data(), // Input dual sol we want to transpose
                                  problem_ptr->n_constraints, // B_ROWS <=> A_NUM_COLS (A_NUM_ROW if A transposed, here yes) <=> n_constraints
                                  reusable_device_scalar_value_0_.data(), // beta <=> 0.0
                                  cusparse_view_.batch_dual_solutions_data_transposed_.data(), // Output transposed dual solutions we want to use in SpMM
                                  climber_strategies_.size(), // B_NUM_ROWS <=> climber_strategies_.size()
                                  cusparse_view_.batch_dual_solutions_data_transposed_.data(), // Output transposed current AtYs we want to use in SpMM
                                  climber_strategies_.size()) ); // B_NUM_ROWS <=> climber_strategies_.size()
      }
      RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(handle_ptr_->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       reusable_device_scalar_value_1_.data(),
                                                       cusparse_view_.A_T,
                                                       cusparse_view_.batch_dual_solutions,
                                                       reusable_device_scalar_value_0_.data(),
                                                       cusparse_view_.batch_current_AtYs,
                                                       CUSPARSE_SPMM_CSR_ALG3,
                                                       (f_t*)cusparse_view_.buffer_transpose_batch.data(),
                                                       stream_view_));
      if (use_row_row)
      {
        RAFT_CUBLAS_TRY( cublasDgeam(handle_ptr_->get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                  problem_ptr->n_variables, // mC <=> A final rows <=> A NUM_ROW (A_NUM_COL if A transposed, here yes) <=> n_variables
                                  climber_strategies_.size(), // nC <=> B NUM_COL <=> climber_strategies_.size()
                                  reusable_device_scalar_value_1_.data(), // alpha <=> 1.0
                                  cusparse_view_.batch_current_AtYs_data_transposed_.data(), // Transposed output of SpMM 
                                  climber_strategies_.size(), // nC <=> B NUM_COL <=> climber_strategies_.size()
                                  reusable_device_scalar_value_0_.data(), // beta <=> 0.0
                                   nullptr, 
                                   problem_ptr->n_variables, // mC <=> A final rows <=> A NUM_ROW (A_NUM_COL if A transposed, here yes) <=> n_variables
                                  current_saddle_point_state_.get_current_AtY().data(), // ReTransposed output to be used COL wise 
                                  problem_ptr->n_variables) ); // mC <=> A final rows <=> A NUM_ROW (A_NUM_COL if A transposed, here yes) <=> n_variables
      }
    }
    else
    {
    for (size_t i = 0; i < climber_strategies_.size(); i++) {
        RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       reusable_device_scalar_value_1_.data(),
                                                       cusparse_view_.A_T,
                                                       cusparse_view_.dual_solution_vector[i],
                                                       reusable_device_scalar_value_0_.data(),
                                                       cusparse_view_.current_AtYs_vector[i],
                                                       CUSPARSE_SPMV_CSR_ALG2,
                                                       (f_t*)cusparse_view_.buffer_transpose.data(),
                                                       stream_view_));
      }
    }
  }
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::compute_A_x()
{
  // A @ x
  if (!batch_mode_)
  {
    RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       reusable_device_scalar_value_1_.data(),
                                       cusparse_view_.A,
                                       cusparse_view_.reflected_primal_solution,
                                       reusable_device_scalar_value_0_.data(),
                                       cusparse_view_.dual_gradient,
                                       CUSPARSE_SPMV_CSR_ALG2,
                                       (f_t*)cusparse_view_.buffer_non_transpose.data(),
                                       stream_view_));
  }
  else
  {
    if (!deterministic_batch_pdlp)
    {
      // When using row row we need to transpose the input / output before / after the SpMM
      if (use_row_row)
      {
        RAFT_CUBLAS_TRY( cublasDgeam(handle_ptr_->get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                  climber_strategies_.size(), 
                                  problem_ptr->n_variables,
                                  reusable_device_scalar_value_1_.data(), 
                                  reflected_primal_.data(), 
                                  problem_ptr->n_variables,
                                  reusable_device_scalar_value_0_.data(), 
                                  cusparse_view_.batch_reflected_primal_solutions_data_transposed_.data(), 
                                  climber_strategies_.size(),
                                  cusparse_view_.batch_reflected_primal_solutions_data_transposed_.data(), 
                                  climber_strategies_.size()) );
      }
      RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(handle_ptr_->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       reusable_device_scalar_value_1_.data(),
                                                       cusparse_view_.A,
                                                       cusparse_view_.batch_reflected_primal_solutions,
                                                       reusable_device_scalar_value_0_.data(),
                                                       cusparse_view_.batch_dual_gradients,
                                                       CUSPARSE_SPMM_CSR_ALG3,
                                                       (f_t*)cusparse_view_.buffer_non_transpose_batch.data(),
                                                       stream_view_));
      if (use_row_row)
      {
        RAFT_CUBLAS_TRY( cublasDgeam(handle_ptr_->get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                  problem_ptr->n_constraints, // mC <=> A final rows <=> A NUM_ROW (A NUM_COL if A transposed) <=> n_constraints
                                  climber_strategies_.size(), // nC <=> B NUM_COL <=> climber_strategies_.size()
                                  reusable_device_scalar_value_1_.data(), // alpha <=> 1.0
                                  cusparse_view_.batch_dual_gradients_data_transposed_.data(), // Transposed SpMM output 
                                  climber_strategies_.size(), // nC <=> B NUM_COL <=> climber_strategies_.size()
                                  reusable_device_scalar_value_0_.data(), // beta <=> 0.0
                                  nullptr, // B not used 
                                  problem_ptr->n_constraints, // mC <=> A final rows <=> A NUM_ROW (A NUM_COL if A transposed) <=> n_constraints
                                  current_saddle_point_state_.get_dual_gradient().data(), // ReTransposed output to be use COL wise 
                                  problem_ptr->n_constraints) // mC <=> A final rows <=> A NUM_ROW (A NUM_COL if A transposed) <=> n_constraints
                                );
      }
    }
    else
    {
      for (size_t i = 0; i < climber_strategies_.size(); ++i)
      {
        RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        reusable_device_scalar_value_1_.data(),
                                        cusparse_view_.A,
                                        cusparse_view_.reflected_primal_solution_vector[i],
                                        reusable_device_scalar_value_0_.data(),
                                        cusparse_view_.dual_gradients_vector[i],
                                        CUSPARSE_SPMV_CSR_ALG2,
                                        (f_t*)cusparse_view_.buffer_non_transpose.data(),
                                        stream_view_));
      }
    }
  }
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::compute_primal_projection_with_gradient(
  rmm::device_uvector<f_t>& primal_step_size)
{
  // Applying *c -* A_t @ y
  // x-(tau*primal_gradient)
  // project by max(min(x[i], upperbound[i]),lowerbound[i])
  // compute delta_primal x'-x

  using f_t2 = typename type_2<f_t>::type;
  // All is fused in a single call to limit number of read / write in memory
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(current_saddle_point_state_.get_primal_solution().data(),
                          problem_ptr->objective_coefficients.data(),
                          current_saddle_point_state_.get_current_AtY().data(),
                          problem_ptr->variable_bounds.data()),
    thrust::make_zip_iterator(potential_next_primal_solution_.data(),
                              current_saddle_point_state_.get_delta_primal().data(),
                              tmp_primal_.data()),
    primal_size_h_,
    primal_projection<f_t, f_t2>(primal_step_size.data()),
    stream_view_);
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::compute_next_primal_dual_solution(
  rmm::device_uvector<f_t>& primal_step_size,
  i_t iterations_since_last_restart,
  bool last_restart_was_average,
  rmm::device_uvector<f_t>& dual_step_size,
  i_t total_pdlp_iterations)
{
  raft::common::nvtx::range fun_scope("compute_next_primal_solution");
#ifdef PDLP_DEBUG_MODE
  std::cout << "  compute_next_primal_solution:" << std::endl;
#endif

  // proj(x-(tau(c-K^Ty)))
  // K = A, tau = primal_step_size, x = primal_solution, y = dual_solution

  // QP if quadratic program: proj(x-(tau(Q*x + c-K^Ty)))

  // Computation should only take place during very first iteration or after each resart to
  // average (after a restart to average, previous A_t @ y is not valid anymore since it was on
  // current)
  // Indeed, adaptative_step_size has already computed what was next (now current) A_t @ y,
  // so we don't need to recompute it here
  if (total_pdhg_iterations_ == 0 ||
      (iterations_since_last_restart == 0 && last_restart_was_average)) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "    Very first or first iteration since last restart and was average, "
                 "recomputing A_t * Y"
              << std::endl;
#endif

    // Primal and dual steps are captured in a cuda graph since called very often
    if (!graph_all.is_initialized(total_pdlp_iterations)) {
      graph_all.start_capture(total_pdlp_iterations);
      // First compute only A_t @ y, needed later in adaptative step size
      compute_At_y();
      // Compute fused primal gradient with projection
      compute_primal_projection_with_gradient(primal_step_size);
      // Compute next dual solution
      compute_next_dual_solution(dual_step_size);
      graph_all.end_capture(total_pdlp_iterations);
    }
    graph_all.launch(total_pdlp_iterations);
  } else {
#ifdef PDLP_DEBUG_MODE
    std::cout << "    Not computing A_t * Y" << std::endl;
#endif
    // A_t * y was already computed in previous iteration
    if (!graph_prim_proj_gradient_dual.is_initialized(total_pdlp_iterations)) {
      graph_prim_proj_gradient_dual.start_capture(total_pdlp_iterations);
      compute_primal_projection_with_gradient(primal_step_size);
      compute_next_dual_solution(dual_step_size);
      graph_prim_proj_gradient_dual.end_capture(total_pdlp_iterations);
    }
    graph_prim_proj_gradient_dual.launch(total_pdlp_iterations);
  }
}

template <typename f_t>
struct primal_reflected_major_projection {
  using f_t2 = typename type_2<f_t>::type;
  primal_reflected_major_projection(const f_t* scalar) : scalar_{scalar} {}
  HDI thrust::tuple<f_t, f_t, f_t> operator()(f_t current_primal,
                                              f_t objective,
                                              f_t Aty,
                                              f_t2 bounds)
  {
    cuopt_assert(*scalar_ != f_t(0.0), "Scalar can't be 0");
    const f_t next         = current_primal - *scalar_ * (objective - Aty);
    const f_t next_clamped = raft::max<f_t>(raft::min<f_t>(next, bounds.y), bounds.x);
    return {
      next_clamped, (next_clamped - next) / *scalar_, f_t(2.0) * next_clamped - current_primal};
  }
  const f_t* scalar_;
};

template <typename f_t>
struct primal_reflected_major_projection_batch {
  using f_t2 = typename type_2<f_t>::type;
  HDI thrust::tuple<f_t, f_t, f_t> operator()(f_t current_primal,
                                              f_t objective,
                                              f_t Aty,
                                              f_t2 bounds,
                                              f_t primal_step_size)
  {
    cuopt_assert(primal_step_size != f_t(0.0), "Scalar can't be 0");
    const f_t next         = current_primal - primal_step_size * (objective - Aty);
    const f_t next_clamped = raft::max<f_t>(raft::min<f_t>(next, bounds.y), bounds.x);
    return {
      next_clamped, (next_clamped - next) / primal_step_size, f_t(2.0) * next_clamped - current_primal};
  }
};


template <typename f_t>
struct primal_reflected_projection {
  using f_t2 = typename type_2<f_t>::type;
  primal_reflected_projection(const f_t* scalar) : scalar_{scalar} {}
  HDI f_t operator()(f_t current_primal, f_t objective, f_t Aty, f_t2 bounds)
  {
    const f_t next         = current_primal - *scalar_ * (objective - Aty);
    const f_t next_clamped = raft::max<f_t>(raft::min<f_t>(next, bounds.y), bounds.x);
    return f_t(2.0) * next_clamped - current_primal;
  }
  const f_t* scalar_;
};

template <typename f_t>
struct primal_reflected_projection_batch {
  using f_t2 = typename type_2<f_t>::type;
  HDI f_t operator()(f_t current_primal, f_t objective, f_t Aty, f_t2 bounds, f_t primal_step_size)
  {
    const f_t next         = current_primal - primal_step_size * (objective - Aty);
    const f_t next_clamped = raft::max<f_t>(raft::min<f_t>(next, bounds.y), bounds.x);
    return f_t(2.0) * next_clamped - current_primal;
  }
};

template <typename f_t>
struct dual_reflected_major_projection {
  dual_reflected_major_projection(const f_t* scalar) : scalar_{scalar} {}
  HDI thrust::tuple<f_t, f_t> operator()(f_t current_dual,
                                         f_t Ax,
                                         f_t lower_bound,
                                         f_t upper_bounds)
  {
    cuopt_assert(*scalar_ != f_t(0.0), "Scalar can't be 0");
    const f_t tmp       = current_dual / *scalar_ - Ax;
    const f_t tmp_proj  = raft::max<f_t>(-upper_bounds, raft::min<f_t>(tmp, -lower_bound));
    const f_t next_dual = (tmp - tmp_proj) * *scalar_;
    return {next_dual, f_t(2.0) * next_dual - current_dual};
  }

  const f_t* scalar_;
};

template <typename f_t>
struct dual_reflected_major_projection_batch {
  HDI thrust::tuple<f_t, f_t> operator()(f_t current_dual,
                                         f_t Ax,
                                         f_t lower_bound,
                                         f_t upper_bounds,
                                         f_t dual_step_size)
  {
    cuopt_assert(dual_step_size != f_t(0.0), "Scalar can't be 0");
    const f_t tmp       = current_dual / dual_step_size - Ax;
    const f_t tmp_proj  = raft::max<f_t>(-upper_bounds, raft::min<f_t>(tmp, -lower_bound));
    const f_t next_dual = (tmp - tmp_proj) * dual_step_size;
    return {next_dual, f_t(2.0) * next_dual - current_dual};
  }
};

template <typename f_t>
struct dual_reflected_projection {
  dual_reflected_projection(const f_t* scalar) : scalar_{scalar} {}
  HDI f_t operator()(f_t current_dual, f_t Ax, f_t lower_bound, f_t upper_bounds)
  {
    cuopt_assert(*scalar_ != f_t(0.0), "Scalar can't be 0");
    const f_t tmp       = current_dual / *scalar_ - Ax;
    const f_t tmp_proj  = raft::max<f_t>(-upper_bounds, raft::min<f_t>(tmp, -lower_bound));
    const f_t next_dual = (tmp - tmp_proj) * *scalar_;
    return f_t(2.0) * next_dual - current_dual;
  }

  const f_t* scalar_;
};

template <typename f_t>
struct dual_reflected_projection_batch {
  HDI f_t operator()(f_t current_dual, f_t Ax, f_t lower_bound, f_t upper_bounds, f_t dual_step_size)
  {
    cuopt_assert(dual_step_size != f_t(0.0), "Scalar can't be 0");
    const f_t tmp       = current_dual / dual_step_size - Ax;
    const f_t tmp_proj  = raft::max<f_t>(-upper_bounds, raft::min<f_t>(tmp, -lower_bound));
    const f_t next_dual = (tmp - tmp_proj) * dual_step_size;
    return f_t(2.0) * next_dual - current_dual;
  }
};

template <typename i_t, typename f_t>
__global__ void refine_primal_projection_major_batch_kernel(i_t batch_size,
                                                            i_t n_variables,
                                                            const i_t* idx,
                                                            const f_t* lower,
                                                            const f_t* upper,
                                                            const f_t* current_primal,
                                                            const f_t* objective,
                                                            const f_t* Aty,
                                                            const f_t* primal_step_size,
                                                            f_t* potential_next,
                                                            f_t* dual_slack,
                                                            f_t* reflected_primal)
{
  int climber_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (climber_id >= batch_size) return;

  i_t var_idx = idx[climber_id];
  f_t l       = lower[climber_id];
  f_t u       = upper[climber_id];

  size_t global_idx = (size_t)climber_id * n_variables + var_idx;

  f_t x     = current_primal[global_idx];
  f_t c     = objective[var_idx];
  f_t y_aty = Aty[global_idx];
  f_t tau   = primal_step_size[climber_id];

  auto [next_clamped, delta_primal, reflected_primal_value] = primal_reflected_major_projection_batch<f_t>{}(x, c, y_aty, {l, u}, tau);

  potential_next[global_idx]   = next_clamped;
  dual_slack[global_idx]       = delta_primal;
  reflected_primal[global_idx] = reflected_primal_value;
}

template <typename i_t, typename f_t>
__global__ void refine_primal_projection_batch_kernel(i_t batch_size,
                                                      i_t n_variables,
                                                      const i_t* idx,
                                                      const f_t* lower,
                                                      const f_t* upper,
                                                      const f_t* current_primal,
                                                      const f_t* objective,
                                                      const f_t* Aty,
                                                      const f_t* primal_step_size,
                                                      f_t* reflected_primal)
{
  int climber_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (climber_id >= batch_size) return;

  i_t var_idx = idx[climber_id];
  f_t l       = lower[climber_id];
  f_t u       = upper[climber_id];

  size_t global_idx = (size_t)climber_id * n_variables + var_idx;

  f_t x     = current_primal[global_idx];
  f_t c     = objective[var_idx];
  f_t y_aty = Aty[global_idx];
  f_t tau   = primal_step_size[climber_id];

  reflected_primal[global_idx] = primal_reflected_projection_batch<f_t>{}(x, c, y_aty, {l, u}, tau);
}

template <typename i_t, typename f_t>
__global__ void refine_initial_primal_projection_kernel(i_t batch_size,
                                                        i_t n_variables,
                                                        const i_t* idx,
                                                        const f_t* lower,
                                                        const f_t* upper,
                                                        f_t* primal_solution)
{
  int climber_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (climber_id >= batch_size) return;

  i_t var_idx = idx[climber_id];
  f_t l       = lower[climber_id];
  f_t u       = upper[climber_id];

  size_t global_idx           = (size_t)climber_id * n_variables + var_idx;
  using f_t2 = typename type_2<f_t>::type;
  primal_solution[global_idx] = clamp<f_t, f_t2>{}(primal_solution[global_idx], {l, u});
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::refine_initial_primal_projection()
{
  if (new_bounds_idx_.size() == 0) return;
  const auto [grid_size, block_size] = kernel_config_from_batch_size(climber_strategies_.size());
  refine_initial_primal_projection_kernel<<<grid_size, block_size, 0, stream_view_>>>(
    (i_t)climber_strategies_.size(),
    problem_ptr->n_variables,
    new_bounds_idx_.data(),
    new_bounds_lower_.data(),
    new_bounds_upper_.data(),
    current_saddle_point_state_.get_primal_solution().data());
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::compute_next_primal_dual_solution_reflected(
  rmm::device_uvector<f_t>& primal_step_size,
  rmm::device_uvector<f_t>& dual_step_size,
  bool should_major)
{
  raft::common::nvtx::range fun_scope("compute_next_primal_dual_solution_reflected");

  using f_t2 = typename type_2<f_t>::type;

  // Compute next primal solution reflected

  if (should_major) {
    if (!graph_all.is_initialized(should_major)) {
      graph_all.start_capture(should_major);

      compute_At_y();
      if (!batch_mode_)
      {
        cub::DeviceTransform::Transform(
        cuda::std::make_tuple(current_saddle_point_state_.get_primal_solution().data(),
        problem_ptr->objective_coefficients.data(),
        current_saddle_point_state_.get_current_AtY().data(),
        problem_ptr->variable_bounds.data()),
        thrust::make_zip_iterator(
          potential_next_primal_solution_.data(), dual_slack_.data(), reflected_primal_.data()),
          primal_size_h_,
          primal_reflected_major_projection<f_t>(primal_step_size.data()),
          stream_view_);
      }
      else
      {
        // TODO batch mode: handle different objective coefficient and variable bounds
          cub::DeviceTransform::Transform(
        cuda::std::make_tuple(current_saddle_point_state_.get_primal_solution().data(),
                              problem_wrap_container(problem_ptr->objective_coefficients),
                              current_saddle_point_state_.get_current_AtY().data(),
                              problem_wrap_container(problem_ptr->variable_bounds),
                            thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            batch_wrapped_iterator<f_t>(primal_step_size.data(), primal_size_h_))
                          ),
          thrust::make_zip_iterator(
            potential_next_primal_solution_.data(), dual_slack_.data(), reflected_primal_.data()),
          potential_next_primal_solution_.size(),
          primal_reflected_major_projection_batch<f_t>(),
          stream_view_);

        if (new_bounds_idx_.size() != 0) {
          const auto [grid_size, block_size] =
            kernel_config_from_batch_size(climber_strategies_.size());
          refine_primal_projection_major_batch_kernel<<<grid_size, block_size, 0, stream_view_>>>(
            (i_t)climber_strategies_.size(),
            problem_ptr->n_variables,
            new_bounds_idx_.data(),
            new_bounds_lower_.data(),
            new_bounds_upper_.data(),
            current_saddle_point_state_.get_primal_solution().data(),
            problem_ptr->objective_coefficients.data(),
            current_saddle_point_state_.get_current_AtY().data(),
            primal_step_size.data(),
            potential_next_primal_solution_.data(),
            dual_slack_.data(),
            reflected_primal_.data());
        }
      }
#ifdef CUPDLP_DEBUG_MODE
      print("potential_next_primal_solution_", potential_next_primal_solution_);
      print("reflected_primal_", reflected_primal_);
      print("dual_slack_", dual_slack_);
#endif

      // Compute next dual
      compute_A_x();

      if (!batch_mode_)
      {
        cub::DeviceTransform::Transform(
        cuda::std::make_tuple(current_saddle_point_state_.get_dual_solution().data(),
                              current_saddle_point_state_.get_dual_gradient().data(),
                              problem_ptr->constraint_lower_bounds.data(),
                              problem_ptr->constraint_upper_bounds.data()),
        thrust::make_zip_iterator(potential_next_dual_solution_.data(), reflected_dual_.data()),
        dual_size_h_,
        dual_reflected_major_projection<f_t>(dual_step_size.data()),
        stream_view_);
      }
      else
      {
        // TODO batch mode: handle different constraint bound
        cub::DeviceTransform::Transform(
        cuda::std::make_tuple(current_saddle_point_state_.get_dual_solution().data(),
                              current_saddle_point_state_.get_dual_gradient().data(),
                              problem_wrap_container(
                            problem_ptr->constraint_lower_bounds),
                              problem_wrap_container(
                            problem_ptr->constraint_upper_bounds),
                            thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            batch_wrapped_iterator<f_t>(dual_step_size.data(), dual_size_h_))),
        thrust::make_zip_iterator(potential_next_dual_solution_.data(), reflected_dual_.data()),
        potential_next_dual_solution_.size(),
        dual_reflected_major_projection_batch<f_t>(),
        stream_view_);
      }

#ifdef CUPDLP_DEBUG_MODE
      print("potential_next_dual_solution_", potential_next_dual_solution_);
      print("reflected_dual_", reflected_dual_);
#endif
      graph_all.end_capture(should_major);
    }
    graph_all.launch(should_major);

  } else {
    if (!graph_all.is_initialized(should_major)) {
      graph_all.start_capture(should_major);

      // Compute next primal
      compute_At_y();

#ifdef CUPDLP_DEBUG_MODE
print("current_saddle_point_state_.get_primal_solution()", current_saddle_point_state_.get_primal_solution());
print("problem_ptr->objective_coefficients", problem_ptr->objective_coefficients);
print("current_saddle_point_state_.get_current_AtY()", current_saddle_point_state_.get_current_AtY());
#endif

      if (!batch_mode_)
      {
        cub::DeviceTransform::Transform(
        cuda::std::make_tuple(current_saddle_point_state_.get_primal_solution().data(),
                              problem_ptr->objective_coefficients.data(),
                              current_saddle_point_state_.get_current_AtY().data(),
                              problem_ptr->variable_bounds.data()),
        reflected_primal_.data(),
        primal_size_h_,
        primal_reflected_projection<f_t>(primal_step_size.data()),
        stream_view_);
      }
      else
      {
        cub::DeviceTransform::Transform(
        cuda::std::make_tuple(current_saddle_point_state_.get_primal_solution().data(),
                              problem_wrap_container(problem_ptr->objective_coefficients),
                              current_saddle_point_state_.get_current_AtY().data(),
                              problem_wrap_container(problem_ptr->variable_bounds),
                            thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            batch_wrapped_iterator<f_t>(primal_step_size.data(), primal_size_h_))
                          ),
          reflected_primal_.data(),
          reflected_primal_.size(),
          primal_reflected_projection_batch<f_t>(),
          stream_view_);

        if (new_bounds_idx_.size() != 0) {
          const auto [grid_size, block_size] =
            kernel_config_from_batch_size(climber_strategies_.size());
          refine_primal_projection_batch_kernel<<<grid_size, block_size, 0, stream_view_>>>(
            (i_t)climber_strategies_.size(),
            problem_ptr->n_variables,
            new_bounds_idx_.data(),
            new_bounds_lower_.data(),
            new_bounds_upper_.data(),
            current_saddle_point_state_.get_primal_solution().data(),
            problem_ptr->objective_coefficients.data(),
            current_saddle_point_state_.get_current_AtY().data(),
            primal_step_size.data(),
            reflected_primal_.data());
        }
      }
#ifdef CUPDLP_DEBUG_MODE
print("reflected_primal_", reflected_primal_);
print("current_saddle_point_state_.get_dual_solution()", current_saddle_point_state_.get_dual_solution());
print("current_saddle_point_state_.get_dual_gradient()", current_saddle_point_state_.get_dual_gradient());
print("problem_ptr->constraint_lower_bounds", problem_ptr->constraint_lower_bounds);
print("problem_ptr->constraint_upper_bounds", problem_ptr->constraint_upper_bounds);
print("dual_step_size", dual_step_size);
#endif

      // Compute next dual
      compute_A_x();

      if (!batch_mode_)
      {
        cub::DeviceTransform::Transform(
          cuda::std::make_tuple(current_saddle_point_state_.get_dual_solution().data(),
                                current_saddle_point_state_.get_dual_gradient().data(),
                                problem_ptr->constraint_lower_bounds.data(),
                                problem_ptr->constraint_upper_bounds.data()),
          reflected_dual_.data(),
          dual_size_h_,
          dual_reflected_projection<f_t>(dual_step_size.data()),
          stream_view_);
      }
      else
      {
        cub::DeviceTransform::Transform(
        cuda::std::make_tuple(current_saddle_point_state_.get_dual_solution().data(),
                              current_saddle_point_state_.get_dual_gradient().data(),
                              problem_wrap_container(problem_ptr->constraint_lower_bounds),
                              problem_wrap_container(problem_ptr->constraint_upper_bounds),
                            thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            batch_wrapped_iterator<f_t>(dual_step_size.data(), dual_size_h_))),
        reflected_dual_.data(),
        reflected_dual_.size(),
        dual_reflected_projection_batch<f_t>(),
        stream_view_);
      }
#ifdef CUPDLP_DEBUG_MODE
      print("reflected_dual_", reflected_dual_);
#endif
      graph_all.end_capture(should_major);
    }
    graph_all.launch(should_major);
  }
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::take_step(rmm::device_uvector<f_t>& primal_step_size,
                                        rmm::device_uvector<f_t>& dual_step_size,
                                        i_t iterations_since_last_restart,
                                        bool last_restart_was_average,
                                        i_t total_pdlp_iterations,
                                        bool is_major_iteration)
{
#ifdef PDLP_DEBUG_MODE
  std::cout << "Take Step:" << std::endl;
#endif

  if (!hyper_params_.use_reflected_primal_dual) {
    cuopt_expects(!batch_mode_, error_type_t::ValidationError, "Batch mode not supported for non reflected primal dual");
    compute_next_primal_dual_solution(primal_step_size,
                                      iterations_since_last_restart,
                                      last_restart_was_average,
                                      dual_step_size,
                                      total_pdlp_iterations);
  } else {
    compute_next_primal_dual_solution_reflected(
      primal_step_size,
      dual_step_size,
      is_major_iteration ||
        ((total_pdlp_iterations + 2) % conditional_major<i_t>(total_pdlp_iterations + 2)) == 0);
  }
  total_pdhg_iterations_ += 1;
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::update_solution(
  cusparse_view_t<i_t, f_t>& current_op_problem_evaluation_cusparse_view_)
{
  raft::common::nvtx::range fun_scope("update_solution");

  // Instead of copying, use a swap (that moves pointers)
  // It's ok because the next will be overwritten next iteration anyways
  // No need to sync, compute_step_sizes has already synced the host

  std::swap(current_saddle_point_state_.primal_solution_, potential_next_primal_solution_);
  std::swap(current_saddle_point_state_.dual_solution_, potential_next_dual_solution_);
  // Accepted (valid step size) next_Aty will be current Aty next PDHG iteration, saves an SpMV
  std::swap(current_saddle_point_state_.current_AtY_, current_saddle_point_state_.next_AtY_);

  // Update cusparse views to point to the new values, cost is marginal
  RAFT_CUSPARSE_TRY(cusparseDnVecSetValues(cusparse_view_.current_AtY,
                                           current_saddle_point_state_.current_AtY_.data()));
  RAFT_CUSPARSE_TRY(
    cusparseDnVecSetValues(cusparse_view_.next_AtY, current_saddle_point_state_.next_AtY_.data()));
  RAFT_CUSPARSE_TRY(cusparseDnVecSetValues(cusparse_view_.potential_next_dual_solution,
                                           potential_next_dual_solution_.data()));
  RAFT_CUSPARSE_TRY(cusparseDnVecSetValues(cusparse_view_.primal_solution,
                                           current_saddle_point_state_.primal_solution_.data()));
  RAFT_CUSPARSE_TRY(cusparseDnVecSetValues(cusparse_view_.dual_solution,
                                           current_saddle_point_state_.dual_solution_.data()));
  RAFT_CUSPARSE_TRY(
    cusparseDnVecSetValues(current_op_problem_evaluation_cusparse_view_.primal_solution,
                           current_saddle_point_state_.primal_solution_.data()));
  RAFT_CUSPARSE_TRY(
    cusparseDnVecSetValues(current_op_problem_evaluation_cusparse_view_.dual_solution,
                           current_saddle_point_state_.dual_solution_.data()));
}

template <typename i_t, typename f_t>
saddle_point_state_t<i_t, f_t>& pdhg_solver_t<i_t, f_t>::get_saddle_point_state()
{
  return current_saddle_point_state_;
}

template <typename i_t, typename f_t>
cusparse_view_t<i_t, f_t>& pdhg_solver_t<i_t, f_t>::get_cusparse_view()
{
  return cusparse_view_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_primal_tmp_resource()
{
  return tmp_primal_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_dual_tmp_resource()
{
  return tmp_dual_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_potential_next_primal_solution()
{
  return potential_next_primal_solution_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_dual_slack()
{
  return dual_slack_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_potential_next_primal_solution() const
{
  return potential_next_primal_solution_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_potential_next_dual_solution() const
{
  return potential_next_dual_solution_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_reflected_dual() const
{
  return reflected_dual_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_reflected_primal() const
{
  return reflected_primal_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_potential_next_dual_solution()
{
  return potential_next_dual_solution_;
}

template <typename i_t, typename f_t>
i_t pdhg_solver_t<i_t, f_t>::get_total_pdhg_iterations()
{
  return total_pdhg_iterations_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_primal_solution()
{
  return current_saddle_point_state_.get_primal_solution();
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_dual_solution()
{
  return current_saddle_point_state_.get_dual_solution();
}

#if MIP_INSTANTIATE_FLOAT
template class pdhg_solver_t<int, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class pdhg_solver_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
