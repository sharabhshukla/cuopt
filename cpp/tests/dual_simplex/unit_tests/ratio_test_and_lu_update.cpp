/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bound_flipping_ratio_test.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/right_looking_lu.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/types.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace cuopt::linear_programming::dual_simplex::test {

// ---------------------------------------------------------------------------
// BFRT smoke test: verify compute_step_length returns a valid entering
// variable for a constructed scenario with bounded nonbasic variables.
//
// Regression coverage for C-DS-3 (Harris tie-break used delta_z_[candidate]
// instead of delta_z_[nonbasic_list_[candidate]]).  The scenario sets up
// 4 nonbasic variables at their lower bounds with equal breakpoints but
// different |delta_z| magnitudes.  The Harris tie-break should select the
// variable with the largest |delta_z|.
// ---------------------------------------------------------------------------
TEST(bound_flipping_ratio_test, harris_tiebreak_selects_largest_pivot)
{
  using i_t = int;
  using f_t = double;

  simplex_solver_settings_t<i_t, f_t> settings;
  settings.zero_tol = 1e-12;

  // 6 variables (n=6), 2 basic (m=2), 4 nonbasic
  const i_t m = 2;
  const i_t n = 6;

  // nonbasic_list: positions 0..3 map to variables 2,3,4,5
  std::vector<i_t> nonbasic_list = {2, 3, 4, 5};
  // nonbasic_mark: variable j -> position in nonbasic_list
  std::vector<i_t> nonbasic_mark(n, -1);
  for (i_t k = 0; k < static_cast<i_t>(nonbasic_list.size()); ++k) {
    nonbasic_mark[nonbasic_list[k]] = k;
  }

  // All nonbasic at lower bound, dual feasible (z > 0)
  std::vector<variable_status_t> vstatus(n, variable_status_t::BASIC);
  for (i_t k = 0; k < static_cast<i_t>(nonbasic_list.size()); ++k) {
    vstatus[nonbasic_list[k]] = variable_status_t::NONBASIC_LOWER;
  }

  // z[j] = |delta_z[j]| so that breakpoints are all at ratio ≈ 1.0
  // delta_z is negative (decreasing z toward 0 → dual infeasibility)
  std::vector<f_t> z(n, 0.0);
  std::vector<f_t> delta_z(n, 0.0);
  std::vector<i_t> delta_z_indices;
  // Use large values so dual_tol / |delta_z| differences are below zero_tol
  z[2]      = 1e6;
  z[3]      = 1e6;
  z[4]      = 5e6;
  z[5]      = 2e6;
  delta_z[2] = -1e6;
  delta_z[3] = -1e6;
  delta_z[4] = -5e6;
  delta_z[5] = -2e6;
  delta_z_indices = {2, 3, 4, 5};

  std::vector<f_t> lower(n, 0.0);
  std::vector<f_t> upper(n, 0.0);
  // Bounded nonbasic variables
  for (i_t k = 0; k < static_cast<i_t>(nonbasic_list.size()); ++k) {
    upper[nonbasic_list[k]] = 10.0;
  }
  std::vector<uint8_t> bounded_variables(n, 0);
  for (i_t k = 0; k < static_cast<i_t>(nonbasic_list.size()); ++k) {
    bounded_variables[nonbasic_list[k]] = 1;
  }

  // Initial slope: small enough that after the first bound flip, slope < 0
  // (slope -= |delta_z[j]| * (upper[j] - lower[j]))
  // For var 4: delta_slope = 5e6 * 10 = 5e7 → set initial_slope = 1e7
  const f_t initial_slope = 1e7;

  bound_flipping_ratio_test_t<i_t, f_t> bfrt(
    settings,
    0.0,
    m,
    n,
    initial_slope,
    lower,
    upper,
    bounded_variables,
    vstatus,
    nonbasic_list,
    z,
    delta_z,
    delta_z_indices,
    nonbasic_mark);

  f_t step_length   = 0.0;
  i_t nonbasic_entering = -1;
  const i_t entering_index = bfrt.compute_step_length(step_length, nonbasic_entering);

  // The entering variable should be valid (>= 0)
  EXPECT_GE(entering_index, 0) << "BFRT should find an entering variable";

  // With the Harris tie-break fix, the entering variable should be 4
  // (the one with the largest |delta_z| = 5e6 among tied breakpoints).
  // Without the fix, the bug could select variable 5 instead.
  EXPECT_EQ(entering_index, 4)
    << "Harris tie-break should select var 4 (largest |delta_z| = 5e6)";
}

// ---------------------------------------------------------------------------
// LU factorization + basis_update_mpf_t solve test.
//
// Creates a small 3x3 basis matrix, factorizes it, constructs a
// basis_update_mpf_t, and verifies B*x = b solves correctly.
// ---------------------------------------------------------------------------
TEST(basis_update_mpf, factorize_and_solve)
{
  using i_t = int;
  using f_t = double;

  simplex_solver_settings_t<i_t, f_t> settings;
  settings.time_limit = std::numeric_limits<f_t>::infinity();

  // 3x3 basis: B = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
  // This is a symmetric positive definite matrix (good for LU stability)
  const i_t m           = 3;
  const i_t n           = 3;
  const i_t nnz         = 7;
  csc_matrix_t<i_t, f_t> A(m, n, nnz);
  A.col_start[0] = 0;
  A.col_start[1] = 2;
  A.col_start[2] = 5;
  A.col_start[3] = 7;
  // Column 0: rows 0,1
  A.i[0] = 0; A.x[0] = 2.0;
  A.i[1] = 1; A.x[1] = 1.0;
  // Column 1: rows 0,1,2
  A.i[2] = 0; A.x[2] = 1.0;
  A.i[3] = 1; A.x[3] = 3.0;
  A.i[4] = 2; A.x[4] = 1.0;
  // Column 2: rows 1,2
  A.i[5] = 1; A.x[5] = 1.0;
  A.i[6] = 2; A.x[6] = 2.0;

  std::vector<i_t> basic_list = {0, 1, 2};

  csc_matrix_t<i_t, f_t> L(m, m, 1);
  csc_matrix_t<i_t, f_t> U(m, m, 1);
  std::vector<i_t> p, pinv, q, deficient, slacks_needed;
  f_t work_estimate = 0.0;

  i_t rank = factorize_basis(
    A, settings, basic_list, 0.0, L, U, p, pinv, q, deficient, slacks_needed, work_estimate);

  ASSERT_EQ(rank, m) << "Factorization should be full rank";

  // Construct basis_update_mpf_t from the factorization
  basis_update_mpf_t<i_t, f_t> basis_update(L, U, p, settings.refactor_frequency);

  // Test B*x = b with b = [5, 10, 5]
  // Expected: x = [1, 2, 1] (verified: 2*1+1*2=4≠5... let me recompute)
  // B*x = [2*1 + 1*2 + 0*1, 1*1 + 3*2 + 1*1, 0*1 + 1*2 + 2*1] = [4, 8, 4] ≠ [5,10,5]
  // Let b = [4, 8, 4], expected x = [1, 2, 1]
  std::vector<f_t> b = {4.0, 8.0, 4.0};
  std::vector<f_t> x(m, 0.0);
  basis_update.b_solve(b, x);

  // Verify solution
  const f_t tol = 1e-10;
  EXPECT_NEAR(x[0], 1.0, tol);
  EXPECT_NEAR(x[1], 2.0, tol);
  EXPECT_NEAR(x[2], 1.0, tol);
}

// ---------------------------------------------------------------------------
// Threshold partial pivoting regression test (H-DS-9).
//
// Verifies that factorize_basis uses settings.threshold_partial_pivoting_tol
// (0.1 by default) rather than the old hardcoded 1e-12.  We check this
// indirectly by confirming that the factorization succeeds and produces
// correct solves for a matrix that would be unstable under the old tolerance.
// ---------------------------------------------------------------------------
TEST(basis_factorization, threshold_pivoting_uses_settings)
{
  using i_t = int;
  using f_t = double;

  simplex_solver_settings_t<i_t, f_t> settings;
  settings.time_limit            = std::numeric_limits<f_t>::infinity();
  settings.eliminate_singletons  = false;

  // Verify the setting is accessible and has the expected default
  EXPECT_NEAR(settings.threshold_partial_pivoting_tol, 0.1, 1e-15);

  // 2x2 matrix with a small pivot that would be accepted under 1e-12
  // but rejected under 0.1 threshold partial pivoting
  const i_t m   = 2;
  const i_t n   = 2;
  const i_t nnz = 4;
  csc_matrix_t<i_t, f_t> A(m, n, nnz);
  A.col_start[0] = 0;
  A.col_start[1] = 2;
  A.col_start[2] = 4;
  A.i[0] = 0; A.x[0] = 1e-8;   // small pivot in column 0
  A.i[1] = 1; A.x[1] = 1.0;
  A.i[2] = 0; A.x[2] = 1.0;
  A.i[3] = 1; A.x[3] = 1.0;

  std::vector<i_t> basic_list = {0, 1};

  csc_matrix_t<i_t, f_t> L(m, m, 1);
  csc_matrix_t<i_t, f_t> U(m, m, 1);
  std::vector<i_t> p, pinv, q, deficient, slacks_needed;
  f_t work_estimate = 0.0;

  // With threshold_partial_pivoting_tol = 0.1, the factorization should
  // pivot on the larger element (1.0 in column 0, row 1) rather than
  // the small 1e-8 element, producing a stable factorization.
  i_t rank = factorize_basis(
    A, settings, basic_list, 0.0, L, U, p, pinv, q, deficient, slacks_needed, work_estimate);

  ASSERT_EQ(rank, m) << "Factorization should be full rank";

  // Verify the solve is accurate
  basis_update_mpf_t<i_t, f_t> basis_update(L, U, p, settings.refactor_frequency);

  // B * [1, 1] = [1e-8 + 1, 1 + 1] = [1.00000001, 2]
  std::vector<f_t> b = {1e-8 + 1.0, 2.0};
  std::vector<f_t> x(m, 0.0);
  basis_update.b_solve(b, x);

  const f_t tol = 1e-8;
  EXPECT_NEAR(x[0], 1.0, tol);
  EXPECT_NEAR(x[1], 1.0, tol);
}

}  // namespace cuopt::linear_programming::dual_simplex::test
