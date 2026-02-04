/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <mip/mip_constants.hpp>
#include <mip/presolve/gf2_presolve.hpp>
#include <mip/presolve/third_party_presolve.hpp>
#include <utilities/logger.hpp>
#include <utilities/timer.hpp>

#include <raft/common/nvtx.hpp>

#if !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"  // ignore boost error for pip wheel build
#endif
#include <papilo/core/Presolve.hpp>
#include <papilo/core/ProblemBuilder.hpp>
#if !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace cuopt::linear_programming::detail {

static papilo::PostsolveStorage<double> post_solve_storage_;
static bool maximize_ = false;

template <typename i_t, typename f_t>
papilo::Problem<f_t> build_papilo_problem(const optimization_problem_t<i_t, f_t>& op_problem,
                                          problem_category_t category)
{
  raft::common::nvtx::range fun_scope("Build papilo problem");
  // Build papilo problem from optimization problem
  papilo::ProblemBuilder<f_t> builder;

  // Get problem dimensions
  const i_t num_cols = op_problem.get_n_variables();
  const i_t num_rows = op_problem.get_n_constraints();
  const i_t nnz      = op_problem.get_nnz();

  CUOPT_LOG_INFO("========== BUILD_PAPILO_PROBLEM START ==========");
  CUOPT_LOG_INFO("[INPUT] num_cols (variables): " << num_cols);
  CUOPT_LOG_INFO("[INPUT] num_rows (constraints): " << num_rows);
  CUOPT_LOG_INFO("[INPUT] nnz (non-zeros): " << nnz);
  CUOPT_LOG_INFO("[INPUT] category: " << (category == problem_category_t::MIP ? "MIP" : "LP"));

  builder.reserve(nnz, num_rows, num_cols);

  // Get problem data from optimization problem
  const auto& coefficients = op_problem.get_constraint_matrix_values();
  const auto& offsets      = op_problem.get_constraint_matrix_offsets();
  const auto& variables    = op_problem.get_constraint_matrix_indices();
  const auto& obj_coeffs   = op_problem.get_objective_coefficients();
  const auto& var_lb       = op_problem.get_variable_lower_bounds();
  const auto& var_ub       = op_problem.get_variable_upper_bounds();
  const auto& bounds       = op_problem.get_constraint_bounds();
  const auto& row_types    = op_problem.get_row_types();
  const auto& constr_lb    = op_problem.get_constraint_lower_bounds();
  const auto& constr_ub    = op_problem.get_constraint_upper_bounds();
  const auto& var_types    = op_problem.get_variable_types();

  // Copy data to host
  std::vector<f_t> h_coefficients(coefficients.size());
  auto stream_view = op_problem.get_handle_ptr()->get_stream();
  raft::copy(h_coefficients.data(), coefficients.data(), coefficients.size(), stream_view);
  std::vector<i_t> h_offsets(offsets.size());
  raft::copy(h_offsets.data(), offsets.data(), offsets.size(), stream_view);
  std::vector<i_t> h_variables(variables.size());
  raft::copy(h_variables.data(), variables.data(), variables.size(), stream_view);
  std::vector<f_t> h_obj_coeffs(obj_coeffs.size());
  raft::copy(h_obj_coeffs.data(), obj_coeffs.data(), obj_coeffs.size(), stream_view);
  std::vector<f_t> h_var_lb(var_lb.size());
  raft::copy(h_var_lb.data(), var_lb.data(), var_lb.size(), stream_view);
  std::vector<f_t> h_var_ub(var_ub.size());
  raft::copy(h_var_ub.data(), var_ub.data(), var_ub.size(), stream_view);
  std::vector<f_t> h_bounds(bounds.size());
  raft::copy(h_bounds.data(), bounds.data(), bounds.size(), stream_view);
  std::vector<char> h_row_types(row_types.size());
  raft::copy(h_row_types.data(), row_types.data(), row_types.size(), stream_view);
  std::vector<f_t> h_constr_lb(constr_lb.size());
  raft::copy(h_constr_lb.data(), constr_lb.data(), constr_lb.size(), stream_view);
  std::vector<f_t> h_constr_ub(constr_ub.size());
  raft::copy(h_constr_ub.data(), constr_ub.data(), constr_ub.size(), stream_view);
  std::vector<var_t> h_var_types(var_types.size());
  raft::copy(h_var_types.data(), var_types.data(), var_types.size(), stream_view);

  maximize_ = op_problem.get_sense();
  CUOPT_LOG_INFO("[OBJECTIVE] maximize: " << (maximize_ ? "true" : "false"));
  CUOPT_LOG_INFO("[OBJECTIVE] original offset: " << op_problem.get_objective_offset());

  if (maximize_) {
    CUOPT_LOG_INFO("[OBJECTIVE] Flipping objective coefficients for maximization");
    f_t orig_sum = 0, flipped_sum = 0;
    for (size_t i = 0; i < h_obj_coeffs.size(); ++i) {
      orig_sum += h_obj_coeffs[i];
      h_obj_coeffs[i] = -h_obj_coeffs[i];
      flipped_sum += h_obj_coeffs[i];
    }
    CUOPT_LOG_INFO("[OBJECTIVE] Sum before flip: " << orig_sum << ", after flip: " << flipped_sum);
  }

  auto constr_bounds_empty = h_constr_lb.empty() && h_constr_ub.empty();
  CUOPT_LOG_INFO("[CONSTRAINTS] bounds_empty: " << (constr_bounds_empty ? "true" : "false"));

  if (constr_bounds_empty) {
    CUOPT_LOG_INFO("[CONSTRAINTS] Building bounds from row types");
    int count_L = 0, count_G = 0, count_E = 0;
    for (size_t i = 0; i < h_row_types.size(); ++i) {
      if (h_row_types[i] == 'L') {
        h_constr_lb.push_back(-std::numeric_limits<f_t>::infinity());
        h_constr_ub.push_back(h_bounds[i]);
        count_L++;
      } else if (h_row_types[i] == 'G') {
        h_constr_lb.push_back(h_bounds[i]);
        h_constr_ub.push_back(std::numeric_limits<f_t>::infinity());
        count_G++;
      } else if (h_row_types[i] == 'E') {
        h_constr_lb.push_back(h_bounds[i]);
        h_constr_ub.push_back(h_bounds[i]);
        count_E++;
      }
    }
    CUOPT_LOG_INFO("[CONSTRAINTS] Row types: L=" << count_L << ", G=" << count_G << ", E=" << count_E);
  } else {
    CUOPT_LOG_INFO("[CONSTRAINTS] Using existing bounds: " << h_constr_lb.size() << " lower, " << h_constr_ub.size() << " upper");
  }

  builder.setNumCols(num_cols);
  builder.setNumRows(num_rows);

  builder.setObjAll(h_obj_coeffs);
  f_t papilo_offset = maximize_ ? -op_problem.get_objective_offset()
                                : op_problem.get_objective_offset();
  builder.setObjOffset(papilo_offset);
  CUOPT_LOG_INFO("[OBJECTIVE] Offset sent to Papilo: " << papilo_offset << " (flipped: " << (maximize_ ? "yes" : "no") << ")");

  if (!h_var_lb.empty() && !h_var_ub.empty()) {
    builder.setColLbAll(h_var_lb);
    builder.setColUbAll(h_var_ub);
    if (op_problem.get_variable_names().size() == h_var_lb.size()) {
      builder.setColNameAll(op_problem.get_variable_names());
    }
    CUOPT_LOG_INFO("[VARIABLES] Set " << h_var_lb.size() << " variable bounds");

    // Log first few variable bounds for verification
    int log_count = std::min(5, (int)h_var_lb.size());
    for (int i = 0; i < log_count; ++i) {
      CUOPT_LOG_INFO("[VARIABLES] Var " << i << ": [" << h_var_lb[i] << ", " << h_var_ub[i] << "]");
    }
  }

  int int_count = 0, cont_count = 0;
  for (size_t i = 0; i < h_var_types.size(); ++i) {
    bool is_int = h_var_types[i] == var_t::INTEGER;
    builder.setColIntegral(i, is_int);
    if (is_int) int_count++; else cont_count++;
  }
  CUOPT_LOG_INFO("[VARIABLES] Types: " << int_count << " integer, " << cont_count << " continuous");

  if (!h_constr_lb.empty() && !h_constr_ub.empty()) {
    builder.setRowLhsAll(h_constr_lb);
    builder.setRowRhsAll(h_constr_ub);
    CUOPT_LOG_INFO("[CONSTRAINTS] Set " << h_constr_lb.size() << " constraint bounds");

    // Log first few constraint bounds for verification
    int log_count = std::min(5, (int)h_constr_lb.size());
    for (int i = 0; i < log_count; ++i) {
      CUOPT_LOG_INFO("[CONSTRAINTS] Row " << i << ": [" << h_constr_lb[i] << ", " << h_constr_ub[i] << "]");
    }
  }

  std::vector<papilo::RowFlags> h_row_flags(h_constr_lb.size());
  std::vector<std::tuple<i_t, i_t, f_t>> h_entries;
  // Add constraints row by row
  CUOPT_LOG_INFO("[CSR_MATRIX] Building entries from CSR format");

  int lhs_inf_count = 0, rhs_inf_count = 0;
  for (size_t i = 0; i < h_constr_lb.size(); ++i) {
    // Get row entries
    i_t row_start   = h_offsets[i];
    i_t row_end     = h_offsets[i + 1];
    i_t num_entries = row_end - row_start;

    if (i < 3) {  // Log first 3 rows in detail
      CUOPT_LOG_INFO("[CSR_MATRIX] Row " << i << " has " << num_entries << " entries (offset [" << row_start << ", " << row_end << "])");
      for (size_t j = 0; j < num_entries && j < 10; ++j) {
        CUOPT_LOG_INFO("[CSR_MATRIX]   Entry: var=" << h_variables[row_start + j] << ", coeff=" << h_coefficients[row_start + j]);
      }
    }

    for (size_t j = 0; j < num_entries; ++j) {
      h_entries.push_back(
        std::make_tuple(i, h_variables[row_start + j], h_coefficients[row_start + j]));
    }

    if (h_constr_lb[i] == -std::numeric_limits<f_t>::infinity()) {
      h_row_flags[i].set(papilo::RowFlag::kLhsInf);
      lhs_inf_count++;
    } else {
      h_row_flags[i].unset(papilo::RowFlag::kLhsInf);
    }
    if (h_constr_ub[i] == std::numeric_limits<f_t>::infinity()) {
      h_row_flags[i].set(papilo::RowFlag::kRhsInf);
      rhs_inf_count++;
    } else {
      h_row_flags[i].unset(papilo::RowFlag::kRhsInf);
    }

    if (h_constr_lb[i] == -std::numeric_limits<f_t>::infinity()) { h_constr_lb[i] = 0; }
    if (h_constr_ub[i] == std::numeric_limits<f_t>::infinity()) { h_constr_ub[i] = 0; }
  }
  CUOPT_LOG_INFO("[CSR_MATRIX] Total entries: " << h_entries.size());
  CUOPT_LOG_INFO("[CSR_MATRIX] Infinity flags: LHS=" << lhs_inf_count << ", RHS=" << rhs_inf_count);

  int var_lb_inf_count = 0, var_ub_inf_count = 0;
  for (size_t i = 0; i < h_var_lb.size(); ++i) {
    bool lb_inf = h_var_lb[i] == -std::numeric_limits<f_t>::infinity();
    bool ub_inf = h_var_ub[i] == std::numeric_limits<f_t>::infinity();
    builder.setColLbInf(i, lb_inf);
    builder.setColUbInf(i, ub_inf);
    if (lb_inf) { builder.setColLb(i, 0); var_lb_inf_count++; }
    if (ub_inf) { builder.setColUb(i, 0); var_ub_inf_count++; }
  }
  CUOPT_LOG_INFO("[VARIABLES] Infinity bounds: LB=" << var_lb_inf_count << ", UB=" << var_ub_inf_count);

  CUOPT_LOG_INFO("[BUILD] Calling builder.build()...");
  auto problem = builder.build();
  CUOPT_LOG_INFO("[BUILD] Papilo problem built successfully");

  if (h_entries.size()) {
    auto constexpr const sorted_entries = true;
    // MIP reductions like clique merging and substituition require more fillin
    const double spare_ratio      = category == problem_category_t::MIP ? 4.0 : 2.0;
    const int min_inter_row_space = category == problem_category_t::MIP ? 30 : 4;
    CUOPT_LOG_INFO("[CSR_MATRIX] Creating sparse storage: spare_ratio=" << spare_ratio << ", min_inter_row_space=" << min_inter_row_space);

    auto csr_storage = papilo::SparseStorage<f_t>(
      h_entries, num_rows, num_cols, sorted_entries, spare_ratio, min_inter_row_space);
    problem.setConstraintMatrix(csr_storage, h_constr_lb, h_constr_ub, h_row_flags);
    CUOPT_LOG_INFO("[CSR_MATRIX] Constraint matrix set in Papilo problem");

    papilo::ConstraintMatrix<f_t>& matrix = problem.getConstraintMatrix();
    int equation_count = 0;
    for (int i = 0; i < problem.getNRows(); ++i) {
      papilo::RowFlags rowFlag = matrix.getRowFlags()[i];
      if (!rowFlag.test(papilo::RowFlag::kRhsInf) && !rowFlag.test(papilo::RowFlag::kLhsInf) &&
          matrix.getLeftHandSides()[i] == matrix.getRightHandSides()[i]) {
        matrix.getRowFlags()[i].set(papilo::RowFlag::kEquation);
        equation_count++;
      }
    }
    CUOPT_LOG_INFO("[CSR_MATRIX] Marked " << equation_count << " rows as equations");
  }

  CUOPT_LOG_INFO("========== BUILD_PAPILO_PROBLEM END ==========");
  return problem;
}

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> build_optimization_problem(
  papilo::Problem<f_t> const& papilo_problem,
  raft::handle_t const* handle_ptr,
  problem_category_t category)
{
  raft::common::nvtx::range fun_scope("Build optimization problem");
  optimization_problem_t<i_t, f_t> op_problem(handle_ptr);

  CUOPT_LOG_INFO("========== BUILD_OPTIMIZATION_PROBLEM START ==========");
  CUOPT_LOG_INFO("[INPUT] Papilo problem size: " << papilo_problem.getNCols() << " vars, " << papilo_problem.getNRows() << " rows");
  CUOPT_LOG_INFO("[INPUT] maximize_ (static): " << (maximize_ ? "true" : "false"));

  auto obj = papilo_problem.getObjective();
  CUOPT_LOG_INFO("[OBJECTIVE] Offset from Papilo: " << obj.offset);
  CUOPT_LOG_INFO("[OBJECTIVE] WARNING: Both branches of ternary are same! Setting: " << obj.offset);

  op_problem.set_objective_offset(maximize_ ? obj.offset : obj.offset);
  op_problem.set_maximize(maximize_);
  op_problem.set_problem_category(category);

  if (papilo_problem.getNRows() == 0 && papilo_problem.getNCols() == 0) {
    CUOPT_LOG_INFO("[EMPTY] Problem is empty after presolve - returning minimal problem");
    // FIXME: Shouldn't need to set offsets
    std::vector<i_t> h_offsets{0};
    std::vector<i_t> h_indices{};
    std::vector<f_t> h_values{};
    op_problem.set_csr_constraint_matrix(h_values.data(),
                                         h_values.size(),
                                         h_indices.data(),
                                         h_indices.size(),
                                         h_offsets.data(),
                                         h_offsets.size());

    CUOPT_LOG_INFO("========== BUILD_OPTIMIZATION_PROBLEM END (EMPTY) ==========");
    return op_problem;
  }

  if (maximize_) {
    CUOPT_LOG_INFO("[OBJECTIVE] Flipping objective coefficients back (was flipped for Papilo)");
    f_t papilo_sum = 0, flipped_sum = 0;
    for (size_t i = 0; i < obj.coefficients.size(); ++i) {
      papilo_sum += obj.coefficients[i];
      obj.coefficients[i] = -obj.coefficients[i];
      flipped_sum += obj.coefficients[i];
    }
    CUOPT_LOG_INFO("[OBJECTIVE] Sum from Papilo: " << papilo_sum << ", after flip back: " << flipped_sum);
  }
  op_problem.set_objective_coefficients(obj.coefficients.data(), obj.coefficients.size());
  CUOPT_LOG_INFO("[OBJECTIVE] Set " << obj.coefficients.size() << " objective coefficients");

  auto& constraint_matrix = papilo_problem.getConstraintMatrix();
  auto row_lower          = constraint_matrix.getLeftHandSides();
  auto row_upper          = constraint_matrix.getRightHandSides();
  auto col_lower          = papilo_problem.getLowerBounds();
  auto col_upper          = papilo_problem.getUpperBounds();

  CUOPT_LOG_INFO("[CONSTRAINTS] Restoring infinity flags in bounds");
  auto row_flags = constraint_matrix.getRowFlags();
  int lhs_inf_restored = 0, rhs_inf_restored = 0;
  for (size_t i = 0; i < row_flags.size(); i++) {
    if (row_flags[i].test(papilo::RowFlag::kLhsInf)) {
      row_lower[i] = -std::numeric_limits<f_t>::infinity();
      lhs_inf_restored++;
    }
    if (row_flags[i].test(papilo::RowFlag::kRhsInf)) {
      row_upper[i] = std::numeric_limits<f_t>::infinity();
      rhs_inf_restored++;
    }
  }
  CUOPT_LOG_INFO("[CONSTRAINTS] Restored infinity: LHS=" << lhs_inf_restored << ", RHS=" << rhs_inf_restored);

  op_problem.set_constraint_lower_bounds(row_lower.data(), row_lower.size());
  op_problem.set_constraint_upper_bounds(row_upper.data(), row_upper.size());
  CUOPT_LOG_INFO("[CONSTRAINTS] Set " << row_lower.size() << " constraint bounds");

  // Log first few for verification
  int log_count = std::min(5, (int)row_lower.size());
  for (int i = 0; i < log_count; ++i) {
    CUOPT_LOG_INFO("[CONSTRAINTS] Row " << i << ": [" << row_lower[i] << ", " << row_upper[i] << "]");
  }

  auto [index_range, nrows] = constraint_matrix.getRangeInfo();

  CUOPT_LOG_INFO("[CSR_MATRIX] ========== CRITICAL SECTION ==========");
  CUOPT_LOG_INFO("[CSR_MATRIX] nrows: " << nrows);
  CUOPT_LOG_INFO("[CSR_MATRIX] NNZ from Papilo: " << constraint_matrix.getNnz());

  // papilo indices do not start from 0 after presolve
  size_t start = index_range[0].start;
  size_t end = index_range[nrows - 1].end;
  CUOPT_LOG_INFO("[CSR_MATRIX] Index range: start=" << start << ", end=" << end);
  CUOPT_LOG_INFO("[CSR_MATRIX] WARNING: Assuming contiguous range from start to end!");

  std::vector<i_t> offsets(nrows + 1);
  for (i_t i = 0; i < nrows; i++) {
    offsets[i] = index_range[i].start - start;
  }
  offsets[nrows] = index_range[nrows - 1].end - start;

  // Log first few offsets
  int offset_log_count = std::min(5, (int)nrows + 1);
  CUOPT_LOG_INFO("[CSR_MATRIX] First " << offset_log_count << " offsets:");
  for (int i = 0; i < offset_log_count; ++i) {
    CUOPT_LOG_INFO("[CSR_MATRIX]   offsets[" << i << "] = " << offsets[i]);
  }

  i_t nnz = constraint_matrix.getNnz();
  CUOPT_LOG_INFO("[CSR_MATRIX] Computed offsets[nrows] = " << offsets[nrows] << ", expected NNZ = " << nnz);

  // CRITICAL CHECK: Verify contiguity assumption
  size_t expected_nnz_from_range = end - start;
  if (expected_nnz_from_range != nnz) {
    CUOPT_LOG_WARN("[CSR_MATRIX] **BUG DETECTED**: NON-CONTIGUOUS CSR MATRIX!");
    CUOPT_LOG_WARN("[CSR_MATRIX] Expected NNZ from range: " << expected_nnz_from_range);
    CUOPT_LOG_WARN("[CSR_MATRIX] Actual NNZ: " << nnz);
    CUOPT_LOG_WARN("[CSR_MATRIX] This means the coefficient/column arrays have gaps!");
  }

  // Check for gaps in index_range
  for (i_t i = 0; i < nrows - 1; i++) {
    size_t current_end = index_range[i].end;
    size_t next_start = index_range[i + 1].start;
    if (current_end != next_start && i < 5) {
      CUOPT_LOG_WARN("[CSR_MATRIX] GAP DETECTED: Row " << i << " ends at " << current_end << ", Row " << (i+1) << " starts at " << next_start);
    }
  }

  assert(offsets[nrows] == nnz);

  const int* cols   = constraint_matrix.getConstraintMatrix().getColumns();
  const f_t* coeffs = constraint_matrix.getConstraintMatrix().getValues();

  CUOPT_LOG_INFO("[CSR_MATRIX] Extracting coefficients/columns starting at index " << start);
  CUOPT_LOG_INFO("[CSR_MATRIX] First 3 rows from Papilo:");
  for (int i = 0; i < std::min(3, (int)nrows); ++i) {
    size_t row_start = index_range[i].start;
    size_t row_end = index_range[i].end;
    CUOPT_LOG_INFO("[CSR_MATRIX]   Row " << i << " range [" << row_start << ", " << row_end << ")");
    for (size_t j = row_start; j < row_end && j < row_start + 10; ++j) {
      CUOPT_LOG_INFO("[CSR_MATRIX]     Entry: col=" << cols[j] << ", coeff=" << coeffs[j]);
    }
  }

  op_problem.set_csr_constraint_matrix(
    &(coeffs[start]), nnz, &(cols[start]), nnz, offsets.data(), nrows + 1);
  CUOPT_LOG_INFO("[CSR_MATRIX] CSR matrix set in optimization problem");

  auto col_flags = papilo_problem.getColFlags();
  std::vector<var_t> var_types(col_flags.size());
  int int_vars = 0, cont_vars = 0, lb_inf = 0, ub_inf = 0;
  for (size_t i = 0; i < col_flags.size(); i++) {
    bool is_int = col_flags[i].test(papilo::ColFlag::kIntegral);
    var_types[i] = is_int ? var_t::INTEGER : var_t::CONTINUOUS;
    if (is_int) int_vars++; else cont_vars++;

    if (col_flags[i].test(papilo::ColFlag::kLbInf)) {
      col_lower[i] = -std::numeric_limits<f_t>::infinity();
      lb_inf++;
    }
    if (col_flags[i].test(papilo::ColFlag::kUbInf)) {
      col_upper[i] = std::numeric_limits<f_t>::infinity();
      ub_inf++;
    }
  }

  CUOPT_LOG_INFO("[VARIABLES] Types: " << int_vars << " integer, " << cont_vars << " continuous");
  CUOPT_LOG_INFO("[VARIABLES] Infinity bounds: LB=" << lb_inf << ", UB=" << ub_inf);

  op_problem.set_variable_lower_bounds(col_lower.data(), col_lower.size());
  op_problem.set_variable_upper_bounds(col_upper.data(), col_upper.size());
  op_problem.set_variable_types(var_types.data(), var_types.size());

  CUOPT_LOG_INFO("[VARIABLES] Set " << col_lower.size() << " variable bounds and types");

  // Log first few variable bounds
  int var_log_count = std::min(5, (int)col_lower.size());
  for (int i = 0; i < var_log_count; ++i) {
    CUOPT_LOG_INFO("[VARIABLES] Var " << i << ": [" << col_lower[i] << ", " << col_upper[i] << "] " << (var_types[i] == var_t::INTEGER ? "INT" : "CONT"));
  }

  CUOPT_LOG_INFO("========== BUILD_OPTIMIZATION_PROBLEM END ==========");
  return op_problem;
}

void check_presolve_status(const papilo::PresolveStatus& status)
{
  switch (status) {
    case papilo::PresolveStatus::kUnchanged:
      CUOPT_LOG_INFO("Presolve status: did not result in any changes");
      break;
    case papilo::PresolveStatus::kReduced:
      CUOPT_LOG_INFO("Presolve status: reduced the problem");
      break;
    case papilo::PresolveStatus::kUnbndOrInfeas:
      CUOPT_LOG_INFO("Presolve status: found an unbounded or infeasible problem");
      break;
    case papilo::PresolveStatus::kInfeasible:
      CUOPT_LOG_INFO("Presolve status: found an infeasible problem");
      break;
    case papilo::PresolveStatus::kUnbounded:
      CUOPT_LOG_INFO("Presolve status: found an unbounded problem");
      break;
  }
}

void check_postsolve_status(const papilo::PostsolveStatus& status)
{
  switch (status) {
    case papilo::PostsolveStatus::kOk: CUOPT_LOG_INFO("Post-solve status: succeeded"); break;
    case papilo::PostsolveStatus::kFailed:
      CUOPT_LOG_INFO(
        "Post-solve status: Post solved solution violates constraints. This is most likely due to "
        "different tolerances.");
      break;
  }
}

template <typename f_t>
void set_presolve_methods(papilo::Presolve<f_t>& presolver,
                          problem_category_t category,
                          bool dual_postsolve)
{
  using uptr = std::unique_ptr<papilo::PresolveMethod<f_t>>;

  if (category == problem_category_t::MIP) {
    // cuOpt custom GF2 presolver
    presolver.addPresolveMethod(uptr(new cuopt::linear_programming::detail::GF2Presolve<f_t>()));
  }
  // fast presolvers
  presolver.addPresolveMethod(uptr(new papilo::SingletonCols<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::CoefficientStrengthening<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ConstraintPropagation<f_t>()));

  // medium presolvers
  presolver.addPresolveMethod(uptr(new papilo::FixContinuous<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::SimpleProbing<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ParallelRowDetection<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ParallelColDetection<f_t>()));

  presolver.addPresolveMethod(uptr(new papilo::SingletonStuffing<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::DualFix<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::SimplifyInequalities<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::CliqueMerging<f_t>()));

  // exhaustive presolvers
  presolver.addPresolveMethod(uptr(new papilo::ImplIntDetection<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::DominatedCols<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::Probing<f_t>()));

  if (!dual_postsolve) {
    presolver.addPresolveMethod(uptr(new papilo::DualInfer<f_t>()));
    presolver.addPresolveMethod(uptr(new papilo::SimpleSubstitution<f_t>()));
    presolver.addPresolveMethod(uptr(new papilo::Sparsify<f_t>()));
    presolver.addPresolveMethod(uptr(new papilo::Substitution<f_t>()));
  } else {
    CUOPT_LOG_INFO("Disabling the presolver methods that do not support dual postsolve");
  }
}

template <typename i_t, typename f_t>
void set_presolve_options(papilo::Presolve<f_t>& presolver,
                          problem_category_t category,
                          f_t absolute_tolerance,
                          f_t relative_tolerance,
                          double time_limit,
                          bool dual_postsolve,
                          i_t num_cpu_threads)
{
  presolver.getPresolveOptions().tlim    = time_limit;
  presolver.getPresolveOptions().threads = num_cpu_threads;  //  user setting or  0 (automatic)
  presolver.getPresolveOptions().feastol = absolute_tolerance; // From function parameter
  if (dual_postsolve) {
    presolver.getPresolveOptions().componentsmaxint = -1;
    presolver.getPresolveOptions().detectlindep     = 0;
  }
}

template <typename f_t>
void set_presolve_parameters(papilo::Presolve<f_t>& presolver,
                             problem_category_t category,
                             int nrows,
                             int ncols)
{
  // It looks like a copy. But this copy has the pointers to relevant variables in papilo
  auto params = presolver.getParameters();
  if (category == problem_category_t::MIP) {
    // Papilo has work unit measurements for probing. Because of this when the first batch fails to
    // produce any reductions, the algorithm stops. To avoid stopping the algorithm, we set a
    // minimum badge size to a huge value. The time limit makes sure that we exit if it takes too
    // long
    bool is_large = ncols > 100000;
    int min_badgesize = is_large ? std::min(5000, ncols / 100) : std::max(ncols / 10, 32);
    int max_clique = is_large ? 5 : 20;
    min_badgesize = std::max(ncols / 2, 32);
    params.setParameter("probing.minbadgesize", min_badgesize);
    params.setParameter("cliquemerging.enabled", true);
    params.setParameter("cliquemerging.maxcalls", 50);

  }
}

template <typename i_t, typename f_t>
std::optional<third_party_presolve_result_t<i_t, f_t>> third_party_presolve_t<i_t, f_t>::apply(
  optimization_problem_t<i_t, f_t> const& op_problem,
  problem_category_t category,
  bool dual_postsolve,
  f_t absolute_tolerance,
  f_t relative_tolerance,
  double time_limit,
  i_t num_cpu_threads)
{
  papilo::Problem<f_t> papilo_problem = build_papilo_problem(op_problem, category);

  CUOPT_LOG_INFO("Original problem: %d constraints, %d variables, %d nonzeros",
                 papilo_problem.getNRows(),
                 papilo_problem.getNCols(),
                 papilo_problem.getConstraintMatrix().getNnz());

  CUOPT_LOG_INFO("Calling Papilo presolver");
  if (category == problem_category_t::MIP) { dual_postsolve = false; }
  papilo::Presolve<f_t> presolver;
  set_presolve_methods<f_t>(presolver, category, dual_postsolve);
  set_presolve_options<i_t, f_t>(presolver,
                                 category,
                                 absolute_tolerance,
                                 relative_tolerance,
                                 time_limit,
                                 dual_postsolve,
                                 num_cpu_threads);
  set_presolve_parameters<f_t>(
    presolver, category, op_problem.get_n_constraints(), op_problem.get_n_variables());

  // Disable papilo logs
  presolver.setVerbosityLevel(papilo::VerbosityLevel::kQuiet);

  auto result = presolver.apply(papilo_problem);
  check_presolve_status(result.status);
  if (result.status == papilo::PresolveStatus::kInfeasible ||
      result.status == papilo::PresolveStatus::kUnbndOrInfeas) {
    return std::nullopt;
  }
  post_solve_storage_ = result.postsolve;
  CUOPT_LOG_INFO("Presolve removed: %d constraints, %d variables, %d nonzeros",
                 op_problem.get_n_constraints() - papilo_problem.getNRows(),
                 op_problem.get_n_variables() - papilo_problem.getNCols(),
                 op_problem.get_nnz() - papilo_problem.getConstraintMatrix().getNnz());
  CUOPT_LOG_INFO("Presolved problem: %d constraints, %d variables, %d nonzeros",
                 papilo_problem.getNRows(),
                 papilo_problem.getNCols(),
                 papilo_problem.getConstraintMatrix().getNnz());

  // Check if presolve found the optimal solution (problem fully reduced)
  if (papilo_problem.getNRows() == 0 && papilo_problem.getNCols() == 0) {
    CUOPT_LOG_INFO("Optimal solution found during presolve");
  }

  auto opt_problem =
    build_optimization_problem<i_t, f_t>(papilo_problem, op_problem.get_handle_ptr(), category);
  auto col_flags = papilo_problem.getColFlags();
  std::vector<i_t> implied_integer_indices;
  for (size_t i = 0; i < col_flags.size(); i++) {
    if (col_flags[i].test(papilo::ColFlag::kImplInt)) implied_integer_indices.push_back(i);
  }

  return std::make_optional(
    third_party_presolve_result_t<i_t, f_t>{opt_problem, implied_integer_indices});
}

template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::undo(rmm::device_uvector<f_t>& primal_solution,
                                            rmm::device_uvector<f_t>& dual_solution,
                                            rmm::device_uvector<f_t>& reduced_costs,
                                            problem_category_t category,
                                            bool status_to_skip,
                                            bool dual_postsolve,
                                            rmm::cuda_stream_view stream_view)
{
  if (status_to_skip) { return; }
  std::vector<f_t> primal_sol_vec_h(primal_solution.size());
  raft::copy(primal_sol_vec_h.data(), primal_solution.data(), primal_solution.size(), stream_view);
  std::vector<f_t> dual_sol_vec_h(dual_solution.size());
  raft::copy(dual_sol_vec_h.data(), dual_solution.data(), dual_solution.size(), stream_view);
  std::vector<f_t> reduced_costs_vec_h(reduced_costs.size());
  raft::copy(reduced_costs_vec_h.data(), reduced_costs.data(), reduced_costs.size(), stream_view);
  papilo::Solution<f_t> reduced_sol(primal_sol_vec_h);
  if (dual_postsolve) {
    reduced_sol.dual         = dual_sol_vec_h;
    reduced_sol.reducedCosts = reduced_costs_vec_h;
    reduced_sol.type         = papilo::SolutionType::kPrimalDual;
  }
  papilo::Solution<f_t> full_sol;

  papilo::Message Msg{};
  Msg.setVerbosityLevel(papilo::VerbosityLevel::kQuiet);
  papilo::Postsolve<f_t> post_solver{Msg, post_solve_storage_.getNum()};

  bool is_optimal = false;
  auto status     = post_solver.undo(reduced_sol, full_sol, post_solve_storage_, is_optimal);
  check_postsolve_status(status);

  primal_solution.resize(full_sol.primal.size(), stream_view);
  dual_solution.resize(full_sol.dual.size(), stream_view);
  reduced_costs.resize(full_sol.reducedCosts.size(), stream_view);
  raft::copy(primal_solution.data(), full_sol.primal.data(), full_sol.primal.size(), stream_view);
  raft::copy(dual_solution.data(), full_sol.dual.data(), full_sol.dual.size(), stream_view);
  raft::copy(
    reduced_costs.data(), full_sol.reducedCosts.data(), full_sol.reducedCosts.size(), stream_view);
}

#if MIP_INSTANTIATE_FLOAT
template class third_party_presolve_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class third_party_presolve_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
