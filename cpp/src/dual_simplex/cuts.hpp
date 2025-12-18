/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/sparse_vector.hpp>
#include <dual_simplex/types.hpp>
#include <dual_simplex/user_problem.hpp>


#include <cmath>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
f_t minimum_violation(const csr_matrix_t<i_t, f_t>& C,
                      const std::vector<f_t>& cut_rhs,
                      const std::vector<f_t>& x)
{
  // Check to see that this is a cut i.e C*x > d
  std::vector<f_t> Cx(C.m);
  csc_matrix_t<i_t, f_t> C_col(C.m, C.n, 0);
  C.to_compressed_col(C_col);
  matrix_vector_multiply(C_col, 1.0, x, 0.0, Cx);
  f_t min_cut_violation = inf;
  for (i_t k = 0; k < Cx.size(); k++) {
    if (Cx[k] <= cut_rhs[k]) {
      printf("C*x <= d for cut %d. C*x %e rhs %e\n", k, Cx[k], cut_rhs[k]);
    }
    min_cut_violation = std::min(min_cut_violation, Cx[k] - cut_rhs[k]);
  }
  return min_cut_violation;
}

template <typename i_t, typename f_t>
class cut_pool_t {
 public:
  cut_pool_t(i_t original_vars, const simplex_solver_settings_t<i_t, f_t>& settings)
    : original_vars_(original_vars),
      settings_(settings),
      cut_storage_(0, original_vars, 0),
      rhs_storage_(0),
      cut_age_(0),
      scored_cuts_(0)
  {
  }

  // Add a cut in the form: cut'*x >= rhs.
  // We expect that the cut is violated by the current relaxation
  // cut'*xstart < rhs
  void add_cut(i_t n, const sparse_vector_t<i_t, f_t>& cut, f_t rhs);

  void score_cuts(std::vector<f_t>& x_relax);

  // We return the cuts in the form best_cuts*x <= best_rhs
  i_t get_best_cuts(csr_matrix_t<i_t, f_t>& best_cuts, std::vector<f_t>& best_rhs);

  void age_cuts();

  void drop_cuts();

  i_t pool_size() const { return cut_storage_.m; }

 private:
  f_t cut_distance(i_t row, const std::vector<f_t>& x, f_t& cut_violation, f_t &cut_norm);
  f_t cut_density(i_t row);
  f_t cut_orthogonality(i_t i, i_t j);

  i_t original_vars_;
  const simplex_solver_settings_t<i_t, f_t>& settings_;

  csr_matrix_t<i_t, f_t> cut_storage_;
  std::vector<f_t> rhs_storage_;
  std::vector<i_t> cut_age_;

  i_t scored_cuts_;
  std::vector<f_t> cut_distances_;
  std::vector<f_t> cut_norms_;
  std::vector<f_t> cut_orthogonality_;
  std::vector<f_t> cut_scores_;
  std::vector<i_t> best_cuts_;
};

template <typename i_t, typename f_t>
class cut_generation_t {
 public:
  cut_generation_t(cut_pool_t<i_t, f_t>& cut_pool) : cut_pool_(cut_pool) {}


  void generate_cuts(const lp_problem_t<i_t, f_t>& lp,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     csc_matrix_t<i_t, f_t>& Arow,
                     const std::vector<variable_type_t>& var_types,
                     basis_update_mpf_t<i_t, f_t>& basis_update,
                     const std::vector<f_t>& xstar,
                     const std::vector<i_t>& basic_list,
                     const std::vector<i_t>& nonbasic_list);
 private:

  void generate_gomory_cuts(const lp_problem_t<i_t, f_t>& lp,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            csc_matrix_t<i_t, f_t>& Arow,
                            const std::vector<variable_type_t>& var_types,
                            basis_update_mpf_t<i_t, f_t>& basis_update,
                            const std::vector<f_t>& xstar,
                            const std::vector<i_t>& basic_list,
                            const std::vector<i_t>& nonbasic_list);

  void generate_mir_cuts(const lp_problem_t<i_t, f_t>& lp,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         csc_matrix_t<i_t, f_t>& Arow,
                         const std::vector<variable_type_t>& var_types,
                         const std::vector<f_t>& xstar);
  cut_pool_t<i_t, f_t>& cut_pool_;
};

template <typename i_t, typename f_t>
class mixed_integer_gomory_base_inequality_t {
 public:
  mixed_integer_gomory_base_inequality_t(const lp_problem_t<i_t, f_t>& lp,
                                         basis_update_mpf_t<i_t, f_t>& basis_update,
                                         const std::vector<i_t> nonbasic_list)
    : b_bar_(lp.num_rows, 0.0),
      nonbasic_mark_(lp.num_cols, 0),
      x_workspace_(lp.num_cols, 0.0),
      x_mark_(lp.num_cols, 0)
  {
    basis_update.b_solve(lp.rhs, b_bar_);
    for (i_t j : nonbasic_list) {
      nonbasic_mark_[j] = 1;
    }
  }

  // Generates the base inequalities: C*x == d that will be turned into cuts
  i_t generate_base_inequality(const lp_problem_t<i_t, f_t>& lp,
                               const simplex_solver_settings_t<i_t, f_t>& settings,
                               csc_matrix_t<i_t, f_t>& Arow,
                               const std::vector<variable_type_t>& var_types,
                               basis_update_mpf_t<i_t, f_t>& basis_update,
                               const std::vector<f_t>& xstar,
                               const std::vector<i_t>& basic_list,
                               const std::vector<i_t>& nonbasic_list,
                               i_t i,
                               sparse_vector_t<i_t, f_t>& inequality,
                               f_t& inequality_rhs);

 private:
  std::vector<f_t> b_bar_;
  std::vector<f_t> nonbasic_mark_;
  std::vector<f_t> x_workspace_;
  std::vector<i_t> x_mark_;
};

template <typename i_t, typename f_t>
class mixed_integer_rounding_cut_t {
 public:
  mixed_integer_rounding_cut_t(i_t num_vars, const simplex_solver_settings_t<i_t, f_t>& settings)
    : num_vars_(num_vars),
      settings_(settings),
      x_workspace_(num_vars, 0.0),
      x_mark_(num_vars, 0),
      has_lower_(num_vars, 0),
      has_upper_(num_vars, 0),
      needs_complement_(false)
  {
  }

  void initialize(const lp_problem_t<i_t, f_t>& lp, const std::vector<f_t>& xstar);

  i_t generate_cut(const sparse_vector_t<i_t, f_t>& a,
                   f_t beta,
                   const std::vector<f_t>& upper_bounds,
                   const std::vector<f_t>& lower_bounds,
                   const std::vector<variable_type_t>& var_types,
                   sparse_vector_t<i_t, f_t>& cut,
                   f_t& cut_rhs);

 private:
  i_t num_vars_;
  const simplex_solver_settings_t<i_t, f_t>& settings_;
  std::vector<f_t> x_workspace_;
  std::vector<i_t> x_mark_;
  std::vector<i_t> has_lower_;
  std::vector<i_t> has_upper_;
  bool needs_complement_;
};

template <typename i_t, typename f_t>
i_t add_cuts(const simplex_solver_settings_t<i_t, f_t>& settings,
             const csr_matrix_t<i_t, f_t>& cuts,
             const std::vector<f_t>& cut_rhs,
             lp_problem_t<i_t, f_t>& lp,
             lp_solution_t<i_t, f_t>& solution,
             basis_update_mpf_t<i_t, f_t>& basis_update,
             std::vector<i_t>& basic_list,
             std::vector<i_t>& nonbasic_list,
             std::vector<variable_status_t>& vstatus,
             std::vector<f_t>& edge_norms);

template <typename i_t, typename f_t>
void remove_cuts(lp_problem_t<i_t, f_t>& lp,
                 const simplex_solver_settings_t<i_t, f_t>& settings,
                 csc_matrix_t<i_t, f_t>& Arow,
                 i_t original_rows,
                 std::vector<variable_type_t>& var_types,
                 std::vector<variable_status_t>& vstatus,
                 std::vector<f_t>& x,
                 std::vector<f_t>& y,
                 std::vector<f_t>& z,
                 std::vector<i_t>& basic_list,
                 std::vector<i_t>& nonbasic_list,
                 basis_update_mpf_t<i_t, f_t>& basis_update);

}

