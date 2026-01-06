/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/cuts.hpp>
#include <dual_simplex/dense_matrix.hpp>


namespace cuopt::linear_programming::dual_simplex {


template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::add_cut(cut_type_t cut_type, const sparse_vector_t<i_t, f_t>& cut, f_t rhs)
{
  // TODO: Need to deduplicate cuts and only add if the cut is not already in the pool

  for (i_t p = 0; p < cut.i.size(); p++) {
    const i_t j = cut.i[p];
    if (j >= original_vars_) {
      settings_.log.printf(
        "Cut has variable %d that is greater than original_vars_ %d\n", j, original_vars_);
      return;
    }
  }

  sparse_vector_t<i_t, f_t> cut_squeezed;
  cut.squeeze(cut_squeezed);
  cut_storage_.append_row(cut_squeezed);
  //settings_.log.printf("Added cut %d to pool\n", cut_storage_.m - 1);
  rhs_storage_.push_back(rhs);
  cut_type_.push_back(cut_type);
  cut_age_.push_back(0);
}


template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_distance(i_t row, const std::vector<f_t>& x, f_t& cut_violation, f_t &cut_norm)
{
  const i_t row_start = cut_storage_.row_start[row];
  const i_t row_end = cut_storage_.row_start[row + 1];
  f_t cut_x = 0.0;
  f_t dot = 0.0;
  for (i_t p = row_start; p < row_end; p++) {
    const i_t j = cut_storage_.j[p];
    const f_t cut_coeff = cut_storage_.x[p];
    cut_x += cut_coeff * x[j];
    dot += cut_coeff * cut_coeff;
  }
  cut_violation = rhs_storage_[row] - cut_x;
  cut_norm = std::sqrt(dot);
  const f_t distance = cut_violation / cut_norm;
  return distance;
}

template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_density(i_t row)
{
  const i_t row_start = cut_storage_.row_start[row];
  const i_t row_end = cut_storage_.row_start[row + 1];
  const i_t cut_nz = row_end - row_start;
  const i_t original_vars = original_vars_;
  return static_cast<f_t>(cut_nz) / original_vars;
}

template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_orthogonality(i_t i,  i_t j)
{
  const i_t i_start = cut_storage_.row_start[i];
  const i_t i_end = cut_storage_.row_start[i + 1];
  const i_t i_nz = i_end - i_start;
  const i_t j_start = cut_storage_.row_start[j];
  const i_t j_end = cut_storage_.row_start[j + 1];
  const i_t j_nz = j_end - j_start;

  f_t dot = sparse_dot(cut_storage_.j.data() + i_start, cut_storage_.x.data() + i_start, i_nz,
                       cut_storage_.j.data() + j_start, cut_storage_.x.data() + j_start, j_nz);

  f_t norm_i = cut_norms_[i];
  f_t norm_j = cut_norms_[j];
  return 1.0 - std::abs(dot) / (norm_i * norm_j);
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::score_cuts(std::vector<f_t>& x_relax)
{
  const f_t weight_distance = 1.0;
  const f_t weight_orthogonality = 1.0;
  cut_distances_.resize(cut_storage_.m, 0.0);
  cut_norms_.resize(cut_storage_.m, 0.0);
  cut_orthogonality_.resize(cut_storage_.m, 1);
  cut_scores_.resize(cut_storage_.m, 0.0);
  for (i_t i = 0; i < cut_storage_.m; i++) {
    f_t violation;
    cut_distances_[i] = cut_distance(i, x_relax, violation, cut_norms_[i]);
    cut_scores_[i] = weight_distance * cut_distances_[i]  + weight_orthogonality * cut_orthogonality_[i];
    //settings_.log.printf("Cut %d distance %e violation %e orthogonality %e score %e\n", i, cut_distances_[i], violation, cut_orthogonality_[i], cut_scores_[i]);
  }

  std::vector<i_t> sorted_indices(cut_storage_.m);
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(), [&](i_t a, i_t b) {
    return cut_scores_[a] > cut_scores_[b];
  });

  std::vector<i_t> indices;
  indices.reserve(sorted_indices.size());


  const i_t max_cuts = 2000;
  const f_t min_orthogonality = 0.5;
  const f_t min_cut_distance = 1e-4;
  best_cuts_.reserve(std::min(max_cuts, cut_storage_.m));
  best_cuts_.clear();
  scored_cuts_ = 0;

  while (scored_cuts_ < max_cuts && !sorted_indices.empty()) {
    const i_t i = sorted_indices[0];

    if (cut_distances_[i] <= min_cut_distance) {
        break;
    }

    if (cut_age_[i] > 0) {
        settings_.log.printf("Adding cut with age %d\n", cut_age_[i]);
    }
    //settings_.log.printf("Scored cuts %d. Adding cut %d score %e\n", scored_cuts_, i, cut_scores_[i]);

    best_cuts_.push_back(i);
    scored_cuts_++;

    // Recompute the orthogonality for the remaining cuts
    for (i_t k = 1; k < sorted_indices.size(); k++) {
      const i_t j = sorted_indices[k];
      cut_orthogonality_[j] = std::min(cut_orthogonality_[j], cut_orthogonality(i, j));
      if (cut_orthogonality_[j] >= min_orthogonality) {
        indices.push_back(j);
        cut_scores_[j] = weight_distance * cut_distances_[j] + weight_orthogonality * cut_orthogonality_[j];
        //settings_.log.printf("Recomputed cut %d score %e\n", j, cut_scores_[j]);
      }
    }

    sorted_indices = indices;
    indices.clear();
    //settings_.log.printf("Sorting %d cuts\n", sorted_indices.size());

    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](i_t a, i_t b) {
        return cut_scores_[a] > cut_scores_[b];
    });
  }
}

template <typename i_t, typename f_t>
i_t cut_pool_t<i_t, f_t>::get_best_cuts(csr_matrix_t<i_t, f_t>& best_cuts, std::vector<f_t>& best_rhs, std::vector<cut_type_t>& best_cut_types)
{
  best_cuts.m = 0;
  best_cuts.n = original_vars_;
  best_cuts.row_start.clear();
  best_cuts.j.clear();
  best_cuts.x.clear();
  best_cuts.row_start.reserve(scored_cuts_ + 1);
  best_cuts.row_start.push_back(0);

  for (i_t i: best_cuts_) {
    sparse_vector_t<i_t, f_t> cut(cut_storage_, i);
    cut.negate();
    best_cuts.append_row(cut);
    //settings_.log.printf("Best cuts nz %d\n", best_cuts.row_start[best_cuts.m]);
    best_rhs.push_back(-rhs_storage_[i]);
    best_cut_types.push_back(cut_type_[i]);
  }

  return static_cast<i_t>(best_cuts_.size());
}


template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::age_cuts()
{
  for (i_t i = 0; i < cut_age_.size(); i++) {
    cut_age_[i]++;
  }
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::drop_cuts()
{
   // TODO: Implement this
}

template <typename i_t, typename f_t>
knapsack_generation_t<i_t, f_t>::knapsack_generation_t(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csc_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types)
{
  knapsack_constraints_.reserve(lp.num_rows);

  is_slack_.resize(lp.num_cols, 0);
  for (i_t j : new_slacks) {
    is_slack_[j] = 1;
  }

  for (i_t i = 0; i < lp.num_rows; i++) {
    const i_t row_start = Arow.col_start[i];
    const i_t row_end   = Arow.col_start[i + 1];
    if (row_end - row_start < 3) { continue; }
    bool is_knapsack    = true;
    f_t sum_pos         = 0.0;
    //printf("i %d ", i);
    for (i_t p = row_start; p < row_end; p++) {
      const i_t j = Arow.i[p];
      if (is_slack_[j]) { continue; }
      const f_t aj = Arow.x[p];
      //printf(" j %d (%e < %e) aj %e\n", j, lp.lower[j], lp.upper[j], aj);
      if (std::abs(aj - std::round(aj)) > settings.integer_tol) {
        is_knapsack = false;
        break;
      }
      if (var_types[j] != variable_type_t::INTEGER || lp.lower[j] != 0.0 || lp.upper[j] != 1.0) {
        is_knapsack = false;
        break;
      }
      if (aj < 0.0) {
        is_knapsack = false;
        break;
      }
      sum_pos += aj;
    }
   // printf("sum_pos %e\n", sum_pos);

    if (is_knapsack) {
      const f_t beta = lp.rhs[i];
      printf("Knapsack constraint %d beta %e sum_pos %e\n", i, beta, sum_pos);
      if (std::abs(beta - std::round(beta)) <= settings.integer_tol) {
        if (beta >= 0.0 && beta <= sum_pos) {
          knapsack_constraints_.push_back(i);
        }
      }
    }
  }

  i_t num_knapsack_constraints = knapsack_constraints_.size();
  settings.log.printf("Number of knapsack constraints %d\n", num_knapsack_constraints);
}

template <typename i_t, typename f_t>
i_t knapsack_generation_t<i_t, f_t>::generate_knapsack_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csc_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  i_t knapsack_row,
  sparse_vector_t<i_t, f_t>& cut,
  f_t& cut_rhs)
{
  // Get the row associated with the knapsack constraint
  sparse_vector_t<i_t, f_t> knapsack_inequality(Arow, knapsack_row);
  f_t knapsack_rhs = lp.rhs[knapsack_row];

  // Remove the slacks from the inequality
  f_t seperation_rhs = 0.0;
  printf(" Knapsack : ");
  for (i_t k = 0; k < knapsack_inequality.i.size(); k++) {
    const i_t j = knapsack_inequality.i[k];
    if (is_slack_[j]) {
      knapsack_inequality.x[k] = 0.0;
    } else {
      printf(" %g x%d +", knapsack_inequality.x[k], j);
      seperation_rhs += knapsack_inequality.x[k];
    }
  }
  printf(" <= %g\n", knapsack_rhs);
  seperation_rhs -= (knapsack_rhs + 1);

  printf("\t");
  for (i_t k = 0; k < knapsack_inequality.i.size(); k++) {
    const i_t j = knapsack_inequality.i[k];
    if (!is_slack_[j]) {
        if (std::abs(xstar[j]) > 1e-3) {
          printf("x_relax[%d]= %g ", j, xstar[j]);
        }
    }
  }
  printf("\n");

  printf("seperation_rhs %g\n", seperation_rhs);
  if (seperation_rhs <= 0.0) { return -1; }

  std::vector<f_t> values;
  values.resize(knapsack_inequality.i.size() - 1);
  std::vector<f_t> weights;
  weights.resize(knapsack_inequality.i.size() - 1);
  i_t h                  = 0;
  f_t objective_constant = 0.0;
  for (i_t k = 0; k < knapsack_inequality.i.size(); k++) {
    const i_t j = knapsack_inequality.i[k];
    if (!is_slack_[j]) {
      const f_t vj = 1.0 - xstar[j];
      objective_constant += vj;
      values[h]  = vj;
      weights[h] = knapsack_inequality.x[k];
      h++;
    }
  }
  std::vector<f_t> solution;
  solution.resize(knapsack_inequality.i.size() - 1);

  printf("Calling solve_knapsack_problem\n");
  f_t objective = solve_knapsack_problem(values, weights, seperation_rhs, solution);
  if (objective != objective) { return -1; }
  printf("objective %e objective_constant %e\n", objective, objective_constant);

  f_t seperation_value = -objective + objective_constant;
  printf("seperation_value %e\n", seperation_value);
  const f_t tol = 1e-6;
  if (seperation_value >= 1.0 - tol) { return -1; }

  i_t cover_size = 0;
  for (i_t k = 0; k < solution.size(); k++) {
    if (solution[k] == 0.0) { cover_size++; }
  }

  cut.i.clear();
  cut.x.clear();
  cut.i.reserve(cover_size);
  cut.x.reserve(cover_size);

  h = 0;
  for (i_t k = 0; k < knapsack_inequality.i.size(); k++) {
    const i_t j = knapsack_inequality.i[k];
    if (!is_slack_[j]) {
      if (solution[h] == 0.0) {
        cut.i.push_back(j);
        cut.x.push_back(-1.0);
      }
      h++;
    }
  }
  cut_rhs = -cover_size + 1;
  cut.sort();

  // The cut is in the form: - sum_{j in cover} x_j >= -cover_size + 1
  // Which is equivalent to: sum_{j in cover} x_j <= cover_size - 1

  // Verify the cut is violated
  f_t dot = cut.dot(xstar);
  f_t violation = dot - cut_rhs;
  printf("Knapsack cut %d violation %e < 0\n", knapsack_row, violation);

  if (violation <= tol) { return -1; }
  return 0;
}

template <typename i_t, typename f_t>
f_t knapsack_generation_t<i_t, f_t>::greedy_knapsack_problem(const std::vector<f_t>& values,
                                                             const std::vector<f_t>& weights,
                                                             f_t rhs,
                                                             std::vector<f_t>& solution)
{
  i_t n = weights.size();
  solution.assign(n, 0.0);

  // Build permutation
  std::vector<i_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0);

  std::vector<f_t> ratios;
  ratios.resize(n);
  for (i_t i = 0; i < n; i++) {
    ratios[i] = values[i] / weights[i];
  }

  // Sort by value / weight ratio
  std::sort(perm.begin(), perm.end(), [&](i_t i, i_t j) { return ratios[i] > ratios[j]; });

  // Greedy select items with the best value / weight ratio until the remaining capacity is exhausted
  f_t remaining   = rhs;
  f_t total_value = 0.0;

  for (i_t j : perm) {
    if (weights[j] <= remaining) {
      solution[j] = 1.0;
      remaining -= weights[j];
      total_value += values[j];
    }
  }

  // Best single-item fallback
  f_t best_single_value = 0.0;
  i_t best_single_idx   = -1;

  for (i_t j = 0; j < n; ++j) {
    if (weights[j] <= rhs && values[j] > best_single_value) {
      best_single_value = values[j];
      best_single_idx   = j;
    }
  }

  if (best_single_value > total_value) {
    solution.assign(n, 0.0);
    solution[best_single_idx] = 1.0;
    return best_single_value;
  }

  return total_value;
}

template <typename i_t, typename f_t>
f_t knapsack_generation_t<i_t, f_t>::solve_knapsack_problem(const std::vector<f_t>& values,
                                                            const std::vector<f_t>& weights,
                                                            f_t rhs,
                                                            std::vector<f_t>& solution)
{
  // Solve the knapsack problem
  // maximize sum_{j=0}^n values[j] * solution[j]
  // subject to sum_{j=0}^n weights[j] * solution[j] <= rhs
  // values: values of the items
  // weights: weights of the items
  // return the value of the solution

  // Using approximate dynamic programming

  i_t n = weights.size();
  f_t objective = std::numeric_limits<f_t>::quiet_NaN();

  // Compute the maximum value
  f_t vmax = *std::max_element(values.begin(), values.end());

  // Check if all the values are integers
  bool all_integers = true;
  const f_t integer_tol = 1e-5;
  for (i_t j = 0; j < n; j++) {
    if (std::abs(values[j] - std::round(values[j])) > integer_tol) {
        all_integers = false;
        break;
    }
  }

  printf("all_integers %d\n", all_integers);

  // Compute the scaling factor and comptue the scaled integer values
  f_t scale = 1.0;
  std::vector<i_t> scaled_values(n);
  if (all_integers) {
    for (i_t j = 0; j < n; j++) {
      scaled_values[j] = static_cast<i_t>(std::floor(values[j]));
    }
  } else {
    const f_t epsilon = 0.1;
    scale             = epsilon * vmax / static_cast<f_t>(n);
    if (scale <= 0.0) { return std::numeric_limits<f_t>::quiet_NaN(); }
    printf("scale %g epsilon %g vmax %g n %d\n", scale, epsilon, vmax, n);
    for (i_t i = 0; i < n; ++i) {
      scaled_values[i] = static_cast<i_t>(std::floor(values[i] / scale));
      //printf("scaled_values[%d] %d values[%d] %g\n", i, scaled_values[i], i, values[i]);
    }
  }

  i_t sum_value = std::accumulate(scaled_values.begin(), scaled_values.end(), 0);
  const i_t INT_INF = std::numeric_limits<i_t>::max() / 2;
  printf("sum value %d\n", sum_value);
  const i_t max_size = 10000;
  if (sum_value <= 0.0 || sum_value >= max_size) {
    printf("sum value %d is negative or too large using greedy solution\n", sum_value);
    return greedy_knapsack_problem(values, weights, rhs, solution);
  }

  // dp(j, v) = minimum weight using first j items to get value v
  dense_matrix_t<i_t, i_t> dp(n + 1, sum_value + 1, INT_INF);
  dense_matrix_t<i_t, uint8_t> take(n + 1, sum_value + 1, 0);
  dp(0, 0) = 0;
  printf("start dp\n");

  // 4. Dynamic programming
  for (int j = 1; j <= n; ++j) {
    for (int v = 0; v <= sum_value; ++v) {
      // Do not take item i-1
      dp(j, v) = dp(j - 1, v);

      // Take item j-1 if possible
      if (v >= scaled_values[j - 1]) {
        i_t candidate = dp(j - 1, v - scaled_values[j - 1]) + static_cast<i_t>(std::floor(weights[j - 1]));
        if (candidate < dp(j, v)) {
          dp(j, v)   = candidate;
          take(j, v) = 1;
        }
      }
    }
  }

  // 5. Find best achievable value within capacity
  i_t best_value = 0;
  for (i_t v = 0; v <= sum_value; ++v) {
    if (dp(n, v) <= rhs) { best_value = v; }
  }

  // 6. Backtrack to recover solution
  i_t v = best_value;
  for (i_t j = n; j >= 1; --j) {
    if (take(j, v)) {
      solution[j - 1] = 1.0;
      v -= scaled_values[j - 1];
    } else {
      solution[j - 1] = 0.0;
    }
  }

  objective = best_value * scale;
  return objective;
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_cuts(const lp_problem_t<i_t, f_t>& lp,
                                               const simplex_solver_settings_t<i_t, f_t>& settings,
                                               csc_matrix_t<i_t, f_t>& Arow,
                                               const std::vector<i_t>& new_slacks,
                                               const std::vector<variable_type_t>& var_types,
                                               basis_update_mpf_t<i_t, f_t>& basis_update,
                                               const std::vector<f_t>& xstar,
                                               const std::vector<i_t>& basic_list,
                                               const std::vector<i_t>& nonbasic_list)
{
  // Generate Gomory Cuts
  generate_gomory_cuts(
    lp, settings, Arow, new_slacks, var_types, basis_update, xstar, basic_list, nonbasic_list);
  //settings.log.printf("Generated Gomory cuts\n");

  // Generate Knapsack cuts
  generate_knapsack_cuts(lp, settings, Arow, new_slacks, var_types, xstar);
  //settings.log.printf("Generated Knapsack cuts\n");

 // Generate MIR cuts
 // generate_mir_cuts(lp, settings, Arow, var_types, xstar);
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_knapsack_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csc_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar)
{
  if (knapsack_generation_.num_knapsack_constraints() > 0) {
    for (i_t knapsack_row : knapsack_generation_.get_knapsack_constraints()) {
      sparse_vector_t<i_t, f_t> cut(lp.num_cols, 0);
      f_t cut_rhs;
      i_t knapsack_status = knapsack_generation_.generate_knapsack_cuts(
        lp, settings, Arow, new_slacks, var_types, xstar, knapsack_row, cut, cut_rhs);
      if (knapsack_status == 0) {
        settings.log.printf("Adding Knapsack cut %d\n", knapsack_row);
        cut_pool_.add_cut(cut_type_t::KNAPSACK, cut, cut_rhs);
      } else {
        settings.log.printf("Knapsack cut %d is not violated. Skipping\n", knapsack_row);
      }
    }
  }
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_mir_cuts(const lp_problem_t<i_t, f_t>& lp,
                                                   const simplex_solver_settings_t<i_t, f_t>& settings,
                                                   csc_matrix_t<i_t, f_t>& Arow,
                                                   const std::vector<i_t>& new_slacks,
                                                   const std::vector<variable_type_t>& var_types,
                                                   const std::vector<f_t>& xstar)
{
  mixed_integer_rounding_cut_t<i_t, f_t> mir(lp.num_cols, settings);
  mir.initialize(lp, new_slacks, xstar);

  for (i_t i = 0; i < lp.num_rows; i++) {
    sparse_vector_t<i_t, f_t> inequality(Arow, i);
    f_t inequality_rhs = lp.rhs[i];

    const i_t row_start = Arow.col_start[i];
    const i_t row_end = Arow.col_start[i + 1];
    i_t last_slack = -1;
    for (i_t p = row_start; p < row_end; p++) {
      const i_t j = Arow.i[p];
      const f_t a = Arow.x[p];
      if (var_types[j] == variable_type_t::CONTINUOUS && a == 1.0 && lp.lower[j] == 0.0) {
        last_slack = j;
      }
    }

    if (last_slack != -1) {
        // Remove the slack from the equality to get an inequality
        for (i_t k = 0; k < inequality.i.size(); k++) {
          const i_t j = inequality.i[k];
          if (j == last_slack) {
            inequality.x[k] = 0.0;
          }
        }

        // inequaility'*x <= inequality_rhs
        // But for MIR we need: inequality'*x >= inequality_rhs
        inequality_rhs *= -1;
        inequality.negate();

        sparse_vector_t<i_t, f_t> cut(lp.num_cols, 0);
        f_t cut_rhs;
        i_t mir_status = mir.generate_cut(inequality, inequality_rhs, lp.upper, lp.lower, var_types, cut, cut_rhs);
        if (mir_status == 0) {
          f_t dot = 0.0;
          f_t cut_norm = 0.0;
          for (i_t k = 0; k < cut.i.size(); k++) {
            const i_t jj = cut.i[k];
            const f_t aj = cut.x[k];
            dot += aj * xstar[jj];
            cut_norm += aj * aj;
          }
          if (dot >= cut_rhs) {
            continue;
          }
        }

        settings.log.printf("Adding MIR cut %d\n", i);
        cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_ROUNDING, cut, cut_rhs);
    }
  }
}


template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_gomory_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csc_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  basis_update_mpf_t<i_t, f_t>& basis_update,
  const std::vector<f_t>& xstar,
  const std::vector<i_t>& basic_list,
  const std::vector<i_t>& nonbasic_list)
{
  mixed_integer_gomory_base_inequality_t<i_t, f_t> gomory(lp, basis_update, nonbasic_list);
  mixed_integer_rounding_cut_t<i_t, f_t> mir(lp.num_cols, settings);

  mir.initialize(lp, new_slacks, xstar);

  for (i_t i = 0; i < lp.num_rows; i++) {
    sparse_vector_t<i_t, f_t> inequality(lp.num_cols, 0);
    f_t inequality_rhs;
    const i_t j = basic_list[i];
    if (var_types[j] != variable_type_t::INTEGER) { continue; }
    const f_t x_j = xstar[j];
    if (std::abs(x_j - std::round(x_j)) < settings.integer_tol) { continue; }
    i_t gomory_status = gomory.generate_base_inequality(lp,
                                                        settings,
                                                        Arow,
                                                        var_types,
                                                        basis_update,
                                                        xstar,
                                                        basic_list,
                                                        nonbasic_list,
                                                        i,
                                                        inequality,
                                                        inequality_rhs);
    if (gomory_status == 0) {
      // Given the base inequality, generate a MIR cut
      sparse_vector_t<i_t, f_t> cut_A(lp.num_cols, 0);
      f_t cut_A_rhs;
      i_t mir_status =
        mir.generate_cut(inequality, inequality_rhs, lp.upper, lp.lower, var_types, cut_A, cut_A_rhs);
      bool A_valid = false;
      f_t cut_A_distance = 0.0;
      if (mir_status == 0) {
        if (cut_A.i.size() == 0) {
          settings.log.printf("No coefficients in cut A\n");
          continue;
        }
        mir.substitute_slacks(lp, Arow, cut_A, cut_A_rhs);
        if (cut_A.i.size() == 0) {
          settings.log.printf("No coefficients in cut A after substituting slacks\n");
          A_valid = false;
        } else {
          // Check that the cut is violated
          f_t dot      = cut_A.dot(xstar);
          f_t cut_norm = cut_A.norm2_squared();
          if (dot >= cut_A_rhs) {
            settings.log.printf("Cut %d is not violated. Skipping\n", i);
            continue;
          }
          cut_A_distance = (cut_A_rhs - dot) / std::sqrt(cut_norm);
          A_valid        = true;
        }
        //cut_pool_.add_cut(lp.num_cols, cut, cut_rhs);
      }

      // Negate the base inequality
      inequality.negate();
      inequality_rhs *= -1;

      sparse_vector_t<i_t, f_t> cut_B(lp.num_cols, 0);
      f_t cut_B_rhs;

      mir_status =
        mir.generate_cut(inequality, inequality_rhs, lp.upper, lp.lower, var_types, cut_B, cut_B_rhs);
      bool B_valid = false;
      f_t cut_B_distance = 0.0;
      if (mir_status == 0) {
        if (cut_B.i.size() == 0) {
          settings.log.printf("No coefficients in cut B\n");
          continue;
        }
        mir.substitute_slacks(lp, Arow, cut_B, cut_B_rhs);
        if (cut_B.i.size() == 0) {
          settings.log.printf("No coefficients in cut B after substituting slacks\n");
          B_valid = false;
        } else {
          // Check that the cut is violated
          f_t dot      = cut_B.dot(xstar);
          f_t cut_norm = cut_B.norm2_squared();
          if (dot >= cut_B_rhs) {
            settings.log.printf("Cut %d is not violated. Skipping\n", i);
            continue;
          }
          cut_B_distance = (cut_B_rhs - dot) / std::sqrt(cut_norm);
          B_valid        = true;
        }
        // cut_pool_.add_cut(lp.num_cols, cut_B, cut_B_rhs);
      }

      if ((cut_A_distance > cut_B_distance) && A_valid) {
        //printf("Adding Gomory cut A: nz %d distance %e valid %d\n", cut_A.i.size(), cut_A_distance, A_valid);
        cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_GOMORY, cut_A, cut_A_rhs);
      } else if (B_valid) {
        //printf("Adding Gomory cut B: nz %d distance %e valid %d\n", cut_B.i.size(), cut_B_distance, B_valid);
        cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_GOMORY, cut_B, cut_B_rhs);
      }
    }
  }
}

template <typename i_t, typename f_t>
i_t mixed_integer_gomory_base_inequality_t<i_t, f_t>::generate_base_inequality(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csc_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  basis_update_mpf_t<i_t, f_t>& basis_update,
  const std::vector<f_t>& xstar,
  const std::vector<i_t>& basic_list,
  const std::vector<i_t>& nonbasic_list,
  i_t i,
  sparse_vector_t<i_t, f_t>& inequality,
  f_t& inequality_rhs)
{
  // Let's look for Gomory cuts
    const i_t j = basic_list[i];
    if (var_types[j] != variable_type_t::INTEGER) { return -1; }
    const f_t x_j = xstar[j];
    if (std::abs(x_j - std::round(x_j)) < settings.integer_tol) { return -1; }
#ifdef PRINT_CUT_INFO
    settings_.log.printf("Generating cut for variable %d relaxed value %e row %d\n", j, x_j, i);
#endif
#ifdef PRINT_BASIS
    for (i_t h = 0; h < basic_list.size(); h++) {
      settings_.log.printf("basic_list[%d] = %d\n", h, basic_list[h]);
    }
#endif

    // Solve B^T u_bar = e_i
    sparse_vector_t<i_t, f_t> e_i(lp.num_rows, 1);
    e_i.i[0] = i;
    e_i.x[0] = 1.0;
    sparse_vector_t<i_t, f_t> u_bar(lp.num_rows, 0);
    basis_update.b_transpose_solve(e_i, u_bar);


#ifdef CHECK_B_TRANSPOSE_SOLVE
    std::vector<f_t> u_bar_dense(lp.num_rows);
    u_bar.to_dense(u_bar_dense);

    std::vector<f_t> BTu_bar(lp.num_rows);
    b_transpose_multiply(lp, basic_list, u_bar_dense, BTu_bar);
    for (i_t k = 0; k < lp.num_rows; k++) {
      if (k == i) {
        if (std::abs(BTu_bar[k] - 1.0) > 1e-6) {
          settings_.log.printf("BTu_bar[%d] = %e i %d\n", k, BTu_bar[k], i);
          exit(1);
        }
      } else {
        if (std::abs(BTu_bar[k]) > 1e-6) {
          settings_.log.printf("BTu_bar[%d] = %e i %d\n", k, BTu_bar[k], i);
          exit(1);
        }
      }
    }
#endif

    // Compute a_bar = N^T u_bar
    // TODO: This is similar to a function in phase2 of dual simplex. See if it can be reused.
    const i_t nz_ubar = u_bar.i.size();
    std::vector<i_t> abar_indices;
    abar_indices.reserve(nz_ubar);
    for (i_t k = 0; k < nz_ubar; k++) {
      const i_t ii        = u_bar.i[k];
      const f_t u_bar_i   = u_bar.x[k];
      const i_t row_start = Arow.col_start[ii];
      const i_t row_end   = Arow.col_start[ii + 1];
      for (i_t p = row_start; p < row_end; p++) {
        const i_t jj = Arow.i[p];
        if (nonbasic_mark_[jj] == 1) {
          x_workspace_[jj] += u_bar_i * Arow.x[p];
          if (!x_mark_[jj]) {
            x_mark_[jj] = 1;
            abar_indices.push_back(jj);
          }
        }
      }
    }

    sparse_vector_t<i_t, f_t> a_bar(lp.num_cols, abar_indices.size() + 1);
    for (i_t k = 0; k < abar_indices.size(); k++) {
      const i_t jj = abar_indices[k];
      a_bar.i[k]   = jj;
      a_bar.x[k]   = x_workspace_[jj];
    }

    // Clear the workspace
    for (i_t jj : abar_indices) {
      x_workspace_[jj] = 0.0;
      x_mark_[jj]      = 0;
    }
    abar_indices.clear();

    // We should now have the base inequality
    // x_j + a_bar^T x_N >= b_bar_i
    // We add x_j into a_bar so that everything is in a single sparse_vector_t
    a_bar.i[a_bar.i.size() - 1] = j;
    a_bar.x[a_bar.x.size() - 1] = 1.0;

#ifdef CHECK_A_BAR_DENSE_DOT
    std::vector<f_t> a_bar_dense(lp.num_cols);
    a_bar.to_dense(a_bar_dense);

    f_t a_bar_dense_dot = dot<i_t, f_t>(a_bar_dense, xstar);
    if (std::abs(a_bar_dense_dot - b_bar[i]) > 1e-6) {
      settings_.log.printf("a_bar_dense_dot = %e b_bar[%d] = %e\n", a_bar_dense_dot, i, b_bar[i]);
      settings_.log.printf("x_j %e b_bar_i %e\n", x_j, b_bar[i]);
      exit(1);
    }
#endif

    // We have that x_j + a_bar^T x_N == b_bar_i
    // So x_j + a_bar^T x_N >= b_bar_i
    // And x_j + a_bar^T x_N <= b_bar_i
    // Or -x_j - a_bar^T x_N >= -b_bar_i

#ifdef PRINT_CUT
    {
      settings_.log.printf("Cut %d\n", i);
      for (i_t k = 0; k < a_bar.i.size(); k++) {
        const i_t jj = a_bar.i[k];
        const f_t aj = a_bar.x[k];
        settings_.log.printf("(%d, %e) ", jj, aj);
      }
      settings_.log.printf("\nEnd cut %d b_bar[%d] = %e\n", i, b_bar[i]);
    }
#endif

    // Skip cuts that are shallow
    const f_t shallow_tol = 1e-2;
    if (std::abs(x_j - std::round(x_j)) < shallow_tol) {
      //settings_.log.printf("Skipping shallow cut %d. b_bar[%d] = %e x_j %e\n", i, i, b_bar[i], x_j);
      return -1;
    }

    const f_t f_val = b_bar_[i] - std::floor(b_bar_[i]);
    if (f_val < 0.01 || f_val > 0.99) {
      //settings_.log.printf("Skipping cut %d. b_bar[%d] = %e f_val %e\n", i, i, b_bar[i], f_val);
      return -1;
    }

#ifdef PRINT_BASE_INEQUALITY
    // Print out the base inequality
    for (i_t k = 0; k < a_bar.i.size(); k++) {
      const i_t jj = a_bar.i[k];
      const f_t aj = a_bar.x[k];
      settings_.log.printf("a_bar[%d] = %e\n", k, aj);
    }
    settings_.log.printf("b_bar[%d] = %e\n", i, b_bar[i]);
#endif

    inequality = a_bar;
    inequality_rhs = b_bar_[i];

    return 0;
}

template <typename i_t, typename f_t>
void mixed_integer_rounding_cut_t<i_t, f_t>::initialize(const lp_problem_t<i_t, f_t>& lp,
                                                        const std::vector<i_t>& new_slacks,
                                                        const std::vector<f_t>& xstar)
{

  if (lp.num_cols != num_vars_) {
    num_vars_ = lp.num_cols;
    x_workspace_.resize(num_vars_, 0.0);
    x_mark_.resize(num_vars_, 0);
    has_lower_.resize(num_vars_, 0);
    has_upper_.resize(num_vars_, 0);
  }

  is_slack_.clear();
  is_slack_.resize(num_vars_, 0);
  slack_rows_.clear();
  slack_rows_.resize(num_vars_, 0);

  for (i_t j : new_slacks) {
    is_slack_[j] = 1;
    const i_t col_start = lp.A.col_start[j];
    const i_t i = lp.A.i[col_start];
    slack_rows_[j] = i;
    if (lp.A.x[col_start] != 1.0) {
      printf("Initialize: Slack row %d has non-unit coefficient %e for variable %d\n", i, lp.A.x[col_start], j);
      exit(1);
    }
  }

  needs_complement_ = false;
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (lp.lower[j] < 0) {
      settings_.log.printf("Variable %d has negative lower bound %e\n", j, lp.lower[j]);
      exit(1);
    }
    const f_t uj = lp.upper[j];
    const f_t lj = lp.lower[j];
    if (uj != inf || lj != 0.0) { needs_complement_ = true; }
    const f_t xstar_j = xstar[j];
    if (uj < inf) {
      if (uj - xstar_j <= xstar_j - lj) {
        has_upper_[j] = 1;
      } else {
        has_lower_[j] = 1;
      }
      continue;
    }

    if (lj > -inf) { has_lower_[j] = 1; }
  }
}

template <typename i_t, typename f_t>
i_t mixed_integer_rounding_cut_t<i_t, f_t>::generate_cut(
  const sparse_vector_t<i_t, f_t>& a,
  f_t beta,
  const std::vector<f_t>& upper_bounds,
  const std::vector<f_t>& lower_bounds,
  const std::vector<variable_type_t>& var_types,
  sparse_vector_t<i_t, f_t>& cut,
  f_t& cut_rhs)
{
  auto f = [](f_t q_1, f_t q_2) -> f_t {
    f_t q_1_hat = q_1 - std::floor(q_1);
    f_t q_2_hat = q_2 - std::floor(q_2);
    return std::min(q_1_hat, q_2_hat) + q_2_hat * std::floor(q_1);
  };

  auto h = [](f_t q) -> f_t { return std::max(q, 0.0); };

  std::vector<i_t> cut_indices;
  cut_indices.reserve(a.i.size());
  f_t R;
  if (!needs_complement_) {
    R = (beta - std::floor(beta)) * std::ceil(beta);

    for (i_t k = 0; k < a.i.size(); k++) {
      const i_t jj = a.i[k];
      f_t aj       = a.x[k];
      if (var_types[jj] == variable_type_t::INTEGER) {
        x_workspace_[jj] += f(aj, beta);
        if (!x_mark_[jj] && x_workspace_[jj] != 0.0) {
          x_mark_[jj] = 1;
          cut_indices.push_back(jj);
        }
      } else {
        x_workspace_[jj] += h(aj);
        if (!x_mark_[jj] && x_workspace_[jj] != 0.0) {
          x_mark_[jj] = 1;
          cut_indices.push_back(jj);
        }
      }
    }
  } else {
    // Compute r
    f_t r = beta;
    for (i_t k = 0; k < a.i.size(); k++) {
      const i_t jj = a.i[k];
      if (has_upper_[jj]) {
        const f_t uj = upper_bounds[jj];
        r -= uj * a.x[k];
        continue;
      }
      if (has_lower_[jj]) {
        const f_t lj = lower_bounds[jj];
        r -= lj * a.x[k];
      }
    }

    // Compute R
    R = std::ceil(r) * (r - std::floor(r));
    for (i_t k = 0; k < a.i.size(); k++) {
      const i_t jj = a.i[k];
      const f_t aj = a.x[k];
      if (has_upper_[jj]) {
        const f_t uj = upper_bounds[jj];
        if (var_types[jj] == variable_type_t::INTEGER) {
          R -= f(-aj, r) * uj;
        } else {
          R -= h(-aj) * uj;
        }
      } else if (has_lower_[jj]) {
        const f_t lj = lower_bounds[jj];
        if (var_types[jj] == variable_type_t::INTEGER) {
          R += f(aj, r) * lj;
        } else {
          R += h(aj) * lj;
        }
      }
    }

    // Compute the cut coefficients
    for (i_t k = 0; k < a.i.size(); k++) {
      const i_t jj = a.i[k];
      const f_t aj = a.x[k];
      if (has_upper_[jj]) {
        if (var_types[jj] == variable_type_t::INTEGER) {
          // Upper intersect I
          x_workspace_[jj] -= f(-aj, r);
          if (!x_mark_[jj] && x_workspace_[jj] != 0.0) {
            x_mark_[jj] = 1;
            cut_indices.push_back(jj);
          }
        } else {
          // Upper intersect C
          f_t h_j = h(-aj);
          if (h_j != 0.0) {
            x_workspace_[jj] -= h_j;
            if (!x_mark_[jj]) {
              x_mark_[jj] = 1;
              cut_indices.push_back(jj);
            }
          }
        }
      } else if (var_types[jj] == variable_type_t::INTEGER) {
        // I \ Upper
        x_workspace_[jj] += f(aj, r);
        if (!x_mark_[jj] && x_workspace_[jj] != 0.0) {
          x_mark_[jj] = 1;
          cut_indices.push_back(jj);
        }
      } else {
        // C \ Upper
        f_t h_j = h(aj);
        if (h_j != 0.0) {
          x_workspace_[jj] += h_j;
          if (!x_mark_[jj]) {
            x_mark_[jj] = 1;
            cut_indices.push_back(jj);
          }
        }
      }
    }
  }

  cut.i.reserve(cut_indices.size());
  cut.x.reserve(cut_indices.size());
  for (i_t k = 0; k < cut_indices.size(); k++) {
    const i_t jj = cut_indices[k];

    // Check for small coefficients
    const f_t aj = x_workspace_[jj];
    if (std::abs(aj) < 1e-6) {
      if (aj >= 0.0 && upper_bounds[jj] < inf) {
        // Move this to the right-hand side
        R -= aj * upper_bounds[jj];
        continue;
      } else if (aj <= 0.0 && lower_bounds[jj] > -inf) {
        R += aj * lower_bounds[jj];
        continue;
      } else {
      }
    }
    cut.i.push_back(jj);
    cut.x.push_back(x_workspace_[jj]);
  }

  // Clear the workspace
  for (i_t jj : cut_indices) {
    x_workspace_[jj] = 0.0;
    x_mark_[jj]      = 0;
  }


  // The new cut is: g'*x >= R
  // But we want to have it in the form h'*x <= b
  cut.sort();

  cut_rhs = R;

  if (cut.i.size() == 0) {
    settings_.log.printf("No coefficients in cut\n");
    return -1;
  }

  return 0;
}

template <typename i_t, typename f_t>
void mixed_integer_rounding_cut_t<i_t, f_t>::substitute_slacks(const lp_problem_t<i_t, f_t>& lp,
                                                               csc_matrix_t<i_t, f_t>& Arow,
                                                               sparse_vector_t<i_t, f_t>& cut,
                                                               f_t& cut_rhs)
{
  // Remove slacks from the cut
  // So that the cut is only over the original variables
  bool found_slack = false;
  i_t cut_nz = 0;
  std::vector<i_t> cut_indices;
  cut_indices.reserve(cut.i.size());

#if 1
  for (i_t j = 0; j < x_workspace_.size(); j++) {
    if (x_workspace_[j] != 0.0) {
      printf("Dirty x_workspace_[%d] = %e\n", j, x_workspace_[j]);
      exit(1);
    }
    if (x_mark_[j] != 0) {
      printf("Dirty x_mark_[%d] = %d\n", j, x_mark_[j]);
      exit(1);
    }
  }
#endif



  for (i_t k = 0; k < cut.i.size(); k++) {
    const i_t j  = cut.i[k];
    const f_t cj = cut.x[k];
    if (is_slack_[j]) {
      found_slack = true;

      // Do the substitution
      // Slack variable s_j participates in row i of the constraint matrix
      // Row i is of the form:
      // sum_{k != j} A(i, k) * x_k + s_j = rhs_i
      /// So we have that
      // s_j = rhs_i - sum_{k != j} A(i, k) * x_k

      // Our cut is of the form:
      // sum_{k != j} C(k) * x_k + C(j) * s_j >= cut_rhs
      // So the cut becomes
      // sum_{k != j} C(k) * x_k + C(j) * (rhs_i - sum_{h != j} A(i, h) * x_h) >= cut_rhs
      // This is equivalent to:
      // sum_{k != j} C(k) * x_k + sum_{h != j} -C(j) * A(i, h) * x_h >= cut_rhs - C(j) * rhs_i
      const i_t i         = slack_rows_[j];
      //printf("Found slack %d in cut. lo %e up %e. Slack row %d\n", j, lp.lower[j], lp.upper[j], i);
      cut_rhs -= cj * lp.rhs[i];
      const i_t row_start = Arow.col_start[i];
      const i_t row_end   = Arow.col_start[i + 1];
      for (i_t q = row_start; q < row_end; q++) {
        const i_t h = Arow.i[q];
        if (h != j) {
          const f_t aih = Arow.x[q];
          x_workspace_[h] -= cj * aih;
          if (!x_mark_[h]) {
            x_mark_[h] = 1;
            cut_indices.push_back(h);
            cut_nz++;
          }
        } else {
            const f_t aij = Arow.x[q];
            if (aij != 1.0) {
                printf("Slack row %d has non-unit coefficient %e for variable %d\n", i, aij, j);
                exit(1);
            }
        }
      }

    } else {
      x_workspace_[j] += cj;
      if (!x_mark_[j]) {
        x_mark_[j] = 1;
        cut_indices.push_back(j);
        cut_nz++;
      }
    }
  }

  if (found_slack) {
    //printf("Found slack. Nz increased from %d to %d: %d\n", cut.i.size(), cut_nz, cut_nz - cut.i.size());
    cut.i.reserve(cut_nz);
    cut.x.reserve(cut_nz);
    cut.i.clear();
    cut.x.clear();

    for (i_t k = 0; k < cut_nz; k++) {
      const i_t j = cut_indices[k];

      // Check for small coefficients
      const f_t aj = x_workspace_[j];
      if (std::abs(aj) < 1e-6) {
        if (aj >= 0.0 && lp.upper[j] < inf) {
          // Move this to the right-hand side
          cut_rhs -= aj * lp.upper[j];
          continue;
        } else if (aj <= 0.0 && lp.lower[j] > -inf) {
          cut_rhs += aj * lp.lower[j];
          continue;
        } else {
        }
      }

      cut.i.push_back(j);
      cut.x.push_back(x_workspace_[j]);
    }
    // Sort the cut
    cut.sort();
  }

  // Clear the workspace
  for (i_t jj : cut_indices) {
    x_workspace_[jj] = 0.0;
    x_mark_[jj]      = 0;
  }


#if 1
  for (i_t j = 0; j < x_workspace_.size(); j++) {
    if (x_workspace_[j] != 0.0) {
      printf("Dirty x_workspace_[%d] = %e\n", j, x_workspace_[j]);
      exit(1);
    }
    if (x_mark_[j] != 0) {
      printf("Dirty x_mark_[%d] = %d\n", j, x_mark_[j]);
      exit(1);
    }
  }
#endif
}

template <typename i_t, typename f_t>
i_t add_cuts(const simplex_solver_settings_t<i_t, f_t>& settings,
             const csr_matrix_t<i_t, f_t>& cuts,
             const std::vector<f_t>& cut_rhs,
             lp_problem_t<i_t, f_t>& lp,
             std::vector<i_t>& new_slacks,
             lp_solution_t<i_t, f_t>& solution,
             basis_update_mpf_t<i_t, f_t>& basis_update,
             std::vector<i_t>& basic_list,
             std::vector<i_t>& nonbasic_list,
             std::vector<variable_status_t>& vstatus,
             std::vector<f_t>& edge_norms)

{
  // Given a set of cuts: C*x <= d that are currently violated
  // by the current solution x* (i.e. C*x* > d), this function
  // adds the cuts into the LP and solves again.

#ifdef CHECK_BASIS
  {
    csc_matrix_t<i_t, f_t> Btest(lp.num_rows, lp.num_rows, 1);
    basis_update.multiply_lu(Btest);
    csc_matrix_t<i_t, f_t> B(lp.num_rows, lp.num_rows, 1);
    form_b(lp.A, basic_list, B);
    csc_matrix_t<i_t, f_t> Diff(lp.num_rows, lp.num_rows, 1);
    add(Btest, B, 1.0, -1.0, Diff);
    const f_t err = Diff.norm1();
    settings.log.printf("Before || B - L*U || %e\n", err);
    if (err > 1e-6) { exit(1); }
  }
#endif

  const i_t p = cuts.m;
  if (cut_rhs.size() != static_cast<size_t>(p)) {
    settings.log.printf("cut_rhs must have the same number of rows as cuts\n");
    return -1;
  }
  settings.log.debug("Number of cuts %d\n", p);
  settings.log.debug("Original lp rows %d\n", lp.num_rows);
  settings.log.debug("Original lp cols %d\n", lp.num_cols);

  csr_matrix_t<i_t, f_t> new_A_row(lp.num_rows, lp.num_cols, 1);
  lp.A.to_compressed_row(new_A_row);

  i_t append_status = new_A_row.append_rows(cuts);
  if (append_status != 0) {
    settings.log.printf("append_rows error: %d\n", append_status);
    exit(1);
  }

  csc_matrix_t<i_t, f_t> new_A_col(lp.num_rows + p, lp.num_cols, 1);
  new_A_row.to_compressed_col(new_A_col);

  // Add in slacks variables for the new rows
  lp.lower.resize(lp.num_cols + p);
  lp.upper.resize(lp.num_cols + p);
  lp.objective.resize(lp.num_cols + p);
  i_t nz = new_A_col.col_start[lp.num_cols];
  new_A_col.col_start.resize(lp.num_cols + p + 1);
  new_A_col.i.resize(nz + p);
  new_A_col.x.resize(nz + p);
  i_t k = lp.num_rows;
  for (i_t j = lp.num_cols; j < lp.num_cols + p; j++) {
    new_A_col.col_start[j] = nz;
    new_A_col.i[nz]        = k++;
    new_A_col.x[nz]        = 1.0;
    nz++;
    lp.lower[j]     = 0.0;
    lp.upper[j]     = inf;
    lp.objective[j] = 0.0;
    new_slacks.push_back(j);
  }
  settings.log.debug("Done adding slacks\n");
  new_A_col.col_start[lp.num_cols + p] = nz;
  new_A_col.n                          = lp.num_cols + p;

  lp.A         = new_A_col;
  i_t old_rows = lp.num_rows;
  lp.num_rows += p;
  i_t old_cols = lp.num_cols;
  lp.num_cols += p;

  lp.rhs.resize(lp.num_rows);
  for (i_t k = old_rows; k < old_rows + p; k++) {
    const i_t h = k - old_rows;
    lp.rhs[k]   = cut_rhs[h];
  }
  settings.log.debug("Done adding rhs\n");

  // Construct C_B = C(:, basic_list)
  std::vector<i_t> C_col_degree(lp.num_cols, 0);
  i_t cuts_nz = cuts.row_start[p];
  for (i_t q = 0; q < cuts_nz; q++) {
    const i_t j = cuts.j[q];
    if (j >= lp.num_cols) {
      settings.log.printf("j %d is greater than p %d\n", j, p);
      return -1;
    }
    C_col_degree[j]++;
  }
  settings.log.debug("Done computing C_col_degree\n");

  std::vector<i_t> in_basis(old_cols, -1);
  const i_t num_basic = static_cast<i_t>(basic_list.size());
  i_t C_B_nz          = 0;
  for (i_t k = 0; k < num_basic; k++) {
    const i_t j = basic_list[k];
    if (j < 0 || j >= old_cols) {
      settings.log.printf(
        "basic_list[%d] = %d is out of bounds %d old_cols %d\n", k, j, j, old_cols);
      return -1;
    }
    in_basis[j] = k;
    if (j < cuts.n) { C_B_nz += C_col_degree[j]; }
  }
  settings.log.debug("Done estimating C_B_nz\n");

  csr_matrix_t<i_t, f_t> C_B(p, num_basic, C_B_nz);
  nz = 0;
  for (i_t i = 0; i < p; i++) {
    C_B.row_start[i]    = nz;
    const i_t row_start = cuts.row_start[i];
    const i_t row_end   = cuts.row_start[i + 1];
    for (i_t q = row_start; q < row_end; q++) {
      const i_t j       = cuts.j[q];
      const i_t j_basis = in_basis[j];
      if (j_basis == -1) { continue; }
      C_B.j[nz] = j_basis;
      C_B.x[nz] = cuts.x[q];
      nz++;
    }
  }
  C_B.row_start[p] = nz;

  if (nz != C_B_nz) {
    settings.log.printf("predicted nz %d actual nz %d\n", C_B_nz, nz);
    return -1;
  }
  settings.log.debug("C_B rows %d cols %d nz %d\n", C_B.m, C_B.n, nz);

  // Adjust the basis update to include the new cuts
  basis_update.append_cuts(C_B);

  basic_list.resize(lp.num_rows, 0);
  i_t h = old_cols;
  for (i_t j = old_rows; j < lp.num_rows; j++) {
    basic_list[j] = h++;
  }

#ifdef CHECK_BASIS
  // Check the basis update
  csc_matrix_t<i_t, f_t> Btest(lp.num_rows, lp.num_rows, 1);
  basis_update.multiply_lu(Btest);

  csc_matrix_t<i_t, f_t> B(lp.num_rows, lp.num_rows, 1);
  form_b(lp.A, basic_list, B);

  csc_matrix_t<i_t, f_t> Diff(lp.num_rows, lp.num_rows, 1);
  add(Btest, B, 1.0, -1.0, Diff);
  const f_t err = Diff.norm1();
  settings.log.printf("After || B - L*U || %e\n", err);
  if (err > 1e-6) {
    settings.log.printf("Diff matrix\n");
    // Diff.print_matrix();
    exit(1);
  }
#endif
  // Adjust the vstatus
  vstatus.resize(lp.num_cols);
  for (i_t j = old_cols; j < lp.num_cols; j++) {
    vstatus[j] = variable_status_t::BASIC;
  }

  return 0;
}

template <typename i_t, typename f_t>
void remove_cuts(lp_problem_t<i_t, f_t>& lp,
                 const simplex_solver_settings_t<i_t, f_t>& settings,
                 csc_matrix_t<i_t, f_t>& Arow,
                 std::vector<i_t>& new_slacks,
                 i_t original_rows,
                 std::vector<variable_type_t>& var_types,
                 std::vector<variable_status_t>& vstatus,
                 std::vector<f_t>& x,
                 std::vector<f_t>& y,
                 std::vector<f_t>& z,
                 std::vector<i_t>& basic_list,
                 std::vector<i_t>& nonbasic_list,
                 basis_update_mpf_t<i_t, f_t>& basis_update)
{
  std::vector<i_t> cuts_to_remove;
  cuts_to_remove.reserve(lp.num_rows - original_rows);
  std::vector<i_t> slacks_to_remove;
  slacks_to_remove.reserve(lp.num_rows - original_rows);
  const f_t dual_tol = 1e-10;

  std::vector<i_t> is_slack(lp.num_cols, 0);
  for (i_t j : new_slacks) {
    is_slack[j] = 1;
  }

  for (i_t k = original_rows; k < lp.num_rows; k++) {
    if (std::abs(y[k]) < dual_tol) {
      const i_t row_start = Arow.col_start[k];
      const i_t row_end   = Arow.col_start[k + 1];
      i_t last_slack      = -1;
      const f_t slack_tol = 1e-3;
      for (i_t p = row_start; p < row_end; p++) {
        const i_t j      = Arow.i[p];
        if (is_slack[j]) {
          if (vstatus[j] == variable_status_t::BASIC && x[j] > slack_tol) { last_slack = j; }
        }
      }
      if (last_slack != -1) {
        cuts_to_remove.push_back(k);
        slacks_to_remove.push_back(last_slack);
      }
    }
  }

  if (cuts_to_remove.size() > 0) {
    //settings.log.printf("Removing %d cuts\n", cuts_to_remove.size());
    std::vector<i_t> marked_rows(lp.num_rows, 0);
    for (i_t i : cuts_to_remove) {
      marked_rows[i] = 1;
    }
    std::vector<i_t> marked_cols(lp.num_cols, 0);
    for (i_t j : slacks_to_remove) {
      marked_cols[j] = 1;
    }

    std::vector<f_t> new_rhs(lp.num_rows - cuts_to_remove.size());
    std::vector<f_t> new_solution_y(lp.num_rows - cuts_to_remove.size());
    i_t h = 0;
    for (i_t i = 0; i < lp.num_rows; i++) {
      if (!marked_rows[i]) {
        new_rhs[h]        = lp.rhs[i];
        new_solution_y[h] = y[i];
        h++;
      }
    }

    Arow.remove_columns(marked_rows);
    Arow.transpose(lp.A);

    std::vector<f_t> new_objective(lp.num_cols - slacks_to_remove.size());
    std::vector<f_t> new_lower(lp.num_cols - slacks_to_remove.size());
    std::vector<f_t> new_upper(lp.num_cols - slacks_to_remove.size());
    std::vector<variable_type_t> new_var_types(lp.num_cols - slacks_to_remove.size());
    std::vector<variable_status_t> new_vstatus(lp.num_cols - slacks_to_remove.size());
    std::vector<i_t> new_basic_list;
    new_basic_list.reserve(lp.num_rows - slacks_to_remove.size());
    std::vector<i_t> new_nonbasic_list;
    new_nonbasic_list.reserve(nonbasic_list.size());
    std::vector<f_t> new_solution_x(lp.num_cols - slacks_to_remove.size());
    std::vector<f_t> new_solution_z(lp.num_cols - slacks_to_remove.size());
    std::vector<i_t> new_is_slacks(lp.num_cols - slacks_to_remove.size(), 0);
    h = 0;
    for (i_t k = 0; k < lp.num_cols; k++) {
      if (!marked_cols[k]) {
        new_objective[h]  = lp.objective[k];
        new_lower[h]      = lp.lower[k];
        new_upper[h]      = lp.upper[k];
        new_var_types[h]  = var_types[k];
        new_vstatus[h]    = vstatus[k];
        new_solution_x[h] = x[k];
        new_solution_z[h] = z[k];
        new_is_slacks[h] = is_slack[k];
        if (new_vstatus[h] != variable_status_t::BASIC) {
          new_nonbasic_list.push_back(h);
        } else {
          new_basic_list.push_back(h);
        }
        h++;
      }
    }
    lp.A.remove_columns(marked_cols);
    lp.A.transpose(Arow);
    lp.objective  = new_objective;
    lp.lower      = new_lower;
    lp.upper      = new_upper;
    lp.rhs        = new_rhs;
    var_types     = new_var_types;
    lp.num_cols   = lp.A.n;
    lp.num_rows   = lp.A.m;

    new_slacks.clear();
    new_slacks.reserve(lp.num_cols);
    for (i_t j = 0; j < lp.num_cols; j++) {
        if (new_is_slacks[j]) {
            new_slacks.push_back(j);
        }
    }
    basic_list    = new_basic_list;
    nonbasic_list = new_nonbasic_list;
    vstatus       = new_vstatus;
    x             = new_solution_x;
    y             = new_solution_y;
    z             = new_solution_z;

    settings.log.printf("Removed %d cuts. After removal %d rows %d columns %d nonzeros\n",
                        cuts_to_remove.size(),
                        lp.num_rows,
                        lp.num_cols,
                        lp.A.col_start[lp.A.n]);

    basis_update.resize(lp.num_rows);
    basis_update.refactor_basis(lp.A, settings, basic_list, nonbasic_list, vstatus);
  }
}


#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class cut_pool_t<int, double>;
template class cut_generation_t<int, double>;
template class mixed_integer_gomory_base_inequality_t<int, double>;
template class mixed_integer_rounding_cut_t<int, double>;

template
int add_cuts(const simplex_solver_settings_t<int, double>& settings,
              const csr_matrix_t<int, double>& cuts,
              const std::vector<double>& cut_rhs,
              lp_problem_t<int, double>& lp,
              std::vector<int>& new_slacks,
              lp_solution_t<int, double>& solution,
              basis_update_mpf_t<int, double>& basis_update,
              std::vector<int>& basic_list,
              std::vector<int>& nonbasic_list,
              std::vector<variable_status_t>& vstatus,
              std::vector<double>& edge_norms);

template
void remove_cuts<int, double>(lp_problem_t<int, double>& lp,
                 const simplex_solver_settings_t<int, double>& settings,
                 csc_matrix_t<int, double>& Arow,
                 std::vector<int>& new_slacks,
                 int original_rows,
                 std::vector<variable_type_t>& var_types,
                 std::vector<variable_status_t>& vstatus,
                 std::vector<double>& x,
                 std::vector<double>& y,
                 std::vector<double>& z,
                 std::vector<int>& basic_list,
                 std::vector<int>& nonbasic_list,
                 basis_update_mpf_t<int, double>& basis_update);
#endif

} // namespace cuopt::linear_programming::dual_simplex


