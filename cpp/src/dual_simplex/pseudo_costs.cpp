/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/phase2.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <omp.h>

namespace cuopt::linear_programming::dual_simplex {

namespace {

template <typename i_t, typename f_t>
void strong_branch_helper(i_t start,
                          i_t end,
                          f_t start_time,
                          const lp_problem_t<i_t, f_t>& original_lp,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          const std::vector<variable_type_t>& var_types,
                          const std::vector<i_t>& fractional,
                          f_t root_obj,
                          const std::vector<f_t>& root_soln,
                          const std::vector<variable_status_t>& root_vstatus,
                          const std::vector<f_t>& edge_norms,
                          pseudo_costs_t<i_t, f_t>& pc)
{
  lp_problem_t child_problem = original_lp;

  constexpr bool verbose = false;
  f_t last_log           = tic();
  i_t thread_id          = omp_get_thread_num();
  for (i_t k = start; k < end; ++k) {
    const i_t j = fractional[k];

    for (i_t branch = 0; branch < 2; branch++) {
      // Do the down branch
      if (branch == 0) {
        child_problem.lower[j] = original_lp.lower[j];
        child_problem.upper[j] = std::floor(root_soln[j]);
      } else {
        child_problem.lower[j] = std::ceil(root_soln[j]);
        child_problem.upper[j] = original_lp.upper[j];
      }

      simplex_solver_settings_t<i_t, f_t> child_settings = settings;
      child_settings.set_log(false);
      f_t lp_start_time = tic();
      f_t elapsed_time  = toc(start_time);
      if (elapsed_time > settings.time_limit) { break; }
      child_settings.time_limit      = std::max(0.0, settings.time_limit - elapsed_time);
      child_settings.iteration_limit = 200;
      lp_solution_t<i_t, f_t> solution(original_lp.num_rows, original_lp.num_cols);
      i_t iter                               = 0;
      std::vector<variable_status_t> vstatus = root_vstatus;
      std::vector<f_t> child_edge_norms      = edge_norms;
      dual::status_t status                  = dual_phase2(2,
                                          0,
                                          lp_start_time,
                                          child_problem,
                                          child_settings,
                                          vstatus,
                                          solution,
                                          iter,
                                          child_edge_norms);

      f_t obj = std::numeric_limits<f_t>::quiet_NaN();
      if (status == dual::status_t::DUAL_UNBOUNDED) {
        // LP was infeasible
        obj = std::numeric_limits<f_t>::infinity();
      } else if (status == dual::status_t::OPTIMAL || status == dual::status_t::ITERATION_LIMIT) {
        obj = compute_objective(child_problem, solution.x);
      } else {
        settings.log.debug("Thread id %2d remaining %d variable %d branch %d status %d\n",
                           thread_id,
                           end - 1 - k,
                           j,
                           branch,
                           status);
      }

      if (branch == 0) {
        pc.strong_branch_down[k] = obj - root_obj;
        if (verbose) {
          settings.log.printf("Thread id %2d remaining %d variable %d branch %d obj %e time %.2f\n",
                              thread_id,
                              end - 1 - k,
                              j,
                              branch,
                              obj,
                              toc(start_time));
        }
      } else {
        pc.strong_branch_up[k] = obj - root_obj;
        if (verbose) {
          settings.log.printf(
            "Thread id %2d remaining %d variable %d branch %d obj %e change down %e change up %e "
            "time %.2f\n",
            thread_id,
            end - 1 - k,
            j,
            branch,
            obj,
            pc.strong_branch_down[k],
            pc.strong_branch_up[k],
            toc(start_time));
        }
      }
      if (toc(start_time) > settings.time_limit) { break; }
    }
    if (toc(start_time) > settings.time_limit) { break; }

    const i_t completed = pc.num_strong_branches_completed++;

    if (thread_id == 0 && toc(last_log) > 10) {
      last_log = tic();
      settings.log.printf("%d of %ld strong branches completed in %.1fs\n",
                          completed,
                          fractional.size(),
                          toc(start_time));
    }

    child_problem.lower[j] = original_lp.lower[j];
    child_problem.upper[j] = original_lp.upper[j];

    if (toc(start_time) > settings.time_limit) { break; }
  }
}

template <typename i_t, typename f_t>
f_t trial_branching(const lp_problem_t<i_t, f_t>& original_lp,
                    const simplex_solver_settings_t<i_t, f_t>& settings,
                    const std::vector<variable_type_t>& var_types,
                    const std::vector<variable_status_t>& vstatus,
                    const std::vector<f_t>& edge_norms,
                    const basis_update_mpf_t<i_t, f_t>& basis_factors,
                    const std::vector<i_t>& basic_list,
                    const std::vector<i_t>& nonbasic_list,
                    i_t branch_var,
                    f_t branch_var_lower,
                    f_t branch_var_upper,
                    f_t upper_bound,
                    i_t bnb_lp_iter_per_node,
                    omp_atomic_t<int64_t>& total_lp_iter)
{
  lp_problem_t child_problem      = original_lp;
  child_problem.lower[branch_var] = branch_var_lower;
  child_problem.upper[branch_var] = branch_var_upper;

  simplex_solver_settings_t<i_t, f_t> child_settings = settings;
  child_settings.set_log(false);
  f_t lp_start_time              = tic();
  i_t lp_iter_upper              = settings.reliability_branching_settings.upper_max_lp_iter;
  i_t lp_iter_lower              = settings.reliability_branching_settings.lower_max_lp_iter;
  child_settings.iteration_limit = std::clamp(bnb_lp_iter_per_node, lp_iter_lower, lp_iter_upper);
  child_settings.cut_off         = upper_bound + settings.dual_tol;
  child_settings.inside_mip      = 2;
  child_settings.scale_columns   = false;

  lp_solution_t<i_t, f_t> solution(original_lp.num_rows, original_lp.num_cols);
  i_t iter                                         = 0;
  std::vector<variable_status_t> child_vstatus     = vstatus;
  std::vector<f_t> child_edge_norms                = edge_norms;
  std::vector<i_t> child_basic_list                = basic_list;
  std::vector<i_t> child_nonbasic_list             = nonbasic_list;
  basis_update_mpf_t<i_t, f_t> child_basis_factors = basis_factors;

  dual::status_t status = dual_phase2_with_advanced_basis(2,
                                                          0,
                                                          false,
                                                          lp_start_time,
                                                          child_problem,
                                                          child_settings,
                                                          child_vstatus,
                                                          child_basis_factors,
                                                          child_basic_list,
                                                          child_nonbasic_list,
                                                          solution,
                                                          iter,
                                                          child_edge_norms);
  total_lp_iter += iter;
  settings.log.debug("Trial branching on variable %d. Lo: %e Up: %e. Iter %d. Status %s. Obj %e\n",
                     branch_var,
                     child_problem.lower[branch_var],
                     child_problem.upper[branch_var],
                     iter,
                     dual::status_to_string(status).c_str(),
                     compute_objective(child_problem, solution.x));

  if (status == dual::status_t::DUAL_UNBOUNDED) {
    // LP was infeasible
    return std::numeric_limits<f_t>::infinity();
  } else if (status == dual::status_t::OPTIMAL || status == dual::status_t::ITERATION_LIMIT ||
             status == dual::status_t::CUTOFF) {
    return compute_objective(child_problem, solution.x);
  } else {
    return std::numeric_limits<f_t>::quiet_NaN();
  }
}

}  // namespace

template <typename i_t, typename f_t>
void strong_branching(const lp_problem_t<i_t, f_t>& original_lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      f_t start_time,
                      const std::vector<variable_type_t>& var_types,
                      const std::vector<f_t> root_soln,
                      const std::vector<i_t>& fractional,
                      f_t root_obj,
                      const std::vector<variable_status_t>& root_vstatus,
                      const std::vector<f_t>& edge_norms,
                      pseudo_costs_t<i_t, f_t>& pc)
{
  pc.resize(original_lp.num_cols);
  pc.strong_branch_down.assign(fractional.size(), 0);
  pc.strong_branch_up.assign(fractional.size(), 0);
  pc.num_strong_branches_completed = 0;

  settings.log.printf("Strong branching using %d threads and %ld fractional variables\n",
                      settings.num_threads,
                      fractional.size());

#pragma omp parallel num_threads(settings.num_threads)
  {
    i_t n = std::min<i_t>(4 * settings.num_threads, fractional.size());

    // Here we are creating more tasks than the number of threads
    // such that they can be scheduled dynamically to the threads.
#pragma omp for schedule(dynamic, 1)
    for (i_t k = 0; k < n; k++) {
      i_t start = std::floor(k * fractional.size() / n);
      i_t end   = std::floor((k + 1) * fractional.size() / n);

      constexpr bool verbose = false;
      if (verbose) {
        settings.log.printf("Thread id %d task id %d start %d end %d. size %d\n",
                            omp_get_thread_num(),
                            k,
                            start,
                            end,
                            end - start);
      }

      strong_branch_helper(start,
                           end,
                           start_time,
                           original_lp,
                           settings,
                           var_types,
                           fractional,
                           root_obj,
                           root_soln,
                           root_vstatus,
                           edge_norms,
                           pc);
    }
  }

  pc.update_pseudo_costs_from_strong_branching(fractional, root_soln);
}

template <typename i_t, typename f_t>
void pseudo_costs_t<i_t, f_t>::update_pseudo_costs(mip_node_t<i_t, f_t>* node_ptr,
                                                   f_t leaf_objective)
{
  std::lock_guard<omp_mutex_t> lock(pseudo_cost_mutex[node_ptr->branch_var]);
  const f_t change_in_obj = leaf_objective - node_ptr->lower_bound;
  const f_t frac          = node_ptr->branch_dir == rounding_direction_t::DOWN
                              ? node_ptr->fractional_val - std::floor(node_ptr->fractional_val)
                              : std::ceil(node_ptr->fractional_val) - node_ptr->fractional_val;
  if (node_ptr->branch_dir == rounding_direction_t::DOWN) {
    pseudo_cost_sum_down[node_ptr->branch_var] += change_in_obj / frac;
    pseudo_cost_num_down[node_ptr->branch_var]++;
  } else {
    pseudo_cost_sum_up[node_ptr->branch_var] += change_in_obj / frac;
    pseudo_cost_num_up[node_ptr->branch_var]++;
  }
}

template <typename i_t, typename f_t>
void pseudo_costs_t<i_t, f_t>::initialized(i_t& num_initialized_down,
                                           i_t& num_initialized_up,
                                           f_t& pseudo_cost_down_avg,
                                           f_t& pseudo_cost_up_avg)
{
  num_initialized_down = 0;
  num_initialized_up   = 0;
  pseudo_cost_down_avg = 0;
  pseudo_cost_up_avg   = 0;
  const i_t n          = pseudo_cost_sum_down.size();
  for (i_t j = 0; j < n; j++) {
    std::lock_guard<omp_mutex_t> lock(pseudo_cost_mutex[j]);

    if (pseudo_cost_num_down[j] > 0) {
      num_initialized_down++;
      if (std::isfinite(pseudo_cost_sum_down[j])) {
        pseudo_cost_down_avg += pseudo_cost_sum_down[j] / pseudo_cost_num_down[j];
      }
    }

    if (pseudo_cost_num_up[j] > 0) {
      num_initialized_up++;

      if (std::isfinite(pseudo_cost_sum_up[j])) {
        pseudo_cost_up_avg += pseudo_cost_sum_up[j] / pseudo_cost_num_up[j];
      }
    }
  }
  if (num_initialized_down > 0) {
    pseudo_cost_down_avg /= num_initialized_down;
  } else {
    pseudo_cost_down_avg = 1.0;
  }
  if (num_initialized_up > 0) {
    pseudo_cost_up_avg /= num_initialized_up;
  } else {
    pseudo_cost_up_avg = 1.0;
  }
}

template <typename i_t, typename f_t>
i_t pseudo_costs_t<i_t, f_t>::variable_selection(const std::vector<i_t>& fractional,
                                                 const std::vector<f_t>& solution,
                                                 logger_t& log)
{
  const i_t num_fractional = fractional.size();
  std::vector<f_t> pseudo_cost_up(num_fractional);
  std::vector<f_t> pseudo_cost_down(num_fractional);
  std::vector<f_t> score(num_fractional);

  i_t num_initialized_down;
  i_t num_initialized_up;
  f_t pseudo_cost_down_avg;
  f_t pseudo_cost_up_avg;

  initialized(num_initialized_down, num_initialized_up, pseudo_cost_down_avg, pseudo_cost_up_avg);

  log.debug("PC: num initialized down %d up %d avg down %e up %e\n",
            num_initialized_down,
            num_initialized_up,
            pseudo_cost_down_avg,
            pseudo_cost_up_avg);

  for (i_t k = 0; k < num_fractional; k++) {
    const i_t j = fractional[k];

    pseudo_cost_mutex[j].lock();
    if (pseudo_cost_num_down[j] != 0) {
      pseudo_cost_down[k] = pseudo_cost_sum_down[j] / pseudo_cost_num_down[j];
    } else {
      pseudo_cost_down[k] = pseudo_cost_down_avg;
    }

    if (pseudo_cost_num_up[j] != 0) {
      pseudo_cost_up[k] = pseudo_cost_sum_up[j] / pseudo_cost_num_up[j];
    } else {
      pseudo_cost_up[k] = pseudo_cost_up_avg;
    }
    pseudo_cost_mutex[j].unlock();

    constexpr f_t eps = 1e-6;
    const f_t f_down  = solution[j] - std::floor(solution[j]);
    const f_t f_up    = std::ceil(solution[j]) - solution[j];
    score[k] =
      std::max(f_down * pseudo_cost_down[k], eps) * std::max(f_up * pseudo_cost_up[k], eps);
  }

  i_t branch_var = fractional[0];
  f_t max_score  = -1;
  i_t select     = -1;

  for (i_t k = 0; k < num_fractional; k++) {
    if (score[k] > max_score) {
      max_score  = score[k];
      branch_var = fractional[k];
      select     = k;
    }
  }

  log.debug("Pseudocost branching on %d. Value %e. Score %e.\n",
            branch_var,
            solution[branch_var],
            score[select]);

  return branch_var;
}

template <typename i_t, typename f_t>
i_t pseudo_costs_t<i_t, f_t>::reliable_variable_selection(
  mip_node_t<i_t, f_t>* node_ptr,
  const std::vector<i_t>& fractional,
  const std::vector<f_t>& solution,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  bnb_worker_data_t<i_t, f_t>* worker_data,
  const bnb_stats_t<i_t, f_t>& bnb_stats,
  f_t upper_bound,
  logger_t& log)
{
  i_t branch_var = fractional[0];
  f_t max_score  = -1;
  i_t num_initialized_down;
  i_t num_initialized_up;
  f_t pseudo_cost_down_avg;
  f_t pseudo_cost_up_avg;

  initialized(num_initialized_down, num_initialized_up, pseudo_cost_down_avg, pseudo_cost_up_avg);

  log.printf("PC: num initialized down %d up %d avg down %e up %e\n",
             num_initialized_down,
             num_initialized_up,
             pseudo_cost_down_avg,
             pseudo_cost_up_avg);

  const int64_t bnb_total_lp_iter  = bnb_stats.total_lp_iters;
  const int64_t bnb_nodes_explored = bnb_stats.nodes_explored;
  const i_t bnb_lp_iter_per_node   = bnb_total_lp_iter / bnb_stats.nodes_explored;

  const i_t max_threshold = settings.reliability_branching_settings.max_reliable_threshold;
  const i_t min_threshold = settings.reliability_branching_settings.min_reliable_threshold;
  const i_t iter_factor   = settings.reliability_branching_settings.bnb_lp_factor;
  const i_t iter_offset   = settings.reliability_branching_settings.bnb_lp_offset;
  const int64_t alpha     = iter_factor * bnb_total_lp_iter;
  const int64_t max_iter  = alpha + settings.reliability_branching_settings.bnb_lp_offset;

  i_t reliable_threshold = settings.reliability_branching_settings.reliable_threshold;
  if (reliable_threshold < 0) {
    i_t gamma = (max_iter - sb_total_lp_iter) / (sb_total_lp_iter + 1);
    gamma     = std::min(1, gamma);
    gamma     = std::max<i_t>((alpha - sb_total_lp_iter) / (sb_total_lp_iter + 1), gamma);

    reliable_threshold = (1 - gamma) * min_threshold + gamma * max_threshold;
    reliable_threshold = sb_total_lp_iter < max_iter ? reliable_threshold : 0;
  }

  std::vector<i_t> unreliable_list;
  omp_mutex_t score_mutex;

  for (auto j : fractional) {
    std::lock_guard<omp_mutex_t> lock(pseudo_cost_mutex[j]);

    if (pseudo_cost_num_down[j] < reliable_threshold ||
        pseudo_cost_num_up[j] < reliable_threshold) {
      unreliable_list.push_back(j);
      continue;
    }

    f_t pc_up   = pseudo_cost_num_up[j] > 0 ? pseudo_cost_sum_up[j] / pseudo_cost_num_up[j]
                                            : pseudo_cost_up_avg;
    f_t pc_down = pseudo_cost_sum_down[j] > 0 ? pseudo_cost_sum_down[j] / pseudo_cost_num_down[j]
                                              : pseudo_cost_down_avg;

    constexpr f_t eps = 1e-6;
    const f_t f_down  = solution[j] - std::floor(solution[j]);
    const f_t f_up    = std::ceil(solution[j]) - solution[j];
    f_t score         = std::max(f_down * pc_down, eps) * std::max(f_up * pc_up, eps);

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
    }
  }

  if (unreliable_list.empty()) {
    log.printf(
      "pc branching on %d. Value %e. Score %e\n", branch_var, solution[branch_var], max_score);

    return branch_var;
  }

  const int task_priority      = settings.reliability_branching_settings.task_priority;
  const i_t max_num_candidates = settings.reliability_branching_settings.max_num_candidates;
  const i_t max_lookahead      = settings.reliability_branching_settings.max_lookahead;
  const i_t num_sb_vars        = std::min<size_t>(unreliable_list.size(), max_num_candidates);

  assert(task_priority > 0);
  assert(max_num_candidates > 0);
  assert(max_lookahead > 0);
  assert(num_sb_vars > 0);

  // Shuffle the unreliable list so every variable has the same chance to be selected.
  if (unreliable_list.size() > max_num_candidates) { worker_data->rng.shuffle(unreliable_list); }

  omp_atomic_t<i_t> unchanged = 0;
  const i_t max_tasks         = std::clamp(num_sb_vars / 2, 1, max_lookahead);
  i_t num_tasks               = settings.reliability_branching_settings.num_tasks;
  num_tasks                   = std::clamp(num_tasks, 1, max_tasks);
  assert(num_tasks > 0);

  settings.log.printf(
    "RB iters = %d, B&B iters = %d, unreliable = %d, num_tasks = %d, reliable_threshold = %d\n",
    sb_total_lp_iter.load(),
    bnb_total_lp_iter,
    unreliable_list.size(),
    num_tasks,
    reliable_threshold);

#pragma omp taskloop if (num_tasks > 1) priority(task_priority) num_tasks(num_tasks) untied
  for (int task_id = 0; task_id < num_tasks; ++task_id) {
    size_t start = (double)task_id * num_sb_vars / num_tasks;
    size_t end   = (double)(task_id + 1) * num_sb_vars / num_tasks;

    for (i_t i = start; i < end; ++i) {
      if (unchanged > max_lookahead) { break; }

      const i_t j = unreliable_list[i];
      pseudo_cost_mutex[j].lock();

      if (pseudo_cost_num_down[j] < reliable_threshold) {
        // Do trial branching on the down branch
        f_t obj = trial_branching(worker_data->leaf_problem,
                                  settings,
                                  var_types,
                                  node_ptr->vstatus,
                                  worker_data->leaf_edge_norms,
                                  worker_data->basis_factors,
                                  worker_data->basic_list,
                                  worker_data->nonbasic_list,
                                  j,
                                  worker_data->leaf_problem.lower[j],
                                  std::floor(solution[j]),
                                  upper_bound,
                                  bnb_lp_iter_per_node,
                                  sb_total_lp_iter);
        if (!std::isnan(obj)) {
          f_t change_in_obj = obj - node_ptr->lower_bound;
          f_t change_in_x   = solution[j] - std::floor(solution[j]);
          pseudo_cost_sum_down[j] += change_in_obj / change_in_x;
          pseudo_cost_num_down[j]++;
        }
      }

      if (pseudo_cost_num_up[j] < reliable_threshold) {
        f_t obj = trial_branching(worker_data->leaf_problem,
                                  settings,
                                  var_types,
                                  node_ptr->vstatus,
                                  worker_data->leaf_edge_norms,
                                  worker_data->basis_factors,
                                  worker_data->basic_list,
                                  worker_data->nonbasic_list,
                                  j,
                                  std::ceil(solution[j]),
                                  worker_data->leaf_problem.upper[j],
                                  upper_bound,
                                  bnb_lp_iter_per_node,
                                  sb_total_lp_iter);

        if (!std::isnan(obj)) {
          f_t change_in_obj = obj - node_ptr->lower_bound;
          f_t change_in_x   = std::ceil(solution[j]) - solution[j];
          pseudo_cost_sum_up[j] += change_in_obj / change_in_x;
          pseudo_cost_num_up[j]++;
        }
      }

      f_t pc_up   = pseudo_cost_num_up[j] > 0 ? pseudo_cost_sum_up[j] / pseudo_cost_num_up[j]
                                              : pseudo_cost_up_avg;
      f_t pc_down = pseudo_cost_sum_down[j] > 0 ? pseudo_cost_sum_down[j] / pseudo_cost_num_down[j]
                                                : pseudo_cost_down_avg;

      pseudo_cost_mutex[j].unlock();

      constexpr f_t eps = 1e-6;
      const f_t f_down  = solution[j] - std::floor(solution[j]);
      const f_t f_up    = std::ceil(solution[j]) - solution[j];
      f_t score         = std::max(f_down * pc_down, eps) * std::max(f_up * pc_up, eps);

      score_mutex.lock();
      if (score > max_score) {
        max_score  = score;
        branch_var = j;
        unchanged  = 0;
      } else {
        unchanged++;
      }
      score_mutex.unlock();
    }
  }

  log.printf(
    "pc branching on %d. Value %e. Score %e\n", branch_var, solution[branch_var], max_score);

  return branch_var;
}

template <typename i_t, typename f_t>
f_t pseudo_costs_t<i_t, f_t>::obj_estimate(const std::vector<i_t>& fractional,
                                           const std::vector<f_t>& solution,
                                           f_t lower_bound,
                                           logger_t& log)
{
  const i_t num_fractional = fractional.size();
  f_t estimate             = lower_bound;

  i_t num_initialized_down;
  i_t num_initialized_up;
  f_t pseudo_cost_down_avg;
  f_t pseudo_cost_up_avg;

  initialized(num_initialized_down, num_initialized_up, pseudo_cost_down_avg, pseudo_cost_up_avg);

  for (i_t k = 0; k < num_fractional; k++) {
    const i_t j = fractional[k];

    f_t pseudo_cost_down = 0;
    f_t pseudo_cost_up   = 0;

    pseudo_cost_mutex[j].lock();
    if (pseudo_cost_num_down[j] != 0) {
      pseudo_cost_down = pseudo_cost_sum_down[j] / pseudo_cost_num_down[j];
    } else {
      pseudo_cost_down = pseudo_cost_down_avg;
    }

    if (pseudo_cost_num_up[j] != 0) {
      pseudo_cost_up = pseudo_cost_sum_up[j] / pseudo_cost_num_up[j];
    } else {
      pseudo_cost_up = pseudo_cost_up_avg;
    }
    pseudo_cost_mutex[j].unlock();

    constexpr f_t eps = 1e-6;
    const f_t f_down  = solution[j] - std::floor(solution[j]);
    const f_t f_up    = std::ceil(solution[j]) - solution[j];
    estimate +=
      std::min(std::max(pseudo_cost_down * f_down, eps), std::max(pseudo_cost_up * f_up, eps));
  }

  log.debug("pseudocost estimate = %e\n", estimate);
  return estimate;
}

template <typename i_t, typename f_t>
void pseudo_costs_t<i_t, f_t>::update_pseudo_costs_from_strong_branching(
  const std::vector<i_t>& fractional, const std::vector<f_t>& root_soln)
{
  for (i_t k = 0; k < fractional.size(); k++) {
    const i_t j = fractional[k];
    for (i_t branch = 0; branch < 2; branch++) {
      if (branch == 0) {
        f_t change_in_obj = strong_branch_down[k];
        if (std::isnan(change_in_obj)) { continue; }
        f_t frac = root_soln[j] - std::floor(root_soln[j]);
        pseudo_cost_sum_down[j] += change_in_obj / frac;
        pseudo_cost_num_down[j]++;
      } else {
        f_t change_in_obj = strong_branch_up[k];
        if (std::isnan(change_in_obj)) { continue; }
        f_t frac = std::ceil(root_soln[j]) - root_soln[j];
        pseudo_cost_sum_up[j] += change_in_obj / frac;
        pseudo_cost_num_up[j]++;
      }
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class pseudo_costs_t<int, double>;

template void strong_branching<int, double>(const lp_problem_t<int, double>& original_lp,
                                            const simplex_solver_settings_t<int, double>& settings,
                                            double start_time,
                                            const std::vector<variable_type_t>& var_types,
                                            const std::vector<double> root_soln,
                                            const std::vector<int>& fractional,
                                            double root_obj,
                                            const std::vector<variable_status_t>& root_vstatus,
                                            const std::vector<double>& edge_norms,
                                            pseudo_costs_t<int, double>& pc);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
