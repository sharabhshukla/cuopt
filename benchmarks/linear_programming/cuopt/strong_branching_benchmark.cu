/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <mps_parser/parser.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <cmath>

#include <rmm/mr/pool_memory_resource.hpp>

#include "benchmark_helper.hpp"

template <typename T>
auto host_copy(T const* device_ptr, size_t size, rmm::cuda_stream_view stream_view)
{
  if (!device_ptr) return std::vector<T>{};
  std::vector<T> host_vec(size);
  raft::copy(host_vec.data(), device_ptr, size, stream_view);
  stream_view.synchronize();
  return host_vec;
}

template <typename T>
auto host_copy(rmm::device_uvector<T> const& device_vec)
{
  return host_copy(device_vec.data(), device_vec.size(), device_vec.stream());
}

bool is_frational(double in)
{
  return std::fabs(in - std::round(in)) > 1e-5;
}

std::pair<cuopt::mps_parser::mps_data_model_t<int, double>, std::vector<cuopt::mps_parser::mps_data_model_t<int, double>>> create_batch_problem(const cuopt::mps_parser::mps_data_model_t<int, double>& op_problem, const cuopt::linear_programming::optimization_problem_solution_t<int, double>& solution, bool no_init_lower)
{ 
  std::vector<double> primal_sol = host_copy(solution.get_primal_solution());
  
  std::vector<std::pair<int, double>> pairs;
  for (size_t i = 0; i < op_problem.get_variable_types().size(); ++i)
  {
    auto c = op_problem.get_variable_types()[i];
    // Is integer in the MIP problem and current solution is factional
    if (c == 'I' && is_frational(primal_sol[i]))
      pairs.emplace_back(i, primal_sol[i]);
  }
  
  const int batch_size = pairs.size();

  if (batch_size == 0)
  {
    std::cout << "No fractional var, exiting" << std::endl;
    exit(0);
  }
  std::cout << "Found " << batch_size << " factional integer variables" << std::endl;

  // Create the problem batch to solve them individually

  const int total_size = no_init_lower ? batch_size : batch_size * 2;

  std::vector<cuopt::mps_parser::mps_data_model_t<int, double>> problems(total_size, op_problem);

  // Create the upper bounds
  for (int i = 0; i < batch_size; i++)
    problems[i].get_variable_upper_bounds()[pairs[i].first] = std::floor(pairs[i].second);
  // Create the lower bounds
  if (!no_init_lower)
    for (int i = 0; i < batch_size; i++)
      problems[i + batch_size].get_variable_lower_bounds()[pairs[i].first] = std::ceil(pairs[i].second);
  // Create batch problem on the original problem
  cuopt::mps_parser::mps_data_model_t<int, double> batch_problem(op_problem);
  const auto& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  std::vector<double> new_variable_lower_bounds(variable_lower_bounds.size() * total_size);
  std::vector<double> new_variable_upper_bounds(variable_upper_bounds.size() * total_size);

  for (int i = 0; i < total_size; i++)
    for (size_t j = 0; j < variable_lower_bounds.size(); ++j)
      new_variable_lower_bounds[i * variable_lower_bounds.size() + j] = problems[i].get_variable_lower_bounds()[j];
  for (int i = 0; i < total_size; i++)
    for (size_t j = 0; j < variable_upper_bounds.size(); ++j)
      new_variable_upper_bounds[i * variable_upper_bounds.size() + j] = problems[i].get_variable_upper_bounds()[j];

  batch_problem.set_variable_lower_bounds(new_variable_lower_bounds.data(), new_variable_lower_bounds.size());
  batch_problem.set_variable_upper_bounds(new_variable_upper_bounds.data(), new_variable_upper_bounds.size());

  return {batch_problem, problems};
}

static bool is_incorrect_objective(double reference, double objective)
{
  if (reference == 0) { return std::abs(objective) > 0.001; }
  if (objective == 0) { return std::abs(reference) > 0.001; }
  return std::abs((reference - objective) / reference) > 0.001;
}

template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

void bench(
  const raft::handle_t& handle,
  cuopt::mps_parser::mps_data_model_t<int, double>& original_problem, // Only useful for warm start
  cuopt::mps_parser::mps_data_model_t<int, double>& batch_problem,
  std::vector<cuopt::mps_parser::mps_data_model_t<int, double>>& problems,
  bool compare_with_baseline,
  bool deterministic, // For now useless, need 13.1 for cuSparse deterministic
  bool init_primal_dual,
  bool init_step_size,
  bool init_primal_weight)
{
  // Important that those 2 are created before the sols

  std::vector<cuopt::linear_programming::optimization_problem_solution_t<int, double>> sols;

  rmm::device_uvector<double> initial_primal(0, handle.get_stream());
  rmm::device_uvector<double> initial_dual(0, handle.get_stream());
  double initial_step_size = std::numeric_limits<double>::signaling_NaN();
  double initial_primal_weight = std::numeric_limits<double>::signaling_NaN();

  bool needs_warm_start_solution =
      init_primal_dual || init_step_size || init_primal_weight;

  if (needs_warm_start_solution)
  {
    // Solving the original to get its primal / dual vectors
    // Should not be necessary but weird behavior with conccurent halt is making things crash
    cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings_local;
    settings_local.method = cuopt::linear_programming::method_t::PDLP;
    settings_local.detect_infeasibility = true;

    cuopt::linear_programming::optimization_problem_solution_t<int, double> original_solution = cuopt::linear_programming::solve_lp(&handle, original_problem, settings_local);

    std::cout << "Original problem solved by PDLP in " << original_solution.get_additional_termination_information().solve_time << " using " << original_solution.get_additional_termination_information().number_of_steps_taken << std::endl;
    if (init_primal_dual) {
      initial_primal = rmm::device_uvector<double>(original_solution.get_primal_solution(), original_solution.get_primal_solution().stream());
      initial_dual = rmm::device_uvector<double>(original_solution.get_dual_solution(), original_solution.get_dual_solution().stream());
    }
    if (init_step_size) {
      initial_step_size = original_solution.get_pdlp_warm_start_data().initial_step_size_;
    }
    if (init_primal_weight) {
      initial_primal_weight = original_solution.get_pdlp_warm_start_data().initial_primal_weight_;
    }
  }

  auto start = std::chrono::steady_clock::now();

  if (compare_with_baseline)
  {
    for (size_t i = 0; i < problems.size(); ++i)
    {
      // Should not be necessary but weird behavior with conccurent halt is making things crash
      cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings_local;
      settings_local.method = cuopt::linear_programming::method_t::PDLP;
      settings_local.detect_infeasibility = true;
      settings_local.iteration_limit = 100000;

      if (init_primal_dual)
      {
        settings_local.set_initial_primal_solution(initial_primal.data(), initial_primal.size(), initial_primal.stream());
        settings_local.set_initial_dual_solution(initial_dual.data(), initial_dual.size(), initial_dual.stream());
      }
      if (init_step_size)
      {
        settings_local.set_initial_step_size(initial_step_size);
      }
      if (init_primal_weight)
      {
        settings_local.set_initial_primal_weight(initial_primal_weight);
      }

      sols.emplace_back(cuopt::linear_programming::solve_lp(&handle, problems[i], settings_local, true, true/*, "batch_instances/custom_" + std::to_string(i) + ".mps"*/));
      std::cout << "Version " << i << " solved " << sols[i].get_termination_status_string() << " using " << sols[i].get_additional_termination_information().number_of_steps_taken << std::endl;
    }
    std::cout << "All solved in " << since(start).count() / 1000.0 << std::endl;
  }

  // Should not be necessary but weird behavior with conccurent halt is making things crash/mer
  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings_local;
  settings_local.method = cuopt::linear_programming::method_t::PDLP;
  settings_local.detect_infeasibility = true;
  settings_local.iteration_limit = 100000;

  if (init_primal_dual)
  {
    settings_local.set_initial_primal_solution(initial_primal.data(), initial_primal.size(), initial_primal.stream());
    settings_local.set_initial_dual_solution(initial_dual.data(), initial_dual.size(), initial_dual.stream());
  }
  if (init_step_size)
  {
    settings_local.set_initial_step_size(initial_step_size);
  }
  if (init_primal_weight)
  {
    settings_local.set_initial_primal_weight(initial_primal_weight);
  }

  cuopt::linear_programming::optimization_problem_solution_t<int, double> batch_solution =
  cuopt::linear_programming::solve_lp(&handle, batch_problem, settings_local);

  std::cout << "Batch problem solved in " << batch_solution.get_additional_termination_information().solve_time << " using " << batch_solution.get_additional_termination_information().number_of_steps_taken << std::endl;

  if (compare_with_baseline)
  {
    for (size_t i = 0; i < sols.size(); ++i)
    {
      if (sols[i].get_termination_status() != batch_solution.get_termination_status(i))
              std::cout << "Terminations not equal at: " << i << " " << sols[i].get_termination_status_string() << " " << batch_solution.get_termination_status_string(i) << std::endl;

      if (is_incorrect_objective(sols[i].get_additional_termination_information().primal_objective, batch_solution.get_additional_termination_information(i).primal_objective))
        std::cout << "Objectives not equal at: " << i << " " << sols[i].get_additional_termination_information().primal_objective << " " << batch_solution.get_additional_termination_information(i).primal_objective << std::endl;
    }
  }
}

int main(int argc, char* argv[])
{
  // Initialize raft handle here to make sure it's destroyed very last
  const raft::handle_t handle_;

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings;
  settings.detect_infeasibility = true;


  // Setup up RMM memory pool
  auto memory_resource = make_pool();
  rmm::mr::set_current_device_resource(memory_resource.get());


  std::vector<std::string> problem_list = {"app1-1"};//{"afiro_original"};//{"scpj4scip"};//;

  bool compare_with_baseline = false;
  for (const auto& problem_name : problem_list)
  {
    // Parse MPS file
    cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>("batch_instances/" + problem_name + ".mps");

    // Solve using Dual Simplex to have the least amount of fractional variable
    settings.method = cuopt::linear_programming::method_t::DualSimplex;

    // Solve LP problem
    cuopt::linear_programming::optimization_problem_solution_t<int, double> solution =
    cuopt::linear_programming::solve_lp(&handle_, op_problem, settings);

    std::cout << "Original problem solved in " << solution.get_additional_termination_information().solve_time << " and " << solution.get_additional_termination_information().number_of_steps_taken << " steps" << std::endl;

    // Create a list of problems for each variante and update op_problem to batchify it
    auto [batch_problem, problems] = create_batch_problem(op_problem, solution, true);

    // The five commented bench calls below correspond to warm start combinations:

    // No warm start: all false
    //bench(handle_, op_problem,batch_problem, problems, compare_with_baseline, false /*deterministic*/, false, false, false, false);

    // Primal dual only: init_primal_dual = true
    //bench(handle_, op_problem, batch_problem, problems, compare_with_baseline, false /*deterministic*/, true, false, false, false);

    // Primal dual + step size + primal weight
    //bench(handle_, op_problem, batch_problem, problems, compare_with_baseline, false /*deterministic*/, true, true, true, false);

    // Primal dual + step size only (not primal weight)
    //bench(handle_, op_problem, batch_problem, problems, compare_with_baseline, false /*deterministic*/, true, true, false, false);

    // Primal dual + primal weight + step size + iteration count (the fullest warm start)
    //bench(handle_, op_problem, batch_problem, problems, compare_with_baseline, false /*deterministic*/, true /*primal dual*/, false /*step size*/, false /*primal weight*/);
    bench(handle_, op_problem, batch_problem, problems, compare_with_baseline, false /*deterministic*/, true /*primal dual*/, true /*step size*/, false /*primal weight*/);
    //bench(handle_, op_problem, batch_problem, problems, compare_with_baseline, false /*deterministic*/, true /*primal dual*/, true /*step size*/, true /*primal weight*/);
  }

  return 0;
}