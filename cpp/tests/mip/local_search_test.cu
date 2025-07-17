/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <linear_programming/pdlp.cuh>
#include <linear_programming/restart_strategy/pdlp_restart_strategy.cuh>
#include <linear_programming/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/diversity/diversity_manager.cuh>
#include <mip/feasibility_jump/feasibility_jump.cuh>
#include <mip/local_search/local_search.cuh>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver_context.cuh>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

static void setup_device_symbols(rmm::cuda_stream_view stream_view)
{
  raft::common::nvtx::range fun_scope("Setting device symbol");
  detail::set_adaptive_step_size_hyper_parameters(stream_view);
  detail::set_restart_hyper_parameters(stream_view);
  detail::set_pdlp_hyper_parameters(stream_view);
}

struct fj_tweaks_t {
  double objective_weight = 0;
};

struct fj_state_t {
  detail::solution_t<int, double> solution;
  std::vector<double> solution_vector;
  int minimums;
  double incumbent_objective;
  double incumbent_violation;
};

// Helper function to setup MIP solver and run FJ with given settings and initial solution
static uint32_t run_fp(std::string test_instance,
                       const detail::fj_settings_t& fj_settings,
                       fj_tweaks_t tweaks                   = {},
                       std::vector<double> initial_solution = {})
{
  const raft::handle_t handle_{};
  std::cout << "Running: " << test_instance << std::endl;

  auto path = cuopt::test::get_rapids_dataset_root_dir() + ("/mip/" + test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  init_handler(op_problem.get_handle_ptr());
  // run the problem constructor of MIP, so that we do bounds standardization
  detail::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();

  setup_device_symbols(op_problem.get_handle_ptr()->get_stream());

  detail::pdhg_solver_t<int, double> pdhg_solver(problem.handle_ptr, problem);
  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(&handle_,
                                                               problem,
                                                               10,
                                                               1.0,
                                                               pdhg_solver,
                                                               problem.reverse_coefficients,
                                                               problem.reverse_offsets,
                                                               problem.reverse_constraints,
                                                               true);

  auto settings       = mip_solver_settings_t<int, double>{};
  settings.time_limit = 30.;
  auto timer          = cuopt::timer_t(30);
  detail::mip_solver_t<int, double> solver(problem, settings, scaling, timer);
  problem.tolerances = settings.get_tolerances();

  rmm::device_uvector<double> lp_optimal_solution(problem.n_variables,
                                                  problem.handle_ptr->get_stream());

  detail::lp_state_t<int, double>& lp_state = problem.lp_state;
  // resize because some constructor might be called before the presolve
  lp_state.resize(problem, problem.handle_ptr->get_stream());
  detail::relaxed_lp_settings_t lp_settings{};
  lp_settings.time_limit            = std::numeric_limits<double>::max();
  lp_settings.tolerance             = 1e-6;
  lp_settings.return_first_feasible = false;
  lp_settings.save_state            = true;
  auto lp_result =
    detail::get_relaxed_lp_solution(problem, lp_optimal_solution, lp_state, lp_settings);
  EXPECT_EQ(lp_result.get_termination_status(), pdlp_termination_status_t::Optimal);
  clamp_within_var_bounds(lp_optimal_solution, &problem, problem.handle_ptr);

  detail::local_search_t<int, double> local_search(solver.context, lp_optimal_solution);

  detail::solution_t<int, double> solution(problem);

  printf("Model fingerprint: 0x%x\n", problem.get_fingerprint());

  local_search.fp.config.bounds_prop_timer_min          = std::numeric_limits<double>::max();
  local_search.fp.config.lp_run_time_after_feasible_min = std::numeric_limits<double>::max();
  local_search.fp.config.linproj_time_limit             = std::numeric_limits<double>::max();
  local_search.fp.config.fj_cycle_escape_time_limit     = std::numeric_limits<double>::max();
  local_search.fp.config.lp_verify_time_limit           = std::numeric_limits<double>::max();
  // local_search.fp.timer = timer_t{std::numeric_limits<double>::max()};

  bool is_feasible = false;
  int iterations   = 0;
  while (true) {
    is_feasible = local_search.fp.run_single_fp_descent(solution);
    printf("fp_loop it %d, is_feasible %d\n", iterations, is_feasible);
    // if feasible return true
    if (is_feasible) {
      break;
    }
    // if not feasible, it means it is a cycle
    else {
      is_feasible = local_search.fp.restart_fp(solution);
      if (is_feasible) { break; }
    }
    iterations++;
  }

  std::vector<uint32_t> hashes;
  hashes.push_back(detail::compute_hash(solution.get_host_assignment()));
  printf("hashes: 0x%x, hash of the hash: 0x%x\n", hashes[0], detail::compute_hash(hashes));

  return detail::compute_hash(hashes);
  // return {host_copy(solution_vector, problem.handle_ptr->get_stream()), iterations};
}

static uint32_t run_fp_check_determinism(std::string test_instance, int iter_limit)
{
  int seed = std::getenv("FJ_SEED") ? std::stoi(std::getenv("FJ_SEED")) : 42;

  detail::fj_settings_t fj_settings;
  fj_settings.time_limit             = 30.;
  fj_settings.mode                   = detail::fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = 20000 * 1000;
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = false;
  fj_settings.termination         = detail::fj_termination_flags_t::FJ_TERMINATION_ITERATION_LIMIT;
  fj_settings.iteration_limit     = iter_limit;
  fj_settings.load_balancing_mode = detail::fj_load_balancing_mode_t::ALWAYS_ON;
  fj_settings.seed                = seed;
  cuopt::seed_generator::set_seed(fj_settings.seed);

  return run_fp(test_instance, fj_settings);

  //    auto state     = run_fp(test_instance, fj_settings);
  //    auto& solution = state.solution;

  //    CUOPT_LOG_DEBUG("%s: Solution generated with FJ: is_feasible %d, objective %g (raw %g)",
  //                    test_instance.c_str(),
  //                    solution.get_feasible(),
  //                    solution.get_user_objective(),
  //                    solution.get_objective());

  //    static auto first_val = solution.get_user_objective();

  //    if (abs(solution.get_user_objective() - first_val) > 1) exit(0);
}

TEST(local_search, feasibility_pump_determinism)
{
  cuopt::default_logger().set_pattern("[%n] [%-6l] %v");

  for (const auto& instance : {
         //"thor50dday.mps",
         //"gen-ip054.mps",
         //"50v-10.mps",
         //"seymour1.mps",
         //"rmatr200-p5.mps"
         //"tr12-30.mps",
         "sct2.mps",
         //"uccase9.mps"
       }) {
    // for (int i = 0; i < 10; i++)
    //  while (true) {
    //    run_fp_check_determinism(instance, 1000);
    //  }

    unsigned long seed = std::random_device{}();
    std::cerr << "Tested with seed " << seed << "\n";
    uint32_t gold_hash = 0;
    for (int i = 0; i < 10; ++i) {
      uint32_t hash = run_fp_check_determinism(instance, 1000);
      if (i == 0) {
        gold_hash = hash;
        printf("Gold hash: 0x%x\n", gold_hash);
      } else {
        ASSERT_EQ(hash, gold_hash);
        printf("Hash: 0x%x\n", hash);
      }
    }
  }
}

}  // namespace cuopt::linear_programming::test
