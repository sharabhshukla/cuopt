/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_profiler_host.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "../mip/mip_utils.cuh"

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/feasibility_jump/feasibility_jump.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver_context.cuh>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace cuopt::linear_programming::test {

void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

__global__ void VectorAdd(const float* A, const float* B, float* C, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) C[idx] = A[idx] + B[idx];
}

class cupti_profiler_t {
 public:
  cupti_profiler_t(const std::vector<const char*>& metrics)
    : metricNames_(metrics), pRangeProfilerObject_(nullptr)
  {
  }

  cupti_profiler_t(const cupti_profiler_t&)            = delete;
  cupti_profiler_t& operator=(const cupti_profiler_t&) = delete;

  ~cupti_profiler_t()
  {
    if (!pRangeProfilerObject_) return;
    CUpti_RangeProfiler_Disable_Params disableRangeProfiler = {
      .structSize = CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
    disableRangeProfiler.pRangeProfilerObject = pRangeProfilerObject_;
    cuptiRangeProfilerDisable(&disableRangeProfiler);
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
      .structSize = CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    cuptiProfilerDeInitialize(&profilerDeInitializeParams);
  }

  void initialize_and_enable(CUcontext cuContext)
  {
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {
      .structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    cuptiProfilerInitialize(&profilerInitializeParams);
    CUdevice device;
    cuCtxGetDevice(&device);
    CUpti_Device_GetChipName_Params getChipNameParams = {
      .structSize = CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    getChipNameParams.deviceIndex = (size_t)device;
    cuptiDeviceGetChipName(&getChipNameParams);
    chipName_                                     = getChipNameParams.pChipName;
    CUpti_RangeProfiler_Enable_Params enableRange = {
      .structSize = CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
    enableRange.ctx = cuContext;
    cuptiRangeProfilerEnable(&enableRange);
    pRangeProfilerObject_ = enableRange.pRangeProfilerObject;
  }

  void configure(CUpti_ProfilerRange range, CUpti_ProfilerReplayMode replayMode, size_t numOfRanges)
  {
    create_config_image();
    create_counter_data_image(numOfRanges);

    CUpti_RangeProfiler_SetConfig_Params setConfig = {
      .structSize = CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
    setConfig.pRangeProfilerObject = pRangeProfilerObject_;
    setConfig.configSize           = configImage_.size();
    setConfig.pConfig              = configImage_.data();
    setConfig.counterDataImageSize = counterDataImage_.size();
    setConfig.pCounterDataImage    = counterDataImage_.data();
    setConfig.range                = range;
    setConfig.replayMode           = replayMode;
    setConfig.maxRangesPerPass     = numOfRanges;
    setConfig.numNestingLevels     = 1;
    setConfig.minNestingLevel      = 1;
    setConfig.passIndex            = 0;
    setConfig.targetNestingLevel   = 0;
    cuptiRangeProfilerSetConfig(&setConfig);
  }

  void start()
  {
    CUpti_RangeProfiler_Start_Params p = {.structSize =
                                            CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
    p.pRangeProfilerObject             = pRangeProfilerObject_;
    cuptiRangeProfilerStart(&p);
  }

  void stop()
  {
    CUpti_RangeProfiler_Stop_Params p = {.structSize = CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
    p.pRangeProfilerObject            = pRangeProfilerObject_;
    cuptiRangeProfilerStop(&p);
  }

  std::unordered_map<std::string, size_t> decode()
  {
    CUpti_RangeProfiler_DecodeData_Params decodeData = {
      .structSize = CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
    decodeData.pRangeProfilerObject = pRangeProfilerObject_;
    cuptiRangeProfilerDecodeData(&decodeData);
    CUpti_RangeProfiler_GetCounterDataInfo_Params cdiParams = {
      .structSize = CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
    cdiParams.pCounterDataImage    = counterDataImage_.data();
    cdiParams.counterDataImageSize = counterDataImage_.size();
    cuptiRangeProfilerGetCounterDataInfo(&cdiParams);
    evaluate_all_ranges(cdiParams.numTotalRanges > 10 ? 10 : cdiParams.numTotalRanges);
    return metric_values;
  }

 private:
  void create_config_image()
  {
    CUpti_Profiler_Host_Initialize_Params hostInitializeParams = {
      .structSize = CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    hostInitializeParams.profilerType              = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    hostInitializeParams.pChipName                 = chipName_.c_str();
    hostInitializeParams.pCounterAvailabilityImage = nullptr;
    cuptiProfilerHostInitialize(&hostInitializeParams);
    CUpti_Profiler_Host_Object* pHostObject = hostInitializeParams.pHostObject;

    CUpti_Profiler_Host_ConfigAddMetrics_Params configAddMetricsParams = {
      .structSize = CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
    configAddMetricsParams.pHostObject   = pHostObject;
    configAddMetricsParams.ppMetricNames = metricNames_.data();
    configAddMetricsParams.numMetrics    = metricNames_.size();
    cuptiProfilerHostConfigAddMetrics(&configAddMetricsParams);

    CUpti_Profiler_Host_GetConfigImageSize_Params getConfigImageSizeParams = {
      .structSize = CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
    getConfigImageSizeParams.pHostObject = pHostObject;
    cuptiProfilerHostGetConfigImageSize(&getConfigImageSizeParams);
    configImage_.resize(getConfigImageSizeParams.configImageSize);

    CUpti_Profiler_Host_GetConfigImage_Params getConfigImageParams = {
      .structSize = CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
    getConfigImageParams.pHostObject     = pHostObject;
    getConfigImageParams.pConfigImage    = configImage_.data();
    getConfigImageParams.configImageSize = configImage_.size();
    cuptiProfilerHostGetConfigImage(&getConfigImageParams);

    CUpti_Profiler_Host_GetNumOfPasses_Params getNumOfPassesParam = {
      .structSize = CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
    getNumOfPassesParam.pConfigImage    = configImage_.data();
    getNumOfPassesParam.configImageSize = configImage_.size();
    cuptiProfilerHostGetNumOfPasses(&getNumOfPassesParam);
    printf("Num of Passes: %d\n", (int)getNumOfPassesParam.numOfPasses);
    CUpti_Profiler_Host_Deinitialize_Params deinitializeParams = {
      .structSize = CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    deinitializeParams.pHostObject = pHostObject;
    cuptiProfilerHostDeinitialize(&deinitializeParams);
  }

  void create_counter_data_image(size_t maxNumOfRangesInCounterDataImage)
  {
    CUpti_RangeProfiler_GetCounterDataSize_Params ctDataSize = {
      .structSize = CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
    ctDataSize.pRangeProfilerObject = pRangeProfilerObject_;
    ctDataSize.pMetricNames         = metricNames_.data();
    ctDataSize.numMetrics           = metricNames_.size();
    ctDataSize.maxNumOfRanges       = maxNumOfRangesInCounterDataImage;
    ctDataSize.maxNumRangeTreeNodes = maxNumOfRangesInCounterDataImage;
    cuptiRangeProfilerGetCounterDataSize(&ctDataSize);
    counterDataImage_.resize(ctDataSize.counterDataSize);
    CUpti_RangeProfiler_CounterDataImage_Initialize_Params initCtImg = {
      .structSize = CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initCtImg.pRangeProfilerObject = pRangeProfilerObject_;
    initCtImg.pCounterData         = counterDataImage_.data();
    initCtImg.counterDataSize      = counterDataImage_.size();
    cuptiRangeProfilerCounterDataImageInitialize(&initCtImg);
  }

  void evaluate_range(size_t rangeIndex, CUpti_Profiler_Host_Object* pHostObject)
  {
    std::vector<double> metricValues(metricNames_.size());
    CUpti_Profiler_Host_EvaluateToGpuValues_Params p = {
      .structSize = CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    p.pHostObject          = pHostObject;
    p.pCounterDataImage    = counterDataImage_.data();
    p.counterDataImageSize = counterDataImage_.size();
    p.ppMetricNames        = metricNames_.data();
    p.numMetrics           = metricNames_.size();
    p.rangeIndex           = rangeIndex;
    p.pMetricValues        = metricValues.data();
    cuptiProfilerHostEvaluateToGpuValues(&p);
    for (size_t i = 0; i < metricNames_.size(); ++i)
      metric_values[metricNames_[i]] += (size_t)metricValues[i];
  }

  void evaluate_all_ranges(size_t numOfRanges)
  {
    CUpti_Profiler_Host_Initialize_Params hostInitializeParams = {
      .structSize = CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    hostInitializeParams.profilerType              = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    hostInitializeParams.pChipName                 = chipName_.c_str();
    hostInitializeParams.pCounterAvailabilityImage = nullptr;
    cuptiProfilerHostInitialize(&hostInitializeParams);
    for (size_t i = 0; i < numOfRanges; ++i)
      evaluate_range(i, hostInitializeParams.pHostObject);
    CUpti_Profiler_Host_Deinitialize_Params deinitializeParams = {
      .structSize = CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    deinitializeParams.pHostObject = hostInitializeParams.pHostObject;
    cuptiProfilerHostDeinitialize(&deinitializeParams);
  }

  std::unordered_map<std::string, size_t> metric_values;
  std::vector<const char*> metricNames_;
  CUpti_RangeProfiler_Object* pRangeProfilerObject_;
  std::vector<uint8_t> counterDataImage_, configImage_;
  std::string chipName_;
};

void test_cupti()
{
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::set_current_device_resource(&cuda_mr);

  const raft::handle_t handle_{};
  auto stream               = handle_.get_stream();
  std::string test_instance = "pk1.mps";

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

  const int vectorLen = 100000;
  rmm::device_uvector<float> d_A(vectorLen, stream), d_B(vectorLen, stream), d_C(vectorLen, stream);

  auto random_generator = [](unsigned int seed) {
    return [=] __device__(int idx) {
      thrust::default_random_engine rng(seed);
      thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
      rng.discard(idx);
      return dist(rng);
    };
  };

  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(vectorLen),
                    d_A.begin(),
                    random_generator(1234));
  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(vectorLen),
                    d_B.begin(),
                    random_generator(5678));

  // Get CUDA context after CUDA operations have initialized it
  CUcontext cuContext;
  cuCtxGetCurrent(&cuContext);

  // cupti_profiler_t profiler({
  //     "sm__warps_launched.sum", "l1tex__t_sectors.sum", "l1tex__data_bank_reads.sum",
  //     "l1tex__data_bank_writes.sum", "l1tex__m_xbar2l1tex_read_bytes.sum",
  //     "l1tex__m_l1tex2xbar_write_bytes.sum", "lts__t_sectors_op_read.sum",
  //     "lts__t_sectors_op_write.sum"
  // });
  cupti_profiler_t profiler({"l1tex__t_sectors.sum"});
  profiler.initialize_and_enable(cuContext);
  profiler.configure(CUPTI_AutoRange, CUPTI_KernelReplay, 10);
  profiler.start();

  detail::fj_settings_t fj_settings;
  fj_settings.feasibility_run = false;
  fj_settings.iteration_limit = 10;
  fj_settings.seed            = 42;
  printf("Running FJ\n");
  auto solution = run_fj(problem, fj_settings).solution;

  // VectorAdd<<<(vectorLen + 127) / 128, 128, 0, stream.value()>>>(d_A.data(), d_B.data(),
  // d_C.data(), vectorLen);

  profiler.stop();

  stream.synchronize();

  auto metric_values = profiler.decode();
  for (const auto& [k, v] : metric_values)
    printf("%s: %zu\n", k.c_str(), v);

  const size_t reads         = 2 * vectorLen * sizeof(float);
  const size_t writes        = vectorLen * sizeof(float);
  const size_t sector_reads  = reads / 32;
  const size_t sector_writes = writes / 32;
  const size_t total_sectors = sector_reads + sector_writes;
  EXPECT_EQ(metric_values["l1tex__t_sectors.sum"], total_sectors);
  // EXPECT_EQ(metric_values["l1tex__m_xbar2l1tex_read_bytes.sum"], reads);
  // EXPECT_EQ(metric_values["l1tex__m_l1tex2xbar_write_bytes.sum"], writes);
}

TEST(cupti, test_cupti) { test_cupti(); }

}  // namespace cuopt::linear_programming::test
