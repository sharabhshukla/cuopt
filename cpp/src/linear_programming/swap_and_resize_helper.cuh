/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <utilities/macros.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

#include <thrust/iterator/zip_iterator.h>

#include <cuda/std/tuple>

#include <raft/util/cuda_utils.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
void matrix_swap(rmm::device_uvector<f_t>& matrix, i_t vector_size, i_t left_swap_index, i_t right_swap_index)
{
  const i_t batch_size = matrix.size() / vector_size;

  cuopt_assert(left_swap_index < right_swap_index, "Left swap index must be less than right swap index");
  cuopt_assert(left_swap_index < batch_size, "Left swap index is out of bounds");
  cuopt_assert(right_swap_index < batch_size, "Right swap index is out of bounds");
  cuopt_assert(vector_size > 0, "Vector size must be greater than 0");
  cuopt_assert(batch_size > 0, "Batch size must be greater than 0");
  
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(
      matrix.data() + (left_swap_index * vector_size), 
      matrix.data() + (right_swap_index * vector_size)),
    thrust::make_zip_iterator(
      matrix.data() + (left_swap_index * vector_size),
      matrix.data() + (right_swap_index * vector_size)),
    vector_size,
    [] HD(f_t a, f_t b) { return thrust::make_tuple(b, a); },
    matrix.stream()
  );
}

template <typename host_vector_t>
void host_vector_swap(host_vector_t& host_vector, int left_swap_index, int right_swap_index)
{
  cuopt_assert(left_swap_index < host_vector.size(), "Left swap index is out of bounds");
  cuopt_assert(right_swap_index < host_vector.size(), "Right swap index is out of bounds");
  cuopt_assert(left_swap_index < right_swap_index, "Left swap index must be less than right swap index");

  // Swap the id to swap to the end
  std::swap(host_vector[left_swap_index], host_vector[right_swap_index]);
}

template <typename i_t, typename f_t>
__global__ void device_scalar_swap(cuda::std::span<f_t> vector, i_t left_swap_index, i_t right_swap_index)
{
  cuopt_assert(threadIdx.x == 0 && blockIdx.x == 0, "This kernel should be launched with a single thread");
  cuda::std::swap(vector[left_swap_index], vector[right_swap_index]);
}

template <typename i_t, typename f_t>
void device_vector_swap(rmm::device_uvector<f_t>& device_vector, i_t left_swap_index, i_t right_swap_index)
{
  cuopt_assert(left_swap_index < device_vector.size(), "Left swap index is out of bounds");
  cuopt_assert(right_swap_index < device_vector.size(), "Right swap index is out of bounds");
  cuopt_assert(left_swap_index < right_swap_index, "Left swap index must be less than right swap index");

  // Swap the id to swap to the end
  device_scalar_swap<<<1, 1, 0, device_vector.stream()>>>(cuda::std::span(device_vector.data(), device_vector.size()), left_swap_index, right_swap_index);
}
}