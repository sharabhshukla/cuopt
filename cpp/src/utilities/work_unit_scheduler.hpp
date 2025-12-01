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
#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

namespace cuopt {

struct work_limit_context_t;

class work_unit_scheduler_t {
 public:
  using callback_t = std::function<void()>;

  explicit work_unit_scheduler_t(double sync_interval = 5.0);

  void set_sync_interval(double interval);
  double get_sync_interval() const;

  void register_context(work_limit_context_t& ctx);
  void deregister_context(work_limit_context_t& ctx);
  void on_work_recorded(work_limit_context_t& ctx, double total_work);
  void queue_callback(work_limit_context_t& source,
                      work_limit_context_t& destination,
                      callback_t callback);

 public:
  bool verbose{false};
  double sync_interval_;

 private:
  struct tagged_callback_t {
    double work_unit_tag;
    callback_t callback;

    bool operator>(const tagged_callback_t& other) const
    {
      return work_unit_tag > other.work_unit_tag;
    }
  };

  using callback_queue_t =
    std::priority_queue<tagged_callback_t, std::vector<tagged_callback_t>, std::greater<>>;

  double current_sync_target() const;
  void wait_at_sync_point(work_limit_context_t& ctx, double sync_target);
  void process_callbacks_for_context(work_limit_context_t& ctx, double up_to_work_units);

  std::vector<std::reference_wrapper<work_limit_context_t>> contexts_;
  std::unordered_map<work_limit_context_t*, callback_queue_t> callback_queues_;
  std::unordered_map<work_limit_context_t*, double> last_sync_target_;

  std::mutex mutex_;
  std::condition_variable cv_;
  size_t contexts_at_barrier_{0};
  double current_sync_target_{0};
  size_t barrier_generation_{0};
  size_t exit_generation_{0};
};

}  // namespace cuopt
