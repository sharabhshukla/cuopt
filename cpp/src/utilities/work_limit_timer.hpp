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

#include <algorithm>
#include <string>
#include "timer.hpp"

namespace cuopt {

// In determinism mode, relies on a work limit accumulator; otherwise rely on a timer
// in non-determinism mode: 1s = 1wu
class work_limit_timer_t {
 public:
  work_limit_timer_t() : deterministic(false), work_limit(0), timer(0) {}
  work_limit_timer_t(bool deterministic, double work_limit_)
    : deterministic(deterministic), work_limit(work_limit_), timer(work_limit_)
  {
  }

  bool check_limit(const char* caller = __builtin_FUNCTION(),
                   const char* file   = __builtin_FILE(),
                   int line           = __builtin_LINE()) const noexcept
  {
    if (deterministic) {
      bool finished_now = work_total >= work_limit;
      if (finished_now && !finished) {
        finished                   = true;
        double actual_elapsed_time = timer.elapsed_time();
        // 10% timing error
        if (abs(actual_elapsed_time - work_limit) / work_limit > 0.10) {
          CUOPT_LOG_ERROR(
            "%s:%d: %s(): Work limit timer finished with a large discrepancy: %fs for %fwu",
            file,
            line,
            caller,
            actual_elapsed_time,
            work_limit);
        }
      }
      return finished;
    } else {
      return timer.check_time_limit();
    }
  }

  // in determinism mode, add the work units to the work limit accumulator
  void record_work(double work_units)
  {
    if (deterministic) { work_total += work_units; }
  }

  double remaining_units() const noexcept
  {
    if (deterministic) {
      return work_limit - work_total;
    } else {
      return timer.remaining_time();
    }
  }

  double remaining_time() const noexcept { return remaining_units(); }

  double elapsed_time() const noexcept
  {
    if (deterministic) {
      return work_total;
    } else {
      return timer.elapsed_time();
    }
  }

  bool check_time_limit(const char* caller = __builtin_FUNCTION(),
                        const char* file   = __builtin_FILE(),
                        int line           = __builtin_LINE()) const noexcept
  {
    return check_limit(caller, file, line);
  }

  bool check_half_time() const noexcept
  {
    if (deterministic) {
      return work_total >= work_limit / 2;
    } else {
      return timer.check_half_time();
    }
  }

  double clamp_remaining_time(double desired_time) const noexcept
  {
    return std::min<double>(desired_time, remaining_time());
  }

  double get_time_limit() const noexcept
  {
    if (deterministic) {
      return work_limit;
    } else {
      return timer.get_time_limit();
    }
  }

  void print_debug(std::string msg) const
  {
    if (deterministic) {
      printf("%s work_limit: %f remaining_work: %f elapsed_work: %f \n",
             msg.c_str(),
             work_limit,
             remaining_time(),
             elapsed_time());
    } else {
      timer.print_debug(msg);
    }
  }

  timer_t timer;
  double work_total{};
  double work_limit{};
  mutable bool finished{false};
  bool deterministic{false};
};
}  // namespace cuopt
