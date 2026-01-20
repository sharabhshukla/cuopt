/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "work_unit_scheduler.hpp"

#include "work_limit_timer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <mip/logger.hpp>

namespace cuopt {

work_unit_scheduler_t::work_unit_scheduler_t(double sync_interval) : sync_interval_(sync_interval)
{
}

void work_unit_scheduler_t::register_context(work_limit_context_t& ctx)
{
  std::lock_guard<std::mutex> lock(mutex_);
  contexts_.push_back(ctx);
  callback_queues_[&ctx]  = callback_queue_t{};
  last_sync_target_[&ctx] = 0.0;
  ctx.scheduler           = this;
}

void work_unit_scheduler_t::deregister_context(work_limit_context_t& ctx)
{
  std::lock_guard<std::mutex> lock(mutex_);
  ctx.scheduler = nullptr;
  contexts_.erase(std::remove_if(contexts_.begin(),
                                 contexts_.end(),
                                 [&ctx](const std::reference_wrapper<work_limit_context_t>& ref) {
                                   return &ref.get() == &ctx;
                                 }),
                  contexts_.end());
  callback_queues_.erase(&ctx);
  last_sync_target_.erase(&ctx);
  cv_.notify_all();
}

void work_unit_scheduler_t::on_work_recorded(work_limit_context_t& ctx, double total_work)
{
  // Early exit if scheduler is stopped
  if (stopped_.load()) { return; }

  if (verbose) {
    double sync_target = current_sync_target();
    CUOPT_LOG_DEBUG("[%s] Work recorded: %f, sync_target: %f (gen %zu)",
                    ctx.name.c_str(),
                    total_work,
                    sync_target,
                    barrier_generation_);
  }

  // Loop to handle large work increments that cross multiple sync points
  while (!stopped_.load() && total_work >= current_sync_target()) {
    wait_at_sync_point(ctx, current_sync_target());
  }
}

void work_unit_scheduler_t::set_sync_callback(sync_callback_t callback)
{
  std::lock_guard<std::mutex> lock(mutex_);
  sync_callback_ = std::move(callback);
}

bool work_unit_scheduler_t::is_stopped() const { return stopped_.load(); }

void work_unit_scheduler_t::stop()
{
  stopped_.store(true);
  cv_.notify_all();
}

sync_result_t work_unit_scheduler_t::wait_for_next_sync(work_limit_context_t& ctx)
{
  if (stopped_.load()) { return sync_result_t::STOPPED; }

  // Advance work to next sync point
  double next_sync              = current_sync_target();
  ctx.global_work_units_elapsed = next_sync;
  wait_at_sync_point(ctx, next_sync);

  return stopped_.load() ? sync_result_t::STOPPED : sync_result_t::CONTINUE;
}

// void work_unit_scheduler_t::queue_callback(work_limit_context_t& source,
//                                            work_limit_context_t& destination,
//                                            callback_t callback)
// {
//   std::lock_guard<std::mutex> lock(mutex_);
//   double tag = source.global_work_units_elapsed;
//   auto it    = callback_queues_.find(&destination);
//   if (it != callback_queues_.end()) { it->second.push({tag, std::move(callback)}); }
// }

double work_unit_scheduler_t::current_sync_target() const
{
  if (sync_interval_ <= 0) return std::numeric_limits<double>::infinity();
  return (barrier_generation_ + 1) * sync_interval_;
}

void work_unit_scheduler_t::wait_at_sync_point(work_limit_context_t& ctx, double sync_target)
{
  auto wait_start = std::chrono::high_resolution_clock::now();

  std::unique_lock<std::mutex> lock(mutex_);

  last_sync_target_[&ctx] = sync_target;
  size_t my_generation    = barrier_generation_;
  contexts_at_barrier_++;

  if (verbose) {
    CUOPT_LOG_DEBUG("[%s] Waiting at sync point %.2f (gen %zu, %zu/%zu contexts)",
                    ctx.name.c_str(),
                    sync_target,
                    my_generation,
                    contexts_at_barrier_,
                    contexts_.size());
  }

  if (contexts_at_barrier_ == contexts_.size()) {
    current_sync_target_ = sync_target;
    barrier_generation_++;
    if (verbose) {
      CUOPT_LOG_DEBUG("[%s] All contexts arrived, new generation %zu, notifying",
                      ctx.name.c_str(),
                      barrier_generation_);
    }
    // Execute sync callback if registered (last arrival executes it)
    if (sync_callback_) {
      lock.unlock();
      bool should_stop = sync_callback_(sync_target);
      lock.lock();
      if (should_stop) { stopped_.store(true); }
    }
    cv_.notify_all();
  } else {
    cv_.wait(lock, [&] { return barrier_generation_ != my_generation || stopped_.load(); });
    if (verbose) { CUOPT_LOG_DEBUG("[%s] Woke up from first wait", ctx.name.c_str()); }
    if (stopped_.load()) {
      // Decrement barrier count before returning to avoid leaving others stuck
      contexts_at_barrier_--;
      cv_.notify_all();
      return;
    }
  }

  size_t my_exit_generation = exit_generation_;

  if (verbose) { CUOPT_LOG_DEBUG("[%s] Processing callbacks", ctx.name.c_str()); }
  // lock.unlock();
  // process_callbacks_for_context(ctx, sync_target);
  // lock.lock();
  if (verbose) { CUOPT_LOG_DEBUG("[%s] Done processing callbacks", ctx.name.c_str()); }

  contexts_at_barrier_--;
  if (contexts_at_barrier_ == 0) {
    exit_generation_++;
    if (verbose) {
      CUOPT_LOG_DEBUG("[%s] All contexts finished callbacks at sync point %.2f (exit gen %zu)",
                      ctx.name.c_str(),
                      sync_target,
                      exit_generation_);
    }
    cv_.notify_all();
  } else {
    if (verbose) {
      CUOPT_LOG_DEBUG("[%s] Waiting for other contexts to finish callbacks (%zu remaining)",
                      ctx.name.c_str(),
                      contexts_at_barrier_);
    }
    cv_.wait(lock, [&] { return exit_generation_ != my_exit_generation || stopped_.load(); });
    if (verbose) { CUOPT_LOG_DEBUG("[%s] Woke up from second wait", ctx.name.c_str()); }
  }

  // Track sync time
  auto wait_end    = std::chrono::high_resolution_clock::now();
  double wait_secs = std::chrono::duration<double>(wait_end - wait_start).count();
  ctx.total_sync_time += wait_secs;

  if (verbose) {
    CUOPT_LOG_DEBUG("[%s] Sync complete at %.2f, waited %.2f ms",
                    ctx.name.c_str(),
                    sync_target,
                    wait_secs * 1000.0);
  }
}

// void work_unit_scheduler_t::process_callbacks_for_context(work_limit_context_t& ctx,
//                                                           double up_to_work_units)
// {

// }

}  // namespace cuopt
