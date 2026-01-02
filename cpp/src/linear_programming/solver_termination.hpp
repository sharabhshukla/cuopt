/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "user_interrupt_handler.hpp"

#include <atomic>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming {

/**
 * @brief Controls solver termination based on time limit, user interrupt, and parent termination.
 *
 * This class owns its own timer and automatically registers with the global interrupt handler.
 * It can optionally be linked to a parent termination object (for sub-MIPs) to inherit
 * termination conditions.
 *
 * Usage:
 *   // Root termination (main solver)
 *   solver_termination_t termination(60.0);
 *
 *   // Slave termination (sub-MIP, linked to parent)
 *   solver_termination_t sub_termination(10.0, &parent_termination);
 *
 *   // Or inline with context
 *   solver_termination_t sub_termination(10.0, context.termination);
 */
class solver_termination_t {
 public:
  /**
   * @brief Construct a termination object.
   * @param time_limit Time limit in seconds.
   * @param parent Optional parent termination object to check for termination.
   */
  explicit solver_termination_t(double time_limit, solver_termination_t* parent = nullptr)
    : timer_(time_limit), parent_(parent)
  {
    callback_id_ = user_interrupt_handler_t::instance().register_callback(
      [this]() { request_user_termination(); });
  }

  ~solver_termination_t()
  {
    user_interrupt_handler_t::instance().unregister_callback(callback_id_);
  }

  // Non-copyable, non-movable (due to registered callback with 'this' pointer)
  solver_termination_t(const solver_termination_t&)            = delete;
  solver_termination_t& operator=(const solver_termination_t&) = delete;
  solver_termination_t(solver_termination_t&&)                 = delete;
  solver_termination_t& operator=(solver_termination_t&&)      = delete;

  /**
   * @brief Check if the solver should terminate.
   * @return true if user interrupt requested, time limit reached, or parent terminated.
   */
  bool should_terminate() const
  {
    if (user_interrupt_requested_.load(std::memory_order_relaxed)) { return true; }
    if (timer_.check_time_limit()) { return true; }
    if (parent_ != nullptr && parent_->should_terminate()) { return true; }
    return false;
  }

  /**
   * @brief Check if termination was due to user interrupt.
   * @return true if user requested termination via Ctrl-C.
   */
  bool user_interrupt_requested() const
  {
    return user_interrupt_requested_.load(std::memory_order_relaxed);
  }

  /**
   * @brief Request termination due to user interrupt (e.g., Ctrl-C).
   */
  void request_user_termination()
  {
    user_interrupt_requested_.store(true, std::memory_order_relaxed);
  }

  /**
   * @brief Get remaining time in seconds.
   */
  double remaining_time() const { return timer_.remaining_time(); }

  /**
   * @brief Get elapsed time in seconds.
   */
  double elapsed_time() const { return timer_.elapsed_time(); }

 private:
  timer_t timer_;
  solver_termination_t* parent_;
  std::atomic<bool> user_interrupt_requested_{false};
  size_t callback_id_;
};

}  // namespace cuopt::linear_programming
