/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/logger.hpp>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace cuopt::linear_programming {

/**
 * @brief Global singleton that handles SIGINT (Ctrl-C) and invokes registered callbacks.
 *
 * Components that want to respond to user interrupts register a callback via
 * register_callback() and unregister via unregister_callback().
 *
 * Safety feature: If the user presses Ctrl-C 5 times within 5 seconds,
 * the process is forcefully terminated.
 */
class user_interrupt_handler_t {
 public:
  static user_interrupt_handler_t& instance()
  {
    static user_interrupt_handler_t instance;
    return instance;
  }

  /**
   * @brief Register a callback to be invoked on SIGINT.
   * @param callback Function to call when interrupt is received.
   * @return Registration ID for later removal.
   */
  size_t register_callback(std::function<void()> callback)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t id      = next_id_++;
    callbacks_[id] = std::move(callback);
    return id;
  }

  /**
   * @brief Unregister a previously registered callback.
   * @param id Registration ID returned by register_callback().
   */
  void unregister_callback(size_t id)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.erase(id);
  }

  // Non-copyable, non-movable
  user_interrupt_handler_t(const user_interrupt_handler_t&)            = delete;
  user_interrupt_handler_t& operator=(const user_interrupt_handler_t&) = delete;
  user_interrupt_handler_t(user_interrupt_handler_t&&)                 = delete;
  user_interrupt_handler_t& operator=(user_interrupt_handler_t&&)      = delete;

 private:
  static constexpr int force_quit_threshold      = 5;
  static constexpr int force_quit_window_seconds = 5;

  using time_point = std::chrono::steady_clock::time_point;

  user_interrupt_handler_t() { previous_handler_ = std::signal(SIGINT, &handle_signal); }

  ~user_interrupt_handler_t()
  {
    if (previous_handler_ != SIG_ERR) { std::signal(SIGINT, previous_handler_); }
  }

  static void handle_signal(int /*sig*/)
  {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);

    auto now    = std::chrono::steady_clock::now();
    auto cutoff = now - std::chrono::seconds(force_quit_window_seconds);

    // Remove timestamps older than the window
    while (!self.interrupt_times_.empty() && self.interrupt_times_.front() < cutoff) {
      self.interrupt_times_.pop_front();
    }
    self.interrupt_times_.push_back(now);

    // Force quit if too many interrupts in the window
    if (static_cast<int>(self.interrupt_times_.size()) >= force_quit_threshold) {
      CUOPT_LOG_INFO("Force quit: %d interrupts in %d seconds.",
                     force_quit_threshold,
                     force_quit_window_seconds);
      std::_Exit(128 + SIGINT);
    }

    // Invoke all registered callbacks
    for (const auto& [id, callback] : self.callbacks_) {
      callback();
    }

    auto remaining = force_quit_threshold - static_cast<int>(self.interrupt_times_.size());
    CUOPT_LOG_INFO(
      "Interrupt received. Stopping solver cleanly... (press Ctrl-C %d more time(s) to force quit)",
      remaining);
  }

  std::mutex mutex_;
  std::unordered_map<size_t, std::function<void()>> callbacks_;
  size_t next_id_{0};
  std::deque<time_point> interrupt_times_;
  void (*previous_handler_)(int) = SIG_ERR;
};

}  // namespace cuopt::linear_programming
