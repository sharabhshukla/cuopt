/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * @file memory_instrumentation.hpp
 * @brief Memory access instrumentation utilities
 *
 * This file provides wrapper classes for tracking memory reads and writes.
 *
 * Usage:
 *   - Define CUOPT_ENABLE_MEMORY_INSTRUMENTATION to enable tracking
 *   - When undefined, all instrumentation becomes zero-overhead passthrough
 *     (record_*() calls inline away, no counter storage overhead)
 *
 * Example:
 *   ins_vector<int> vec;  // Instrumented std::vector<int>
 *   vec.push_back(42);
 *   auto val = vec[0];
 *   // When enabled: tracking occurs, counters accumulate
 *   // When disabled: direct passthrough, compiler optimizes away all overhead
 */
// Thank you Cursor!

#pragma once

#include <functional>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#define CUOPT_ENABLE_MEMORY_INSTRUMENTATION 1

#ifdef __NVCC__
#define HDI inline __host__ __device__
#else
#define HDI inline __attribute__((always_inline))
#endif

namespace cuopt {

// Define CUOPT_ENABLE_MEMORY_INSTRUMENTATION to 1 to enable memory tracking
// When 0, instrumentation becomes a zero-overhead passthrough

#if CUOPT_ENABLE_MEMORY_INSTRUMENTATION

// Manifold class to collect statistics from multiple instrumented objects
// Stores raw pointers to counters - works with any instrumented type
class instrumentation_manifold_t {
 public:
  instrumentation_manifold_t() = default;

  // Template add - works with any type that has byte_loads/byte_stores members
  template <typename Instrumented>
  void add(const std::string& description, const Instrumented& instrumented)
  {
    instrumented_.insert_or_assign(
      description, std::make_pair(&instrumented.byte_loads, &instrumented.byte_stores));
  }

  // Collect total loads and stores across all instrumented objects
  std::pair<size_t, size_t> collect()
  {
    size_t total_loads  = 0;
    size_t total_stores = 0;

    for (const auto& [name, counters] : instrumented_) {
      total_loads += *counters.first;
      total_stores += *counters.second;
    }

    return {total_loads, total_stores};
  }

  // Collect per-wrapper statistics
  std::vector<std::tuple<std::string, size_t, size_t>> collect_per_wrapper()
  {
    std::vector<std::tuple<std::string, size_t, size_t>> results;
    results.reserve(instrumented_.size());

    for (const auto& [name, counters] : instrumented_) {
      results.emplace_back(name, *counters.first, *counters.second);
    }

    return results;
  }

  // Collect total loads and stores, then flush counters
  std::pair<size_t, size_t> collect_and_flush()
  {
    auto result = collect();
    flush();
    return result;
  }

  void flush()
  {
    for (const auto& [name, counters] : instrumented_) {
      *const_cast<size_t*>(counters.first)  = 0;
      *const_cast<size_t*>(counters.second) = 0;
    }
  }

 private:
  std::unordered_map<std::string, std::pair<const size_t*, const size_t*>> instrumented_;
};

#else

// No-op manifold when instrumentation is disabled
class instrumentation_manifold_t {
 public:
  instrumentation_manifold_t() = default;
  template <typename Instrumented>
  void add(const std::string&, const Instrumented&)
  {
  }
  std::pair<size_t, size_t> collect() { return {0, 0}; }
  std::vector<std::tuple<std::string, size_t, size_t>> collect_per_wrapper() { return {}; }
  std::pair<size_t, size_t> collect_and_flush() { return {0, 0}; }
  void flush() {}
};

#endif  // CUOPT_ENABLE_MEMORY_INSTRUMENTATION

// Helper traits to detect container capabilities
namespace type_traits_utils {

template <typename T, typename = void>
struct has_reserve : std::false_type {};

template <typename T>
struct has_reserve<T, std::void_t<decltype(std::declval<T>().reserve(size_t{}))>> : std::true_type {
};

template <typename T, typename = void>
struct has_capacity : std::false_type {};

template <typename T>
struct has_capacity<T, std::void_t<decltype(std::declval<T>().capacity())>> : std::true_type {};

template <typename T, typename = void>
struct has_shrink_to_fit : std::false_type {};

template <typename T>
struct has_shrink_to_fit<T, std::void_t<decltype(std::declval<T>().shrink_to_fit())>>
  : std::true_type {};

template <typename T, typename = void>
struct has_push_back : std::false_type {};

template <typename T>
struct has_push_back<
  T,
  std::void_t<decltype(std::declval<T>().push_back(std::declval<typename T::value_type>()))>>
  : std::true_type {};

template <typename T, typename = void>
struct has_emplace_back : std::false_type {};

template <typename T>
struct has_emplace_back<T, std::void_t<decltype(std::declval<T>().emplace_back())>>
  : std::true_type {};

template <typename T, typename = void>
struct has_pop_back : std::false_type {};

template <typename T>
struct has_pop_back<T, std::void_t<decltype(std::declval<T>().pop_back())>> : std::true_type {};

template <typename T, typename = void>
struct has_data : std::false_type {};

template <typename T>
struct has_data<T, std::void_t<decltype(std::declval<T>().data())>> : std::true_type {};

template <typename T, typename = void>
struct has_resize : std::false_type {};

template <typename T>
struct has_resize<T, std::void_t<decltype(std::declval<T>().resize(size_t{}))>> : std::true_type {};

template <typename T, typename = void>
struct has_clear : std::false_type {};

template <typename T>
struct has_clear<T, std::void_t<decltype(std::declval<T>().clear())>> : std::true_type {};

template <typename T, typename = void>
struct has_max_size : std::false_type {};

template <typename T>
struct has_max_size<T, std::void_t<decltype(std::declval<T>().max_size())>> : std::true_type {};

template <typename T, typename = void>
struct has_front : std::false_type {};

template <typename T>
struct has_front<T, std::void_t<decltype(std::declval<T>().front())>> : std::true_type {};

template <typename T, typename = void>
struct has_back : std::false_type {};

template <typename T>
struct has_back<T, std::void_t<decltype(std::declval<T>().back())>> : std::true_type {};

}  // namespace type_traits_utils

#if CUOPT_ENABLE_MEMORY_INSTRUMENTATION

// Memory operation instrumentation wrapper for container-like types
// No inheritance - counters embedded directly for compiler optimization
template <typename T>
struct memop_instrumentation_wrapper_t {
  // Instrumentation counters - embedded directly, no base class
  mutable size_t byte_loads{0};
  mutable size_t byte_stores{0};

  HDI void reset_counters() const { byte_loads = byte_stores = 0; }

  template <typename U>
  HDI void record_load() const
  {
    byte_loads += sizeof(U);
  }

  template <typename U>
  HDI void record_store() const
  {
    byte_stores += sizeof(U);
  }

  template <typename U>
  HDI void record_rmw() const
  {
    byte_loads += sizeof(U);
    byte_stores += sizeof(U);
  }

  // Standard container type traits
  using value_type      = typename T::value_type;
  using size_type       = typename T::size_type;
  using difference_type = typename T::difference_type;
  using reference       = typename T::reference;
  using const_reference = typename T::const_reference;
  using pointer         = typename T::pointer;
  using const_pointer   = typename T::const_pointer;

  static_assert(std::is_trivially_copyable_v<value_type>,
                "value_type must be trivially copyable for memory instrumentation");
  static constexpr size_t type_size = sizeof(value_type);

  // Use native iterators - no instrumentation overhead on iteration
  // Memory access counting is done via operator[] and batch methods
  using iterator               = typename T::iterator;
  using const_iterator         = typename T::const_iterator;
  using reverse_iterator       = typename T::reverse_iterator;
  using const_reverse_iterator = typename T::const_reverse_iterator;

  // Constructors - cache data pointer for device access
  memop_instrumentation_wrapper_t() : array_(), data_ptr_(array_.data()) {}
  memop_instrumentation_wrapper_t(const T& arr) : array_(arr), data_ptr_(array_.data()) {}
  memop_instrumentation_wrapper_t(T&& arr) : array_(std::move(arr)), data_ptr_(array_.data()) {}

  // Forwarding constructor for underlying container initialization
  template <typename Arg,
            typename... Args,
            typename = std::enable_if_t<
              !std::is_same_v<std::decay_t<Arg>, memop_instrumentation_wrapper_t> &&
              !std::is_same_v<std::decay_t<Arg>, T> &&
              (sizeof...(Args) > 0 || !std::is_convertible_v<Arg, T>)>>
  explicit memop_instrumentation_wrapper_t(Arg&& arg, Args&&... args)
    : array_(std::forward<Arg>(arg), std::forward<Args>(args)...), data_ptr_(array_.data())
  {
  }

  // Copy/move - update data pointer cache, reset counters for new instance
  memop_instrumentation_wrapper_t(const memop_instrumentation_wrapper_t& other)
    : byte_loads(0), byte_stores(0), array_(other.array_), data_ptr_(array_.data())
  {
  }
  memop_instrumentation_wrapper_t(memop_instrumentation_wrapper_t&& other) noexcept
    : byte_loads(0), byte_stores(0), array_(std::move(other.array_)), data_ptr_(array_.data())
  {
  }
  memop_instrumentation_wrapper_t& operator=(const memop_instrumentation_wrapper_t& other)
  {
    if (this != &other) {
      array_    = other.array_;
      data_ptr_ = array_.data();
      // Don't copy counters - each instance tracks its own accesses
    }
    return *this;
  }
  memop_instrumentation_wrapper_t& operator=(memop_instrumentation_wrapper_t&& other) noexcept
  {
    if (this != &other) {
      array_    = std::move(other.array_);
      data_ptr_ = array_.data();
      // Don't move counters - each instance tracks its own accesses
    }
    return *this;
  }

  // Element access - return reference directly, count optimistically
  // Use cached data_ptr_ to avoid calling host-only std::vector methods from device code
  HDI reference operator[](size_type index)
  {
    record_store<value_type>();
    return data_ptr_[index];
  }

  HDI value_type operator[](size_type index) const
  {
    record_load<value_type>();
    return data_ptr_[index];
  }

  reference front()
  {
    record_store<value_type>();
    return array_.front();
  }
  value_type front() const
  {
    record_load<value_type>();
    return array_.front();
  }

  reference back()
  {
    record_store<value_type>();
    return array_.back();
  }
  value_type back() const
  {
    record_load<value_type>();
    return array_.back();
  }

  // Raw pointer access - bypasses instrumentation for hot loops
  pointer data() noexcept { return array_.data(); }
  const_pointer data() const noexcept { return array_.data(); }

  // Iterators - native iterators, no instrumentation overhead
  iterator begin() noexcept { return array_.begin(); }
  const_iterator begin() const noexcept { return array_.begin(); }
  const_iterator cbegin() const noexcept { return array_.cbegin(); }

  iterator end() noexcept { return array_.end(); }
  const_iterator end() const noexcept { return array_.end(); }
  const_iterator cend() const noexcept { return array_.cend(); }

  reverse_iterator rbegin() noexcept { return array_.rbegin(); }
  const_reverse_iterator rbegin() const noexcept { return array_.rbegin(); }
  const_reverse_iterator crbegin() const noexcept { return array_.crbegin(); }

  reverse_iterator rend() noexcept { return array_.rend(); }
  const_reverse_iterator rend() const noexcept { return array_.rend(); }
  const_reverse_iterator crend() const noexcept { return array_.crend(); }

  // Capacity - direct forwarding
  bool empty() const noexcept { return array_.empty(); }
  size_type size() const noexcept { return array_.size(); }
  size_type max_size() const noexcept { return array_.max_size(); }
  size_type capacity() const noexcept { return array_.capacity(); }

  void reserve(size_type new_cap)
  {
    array_.reserve(new_cap);
    data_ptr_ = array_.data();
  }
  void shrink_to_fit()
  {
    array_.shrink_to_fit();
    data_ptr_ = array_.data();
  }

  // Modifiers
  void clear() noexcept
  {
    array_.clear();
    data_ptr_ = array_.data();
  }

  void push_back(const value_type& value)
  {
    record_store<value_type>();
    array_.push_back(value);
    data_ptr_ = array_.data();
  }

  void push_back(value_type&& value)
  {
    record_store<value_type>();
    array_.push_back(std::move(value));
    data_ptr_ = array_.data();
  }

  template <typename... Args>
  void emplace_back(Args&&... args)
  {
    record_store<value_type>();
    array_.emplace_back(std::forward<Args>(args)...);
    data_ptr_ = array_.data();
  }

  void pop_back()
  {
    record_load<value_type>();
    array_.pop_back();
    // data_ptr_ unchanged - pop_back doesn't reallocate
  }

  void resize(size_type count)
  {
    size_type old_size = array_.size();
    array_.resize(count);
    data_ptr_ = array_.data();
    if (count > old_size) { byte_stores += (count - old_size) * type_size; }
  }

  void resize(size_type count, const value_type& value)
  {
    size_type old_size = array_.size();
    array_.resize(count, value);
    data_ptr_ = array_.data();
    if (count > old_size) { byte_stores += (count - old_size) * type_size; }
  }

  // Batch counting for manual instrumentation after raw pointer use
  void record_loads(size_t count) const { byte_loads += count * type_size; }
  void record_stores(size_t count) { byte_stores += count * type_size; }

  // Conversion operators
  operator T&() { return array_; }
  operator const T&() const { return array_; }

  T&& release_array() { return std::move(array_); }

  T& underlying() { return array_; }
  const T& underlying() const { return array_; }

 private:
  T array_;
  pointer __restrict__ data_ptr_{nullptr};  // Cached for device access
};

#else  // !CUOPT_ENABLE_MEMORY_INSTRUMENTATION

// Zero-overhead passthrough wrapper when instrumentation is disabled
template <typename T>
struct memop_instrumentation_wrapper_t {
  // No-op instrumentation methods for API compatibility
  HDI void reset_counters() const {}
  template <typename U>
  HDI void record_load() const
  {
  }
  template <typename U>
  HDI void record_store() const
  {
  }
  template <typename U>
  HDI void record_rmw() const
  {
  }
  using value_type             = typename T::value_type;
  using size_type              = typename T::size_type;
  using difference_type        = typename T::difference_type;
  using reference              = typename T::reference;
  using const_reference        = typename T::const_reference;
  using pointer                = typename T::pointer;
  using const_pointer          = typename T::const_pointer;
  using iterator               = typename T::iterator;
  using const_iterator         = typename T::const_iterator;
  using reverse_iterator       = typename T::reverse_iterator;
  using const_reverse_iterator = typename T::const_reverse_iterator;

  // Constructors
  memop_instrumentation_wrapper_t() = default;
  memop_instrumentation_wrapper_t(const T& arr) : array_(arr) {}
  memop_instrumentation_wrapper_t(T&& arr) : array_(std::move(arr)) {}

  template <typename Arg,
            typename... Args,
            typename = std::enable_if_t<
              !std::is_same_v<std::decay_t<Arg>, memop_instrumentation_wrapper_t> &&
              !std::is_same_v<std::decay_t<Arg>, T> &&
              (sizeof...(Args) > 0 || !std::is_convertible_v<Arg, T>)>>
  explicit memop_instrumentation_wrapper_t(Arg&& arg, Args&&... args)
    : array_(std::forward<Arg>(arg), std::forward<Args>(args)...)
  {
  }

  // Default copy/move
  memop_instrumentation_wrapper_t(const memop_instrumentation_wrapper_t&)                = default;
  memop_instrumentation_wrapper_t(memop_instrumentation_wrapper_t&&) noexcept            = default;
  memop_instrumentation_wrapper_t& operator=(const memop_instrumentation_wrapper_t&)     = default;
  memop_instrumentation_wrapper_t& operator=(memop_instrumentation_wrapper_t&&) noexcept = default;

  // Element access - direct passthrough
  reference operator[](size_type index) { return array_[index]; }
  const_reference operator[](size_type index) const { return array_[index]; }

  reference front() { return array_.front(); }
  const_reference front() const { return array_.front(); }

  reference back() { return array_.back(); }
  const_reference back() const { return array_.back(); }

  pointer data() noexcept { return array_.data(); }
  const_pointer data() const noexcept { return array_.data(); }

  // Iterators
  iterator begin() noexcept { return array_.begin(); }
  const_iterator begin() const noexcept { return array_.begin(); }
  const_iterator cbegin() const noexcept { return array_.cbegin(); }

  iterator end() noexcept { return array_.end(); }
  const_iterator end() const noexcept { return array_.end(); }
  const_iterator cend() const noexcept { return array_.cend(); }

  reverse_iterator rbegin() noexcept { return array_.rbegin(); }
  const_reverse_iterator rbegin() const noexcept { return array_.rbegin(); }
  const_reverse_iterator crbegin() const noexcept { return array_.crbegin(); }

  reverse_iterator rend() noexcept { return array_.rend(); }
  const_reverse_iterator rend() const noexcept { return array_.rend(); }
  const_reverse_iterator crend() const noexcept { return array_.crend(); }

  // Capacity
  bool empty() const noexcept { return array_.empty(); }
  size_type size() const noexcept { return array_.size(); }
  size_type max_size() const noexcept { return array_.max_size(); }
  size_type capacity() const noexcept { return array_.capacity(); }

  void reserve(size_type new_cap) { array_.reserve(new_cap); }
  void shrink_to_fit() { array_.shrink_to_fit(); }

  // Modifiers
  void clear() noexcept { array_.clear(); }
  void push_back(const value_type& value) { array_.push_back(value); }
  void push_back(value_type&& value) { array_.push_back(std::move(value)); }

  template <typename... Args>
  void emplace_back(Args&&... args)
  {
    array_.emplace_back(std::forward<Args>(args)...);
  }

  void pop_back() { array_.pop_back(); }
  void resize(size_type count) { array_.resize(count); }
  void resize(size_type count, const value_type& value) { array_.resize(count, value); }

  // Conversion operators
  operator T&() { return array_; }
  operator const T&() const { return array_; }

  T&& release_array() { return std::move(array_); }

  T& underlying() { return array_; }
  const T& underlying() const { return array_; }

  // No-op batch counting stubs for API compatibility
  void record_loads(size_t) const {}
  void record_stores(size_t) {}

 private:
  T array_;
};

#endif  // CUOPT_ENABLE_MEMORY_INSTRUMENTATION

// Convenience alias for instrumented std::vector
template <typename T>
using ins_vector = memop_instrumentation_wrapper_t<std::vector<T>>;

}  // namespace cuopt
