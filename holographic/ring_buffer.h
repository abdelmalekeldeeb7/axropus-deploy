#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

template <typename T, std::size_t Capacity>
class RingBuffer {
public:
  static_assert(Capacity > 0);
  static_assert(std::is_trivially_copyable_v<T>);

  RingBuffer() = default;
  RingBuffer(const RingBuffer &) = delete;
  RingBuffer & operator=(const RingBuffer &) = delete;

  static constexpr std::size_t capacity() noexcept { return Capacity; }

  void push(const T & value) noexcept {
    // Single-writer: publish element then advance the cursor with release semantics so
    // readers that observe the new cursor (acquire) also observe the data write.
    const std::uint64_t idx = write_idx_.load(std::memory_order_relaxed);
    const std::size_t slot = static_cast<std::size_t>(idx % Capacity);
    data_[slot] = value;
    write_idx_.store(idx + 1, std::memory_order_release);
  }

  std::uint64_t write_index() const noexcept {
    return write_idx_.load(std::memory_order_acquire);
  }

  std::size_t size() const noexcept {
    const std::uint64_t w = write_index();
    return (w < Capacity) ? static_cast<std::size_t>(w) : Capacity;
  }

  const T * data() const noexcept { return data_; }
  const T & at(std::uint64_t abs_index) const noexcept { return data_[static_cast<std::size_t>(abs_index % Capacity)]; }

private:
  alignas(64) std::atomic<std::uint64_t> write_idx_{0};
  T data_[Capacity]{};
};
