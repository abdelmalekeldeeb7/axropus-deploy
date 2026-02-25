#pragma once

#include <cstddef>
#include <cstdint>

struct HoloEvent {
  std::uint64_t step_id;
  std::int32_t token_id;
  std::uint8_t thermo_depth_at_commit;
  bool accepted;
  float acceptance_rate_at_commit;
  float entropy_at_commit;
  float speculative_cost;
  float tps_snapshot;
};

constexpr std::size_t kHoloCapacity = 50000;

// Write-only recording API (single writer, multi-reader accessors).
void holo_record(const HoloEvent & e) noexcept;
std::size_t holo_size() noexcept;
const HoloEvent * holo_data() noexcept;

// Confidence oracle:
// - Computes a mean confidence score over the most recent `steps_ahead` events.
// - Returns true iff mean confidence >= required_confidence.
//
// confidence = acceptance_rate * exp(-entropy)
//
// If `required_confidence` is <= 0 or non-finite, defaults to the configured value:
// - KORITH_HOLO_CONFIDENCE (default 0.92)
bool holo_is_future_safe(int steps_ahead, float required_confidence) noexcept;
