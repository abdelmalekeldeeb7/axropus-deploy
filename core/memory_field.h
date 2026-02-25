#pragma once

#include <cstdint>
#include <string>

namespace korith::core {

// MF invariants:
// - Outputs are bounded and monotonic (bounded deltas only).
// - Replay enable/disable must not oscillate within a cooldown window.
// - MF never overrides AMF hard disables (engine enforcement).

// Aggregated, model-agnostic AMF telemetry.
struct MemoryFieldInput {
  std::uint64_t now_ms = 0;

  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t blocks = 0;
  std::uint64_t failures = 0;

  std::uint64_t tokens_saved = 0;
  std::uint64_t ms_saved = 0;
  std::uint64_t restore_ms = 0;

  std::uint64_t admissions = 0;
  std::uint64_t admission_rejects = 0;

  std::uint64_t evictions = 0;
  std::uint64_t evicted_bytes = 0;

  std::uint64_t entries = 0;
  std::uint64_t bytes = 0;
  std::uint64_t max_bytes = 0;

  std::uint64_t oldest_entry_age_ms = 0;
  std::uint64_t negative_roi_streak = 0;
};

// Policy knobs only. No direct execution control.
struct MemoryFieldOutput {
  double min_admit_roi = 0.0;
  double eviction_pressure = 1.0;
  // Bitmask for temporary replay disablement.
  // bit0 = disable all replay
  // other bits reserved for future scopes (e.g., class-based disables)
  std::uint32_t replay_disable_mask = 0;
  std::uint64_t cooldown_ms = 0;
};

struct MemoryFieldState {
  std::uint64_t last_update_ms = 0;
  std::uint64_t requests_since_update = 0;
  std::uint64_t last_replay_flip_ms = 0;
  std::uint32_t replay_flip_count = 0;
  MemoryFieldOutput last{};
  MemoryFieldInput prev_input{};
  bool has_prev_input = false;
};

MemoryFieldOutput memory_field_update(MemoryFieldState * state,
                                      const MemoryFieldInput & input);

// Snapshot helpers for MF restore/apply.
std::string memory_field_state_to_json(const MemoryFieldState & state);
bool memory_field_state_from_json(const std::string & json, MemoryFieldState * state);

}  // namespace korith::core
