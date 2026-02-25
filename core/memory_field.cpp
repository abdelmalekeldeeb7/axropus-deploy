#include "memory_field.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace korith::core {

namespace {

// MF invariants:
// - Outputs are bounded and monotonic (bounded deltas only).
// - Replay enable/disable must not oscillate within a cooldown window.
// - MF never overrides AMF hard disables (engine enforcement).

constexpr std::uint64_t kPolicyUpdateEveryRequests = 2;
constexpr std::uint64_t kPolicyUpdateEveryMs = 100;
constexpr double kMinAdmitRoiMax = 10.0;
constexpr double kEvictionPressureMin = 0.5;
constexpr double kEvictionPressureMax = 3.0;
constexpr double kEvictionPressureMaxDelta = 0.5;
constexpr std::uint64_t kCooldownMaxMs = 60000;

std::uint64_t delta_u64(std::uint64_t now, std::uint64_t prev) {
  return (now >= prev) ? (now - prev) : 0;
}

double clamp_nonneg(double v) {
  if (!std::isfinite(v) || v < 0.0) {
    return 0.0;
  }
  return v;
}

double clamp_unit(double v) {
  if (!std::isfinite(v)) {
    return 0.0;
  }
  if (v < 0.0) {
    return 0.0;
  }
  if (v > 1.0) {
    return 1.0;
  }
  return v;
}

double clamp_range(double v, double lo, double hi) {
  if (!std::isfinite(v)) {
    return lo;
  }
  if (v < lo) {
    return lo;
  }
  if (v > hi) {
    return hi;
  }
  return v;
}

bool small_change(double old_v, double new_v, double rel_eps, double abs_eps) {
  const double diff = std::fabs(new_v - old_v);
  const double scale = std::max(abs_eps, std::fabs(old_v) * rel_eps);
  return diff < scale;
}

}  // namespace

MemoryFieldOutput memory_field_update(MemoryFieldState * state,
                                      const MemoryFieldInput & input) {
  MemoryFieldOutput last{};
  if (state) {
    last = state->last;
  }
  MemoryFieldOutput out = last;

  if (state && state->has_prev_input) {
    const std::uint64_t elapsed =
        (input.now_ms > state->last_update_ms) ? (input.now_ms - state->last_update_ms) : 0;
    const bool cadence_ready =
        (state->requests_since_update >= kPolicyUpdateEveryRequests) ||
        (elapsed >= kPolicyUpdateEveryMs);
    if (!cadence_ready) {
      state->requests_since_update += 1;
      return last;
    }
  }

  std::uint64_t dhits = 0;
  std::uint64_t dmisses = 0;
  std::uint64_t dblocks = 0;
  std::uint64_t dfailures = 0;
  std::uint64_t dms_saved = 0;
  std::uint64_t drestore_ms = 0;

  if (state && state->has_prev_input) {
    dhits = delta_u64(input.hits, state->prev_input.hits);
    dmisses = delta_u64(input.misses, state->prev_input.misses);
    dblocks = delta_u64(input.blocks, state->prev_input.blocks);
    dfailures = delta_u64(input.failures, state->prev_input.failures);
    dms_saved = delta_u64(input.ms_saved, state->prev_input.ms_saved);
    drestore_ms = delta_u64(input.restore_ms, state->prev_input.restore_ms);
  }

  const double hit_denom = static_cast<double>(dhits + dmisses);
  const double hit_rate = (hit_denom > 0.0)
      ? static_cast<double>(dhits) / hit_denom
      : 0.0;
  const double window_roi = (drestore_ms > 0)
      ? static_cast<double>(dms_saved) / static_cast<double>(drestore_ms)
      : 0.0;

  const double pressure = (input.max_bytes > 0)
      ? clamp_unit(static_cast<double>(input.bytes) / static_cast<double>(input.max_bytes))
      : 0.0;

  const bool roi_bad = (drestore_ms > 0) && (window_roi < 1.0);
  const bool roi_very_bad = (drestore_ms > 0) && (window_roi < 0.5);
  const bool roi_good = (drestore_ms > 0) && (window_roi > 1.2);
  const bool hit_good = hit_rate > 0.20;
  const bool stability_good = roi_good || hit_good || (input.negative_roi_streak == 0);

  if (roi_bad || input.negative_roi_streak >= 3 || pressure > 0.90) {
    out.min_admit_roi = std::max(out.min_admit_roi, 1.0);
  } else if (roi_good && hit_good && pressure < 0.70 && out.min_admit_roi > 0.0) {
    out.min_admit_roi *= 0.90;
  }

  if (pressure > 0.90) {
    out.eviction_pressure = 2.0;
  } else if (pressure > 0.75) {
    out.eviction_pressure = 1.5;
  } else if (pressure < 0.50) {
    out.eviction_pressure = 0.8;
  } else {
    out.eviction_pressure = 1.0;
  }

  const bool disable_replay = (input.negative_roi_streak >= 3) || (roi_very_bad && dhits >= 2);
  if (disable_replay) {
    out.replay_disable_mask = 1u;
    out.cooldown_ms = std::max<std::uint64_t>(out.cooldown_ms, 2000u);
  } else if (stability_good) {
    out.replay_disable_mask = 0u;
    out.cooldown_ms = 0u;
  }

  if (out.eviction_pressure > last.eviction_pressure + kEvictionPressureMaxDelta) {
    out.eviction_pressure = last.eviction_pressure + kEvictionPressureMaxDelta;
  } else if (out.eviction_pressure < last.eviction_pressure - kEvictionPressureMaxDelta) {
    out.eviction_pressure = last.eviction_pressure - kEvictionPressureMaxDelta;
  }

  if (state) {
    const std::uint32_t last_mask = last.replay_disable_mask & 0x1u;
    std::uint32_t next_mask = out.replay_disable_mask & 0x1u;
    const std::uint64_t window_ms = std::max<std::uint64_t>(out.cooldown_ms, last.cooldown_ms);
    const std::uint64_t window_ms_effective =
        (window_ms > 0) ? window_ms : kPolicyUpdateEveryMs;

    if (next_mask != last_mask) {
      const std::uint64_t now_ms = input.now_ms;
      std::uint32_t flip_count = state->replay_flip_count;
      if (state->last_replay_flip_ms != 0 &&
          now_ms > state->last_replay_flip_ms &&
          (now_ms - state->last_replay_flip_ms) <= window_ms_effective) {
        if (flip_count < std::numeric_limits<std::uint32_t>::max()) {
          flip_count += 1;
        }
      } else {
        flip_count = 1;
      }
      state->last_replay_flip_ms = now_ms;
      state->replay_flip_count = flip_count;
      if (flip_count > 1) {
        next_mask = 1u;
        out.cooldown_ms = std::max(out.cooldown_ms, window_ms_effective);
      }
    }
    out.replay_disable_mask = next_mask;
  }

  if (small_change(last.min_admit_roi, out.min_admit_roi, 0.10, 0.05)) {
    out.min_admit_roi = last.min_admit_roi;
  }
  if (small_change(last.eviction_pressure, out.eviction_pressure, 0.10, 0.05)) {
    out.eviction_pressure = last.eviction_pressure;
  }
  if (!stability_good && out.cooldown_ms < last.cooldown_ms) {
    out.cooldown_ms = last.cooldown_ms;
    out.replay_disable_mask = last.replay_disable_mask;
  }

  out.min_admit_roi = clamp_range(clamp_nonneg(out.min_admit_roi), 0.0, kMinAdmitRoiMax);
  if (!std::isfinite(out.eviction_pressure) || out.eviction_pressure <= 0.0) {
    out.eviction_pressure = 1.0;
  }
  out.eviction_pressure = clamp_range(out.eviction_pressure, kEvictionPressureMin, kEvictionPressureMax);
  out.replay_disable_mask &= 0x1u;
  if (out.cooldown_ms > kCooldownMaxMs) {
    out.cooldown_ms = kCooldownMaxMs;
  }

  const bool changed =
      (!small_change(last.min_admit_roi, out.min_admit_roi, 0.001, 0.001)) ||
      (!small_change(last.eviction_pressure, out.eviction_pressure, 0.001, 0.001)) ||
      (last.replay_disable_mask != out.replay_disable_mask) ||
      (last.cooldown_ms != out.cooldown_ms);

  if (changed) {
    std::fprintf(stderr,
                 "[MF_POLICY] min_admit_roi=%.2f->%.2f evict_pressure=%.2f->%.2f "
                 "replay_mask=0x%02x cooldown_ms=%llu "
                 "roi=%.2f hits=%llu misses=%llu blocks=%llu failures=%llu pressure=%.2f\n",
                 last.min_admit_roi,
                 out.min_admit_roi,
                 last.eviction_pressure,
                 out.eviction_pressure,
                 static_cast<unsigned>(out.replay_disable_mask),
                 static_cast<unsigned long long>(out.cooldown_ms),
                 window_roi,
                 static_cast<unsigned long long>(dhits),
                 static_cast<unsigned long long>(dmisses),
                 static_cast<unsigned long long>(dblocks),
                 static_cast<unsigned long long>(dfailures),
                 pressure);
    (void) std::fflush(stderr);
  }

  if (state) {
    state->last = out;
    state->prev_input = input;
    state->has_prev_input = true;
    state->last_update_ms = input.now_ms;
    state->requests_since_update = 0;
  }

  return out;
}

namespace {

bool extract_double(const std::string & json, const char * key, double & out) {
  const std::string needle = std::string("\"") + key + "\"";
  const std::size_t pos = json.find(needle);
  if (pos == std::string::npos) {
    return false;
  }
  const std::size_t colon = json.find(':', pos + needle.size());
  if (colon == std::string::npos) {
    return false;
  }
  const char * start = json.c_str() + colon + 1;
  char * end = nullptr;
  const double v = std::strtod(start, &end);
  if (end == start || !std::isfinite(v)) {
    return false;
  }
  out = v;
  return true;
}

bool extract_u64(const std::string & json, const char * key, std::uint64_t & out) {
  const std::string needle = std::string("\"") + key + "\"";
  const std::size_t pos = json.find(needle);
  if (pos == std::string::npos) {
    return false;
  }
  const std::size_t colon = json.find(':', pos + needle.size());
  if (colon == std::string::npos) {
    return false;
  }
  const char * start = json.c_str() + colon + 1;
  char * end = nullptr;
  const unsigned long long v = std::strtoull(start, &end, 10);
  if (end == start) {
    return false;
  }
  out = static_cast<std::uint64_t>(v);
  return true;
}

}  // namespace

std::string memory_field_state_to_json(const MemoryFieldState & state) {
  char buf[512];
  std::snprintf(
      buf,
      sizeof(buf),
      "{"
      "\"last_update_ms\":%llu,"
      "\"requests_since_update\":%llu,"
      "\"last_replay_flip_ms\":%llu,"
      "\"replay_flip_count\":%u,"
      "\"last\":{"
      "\"min_admit_roi\":%.6f,"
      "\"eviction_pressure\":%.6f,"
      "\"replay_disable_mask\":%u,"
      "\"cooldown_ms\":%llu"
      "}"
      "}",
      static_cast<unsigned long long>(state.last_update_ms),
      static_cast<unsigned long long>(state.requests_since_update),
      static_cast<unsigned long long>(state.last_replay_flip_ms),
      static_cast<unsigned>(state.replay_flip_count),
      state.last.min_admit_roi,
      state.last.eviction_pressure,
      static_cast<unsigned>(state.last.replay_disable_mask),
      static_cast<unsigned long long>(state.last.cooldown_ms));
  return std::string(buf);
}

bool memory_field_state_from_json(const std::string & json, MemoryFieldState * state) {
  if (!state) {
    return false;
  }
  MemoryFieldState out{};
  std::uint64_t u64 = 0;
  double f64 = 0.0;

  if (extract_u64(json, "last_update_ms", u64)) {
    out.last_update_ms = u64;
  }
  if (extract_u64(json, "requests_since_update", u64)) {
    out.requests_since_update = u64;
  }
  if (extract_u64(json, "last_replay_flip_ms", u64)) {
    out.last_replay_flip_ms = u64;
  }
  if (extract_u64(json, "replay_flip_count", u64)) {
    out.replay_flip_count = static_cast<std::uint32_t>(u64);
  }

  if (!extract_double(json, "min_admit_roi", f64)) {
    return false;
  }
  out.last.min_admit_roi = f64;
  if (!extract_double(json, "eviction_pressure", f64)) {
    return false;
  }
  out.last.eviction_pressure = f64;
  if (!extract_u64(json, "replay_disable_mask", u64)) {
    return false;
  }
  out.last.replay_disable_mask = static_cast<std::uint32_t>(u64);
  if (!extract_u64(json, "cooldown_ms", u64)) {
    return false;
  }
  out.last.cooldown_ms = u64;

  out.has_prev_input = false;
  *state = out;
  return true;
}

}  // namespace korith::core
