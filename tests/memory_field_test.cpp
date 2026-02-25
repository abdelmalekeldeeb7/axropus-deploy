#include "../core/memory_field.h"

#include <cmath>

namespace {

bool near(double a, double b, double eps = 1e-6) {
  return std::fabs(a - b) <= eps;
}

}  // namespace

int main() {
  using namespace korith::core;

  MemoryFieldState state{};

  MemoryFieldInput in{};
  in.now_ms = 1000;
  in.max_bytes = 1000;
  in.bytes = 500;

  MemoryFieldOutput out = memory_field_update(&state, in);
  if (!near(out.eviction_pressure, 1.0)) {
    return 1;
  }

  MemoryFieldInput in_bad = in;
  in_bad.now_ms = 2000;
  in_bad.hits = 4;
  in_bad.misses = 0;
  in_bad.ms_saved = 80;     // ROI = 0.4
  in_bad.restore_ms = 200;
  in_bad.negative_roi_streak = 1;

  out = memory_field_update(&state, in_bad);
  if (out.min_admit_roi < 1.0) {
    return 2;
  }
  if ((out.replay_disable_mask & 0x1u) == 0u) {
    return 4;
  }
  if (out.cooldown_ms < 2000u) {
    return 5;
  }

  MemoryFieldInput in_good = in_bad;
  in_good.now_ms = 3000;
  in_good.hits = 8;         // +4
  in_good.misses = 0;
  in_good.ms_saved = 280;   // +200 (ROI 2.0)
  in_good.restore_ms = 300; // +100
  in_good.negative_roi_streak = 0;

  out = memory_field_update(&state, in_good);
  if (!near(out.min_admit_roi, 0.9, 1e-3)) {
    return 6;
  }
  if ((out.replay_disable_mask & 0x1u) != 0u) {
    return 8;
  }
  if (out.cooldown_ms != 0u) {
    return 9;
  }

  MemoryFieldInput in_pressure = in_good;
  in_pressure.now_ms = 4000;
  in_pressure.bytes = 950;
  in_pressure.max_bytes = 1000;

  out = memory_field_update(&state, in_pressure);
  if (!near(out.eviction_pressure, 2.0, 1e-6)) {
    return 10;
  }

  return 0;
}
