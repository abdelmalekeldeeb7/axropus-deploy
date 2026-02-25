#include "memory.h"

#include "ring_buffer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>

namespace {

RingBuffer<HoloEvent, kHoloCapacity> g_holo;

}  // namespace

void holo_record(const HoloEvent & e) noexcept {
  g_holo.push(e);
}

std::size_t holo_size() noexcept {
  return g_holo.size();
}

const HoloEvent * holo_data() noexcept {
  return g_holo.data();
}

namespace {

float env_required_confidence() noexcept {
  static float cached = std::numeric_limits<float>::quiet_NaN();
  if (std::isfinite(cached)) {
    return cached;
  }

  float v = 0.92f;
  if (const char * env = std::getenv("KORITH_HOLO_CONFIDENCE")) {
    if (env[0] != '\0') {
      char * end = nullptr;
      const float parsed = std::strtof(env, &end);
      if (end && end != env && *end == '\0' && std::isfinite(parsed)) {
        v = parsed;
      }
    }
  }

  if (!std::isfinite(v)) {
    v = 0.92f;
  }
  v = std::clamp(v, 0.0f, 1.0f);
  cached = v;
  return cached;
}

float safe_accept(float x) noexcept {
  if (!std::isfinite(x)) {
    return 0.0f;
  }
  return std::clamp(x, 0.0f, 1.0f);
}

float safe_entropy(float x) noexcept {
  if (!std::isfinite(x)) {
    return 1.0f;
  }
  return std::clamp(x, 0.0f, 1.0f);
}

}  // namespace

bool holo_is_future_safe(int steps_ahead, float required_confidence) noexcept {
  if (steps_ahead <= 0 || steps_ahead > static_cast<int>(kHoloCapacity)) {
    std::fprintf(stderr, "[HOLO-ORACLE] steps=%d mean_conf=0.00 decision=FALLBACK\n", steps_ahead);
    (void) std::fflush(stderr);
    return false;
  }

  if (!(required_confidence > 0.0f) || !std::isfinite(required_confidence)) {
    required_confidence = env_required_confidence();
  }
  required_confidence = std::clamp(required_confidence, 0.0f, 1.0f);

  const std::uint64_t w = g_holo.write_index();
  if (w < static_cast<std::uint64_t>(steps_ahead)) {
    std::fprintf(stderr, "[HOLO-ORACLE] steps=%d mean_conf=0.00 decision=FALLBACK\n", steps_ahead);
    (void) std::fflush(stderr);
    return false;
  }

  double sum = 0.0;
  for (int i = 0; i < steps_ahead; ++i) {
    const std::uint64_t idx = (w - 1u) - static_cast<std::uint64_t>(i);
    const HoloEvent & e = g_holo.at(idx);

    const double a = static_cast<double>(safe_accept(e.acceptance_rate_at_commit));
    const double h = static_cast<double>(safe_entropy(e.entropy_at_commit));
    sum += a * std::exp(-h);
  }

  const double mean = sum / static_cast<double>(steps_ahead);
  const bool ok = mean >= static_cast<double>(required_confidence);
  std::fprintf(stderr,
               "[HOLO-ORACLE] steps=%d mean_conf=%.2f decision=%s\n",
               steps_ahead,
               mean,
               ok ? "SAFE" : "FALLBACK");
  (void) std::fflush(stderr);
  return ok;
}

extern "C" std::size_t engine_holo_size(void) noexcept {
  return holo_size();
}

extern "C" const HoloEvent * engine_holo_data(void) noexcept {
  return holo_data();
}
