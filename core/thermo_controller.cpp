#include "thermo_controller.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace korith::core {

namespace {

constexpr int kDefaultMaxDepth = 12;
constexpr int kMinDepthFloor = 2;
constexpr int kHardMaxDepth = 32;

// Normalized entropy threshold in [0, 1].
constexpr float kDefaultEntropyThreshold = 0.35f;
constexpr float kEntropySpikeDelta = 0.10f;

int parse_env_i32(const char * name, int default_value) {
  const char * v = std::getenv(name);
  if (!v || v[0] == '\0') {
    return default_value;
  }
  char * end = nullptr;
  const long parsed = std::strtol(v, &end, 10);
  if (!end || end == v || *end != '\0') {
    return default_value;
  }
  if (parsed < 1 || parsed > std::numeric_limits<int>::max()) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

int max_depth_configured() {
  static int cached = 0;
  if (cached != 0) {
    return cached;
  }
  const int env = parse_env_i32("KORITH_MAX_DEPTH", kDefaultMaxDepth);
  cached = std::clamp(env, 1, kHardMaxDepth);
  return cached;
}

float normalized_entropy_from_logits(const float * logits, int32_t n_vocab) {
  if (!logits || n_vocab <= 1) {
    return 0.0f;
  }

  float max_logit = -std::numeric_limits<float>::infinity();
  for (int32_t i = 0; i < n_vocab; ++i) {
    max_logit = std::max(max_logit, logits[i]);
  }
  if (!std::isfinite(max_logit)) {
    return 0.0f;
  }

  double sum = 0.0;
  double sum_z_logz = 0.0;
  for (int32_t i = 0; i < n_vocab; ++i) {
    const double x = static_cast<double>(logits[i] - max_logit);
    const double z = std::exp(x);
    sum += z;
    sum_z_logz += z * x;  // x = log(z)
  }

  if (!(sum > 0.0) || !std::isfinite(sum)) {
    return 0.0f;
  }

  const double h = std::log(sum) - (sum_z_logz / sum);
  const double denom = std::log(static_cast<double>(n_vocab));
  if (!(denom > 1e-12)) {
    return 0.0f;
  }

  double hn = h / denom;
  if (!std::isfinite(hn)) {
    hn = 0.0;
  }
  if (hn < 0.0) {
    hn = 0.0;
  } else if (hn > 1.0) {
    hn = 1.0;
  }
  return static_cast<float>(hn);
}

ThermoState g_state{};

// EWMA moments for accept_ema time-series variance.
bool g_have_accept = false;
float g_accept_m1 = 0.0f;
float g_accept_m2 = 0.0f;

bool g_have_entropy = false;
float g_last_entropy = 0.0f;

}  // namespace

int thermo_next_depth(const Metrics & metrics) {
  const int max_depth = max_depth_configured();

  float accept = metrics.accept_ema;
  if (!std::isfinite(accept)) {
    accept = 0.0f;
  } else if (accept < 0.0f) {
    accept = 0.0f;
  } else if (accept > 1.0f) {
    accept = 1.0f;
  }
  g_state.accept_ema = accept;

  // Short-term acceptance variance: EWMA variance of the accept_ema signal.
  constexpr float kAlphaVar = 0.10f;
  if (!g_have_accept) {
    g_have_accept = true;
    g_accept_m1 = accept;
    g_accept_m2 = accept * accept;
  } else {
    g_accept_m1 = (1.0f - kAlphaVar) * g_accept_m1 + kAlphaVar * accept;
    g_accept_m2 = (1.0f - kAlphaVar) * g_accept_m2 + kAlphaVar * (accept * accept);
  }
  float var = g_accept_m2 - g_accept_m1 * g_accept_m1;
  if (!std::isfinite(var) || var < 0.0f) {
    var = 0.0f;
  }
  g_state.accept_var = var;

  float entropy = 0.0f;
  if (metrics.logits && metrics.n_vocab > 0) {
    entropy = normalized_entropy_from_logits(metrics.logits, metrics.n_vocab);
  } else {
    // Acceptance proxy when logits are unavailable.
    entropy = 1.0f - accept;
  }
  if (!std::isfinite(entropy)) {
    entropy = 0.0f;
  } else if (entropy < 0.0f) {
    entropy = 0.0f;
  } else if (entropy > 1.0f) {
    entropy = 1.0f;
  }
  g_state.entropy = entropy;

  const bool entropy_spike =
      g_have_entropy && (entropy > g_last_entropy + kEntropySpikeDelta) && (entropy > kDefaultEntropyThreshold);
  g_have_entropy = true;
  g_last_entropy = entropy;

  // Depth control policy.
  //
  // Dynamic k adjustment:
  //   accept > 0.85 → increase depth (up to max 12)
  //   accept < 0.65 → decrease depth (down to min 2)
  //   Entropy spike  → decrease depth (safety valve)
  if (g_state.depth < kMinDepthFloor) {
    g_state.depth = kMinDepthFloor;
  }

  if (accept < 0.65f || entropy_spike) {
    g_state.depth = std::max(kMinDepthFloor, g_state.depth - 1);
  } else if (accept > 0.85f && var < 0.01f && entropy < kDefaultEntropyThreshold) {
    g_state.depth = std::min(max_depth, g_state.depth + 1);
  }

  g_state.depth = std::clamp(g_state.depth, kMinDepthFloor, max_depth);
  return g_state.depth;
}

ThermoState thermo_state() {
  return g_state;
}

void thermo_reset() {
  g_state = ThermoState{};
  g_state.depth = kMinDepthFloor;

  g_have_accept = false;
  g_accept_m1 = 0.0f;
  g_accept_m2 = 0.0f;

  g_have_entropy = false;
  g_last_entropy = 0.0f;
}

}  // namespace korith::core
