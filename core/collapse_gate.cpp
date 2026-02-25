#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace korith::core {

struct SpeculativeEngineResult {
  std::uint64_t tokens_generated = 0;
  float confidence = 0.0f;
  std::uint64_t compute_time = 0;
  float power_weight = 0.0f;

  void * merge_ctx = nullptr;
  void * kv_cache = nullptr;
  void (*merge_kv_cache)(void * merge_ctx, void * kv_cache) = nullptr;
  void (*discard_kv_cache)(void * merge_ctx, void * kv_cache) = nullptr;
};

namespace {

float sanitize_confidence(float v) {
  if (!std::isfinite(v)) {
    return 0.0f;
  }
  return std::clamp(v, 0.0f, 1.0f);
}

float sanitize_nonneg(float v) {
  if (!std::isfinite(v) || v < 0.0f) {
    return 0.0f;
  }
  return v;
}

std::uint64_t g_probe_runs = 0;

}  // namespace

int collapse_gate_commit(const std::vector<SpeculativeEngineResult> & engines) {
  std::uint64_t spec_tokens = 0;
  for (const SpeculativeEngineResult & e : engines) {
    spec_tokens += e.tokens_generated;
  }
  if (!engines.empty()) {
    g_probe_runs += 1;
  }

  if (spec_tokens == 0 && g_probe_runs == 0) {
    // Bootstrap only: allow one probe attempt before collapse is permitted.
    g_probe_runs = 1;
    std::fprintf(stderr, "[COLLAPSE_GATE] chosen_engine=-1 benefit=0.000 cost=0.000 net_gain=0.000\n");
    (void) std::fflush(stderr);
    return -1;
  }

  if (engines.empty()) {
    std::fprintf(stderr, "[COLLAPSE_GATE] chosen_engine=-1 benefit=0.000 cost=0.000 net_gain=0.000\n");
    (void) std::fflush(stderr);
    return -1;
  }

  int chosen = -1;
  double best_net = -std::numeric_limits<double>::infinity();
  double chosen_benefit = 0.0;
  double chosen_cost = 0.0;

  for (std::size_t i = 0; i < engines.size(); ++i) {
    const SpeculativeEngineResult & e = engines[i];
    const float conf = sanitize_confidence(e.confidence);
    const float weight = sanitize_nonneg(e.power_weight);

    const double benefit = static_cast<double>(e.tokens_generated) * static_cast<double>(conf);
    const double cost = static_cast<double>(e.compute_time) * static_cast<double>(weight);
    const double net = benefit - cost;

    if (net > best_net) {
      best_net = net;
      chosen = static_cast<int>(i);
      chosen_benefit = benefit;
      chosen_cost = cost;
    }
  }

  const double net_gain = chosen_benefit - chosen_cost;
  std::fprintf(stderr,
               "[COLLAPSE_GATE] chosen_engine=%d benefit=%.3f cost=%.3f net_gain=%.3f\n",
               chosen,
               chosen_benefit,
               chosen_cost,
               net_gain);
  (void) std::fflush(stderr);

  if (chosen < 0) {
    return -1;
  }

  for (std::size_t i = 0; i < engines.size(); ++i) {
    const SpeculativeEngineResult & e = engines[i];
    if (static_cast<int>(i) == chosen) {
      if (e.merge_kv_cache) {
        e.merge_kv_cache(e.merge_ctx, e.kv_cache);
      }
      continue;
    }
    if (e.discard_kv_cache) {
      e.discard_kv_cache(e.merge_ctx, e.kv_cache);
    }
  }

  return chosen;
}

}  // namespace korith::core
