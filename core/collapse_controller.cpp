#include "collapse_controller.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace korith::core {

namespace {

static inline float sanitize_nonneg_f32(float v) {
  if (!std::isfinite(v) || v < 0.0f) {
    return 0.0f;
  }
  return v;
}

static inline float sanitize_confidence_f32(float v) {
  if (!std::isfinite(v)) {
    return 0.0f;
  }
  return std::clamp(v, 0.0f, 1.0f);
}

static inline std::int32_t sanitize_depth_i32(std::int32_t v) {
  if (v < 1) {
    return 1;
  }
  return v;
}

static inline const char * reason_name(CollapseReason r) {
  switch (r) {
    case CollapseReason::kOk:
      return "ok";
    case CollapseReason::kDepthOne:
      return "depth<=1";
    case CollapseReason::kNoSignals:
      return "no_signals";
    case CollapseReason::kSignalsSuggestDepthOne:
      return "signals_depth<=1";
    case CollapseReason::kLowConfidence:
      return "low_confidence";
    case CollapseReason::kLowBenefit:
      return "low_benefit";
    case CollapseReason::kMissingCommitCallback:
      return "missing_commit_callback";
    case CollapseReason::kMissingFallbackCallback:
      return "missing_fallback_callback";
    case CollapseReason::kForcedValidation:
      return "forced_validation";
    default:
      return "unknown";
  }
}

}  // namespace

CollapseDecision CollapseController::decide(const EngineSignal * signals,
                                           std::size_t signal_count,
                                           std::int32_t scheduler_depth) const {
  CollapseDecision out{};
  out.scheduler_depth = sanitize_depth_i32(scheduler_depth);
  out.commit_depth = out.scheduler_depth;

  if (out.scheduler_depth <= 1) {
    out.commit = false;
    out.reason = CollapseReason::kDepthOne;
    return out;
  }

  out.engine_count = signal_count;
  out.min_confidence = 1.0f;
  out.total_cost = 0.0f;
  out.total_benefit = 0.0f;
  out.min_suggested_depth = out.scheduler_depth;

  if (!signals || signal_count == 0) {
    // Default behavior: when no engine signals are provided, honor the scheduler's depth.
    out.commit = true;
    out.reason = CollapseReason::kNoSignals;
    return out;
  }

  for (std::size_t i = 0; i < signal_count; ++i) {
    const EngineSignal & s = signals[i];
    const float conf = sanitize_confidence_f32(s.confidence);
    out.min_confidence = std::min(out.min_confidence, conf);
    const float cost = sanitize_nonneg_f32(s.cost_estimate);
    const float benefit = sanitize_nonneg_f32(s.benefit_estimate);
    out.total_cost += cost;
    out.total_benefit += benefit;
    out.min_suggested_depth = std::min(out.min_suggested_depth, sanitize_depth_i32(s.suggested_depth));
  }

  if (out.min_suggested_depth <= 1) {
    out.commit = false;
    out.reason = CollapseReason::kSignalsSuggestDepthOne;
    return out;
  }

  constexpr float kMinConfidence = 0.50f;
  if (out.min_confidence < kMinConfidence) {
    out.commit = false;
    out.reason = CollapseReason::kLowConfidence;
    return out;
  }

  if (out.total_benefit + 1e-6f < out.total_cost) {
    out.commit = false;
    out.reason = CollapseReason::kLowBenefit;
    return out;
  }

  out.commit = true;
  out.reason = CollapseReason::kOk;
  return out;
}

void CollapseController::execute(Context & ctx,
                                 const EngineSignal * signals,
                                 std::size_t signal_count,
                                 std::int32_t scheduler_depth,
                                 const CollapseCallbacks & callbacks) const {
  CollapseDecision d = decide(signals, signal_count, scheduler_depth);
  if (ctx.baseline_ready && d.reason == CollapseReason::kDepthOne) {
    d.commit = true;
    d.reason = CollapseReason::kNoSignals;
  }

  std::int32_t speculative_tokens = 0;
  if (signals && signal_count > 0) {
    double total = 0.0;
    for (std::size_t i = 0; i < signal_count; ++i) {
      total += static_cast<double>(sanitize_nonneg_f32(signals[i].benefit_estimate));
    }
    speculative_tokens = static_cast<std::int32_t>(std::lround(total));
  }
  const std::int32_t committed_tokens =
      (d.commit && d.reason == CollapseReason::kForcedValidation) ? speculative_tokens : 0;

  if (d.commit) {
    if (!callbacks.commit_speculative) {
      d.commit = false;
      d.reason = CollapseReason::kMissingCommitCallback;
    }
  } else {
    if (!callbacks.fallback_decode) {
      d.reason = CollapseReason::kMissingFallbackCallback;
    }
  }

  if (d.commit) {
    if (d.commit_depth < d.scheduler_depth) {
      std::fprintf(stderr,
                   "[COLLAPSE] clamp_depth=%d->%d reason=%s\n",
                   static_cast<int>(d.scheduler_depth),
                   static_cast<int>(d.commit_depth),
                   reason_name(d.reason));
      (void) std::fflush(stderr);
    }
    std::fprintf(stderr,
                 "[COLLAPSE] commit=true reason=%s scheduler_depth=%d commit_depth=%d engines=%zu "
                 "spec_tokens=%d committed=%d "
                 "min_conf=%.3f cost=%.3f benefit=%.3f suggested_depth=%d\n",
                 reason_name(d.reason),
                 static_cast<int>(d.scheduler_depth),
                 static_cast<int>(d.commit_depth),
                 d.engine_count,
                 static_cast<int>(speculative_tokens),
                 static_cast<int>(committed_tokens),
                 static_cast<double>(d.min_confidence),
                 static_cast<double>(d.total_cost),
                 static_cast<double>(d.total_benefit),
                 static_cast<int>(d.min_suggested_depth));
    (void) std::fflush(stderr);
    callbacks.commit_speculative(ctx, d.commit_depth);
    if (ctx.opaque) {
      const auto * metrics = static_cast<const CollapseMetrics *>(ctx.opaque);
      if (metrics->speculative_tokens > 0) {
        const double speculative = static_cast<double>(metrics->speculative_tokens);
        const double accepted = static_cast<double>(metrics->accepted_tokens);
        const double confidence = (speculative > 0.0) ? (accepted / speculative) : 0.0;
        std::fprintf(stderr,
                     "[COLLAPSE] accepted=%llu speculative=%llu confidence=%.3f\n",
                     static_cast<unsigned long long>(metrics->accepted_tokens),
                     static_cast<unsigned long long>(metrics->speculative_tokens),
                     confidence);
        (void) std::fflush(stderr);
      }
    }
    return;
  }

  std::fprintf(stderr,
               "[COLLAPSE] fallback=true reason=%s scheduler_depth=%d engines=%zu "
               "spec_tokens=%d committed=%d "
               "min_conf=%.3f cost=%.3f benefit=%.3f suggested_depth=%d\n",
               reason_name(d.reason),
               static_cast<int>(d.scheduler_depth),
               d.engine_count,
               static_cast<int>(speculative_tokens),
               static_cast<int>(committed_tokens),
               static_cast<double>(d.min_confidence),
               static_cast<double>(d.total_cost),
               static_cast<double>(d.total_benefit),
               static_cast<int>(d.min_suggested_depth));
  (void) std::fflush(stderr);

  if (callbacks.fallback_decode) {
    callbacks.fallback_decode(ctx);
  }

  if (ctx.opaque) {
    const auto * metrics = static_cast<const CollapseMetrics *>(ctx.opaque);
    if (metrics->speculative_tokens > 0) {
      const double speculative = static_cast<double>(metrics->speculative_tokens);
      const double accepted = static_cast<double>(metrics->accepted_tokens);
      const double confidence = (speculative > 0.0) ? (accepted / speculative) : 0.0;
      std::fprintf(stderr,
                   "[COLLAPSE] accepted=%llu speculative=%llu confidence=%.3f\n",
                   static_cast<unsigned long long>(metrics->accepted_tokens),
                   static_cast<unsigned long long>(metrics->speculative_tokens),
                   confidence);
      (void) std::fflush(stderr);
    }
  }
}

}  // namespace korith::core
