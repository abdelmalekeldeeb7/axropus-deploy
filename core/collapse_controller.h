#pragma once

#include <cstddef>
#include <cstdint>

#include "engine_trait.h"

namespace korith::core {

struct CollapseMetrics {
  std::uint64_t speculative_tokens = 0;
  std::uint64_t accepted_tokens = 0;
};

struct CollapseCallbacks {
  // Commits accepted speculative tokens into the main decode stream.
  //
  // `depth` is the scheduler-provided final depth (optionally clamped by the controller).
  void (*commit_speculative)(Context & ctx, std::int32_t depth) = nullptr;

  // Executes the non-speculative fallback decode path (no speculative commit).
  void (*fallback_decode)(Context & ctx) = nullptr;
};

enum class CollapseReason : std::int32_t {
  kOk = 0,
  kDepthOne = 1,
  kNoSignals = 2,
  kSignalsSuggestDepthOne = 3,
  kLowConfidence = 4,
  kLowBenefit = 5,
  kMissingCommitCallback = 6,
  kMissingFallbackCallback = 7,
  kForcedValidation = 8,
};

struct CollapseDecision {
  bool commit = false;
  std::int32_t scheduler_depth = 1;
  std::int32_t commit_depth = 1;
  CollapseReason reason = CollapseReason::kDepthOne;

  // Aggregated signal metrics (best-effort; sanitized to finite values).
  std::size_t engine_count = 0;
  float min_confidence = 0.0f;
  float total_cost = 0.0f;
  float total_benefit = 0.0f;
  std::int32_t min_suggested_depth = 1;
};

class CollapseController {
public:
  CollapseController() = default;

  // Computes a collapse decision from engine signals + scheduler depth.
  CollapseDecision decide(const EngineSignal * signals,
                          std::size_t signal_count,
                          std::int32_t scheduler_depth) const;

  // Applies the decision by explicitly invoking either `commit_speculative` or `fallback_decode`.
  void execute(Context & ctx,
               const EngineSignal * signals,
               std::size_t signal_count,
               std::int32_t scheduler_depth,
               const CollapseCallbacks & callbacks) const;
};

}  // namespace korith::core
