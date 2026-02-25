#pragma once

#include <cstdint>

namespace korith::core {

// Snapshot of inference/control signals for thermodynamic depth control.
//
// Notes:
// - `logits` may be null; in that case, the controller uses an acceptance proxy for entropy.
// - `n_vocab` is required only when `logits` is non-null.
struct Metrics {
  float accept_ema = 0.0f;
  const float * logits = nullptr;
  int32_t n_vocab = 0;
};

struct ThermoState {
  float accept_ema = 0.0f;
  float accept_var = 0.0f;
  float entropy = 0.0f;
  int depth = 1;
};

// Returns the next speculative depth to use (>= 1).
int thermo_next_depth(const Metrics & metrics);

// Returns the most recent controller state (for logging/telemetry).
ThermoState thermo_state();

// Resets controller state (depth starts at 1).
void thermo_reset();

}  // namespace korith::core
