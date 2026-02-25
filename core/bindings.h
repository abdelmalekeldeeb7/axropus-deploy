#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------- Engine (C ABI) ----------------------------
//
// Minimal persistent inference engine loop.
bool engine_init(const char * model_path);

// Advances generation by up to `batch_tokens` tokens.
// Returns the number of tokens that were actually printed (see metrics notes).
int32_t engine_step(int32_t batch_tokens);

// Updates the metrics subsystem state (monotonic time sampling).
// This is intended to be called by thin runners after `engine_step()`.
void engine_metrics_tick(void);

// Returns a pointer to the most recent logits row produced by the target model.
// The buffer is owned by llama.cpp and remains valid until the next decode call
// on the target context or shutdown.
const float * engine_get_logits(void);

// Optional: speculative decoding telemetry/control. These are safe to call even if
// speculation is disabled (draft model not loaded).
typedef struct engine_spec_stats {
  int32_t spec_depth;
  int32_t _pad0;
  double accept_ratio;
  double accept_ema;
  uint64_t proposed;
  uint64_t accepted;
  double decode_ms_saved;
  double spec_overhead_ms;
} engine_spec_stats;

// Sets the speculative depth (clamped internally). Returns false if the engine is not initialized.
bool engine_set_spec_depth(int32_t depth);

// Retrieves speculative stats for the current engine. Returns false if the engine is not initialized.
bool engine_get_spec_stats(engine_spec_stats * out);

// ----------------------- Holographic memory (C ABI) ----------------------
//
// Read-only accessors to the engine's write-only holographic ring buffer.
//
// The buffer is owned by the engine and remains valid for the lifetime of the
// process. Consumers that need struct layout details should include the C++
// definition from `holographic/memory.h` or define a matching POD layout in
// their own bindings.
struct HoloEvent;

// Returns the number of valid events currently stored in the ring buffer.
size_t engine_holo_size(void);

// Returns a pointer to the backing storage for the ring buffer.
const struct HoloEvent * engine_holo_data(void);

// Frees all engine resources.
void engine_shutdown(void);

// ---------------------------- Metrics (C ABI) ---------------------------
//
// Metrics are derived from a single monotonically-increasing counter:
// the number of tokens that produced non-empty output and were successfully
// written to stdout by the engine.

// Sets the rolling TPS window (milliseconds). Values <= 0 restore the default.
void metrics_set_window_ms(int64_t window_ms);

// Clears the internal rolling-window history (does not reset the shared counter).
void metrics_reset_window(void);

// Returns tokens/sec computed over a rolling time window using a monotonic clock.
double metrics_tps_instant(void);
double metrics_tps_rolling(void);

// Records an instantaneous power reading (watts) for tokens-per-joule computation.
// The caller must supply real power data (no synthetic values).
void metrics_record_power_watts(double watts);

// Returns tokens per joule computed over the same rolling window, or NaN if insufficient samples.
double metrics_tokens_per_joule_rolling(void);

// Returns the current total printed-token counter value.
uint64_t metrics_tokens_printed_total(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
#include <atomic>

// The only global shared between the engine and the metrics subsystem.
// The engine increments this when a token produces non-empty bytes that are
// successfully written to stdout.
extern std::atomic<uint64_t> korith_tokens_printed_total;
#endif
