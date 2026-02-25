#pragma once

#include <cstdint>

// Thread-safe, audit-proof token throughput metrics.
//
// IMPORTANT INTEGRATION RULE:
// - Call `metrics.on_token_printed(n)` immediately after *successfully* writing token bytes to stdout.
// - Do not call it for logits, speculative candidates, or any token that was not actually printed.
class Metrics {
public:
  Metrics();
  Metrics(const Metrics &) = delete;
  Metrics & operator=(const Metrics &) = delete;
  Metrics(Metrics &&) = delete;
  Metrics & operator=(Metrics &&) = delete;
  ~Metrics();

  // Adds `count` to the printed-token counter and updates rolling state.
  void on_token_printed(std::uint32_t count);

  // Refreshes internal rolling state from the current total (does not increment tokens).
  void tick();

  // Tokens/sec over the most recent observed interval.
  double get_instant_tps();

  // Tokens/sec over a rolling window.
  double get_rolling_tps();

  // Total number of printed tokens observed by this process.
  std::uint64_t get_total_tokens() const;

  // Optional helpers used by the C ABI wrappers.
  void set_window_ms(std::int64_t window_ms);
  void reset_window();
  void record_power_watts(double watts);
  double get_tokens_per_joule_rolling();

private:
  struct Impl;
  Impl * impl_ = nullptr;
};

// Global metrics instance intended for direct use by the engine.
extern Metrics metrics;
