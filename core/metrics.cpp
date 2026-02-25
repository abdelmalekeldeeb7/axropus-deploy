#include "bindings.h"
#include "metrics.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>

std::atomic<std::uint64_t> korith_tokens_printed_total{0};

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::chrono::nanoseconds kDefaultWindow = std::chrono::seconds(5);

std::uint64_t saturating_add(std::atomic<std::uint64_t> & counter, std::uint64_t delta) {
  std::uint64_t cur = counter.load(std::memory_order_relaxed);
  while (true) {
    const std::uint64_t next = (std::numeric_limits<std::uint64_t>::max() - cur < delta)
                                   ? std::numeric_limits<std::uint64_t>::max()
                                   : (cur + delta);
    if (counter.compare_exchange_weak(cur, next, std::memory_order_relaxed, std::memory_order_relaxed)) {
      return next;
    }
  }
}

struct TokenEvent {
  Clock::time_point t;
  std::uint64_t total = 0;
};

struct PowerSample {
  Clock::time_point t;
  std::uint64_t total = 0;
  double watts = 0.0;
};

double tps_between(const TokenEvent & a, const TokenEvent & b) {
  const double dt_s = std::chrono::duration<double>(b.t - a.t).count();
  if (dt_s <= 1e-12) {
    return 0.0;
  }
  if (b.total <= a.total) {
    return 0.0;
  }
  return static_cast<double>(b.total - a.total) / dt_s;
}

}  // namespace

struct Metrics::Impl {
  std::mutex mu;
  std::chrono::nanoseconds window = kDefaultWindow;

  // Observations of printed-token totals (events occur only when the total increases).
  std::deque<TokenEvent> events;
  std::uint64_t last_observed_total = 0;

  // Optional power samples to compute tokens/J.
  std::deque<PowerSample> power;

  void reset_locked(std::uint64_t current_total) {
    events.clear();
    power.clear();
    last_observed_total = current_total;
  }

  void trim_locked(const Clock::time_point now) {
    const auto cutoff = now - window;
    while (!events.empty() && events.front().t < cutoff) {
      events.pop_front();
    }
    while (!power.empty() && power.front().t < cutoff) {
      power.pop_front();
    }
  }

  void observe_total_locked(Clock::time_point now, std::uint64_t total) {
    // Counter reset or wrap: drop history to keep monotonic deltas audit-proof.
    if (!events.empty() && total < events.back().total) {
      reset_locked(total);
      return;
    }

    if (total < last_observed_total) {
      reset_locked(total);
      return;
    }

    if (total > last_observed_total) {
      if (!events.empty() && now <= events.back().t) {
        now = events.back().t + std::chrono::nanoseconds(1);
      }
      events.push_back(TokenEvent{now, total});
      last_observed_total = total;
    }

    trim_locked(now);
  }

  void record_power_locked(Clock::time_point now, std::uint64_t total, double watts) {
    if (!std::isfinite(watts) || watts < 0.0) {
      return;
    }

    if (!power.empty() && now <= power.back().t) {
      now = power.back().t + std::chrono::nanoseconds(1);
    }
    power.push_back(PowerSample{now, total, watts});
  }

  double tokens_per_joule_rolling_locked() const {
    if (power.size() < 2) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    double energy_j = 0.0;
    for (std::size_t i = 1; i < power.size(); ++i) {
      const PowerSample & a = power[i - 1];
      const PowerSample & b = power[i];
      const double dt_s = std::chrono::duration<double>(b.t - a.t).count();
      if (dt_s <= 0.0) {
        continue;
      }
      energy_j += 0.5 * (a.watts + b.watts) * dt_s;
    }

    if (!(energy_j > 0.0)) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    const std::uint64_t d_tok = power.back().total - power.front().total;
    if (d_tok == 0) {
      return 0.0;
    }
    return static_cast<double>(d_tok) / energy_j;
  }
};

Metrics::Metrics() : impl_(new Impl()) {}

Metrics::~Metrics() {
  delete impl_;
  impl_ = nullptr;
}

void Metrics::on_token_printed(std::uint32_t count) {
  if (count == 0) {
    return;
  }

  const auto now = Clock::now();
  const std::uint64_t total = saturating_add(korith_tokens_printed_total, static_cast<std::uint64_t>(count));

  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->observe_total_locked(now, total);
}

void Metrics::tick() {
  const auto now = Clock::now();
  const std::uint64_t total = korith_tokens_printed_total.load(std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->observe_total_locked(now, total);
}

double Metrics::get_instant_tps() {
  const auto now = Clock::now();
  const std::uint64_t total = korith_tokens_printed_total.load(std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->observe_total_locked(now, total);

  if (impl_->events.size() < 2) {
    return 0.0;
  }
  const TokenEvent & a = impl_->events[impl_->events.size() - 2];
  const TokenEvent & b = impl_->events[impl_->events.size() - 1];
  return tps_between(a, b);
}

double Metrics::get_rolling_tps() {
  const auto now = Clock::now();
  const std::uint64_t total = korith_tokens_printed_total.load(std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->observe_total_locked(now, total);

  if (impl_->events.size() < 2) {
    return 0.0;
  }
  const TokenEvent & a = impl_->events.front();
  const TokenEvent & b = impl_->events.back();
  return tps_between(a, b);
}

std::uint64_t Metrics::get_total_tokens() const {
  return korith_tokens_printed_total.load(std::memory_order_relaxed);
}

void Metrics::set_window_ms(std::int64_t window_ms) {
  const auto now = Clock::now();
  const std::uint64_t total = korith_tokens_printed_total.load(std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(impl_->mu);
  if (window_ms <= 0) {
    impl_->window = kDefaultWindow;
  } else {
    constexpr std::int64_t kMinWindowMs = 1;
    const std::int64_t clamped = (window_ms < kMinWindowMs) ? kMinWindowMs : window_ms;
    impl_->window = std::chrono::milliseconds(clamped);
  }
  impl_->reset_locked(total);
  impl_->trim_locked(now);
}

void Metrics::reset_window() {
  const std::uint64_t total = korith_tokens_printed_total.load(std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->reset_locked(total);
}

void Metrics::record_power_watts(double watts) {
  const auto now = Clock::now();
  const std::uint64_t total = korith_tokens_printed_total.load(std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->observe_total_locked(now, total);
  impl_->record_power_locked(now, total, watts);
  impl_->trim_locked(now);
}

double Metrics::get_tokens_per_joule_rolling() {
  const auto now = Clock::now();
  const std::uint64_t total = korith_tokens_printed_total.load(std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->observe_total_locked(now, total);
  impl_->trim_locked(now);
  return impl_->tokens_per_joule_rolling_locked();
}

Metrics metrics;

extern "C" {

void engine_metrics_tick(void) {
  metrics.tick();
}

void metrics_set_window_ms(int64_t window_ms) {
  metrics.set_window_ms(window_ms);
}

void metrics_reset_window(void) {
  metrics.reset_window();
}

double metrics_tps_instant(void) {
  return metrics.get_instant_tps();
}

double metrics_tps_rolling(void) {
  return metrics.get_rolling_tps();
}

void metrics_record_power_watts(double watts) {
  metrics.record_power_watts(watts);
}

double metrics_tokens_per_joule_rolling(void) {
  return metrics.get_tokens_per_joule_rolling();
}

std::uint64_t metrics_tokens_printed_total(void) {
  return metrics.get_total_tokens();
}

}  // extern "C"
