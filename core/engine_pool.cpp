#include "engine_pool.h"

#include "bindings.h"
#include "speculative_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace korith::core {

namespace {

std::mutex g_pool_mu;
std::vector<std::unique_ptr<SpeculativeEngine>> g_engines;
std::vector<EngineResult> g_results;
bool g_bootstrap_done = false;

// Cached draft model references for attaching to new engines.
llama_context * g_draft_ctx = nullptr;
llama_batch * g_draft_batch = nullptr;
const llama_vocab * g_draft_vocab = nullptr;
int32_t g_draft_n_vocab = 0;

void attach_draft_to_engine(SpeculativeEngine & engine) {
  if (g_draft_ctx) {
    engine.attach_draft(g_draft_ctx, g_draft_batch, g_draft_vocab, g_draft_n_vocab);
  }
}

void ensure_engine_count(std::size_t count) {
  if (g_engines.size() == count) {
    return;
  }

  if (g_engines.size() < count) {
    const std::size_t start = g_engines.size();
    g_engines.reserve(count);
    for (std::size_t i = start; i < count; ++i) {
      auto engine = std::make_unique<SpeculativeEngine>();
      engine->set_cuda_stream(reinterpret_cast<void *>(static_cast<std::uintptr_t>(i + 1)));
      attach_draft_to_engine(*engine);
      g_engines.push_back(std::move(engine));
    }
  } else {
    g_engines.resize(count);
  }
}

}  // namespace

void engine_pool_run(int scheduler_depth, int max_tokens) {
  const int32_t depth = std::max<int32_t>(1, scheduler_depth);
  std::size_t desired = (depth > 1) ? static_cast<std::size_t>(depth - 1) : 0u;

  std::lock_guard<std::mutex> lock(g_pool_mu);
  if (depth == 1 && g_engines.empty() && !g_bootstrap_done) {
    // Bootstrap only: allow a single speculative engine to run once at depth=2.
    desired = 1u;
    g_bootstrap_done = true;
  }
  ensure_engine_count(desired);

  g_results.clear();
  g_results.resize(desired);

  std::vector<std::thread> threads;
  std::vector<std::uint64_t> times_ns(desired, 0);
  std::vector<std::uint64_t> tokens_generated(desired, 0);
  threads.reserve(desired);

  for (std::size_t i = 0; i < desired; ++i) {
    threads.emplace_back([i, max_tokens, &times_ns, &tokens_generated]() {
      const auto t0 = std::chrono::steady_clock::now();
      g_engines[i]->run_step(max_tokens);
      const auto t1 = std::chrono::steady_clock::now();
      times_ns[i] = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

      const TokenSpan tokens = g_engines[i]->get_tokens();
      tokens_generated[i] = static_cast<std::uint64_t>(tokens.size);
    });
  }

  for (std::thread & t : threads) {
    t.join();
  }

  const double tpj = metrics_tokens_per_joule_rolling();
  const bool have_power = std::isfinite(tpj) && tpj > 1e-12;

  for (std::size_t i = 0; i < desired; ++i) {
    EngineResult & r = g_results[i];
    r.tokens_generated = tokens_generated[i];
    r.execution_time_ns = times_ns[i];

    if (!have_power || r.execution_time_ns == 0 || r.tokens_generated == 0) {
      r.power_watts = std::numeric_limits<float>::quiet_NaN();
      continue;
    }

    const double dt_s = static_cast<double>(r.execution_time_ns) * 1e-9;
    const double energy_j = static_cast<double>(r.tokens_generated) / tpj;
    r.power_watts = static_cast<float>(energy_j / dt_s);
  }
}

const EngineResult * get_engine_results() {
  std::lock_guard<std::mutex> lock(g_pool_mu);
  if (g_results.empty()) {
    return nullptr;
  }
  return g_results.data();
}

std::size_t get_engine_count() {
  std::lock_guard<std::mutex> lock(g_pool_mu);
  return g_results.size();
}

void engine_pool_attach_draft(
    llama_context * ctx,
    llama_batch * batch,
    const llama_vocab * vocab,
    int32_t n_vocab) {
  std::lock_guard<std::mutex> lock(g_pool_mu);
  g_draft_ctx = ctx;
  g_draft_batch = batch;
  g_draft_vocab = vocab;
  g_draft_n_vocab = n_vocab;

  // Attach to all existing engines.
  for (auto & engine : g_engines) {
    if (engine) {
      attach_draft_to_engine(*engine);
    }
  }
}

}  // namespace korith::core
