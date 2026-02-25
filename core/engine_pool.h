#pragma once

#include <cstddef>
#include <cstdint>

// Forward declarations for llama.cpp types.
struct llama_context;
struct llama_batch;
struct llama_vocab;

namespace korith::core {

struct EngineResult {
  std::uint64_t tokens_generated = 0;
  std::uint64_t execution_time_ns = 0;
  float power_watts = 0.0f;
};

// Attach a draft model context to all pool engines.
// The caller retains ownership of all pointers.
void engine_pool_attach_draft(llama_context * ctx, llama_batch * batch, const llama_vocab * vocab, int32_t n_vocab);

void engine_pool_run(int scheduler_depth, int max_tokens);
const EngineResult * get_engine_results();
std::size_t get_engine_count();

}  // namespace korith::core

