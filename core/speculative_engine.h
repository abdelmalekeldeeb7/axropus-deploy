#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// Forward declarations for llama.cpp types (avoids pulling in llama.h).
struct llama_context;
struct llama_batch;
struct llama_vocab;

namespace korith::core {

struct TokenSpan {
  const std::int32_t * data = nullptr;
  std::size_t size = 0;
};

class SpeculativeEngine {
public:
  SpeculativeEngine() = default;
  ~SpeculativeEngine() = default;

  SpeculativeEngine(const SpeculativeEngine &) = delete;
  SpeculativeEngine & operator=(const SpeculativeEngine &) = delete;
  SpeculativeEngine(SpeculativeEngine &&) = default;
  SpeculativeEngine & operator=(SpeculativeEngine &&) = default;

  // Attach a draft model context for real token proposal.
  // When attached, run_step() uses the draft model instead of dummy tokens.
  // The caller retains ownership of all pointers.
  void attach_draft(llama_context * ctx, llama_batch * batch, const llama_vocab * vocab, int32_t n_vocab);
  bool has_draft() const;

  int run_step(int max_tokens);
  float get_confidence() const;
  float get_entropy() const;
  float get_cost() const;
  TokenSpan get_tokens() const;
  std::uint64_t get_compute_time_ns() const;

  void set_cuda_stream(void * stream);
  void * get_cuda_stream() const;

  void reset_kv_view(const void * base_ptr, std::size_t size_bytes);

private:
  // Draft model proposal using the attached context.
  int run_step_draft(int max_tokens);
  // Fallback: dummy token generation (when no draft model is attached).
  int run_step_dummy(int max_tokens);

  struct KvCacheView {
    const void * base = nullptr;
    void * copy = nullptr;
    std::size_t bytes = 0;
    bool has_copy = false;

    ~KvCacheView() {
      if (copy) {
        std::free(copy);
      }
    }

    void reset(const void * base_ptr, std::size_t size_bytes) {
      if (copy) {
        std::free(copy);
      }
      base = base_ptr;
      copy = nullptr;
      bytes = size_bytes;
      has_copy = false;
    }

    const void * data() const {
      return has_copy ? copy : base;
    }

    void * data_mut() {
      if (!has_copy && base && bytes > 0) {
        copy = std::malloc(bytes);
        if (copy) {
          std::memcpy(copy, base, bytes);
          has_copy = true;
        }
      }
      return has_copy ? copy : nullptr;
    }
  };

  KvCacheView kv_view_;
  void * cuda_stream_ = nullptr;
  std::vector<std::int32_t> tokens_;
  float confidence_ = 0.0f;
  float entropy_ = 0.0f;
  float cost_estimate_ = 0.0f;
  std::uint64_t compute_time_ns_ = 0;

  // Draft model references (non-owning).
  llama_context * draft_ctx_ = nullptr;
  llama_batch * draft_batch_ = nullptr;
  const llama_vocab * draft_vocab_ = nullptr;
  int32_t draft_n_vocab_ = 0;
};

}  // namespace korith::core

