#include "speculative_engine.h"

#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

namespace korith::core {

// ---------------------------------------------------------------------------
// attach_draft / has_draft
// ---------------------------------------------------------------------------

void SpeculativeEngine::attach_draft(
    llama_context * ctx,
    llama_batch * batch,
    const llama_vocab * vocab,
    int32_t n_vocab) {
  draft_ctx_ = ctx;
  draft_batch_ = batch;
  draft_vocab_ = vocab;
  draft_n_vocab_ = n_vocab;
}

bool SpeculativeEngine::has_draft() const {
  return draft_ctx_ != nullptr && draft_batch_ != nullptr;
}

// ---------------------------------------------------------------------------
// run_step — dispatch to draft model or dummy fallback
// ---------------------------------------------------------------------------

int SpeculativeEngine::run_step(int max_tokens) {
  const auto t0 = std::chrono::steady_clock::now();

  tokens_.clear();
  if (max_tokens <= 0) {
    compute_time_ns_ = 0;
    confidence_ = 0.0f;
    entropy_ = 0.0f;
    cost_estimate_ = 0.0f;
    return 0;
  }

  int count = 0;
  if (has_draft()) {
    count = run_step_draft(max_tokens);
  } else {
    count = run_step_dummy(max_tokens);
  }

  compute_time_ns_ = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
  cost_estimate_ = static_cast<float>(compute_time_ns_) * 1e-9f;
  return count;
}

// ---------------------------------------------------------------------------
// run_step_draft — propose tokens using the real draft model
// ---------------------------------------------------------------------------

int SpeculativeEngine::run_step_draft(int max_tokens) {
  const int k = std::max(1, std::min(max_tokens, 12));

  tokens_.reserve(static_cast<std::size_t>(k));

  // Greedy sample from current draft logits, then decode to advance.
  double entropy_sum = 0.0;
  int entropy_count = 0;

  for (int i = 0; i < k; ++i) {
    const float * logits = llama_get_logits(draft_ctx_);
    if (!logits) {
      break;
    }

    // Greedy sample: find argmax.
    float max_logit = -std::numeric_limits<float>::infinity();
    llama_token best_tok = 0;
    for (int32_t v = 0; v < draft_n_vocab_; ++v) {
      if (logits[v] > max_logit) {
        max_logit = logits[v];
        best_tok = static_cast<llama_token>(v);
      }
    }

    // Check for EOG.
    if (draft_vocab_ && llama_vocab_is_eog(draft_vocab_, best_tok)) {
      break;
    }

    // Compute normalized entropy for this position.
    {
      double sum_exp = 0.0;
      double sum_z_logz = 0.0;
      for (int32_t v = 0; v < draft_n_vocab_; ++v) {
        const double x = static_cast<double>(logits[v] - max_logit);
        const double z = std::exp(x);
        sum_exp += z;
        sum_z_logz += z * x;
      }
      if (sum_exp > 0.0 && std::isfinite(sum_exp)) {
        const double h = std::log(sum_exp) - (sum_z_logz / sum_exp);
        const double denom = std::log(static_cast<double>(draft_n_vocab_));
        if (denom > 1e-12) {
          double hn = h / denom;
          hn = std::clamp(hn, 0.0, 1.0);
          entropy_sum += hn;
          entropy_count += 1;
        }
      }
    }

    tokens_.push_back(static_cast<std::int32_t>(best_tok));

    // Decode the token through the draft model to advance KV cache.
    // Use direct struct manipulation (llama_batch_clear/add are not public API).
    draft_batch_->n_tokens = 1;
    draft_batch_->token[0] = best_tok;
    draft_batch_->pos[0] = static_cast<llama_pos>(i);  // relative position within this proposal
    draft_batch_->n_seq_id[0] = 1;
    draft_batch_->seq_id[0][0] = 0;
    draft_batch_->logits[0] = 1;
    if (llama_decode(draft_ctx_, *draft_batch_) != 0) {
      // Decode failed — return what we have so far.
      break;
    }
  }

  const int count = static_cast<int>(tokens_.size());

  // Confidence: based on how many tokens we managed to propose vs requested.
  confidence_ = (k > 0) ? (static_cast<float>(count) / static_cast<float>(k)) : 0.0f;

  // Entropy: average normalized entropy across proposed positions.
  entropy_ = (entropy_count > 0)
      ? static_cast<float>(entropy_sum / static_cast<double>(entropy_count))
      : 1.0f;

  return count;
}

// ---------------------------------------------------------------------------
// run_step_dummy — fallback when no draft model is attached
// ---------------------------------------------------------------------------

int SpeculativeEngine::run_step_dummy(int max_tokens) {
  const int token_count = std::max(1, std::min(max_tokens, 4));
  tokens_.reserve(static_cast<std::size_t>(token_count));
  for (int i = 0; i < token_count; ++i) {
    tokens_.push_back(static_cast<std::int32_t>(i));
  }
  confidence_ = 0.5f;
  entropy_ = 1.0f;
  return token_count;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

float SpeculativeEngine::get_confidence() const {
  return confidence_;
}

float SpeculativeEngine::get_entropy() const {
  return entropy_;
}

float SpeculativeEngine::get_cost() const {
  return cost_estimate_;
}

TokenSpan SpeculativeEngine::get_tokens() const {
  return TokenSpan{tokens_.data(), tokens_.size()};
}

std::uint64_t SpeculativeEngine::get_compute_time_ns() const {
  return compute_time_ns_;
}

void SpeculativeEngine::set_cuda_stream(void * stream) {
  cuda_stream_ = stream;
}

void * SpeculativeEngine::get_cuda_stream() const {
  return cuda_stream_;
}

void SpeculativeEngine::reset_kv_view(const void * base_ptr, std::size_t size_bytes) {
  kv_view_.reset(base_ptr, size_bytes);
}

}  // namespace korith::core
