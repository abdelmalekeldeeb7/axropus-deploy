#include "korith.h"

#include "../core/amf_store.h"

#include <llama.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace {

struct KorithState {
  bool backend_inited = false;

  llama_model * model = nullptr;
  llama_context * ctx = nullptr;
  const llama_vocab * vocab = nullptr;
  llama_sampler * sampler = nullptr;

  llama_batch batch{};
  bool batch_inited = false;

  llama_pos pos = 0;
  bool ready = false;     // prompt evaluated and logits available
  bool finished = true;   // true until prompt is evaluated

  korith::core::AmfStore amf;
  korith::core::AmfContext amf_ctx{};
  bool amf_ready = false;
};

KorithState g;
std::mutex g_mu;

void reset_runtime_state() {
  g.pos = 0;
  g.ready = false;
  g.finished = true;
}

void shutdown_unlocked() {
  if (g.sampler) {
    llama_sampler_free(g.sampler);
    g.sampler = nullptr;
  }

  if (g.batch_inited) {
    llama_batch_free(g.batch);
    g.batch = {};
    g.batch_inited = false;
  }

  if (g.ctx) {
    llama_free(g.ctx);
    g.ctx = nullptr;
  }

  if (g.model) {
    llama_model_free(g.model);
    g.model = nullptr;
  }

  g.vocab = nullptr;
  reset_runtime_state();

  if (g.backend_inited) {
    llama_backend_free();
    g.backend_inited = false;
  }
}

bool ensure_capacity_and_decode(const std::vector<llama_token> & tokens) {
  if (!g.ctx || !g.batch_inited) {
    return false;
  }

  const uint32_t n_ctx = llama_n_ctx(g.ctx);
  if (n_ctx > 0 && tokens.size() >= static_cast<std::size_t>(n_ctx)) {
    return false;
  }

  const int32_t n_batch = static_cast<int32_t>(llama_n_batch(g.ctx));
  if (n_batch <= 0) {
    return false;
  }

  std::size_t idx = 0;
  while (idx < tokens.size()) {
    const int32_t chunk = std::min<int32_t>(n_batch, static_cast<int32_t>(tokens.size() - idx));

    g.batch.n_tokens = chunk;
    for (int32_t i = 0; i < chunk; ++i) {
      const std::size_t t = idx + static_cast<std::size_t>(i);

      g.batch.token[i] = tokens[t];
      g.batch.pos[i] = g.pos + i;
      g.batch.n_seq_id[i] = 1;
      g.batch.seq_id[i][0] = 0;
      g.batch.logits[i] = (t + 1 == tokens.size()) ? 1 : 0;
    }

    const int32_t rc = llama_decode(g.ctx, g.batch);
    if (rc != 0) {
      return false;
    }

    g.pos += chunk;
    idx += static_cast<std::size_t>(chunk);
  }

  return true;
}

std::vector<llama_token> tokenize_prompt(const char * prompt) {
  if (!g.vocab || !prompt) {
    return {};
  }

  const std::size_t len = std::strlen(prompt);
  const std::size_t i32_max = static_cast<std::size_t>(std::numeric_limits<int32_t>::max());
  if (len > i32_max) {
    return {};
  }

  constexpr std::size_t extra = 16;
  std::size_t cap = len;
  if (cap > i32_max - extra) {
    cap = i32_max;
  } else {
    cap += extra;
  }

  std::vector<llama_token> out(cap);

  int32_t n = llama_tokenize(g.vocab,
                             prompt,
                             static_cast<int32_t>(len),
                             out.data(),
                             static_cast<int32_t>(out.size()),
                             /* add_special = */ true,
                             /* parse_special = */ true);

  if (n == std::numeric_limits<int32_t>::min()) {
    return {};
  }

  if (n < 0) {
    out.resize(static_cast<std::size_t>(-n));
    n = llama_tokenize(g.vocab,
                       prompt,
                       static_cast<int32_t>(len),
                       out.data(),
                       static_cast<int32_t>(out.size()),
                       /* add_special = */ true,
                       /* parse_special = */ true);
  }

  if (n <= 0) {
    return {};
  }

  out.resize(static_cast<std::size_t>(n));
  return out;
}

}  // namespace

extern "C" {

bool korith_init(const char * model_path, int n_ctx) {
  std::lock_guard<std::mutex> lock(g_mu);
  shutdown_unlocked();

  if (!model_path || model_path[0] == '\0') {
    return false;
  }

  try {
    llama_backend_init();
    g.backend_inited = true;

    llama_model_params mparams = llama_model_default_params();
    if (llama_supports_gpu_offload()) {
      mparams.n_gpu_layers = 999;  // clamped internally
      mparams.main_gpu = 0;
    } else {
      mparams.n_gpu_layers = 0;
    }

    g.model = llama_model_load_from_file(model_path, mparams);
    if (!g.model) {
      shutdown_unlocked();
      return false;
    }

    g.vocab = llama_model_get_vocab(g.model);
    if (!g.vocab) {
      shutdown_unlocked();
      return false;
    }

    llama_context_params cparams = llama_context_default_params();
    const uint32_t req_n_ctx = (n_ctx > 0) ? static_cast<uint32_t>(n_ctx) : 0;  // 0 => model default
    cparams.n_ctx = req_n_ctx;
    cparams.n_seq_max = 1;
    cparams.n_batch = (req_n_ctx > 0) ? std::min<uint32_t>(1024, req_n_ctx) : 1024;
    cparams.n_ubatch = cparams.n_batch;
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;

    g.ctx = llama_init_from_model(g.model, cparams);
    if (!g.ctx) {
      shutdown_unlocked();
      return false;
    }

    const uint32_t ctx_n_batch_u32 = llama_n_batch(g.ctx);
    if (ctx_n_batch_u32 == 0 || ctx_n_batch_u32 > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      shutdown_unlocked();
      return false;
    }

    const int32_t ctx_n_batch = static_cast<int32_t>(ctx_n_batch_u32);
    g.batch = llama_batch_init(ctx_n_batch, 0, 1);
    g.batch_inited = true;

    if (!g.batch.token || !g.batch.pos || !g.batch.n_seq_id || !g.batch.seq_id || !g.batch.logits) {
      shutdown_unlocked();
      return false;
    }

    // llama_batch_init() does not check allocation failures. Validate the full
    // seq_id matrix to avoid UB when writing g.batch.seq_id[i][0].
    for (int32_t i = 0; i < ctx_n_batch; ++i) {
      if (!g.batch.seq_id[i]) {
        shutdown_unlocked();
        return false;
      }
    }
    if (g.batch.seq_id[ctx_n_batch] != nullptr) {
      shutdown_unlocked();
      return false;
    }

    g.sampler = llama_sampler_init_greedy();
    if (!g.sampler) {
      shutdown_unlocked();
      return false;
    }

    g.amf_ctx = {};
    g.amf_ctx.model_hash = korith::core::amf_hash_file(model_path);
    g.amf_ctx.n_ctx = llama_n_ctx(g.ctx);
    g.amf_ctx.kv_version = 1;
    g.amf_ctx.rope_base_bits = korith::core::amf_float_bits(0.0f);
    g.amf_ctx.rope_scale_bits =
        korith::core::amf_float_bits(llama_model_rope_freq_scale_train(g.model));
    g.amf_ready = g.amf.init_from_env();
    if (!g.amf_ready || g.amf_ctx.model_hash == 0) {
      g.amf_ready = false;
      g.amf.log_disabled_once();
    }

    reset_runtime_state();
    return true;
  } catch (...) {
    shutdown_unlocked();
    return false;
  }
}

int korith_tokenize(const char * prompt) {
  std::lock_guard<std::mutex> lock(g_mu);
  try {
    if (!g.ctx || !g.vocab || !g.batch_inited || !g.sampler) {
      return -1;
    }

    // Reset memory so each prompt is an independent run.
    llama_memory_t mem = llama_get_memory(g.ctx);
    if (!mem) {
      return -1;
    }
    llama_memory_clear(mem, /* data = */ true);
    llama_sampler_reset(g.sampler);
    reset_runtime_state();

    std::vector<llama_token> tokens = tokenize_prompt(prompt ? prompt : "");
    if (tokens.empty()) {
      const llama_token bos = llama_vocab_bos(g.vocab);
      if (bos == LLAMA_TOKEN_NULL) {
        return -1;
      }
      tokens.push_back(bos);
    }

    if (g.amf_ready) {
      korith::core::AmfEntry entry{};
      std::vector<llama_token> entry_tokens;
      korith::core::AmfLookupResult lookup =
          g.amf.find_longest_prefix(g.amf_ctx, tokens, &entry, &entry_tokens);
      if (lookup == korith::core::AmfLookupResult::kHit) {
        const auto load_t0 = std::chrono::steady_clock::now();
        std::vector<std::uint8_t> kv_blob;
        if (g.amf.load_kv(entry, &kv_blob)) {
          const std::size_t got = llama_state_set_data(g.ctx, kv_blob.data(), kv_blob.size());
          if (got == kv_blob.size()) {
            g.pos = static_cast<llama_pos>(entry.prefix_len);
            double suffix_ms = 0.0;
            if (entry.prefix_len < tokens.size()) {
              const std::vector<llama_token> suffix(tokens.begin() + entry.prefix_len, tokens.end());
              const auto suffix_t0 = std::chrono::steady_clock::now();
              if (!ensure_capacity_and_decode(suffix)) {
                reset_runtime_state();
                return -1;
              }
              const auto suffix_t1 = std::chrono::steady_clock::now();
              suffix_ms = std::chrono::duration<double, std::milli>(suffix_t1 - suffix_t0).count();
            }
            const auto load_t1 = std::chrono::steady_clock::now();
            const double load_ms =
                std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();
            const double restore_ms = load_ms + suffix_ms;
            const double baseline_ms =
                (entry.saved_ms > 0) ? static_cast<double>(entry.saved_ms) : restore_ms;
            const double saved_ms =
                (baseline_ms > 0.0 && std::isfinite(baseline_ms))
                    ? std::max(0.0, baseline_ms - restore_ms)
                    : 0.0;
            g.amf.note_hit(entry,
                           static_cast<std::uint64_t>(entry.prefix_len),
                           static_cast<std::uint64_t>(restore_ms),
                           static_cast<std::uint64_t>(saved_ms));
            std::fprintf(stderr,
                         "[AMF_HIT] tokens_saved=%u prefix_len=%u load_ms=%.2f suffix_ms=%.2f saved_ms=%.2f\n",
                         entry.prefix_len,
                         entry.prefix_len,
                         load_ms,
                         suffix_ms,
                         saved_ms);
            (void) std::fflush(stderr);
            g.ready = true;
            g.finished = false;
            return static_cast<int>(tokens.size());
          }
        }
        lookup = korith::core::AmfLookupResult::kMissLoadFailed;
      }
      g.amf.note_miss();
      std::fprintf(stderr, "[AMF_MISS] reason=%s\n", korith::core::amf_lookup_reason_name(lookup));
      (void) std::fflush(stderr);
    }

    const auto decode_t0 = std::chrono::steady_clock::now();
    if (!ensure_capacity_and_decode(tokens)) {
      reset_runtime_state();
      return -1;
    }
    const auto decode_t1 = std::chrono::steady_clock::now();

    if (g.amf_ready && tokens.size() >= g.amf.min_tokens()) {
      const std::size_t size = llama_state_get_size(g.ctx);
      if (size > 0) {
        std::vector<std::uint8_t> kv_blob(size);
        const std::size_t got = llama_state_get_data(g.ctx, kv_blob.data(), kv_blob.size());
        if (got == kv_blob.size()) {
          const double saved_ms =
              std::chrono::duration<double, std::milli>(decode_t1 - decode_t0).count();
          (void) g.amf.store_entry(
              g.amf_ctx,
              tokens,
              kv_blob.data(),
              kv_blob.size(),
              static_cast<std::uint64_t>(saved_ms));
        }
      }
    }

    g.ready = true;
    g.finished = false;
    return static_cast<int>(tokens.size());
  } catch (...) {
    reset_runtime_state();
    return -1;
  }
}

int korith_step(void) {
  std::lock_guard<std::mutex> lock(g_mu);
  try {
    if (!g.ctx || !g.vocab || !g.batch_inited || !g.sampler || !g.ready || g.finished) {
      return -1;
    }

    const uint32_t n_ctx = llama_n_ctx(g.ctx);
    if (n_ctx > 0 && static_cast<std::uint64_t>(g.pos) >= n_ctx) {
      g.finished = true;
      return -1;
    }

    // Sample from logits of the last evaluated token (greedy).
    const llama_token token = llama_sampler_sample(g.sampler, g.ctx, -1);
    if (token == LLAMA_TOKEN_NULL) {
      g.finished = true;
      return -1;
    }
    llama_sampler_accept(g.sampler, token);

    if (llama_vocab_is_eog(g.vocab, token)) {
      g.finished = true;
      return -1;
    }

    // Evaluate the sampled token so the next call can sample from new logits.
    g.batch.n_tokens = 1;
    g.batch.token[0] = token;
    g.batch.pos[0] = g.pos;
    g.batch.n_seq_id[0] = 1;
    g.batch.seq_id[0][0] = 0;
    g.batch.logits[0] = 1;

    const int32_t rc = llama_decode(g.ctx, g.batch);
    if (rc != 0) {
      g.finished = true;
      return -1;
    }

    g.pos += 1;
    return static_cast<int>(token);
  } catch (...) {
    g.finished = true;
    return -1;
  }
}

void korith_shutdown(void) {
  std::lock_guard<std::mutex> lock(g_mu);
  shutdown_unlocked();
}

}  // extern "C"
