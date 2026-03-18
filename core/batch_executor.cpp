#include "bindings.h"
#include "cuda_graph_decode.h"
#include "kernels/accept_scan.h"
#include "kernels/spec_verify.h"
#include "spec_plan.h"

#include <llama.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <vector>

#include "../holographic/memory.h"

namespace {

// Benchmark / instrumentation counters (best-effort, no engine ABI changes).
std::atomic<std::uint64_t> g_tokens_committed_without_decode{0};
std::atomic<std::uint64_t> g_tokens_decoded_normally{0};
std::vector<float> g_best_logits_buffer;

static inline std::uint64_t hash_token_prefix_step(std::uint64_t h, llama_token tok) noexcept {
  // FNV-1a over token ids (deterministic, fast).
  constexpr std::uint64_t kPrime = 1099511628211ull;
  const std::uint32_t v = static_cast<std::uint32_t>(tok);
  h ^= static_cast<std::uint64_t>(v);
  h *= kPrime;
  return h;
}

struct HoloKVStoreState {
  const llama_context * ctx = nullptr;
  bool valid = false;
  std::uint64_t prefix_hash = 0;
  std::uint64_t last_used_step = 0;
  int32_t last_depth = 2;
};

constexpr std::uint16_t kAgeFastVerify = 2;
constexpr std::uint16_t kAgeSkip = 4;
constexpr float kConfEmaAlpha = 0.2f;
constexpr float kConfEmaBoost = 0.85f;
constexpr float kConfEmaFreeze = 0.6f;
constexpr std::size_t kMaxSpecTokens = 64;
static_assert(sizeof(llama_token) == sizeof(int32_t), "llama_token must be 32-bit");

bool spec_v3_enabled() {
  static bool initialized = false;
  static bool enabled = false;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_V3")) {
      enabled = (env[0] != '\0') && (env[0] != '0');
    }
  }
  return enabled;
}

int spec_v3_block_min() {
  static bool initialized = false;
  static int value = 8;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_V3_BLOCK_MIN")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end && end != env && *end == '\0') {
          value = static_cast<int>(parsed);
        }
      }
    }
    value = std::clamp(value, 2, static_cast<int>(kMaxSpecTokens));
  }
  return value;
}

int spec_v3_block_max() {
  static bool initialized = false;
  static int value = 16;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_V3_BLOCK_MAX")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end && end != env && *end == '\0') {
          value = static_cast<int>(parsed);
        }
      }
    }
    value = std::clamp(value, 2, static_cast<int>(kMaxSpecTokens));
  }
  return value;
}

bool gpu_accept_scan_enabled() {
  static bool initialized = false;
  static bool enabled = false;
  if (!initialized) {
    initialized = true;

    const bool available = korith::core::gpu_accept_scan_available();
    bool requested = available;
    if (const char * env = std::getenv("KORITH_GPU_ACCEPT_SCAN")) {
      requested = (env[0] != '\0') && (env[0] != '0');
    }
    enabled = requested && available;

    std::fprintf(stderr,
                 "[GPU_ACCEPT_SCAN] enabled=%d backend=%s\n",
                 enabled ? 1 : 0,
                 enabled ? "cuda" : "host");
    (void) std::fflush(stderr);
  }
  return enabled;
}

bool spec_fused_verify_enabled() {
  static bool initialized = false;
  static bool enabled = true;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_FUSED_VERIFY")) {
      enabled = (env[0] != '\0') && (env[0] != '0');
    }
  }
  return enabled;
}

int spec_profit_window() {
  static bool initialized = false;
  static int value = 10;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_PROFIT_WINDOW")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end && end != env && *end == '\0') {
          value = static_cast<int>(parsed);
        }
      }
    }
    value = std::clamp(value, 1, 64);
  }
  return value;
}

int spec_profit_bad_streak_limit() {
  static bool initialized = false;
  static int value = 5;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_PROFIT_BAD_STREAK")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end && end != env && *end == '\0') {
          value = static_cast<int>(parsed);
        }
      }
    }
    value = std::clamp(value, 1, 64);
  }
  return value;
}

int spec_profit_cooldown_steps() {
  static bool initialized = false;
  static int value = 10;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_PROFIT_COOLDOWN_STEPS")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end && end != env && *end == '\0') {
          value = static_cast<int>(parsed);
        }
      }
    }
    value = std::clamp(value, 1, 4096);
  }
  return value;
}

bool spec_adaptive_enabled() {
  static bool initialized = false;
  static bool enabled = true;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_ADAPTIVE")) {
      enabled = (env[0] != '\0') && (env[0] != '0');
    }
  }
  return enabled;
}

int spec_max_verify_engines() {
  static bool initialized = false;
  static int value = -1;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_MAX_ENGINES")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end && end != env && *end == '\0') {
          value = static_cast<int>(parsed);
        }
      }
    }
  }
  return value;
}

int32_t first_mismatch_accept_scan(
    const llama_token * target_tokens,
    const llama_token * draft_tokens,
    int32_t count) {
  if (!target_tokens || !draft_tokens || count <= 0) {
    return -1;
  }

  constexpr int32_t kChunk = 8;
  int32_t i = 0;
  for (; i + kChunk <= count; i += kChunk) {
    std::uint32_t mismatch_mask = 0u;
    for (int32_t j = 0; j < kChunk; ++j) {
      if (target_tokens[i + j] != draft_tokens[i + j]) {
        mismatch_mask |= (1u << static_cast<std::uint32_t>(j));
      }
    }
    if (mismatch_mask != 0u) {
      return i + static_cast<int32_t>(std::countr_zero(mismatch_mask));
    }
  }

  for (; i < count; ++i) {
    if (target_tokens[i] != draft_tokens[i]) {
      return i;
    }
  }

  return -1;
}

bool spec_shadow_when_commit_blocked() {
  static bool initialized = false;
  static bool enabled = false;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_SHADOW_WHEN_COMMIT_BLOCKED")) {
      enabled = (env[0] != '\0') && (env[0] != '0');
    }
  }
  return enabled;
}

float spec_safety_entropy_max() {
  static bool initialized = false;
  static float value = 0.6f;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_SAFETY_MAX_ENTROPY")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const double parsed = std::strtod(env, &end);
        if (end && end != env && std::isfinite(parsed)) {
          value = static_cast<float>(std::clamp(parsed, 0.0, 1.0));
        }
      }
    }
  }
  return value;
}

double spec_safety_relax_accept() {
  static bool initialized = false;
  static double value = 0.9;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_SAFETY_RELAX_ACCEPT")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const double parsed = std::strtod(env, &end);
        if (end && end != env && *end == '\0' && std::isfinite(parsed)) {
          value = std::clamp(parsed, 0.0, 1.0);
        }
      }
    }
  }
  return value;
}

int spec_safety_entropy_streak() {
  static bool initialized = false;
  static int value = 3;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_SAFETY_ENTROPY_STREAK")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end && end != env && *end == '\0') {
          value = static_cast<int>(std::clamp<long>(parsed, 1, 64));
        }
      }
    }
  }
  return value;
}

double spec_safety_min_accept() {
  static bool initialized = false;
  static double value = 0.2;
  if (!initialized) {
    initialized = true;
    const char * env = std::getenv("KORITH_SPEC_SAFETY_MIN_ACCEPT");
    if (!env || env[0] == '\0') {
      env = std::getenv("KORITH_SPEC_MIN_ACCEPT");
    }
    if (env && env[0] != '\0') {
        char * end = nullptr;
        const double parsed = std::strtod(env, &end);
        if (end && end != env && *end == '\0' && std::isfinite(parsed)) {
          value = std::clamp(parsed, 0.0, 1.0);
        }
    }
  }
  return value;
}

enum class FlushReason : std::uint8_t {
  kNone = 0,
  kEos = 1,
  kPressure = 2,
};

struct TokenMeta {
  llama_token token_id = LLAMA_TOKEN_NULL;
  std::uint16_t age = 0;
  std::uint64_t last_verified_step = 0;
};

struct SpecPersistState {
  const llama_context * ctx = nullptr;
  int32_t n_vocab = 0;
  bool valid = false;
  llama_seq_id seq_id = 1;
  bool finish_pending = false;
  bool flush_pending = false;
  FlushReason flush_reason = FlushReason::kNone;
  std::uint64_t step_id = 0;
  float last_entropy = 0.0f;
  double last_accept = 0.0;
  double accept_gate_ema = std::numeric_limits<double>::quiet_NaN();
  bool have_last_entropy = false;
  bool have_last_accept = false;
  std::uint16_t entropy_hot_streak = 0;
  std::array<double, 64> profit_window{};
  std::size_t profit_head = 0;
  std::size_t profit_count = 0;
  double profit_sum = 0.0;
  double baseline_decode_ms_ema = std::numeric_limits<double>::quiet_NaN();
  int32_t negative_effective_streak = 0;
  int32_t profitability_cooldown_steps = 0;
  int32_t profitability_probe_steps = 1;
  int32_t profitability_retry_count = 0;
  bool profitability_hard_disabled = false;
  bool adaptive_request_disabled = false;
  int32_t adaptive_bad_checks = 0;
  std::uint64_t adaptive_window_tokens = 0;
  std::uint64_t adaptive_window_ns = 0;
  double adaptive_window_effective_ms = 0.0;
  std::array<TokenMeta, kMaxSpecTokens> meta{};
  std::array<std::uint8_t, kMaxSpecTokens> speculative{};
  std::array<float, kMaxSpecTokens> entropy{};
  std::vector<llama_token> deferred_draft_tokens;
  std::size_t head = 0;
  std::size_t count = 0;
  std::vector<float> logits;
  std::vector<float> conf_ema;
  std::vector<std::uint8_t> conf_valid;
};

bool decode_one_token(
    llama_context * ctx,
    llama_batch & batch,
    llama_seq_id seq_id,
    llama_pos pos,
    llama_token token,
    bool want_logits,
    bool allow_cuda_graph = false) {
  if (!ctx) {
    return false;
  }

  batch.n_tokens = 1;
  batch.token[0] = token;
  batch.pos[0] = pos;
  batch.n_seq_id[0] = 1;
  batch.seq_id[0][0] = seq_id;
  batch.logits[0] = want_logits ? 1 : 0;

  if (allow_cuda_graph) {
    const llama_model * model = llama_get_model(ctx);
    const int32_t model_dim = model ? llama_model_n_embd(model) : 0;
    if (korith::core::cuda_graph_decode_step(ctx, batch, model_dim, /* submit_us = */ nullptr)) {
      return true;
    }
  }

  const int32_t rc = llama_decode(ctx, batch);
  return rc == 0;
}

int32_t decode_batch_tokens(
    llama_context * ctx,
    llama_batch & batch,
    llama_seq_id seq_id,
    llama_pos start_pos,
    const std::vector<llama_token> & tokens,
    bool logits_all) {
  if (!ctx) {
    return -1;
  }
  if (tokens.empty()) {
    return 0;
  }
  if (tokens.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
    return -1;
  }

  const int32_t n = static_cast<int32_t>(tokens.size());
  batch.n_tokens = n;

  for (int32_t i = 0; i < n; ++i) {
    batch.token[i] = tokens[static_cast<std::size_t>(i)];
    batch.pos[i] = start_pos + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = seq_id;
    batch.logits[i] = logits_all ? 1 : (i + 1 == n ? 1 : 0);
  }

  return llama_decode(ctx, batch);
}

int32_t write_token_piece(const llama_vocab * vocab, llama_token token) {
  if (!vocab) {
    return -1;
  }

  // Fast path: most token pieces are small.
  char stack_buf[256];
  int32_t n = llama_token_to_piece(vocab, token, stack_buf, static_cast<int32_t>(sizeof(stack_buf)), 0, false);
  if (n == 0) {
    return 0;  // nothing to print, but not an error
  }

  if (n > 0) {
    const std::size_t want = static_cast<std::size_t>(n);
    if (std::fwrite(stack_buf, 1, want, stdout) != want) {
      return -1;
    }
    return n;
  }

  // Slow path: query required size and allocate.
  const int32_t need = -n;
  if (need <= 0) {
    return -1;
  }
  std::vector<char> dyn(static_cast<std::size_t>(need));

  n = llama_token_to_piece(vocab, token, dyn.data(), static_cast<int32_t>(dyn.size()), 0, false);
  if (n <= 0) {
    return -1;
  }
  const std::size_t want = static_cast<std::size_t>(n);
  if (std::fwrite(dyn.data(), 1, want, stdout) != want) {
    return -1;
  }
  return n;
}

llama_token sample_greedy_from_logits(const float * logits, int32_t n_vocab) {
  if (!logits || n_vocab <= 0) {
    return LLAMA_TOKEN_NULL;
  }

  int32_t best_id = 0;
  float best_logit = logits[0];
  for (int32_t i = 1; i < n_vocab; ++i) {
    const float v = logits[i];
    if (v > best_logit) {
      best_logit = v;
      best_id = i;
    }
  }
  return static_cast<llama_token>(best_id);
}

uint32_t xorshift32(uint32_t x) {
  // Deterministic, fast RNG for speculative sampling. Not cryptographically secure.
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

double u01_from_u32(uint32_t x) {
  // [0, 1) with 32 bits of resolution.
  return static_cast<double>(x) * (1.0 / 4294967296.0);
}

llama_token sample_dist_from_logits(
    const float * logits,
    int32_t n_vocab,
    uint32_t seed,
    const llama_vocab * vocab,
    bool allow_eog) {
  if (!logits || n_vocab <= 0 || !vocab) {
    return LLAMA_TOKEN_NULL;
  }

  // Softmax sampling without allocating a full probability array:
  // - pass 1: max logit
  // - pass 2: sum(exp(logit-max))
  // - pass 3: sample by cumulative sum
  float max_logit = -std::numeric_limits<float>::infinity();
  for (int32_t i = 0; i < n_vocab; ++i) {
    if (!allow_eog && llama_vocab_is_eog(vocab, static_cast<llama_token>(i))) {
      continue;
    }
    max_logit = std::max(max_logit, logits[i]);
  }

  if (!std::isfinite(max_logit)) {
    return LLAMA_TOKEN_NULL;
  }

  double sum = 0.0;
  for (int32_t i = 0; i < n_vocab; ++i) {
    if (!allow_eog && llama_vocab_is_eog(vocab, static_cast<llama_token>(i))) {
      continue;
    }
    sum += std::exp(static_cast<double>(logits[i] - max_logit));
  }

  if (!(sum > 0.0) || !std::isfinite(sum)) {
    return LLAMA_TOKEN_NULL;
  }

  uint32_t rng = seed;
  if (rng == 0) {
    rng = 1;
  }
  rng = xorshift32(rng);
  const double target = u01_from_u32(rng) * sum;

  double c = 0.0;
  for (int32_t i = 0; i < n_vocab; ++i) {
    if (!allow_eog && llama_vocab_is_eog(vocab, static_cast<llama_token>(i))) {
      continue;
    }
    c += std::exp(static_cast<double>(logits[i] - max_logit));
    if (c >= target) {
      return static_cast<llama_token>(i);
    }
  }

  // Due to floating point rounding, we might fall off the end. Pick the last valid token.
  for (int32_t i = n_vocab - 1; i >= 0; --i) {
    if (!allow_eog && llama_vocab_is_eog(vocab, static_cast<llama_token>(i))) {
      continue;
    }
    return static_cast<llama_token>(i);
  }

  return LLAMA_TOKEN_NULL;
}

llama_token pick_alternate_token(
    const float * logits,
    int32_t n_vocab,
    const llama_vocab * vocab,
    bool allow_eog,
    llama_token avoid) {
  if (!logits || n_vocab <= 0 || !vocab) {
    return LLAMA_TOKEN_NULL;
  }

  int32_t best_id = -1;
  int32_t second_id = -1;
  float best_logit = -std::numeric_limits<float>::infinity();
  float second_logit = -std::numeric_limits<float>::infinity();

  for (int32_t i = 0; i < n_vocab; ++i) {
    if (!allow_eog && llama_vocab_is_eog(vocab, static_cast<llama_token>(i))) {
      continue;
    }
    const float v = logits[i];
    if (v > best_logit) {
      second_logit = best_logit;
      second_id = best_id;
      best_logit = v;
      best_id = i;
    } else if (v > second_logit) {
      second_logit = v;
      second_id = i;
    }
  }

  if (best_id < 0) {
    return LLAMA_TOKEN_NULL;
  }
  if (static_cast<llama_token>(best_id) != avoid) {
    return static_cast<llama_token>(best_id);
  }
  if (second_id < 0) {
    return LLAMA_TOKEN_NULL;
  }
  return static_cast<llama_token>(second_id);
}

float normalized_entropy_from_logits(
    const float * logits,
    int32_t n_vocab,
    const llama_vocab * vocab,
    bool allow_eog) {
  if (!logits || n_vocab <= 1) {
    return 0.0f;
  }

  float max_logit = -std::numeric_limits<float>::infinity();
  int32_t n_allowed = 0;
  for (int32_t i = 0; i < n_vocab; ++i) {
    if (!allow_eog && vocab && llama_vocab_is_eog(vocab, static_cast<llama_token>(i))) {
      continue;
    }
    max_logit = std::max(max_logit, logits[i]);
    n_allowed += 1;
  }

  if (n_allowed <= 1 || !std::isfinite(max_logit)) {
    return 0.0f;
  }

  double sum_z = 0.0;
  double sum_z_x = 0.0;
  for (int32_t i = 0; i < n_vocab; ++i) {
    if (!allow_eog && vocab && llama_vocab_is_eog(vocab, static_cast<llama_token>(i))) {
      continue;
    }
    const double x = static_cast<double>(logits[i] - max_logit);
    const double z = std::exp(x);
    sum_z += z;
    sum_z_x += z * x;
  }

  if (!(sum_z > 0.0) || !std::isfinite(sum_z)) {
    return 0.0f;
  }

  const double h = std::log(sum_z) - (sum_z_x / sum_z);
  const double denom = std::log(static_cast<double>(n_allowed));
  if (!(denom > 1e-12)) {
    return 0.0f;
  }

  double hn = h / denom;
  if (!std::isfinite(hn)) {
    hn = 0.0;
  }
  if (hn < 0.0) {
    hn = 0.0;
  } else if (hn > 1.0) {
    hn = 1.0;
  }
  return static_cast<float>(hn);
}

int32_t max_spec_depth(llama_context * ctx_target) {
  if (!ctx_target) {
    return 1;
  }
  const int32_t n_seq = std::max<int32_t>(1, llama_n_seq_max(ctx_target));
  if (n_seq < 2) {
    return 1;
  }
  return 32;
}

bool rollback_draft_to(llama_context * ctx_draft, llama_pos & pos_draft, llama_pos base_pos_draft) {
  if (!ctx_draft) {
    pos_draft = base_pos_draft;
    return true;
  }
  if (pos_draft <= base_pos_draft) {
    pos_draft = base_pos_draft;
    return true;
  }

  llama_memory_t mem_d = llama_get_memory(ctx_draft);
  if (!mem_d) {
    return false;
  }
  if (!llama_memory_seq_rm(mem_d, /* seq_id = */ 0, base_pos_draft, pos_draft)) {
    return false;
  }
  pos_draft = base_pos_draft;
  return true;
}

}  // namespace

namespace korith::core {

int32_t batch_executor_step(
    llama_context * ctx_target,
    llama_batch & batch_target,
    const llama_vocab * vocab_target,
    int32_t n_vocab,
    llama_context * ctx_draft,
    llama_batch & batch_draft,
    llama_pos & pos_target,
    llama_pos & pos_draft,
    const float *& logits_target,
    const float *& logits_draft,
    bool & finished,
    int32_t batch_tokens,
    bool force_normal_decode,
    bool reuse_kv,
    bool force_speculation,
    int32_t spec_window_limit,
    int32_t & spec_depth,
    const korith::core::SpecPlan & plan,
    double & accept_ema,
    std::uint64_t & spec_proposed,
    std::uint64_t & spec_accepted,
    std::uint64_t & printed_total,
    std::uint64_t & token_prefix_hash,
    std::uint64_t & kv_hit_count,
    std::uint64_t & kv_miss_count,
    std::uint64_t & kv_evict_count,
    double baseline_tps_ema,
    double tps_delta,
    std::uint64_t & spec_elapsed_ns,
    std::uint64_t & spec_draft_ns,
    std::uint64_t & spec_verify_ns,
    std::uint64_t & spec_accept_scan_ns,
    double & spec_effective_ms,
    std::uint64_t & spec_skipped_tokens,
    std::uint64_t & spec_fast_verified_tokens,
    int32_t engine_count_cap,
    bool benchmark_mode,
    std::uint64_t min_tokens_before_eog) {
  if (!ctx_target || !vocab_target || !logits_target) {
    return -1;
  }

  spec_skipped_tokens = 0;
  spec_fast_verified_tokens = 0;
  spec_draft_ns = 0;
  spec_verify_ns = 0;
  spec_accept_scan_ns = 0;
  spec_effective_ms = std::numeric_limits<double>::quiet_NaN();

  if (finished || batch_tokens <= 0) {
    return 0;
  }

  static bool step_debug_enabled = []() {
    const char * env = std::getenv("KORITH_DEBUG_STEP");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  static bool step_debug_config_logged = false;
  if (step_debug_enabled && !step_debug_config_logged) {
    std::fprintf(stderr,
                 "[STEP_DEBUG] bench=%d min_tokens=%llu batch_tokens=%d\n",
                 benchmark_mode ? 1 : 0,
                 static_cast<unsigned long long>(min_tokens_before_eog),
                 batch_tokens);
    (void) std::fflush(stderr);
    step_debug_config_logged = true;
  }

  const uint32_t n_ctx = llama_n_ctx_seq(ctx_target);
  int32_t printed_this_call = 0;
  int32_t generated_this_call = 0;

  using Clock = std::chrono::steady_clock;
  static Clock::time_point holo_tps_last_at = Clock::now();
  static std::uint64_t holo_tps_last_tokens = 0;
  static float holo_tps_cached = 0.0f;
  auto holo_tps_snapshot = [&]() -> float {
    constexpr std::uint64_t kUpdateEveryTokens = 64;

    const std::uint64_t total = printed_total;
    if (total < holo_tps_last_tokens) {
      holo_tps_last_tokens = total;
      holo_tps_last_at = Clock::now();
      holo_tps_cached = 0.0f;
      return holo_tps_cached;
    }

    const std::uint64_t delta = total - holo_tps_last_tokens;
    if (delta < kUpdateEveryTokens) {
      return holo_tps_cached;
    }

    const auto now = Clock::now();
    const double dt_s = std::chrono::duration<double>(now - holo_tps_last_at).count();
    if (dt_s > 1e-9) {
      holo_tps_cached = static_cast<float>(static_cast<double>(delta) / dt_s);
    }
    holo_tps_last_at = now;
    holo_tps_last_tokens = total;
    return holo_tps_cached;
  };

  const bool spec_enabled = (ctx_draft != nullptr) && (logits_draft != nullptr);
  const int32_t n_seq_max = std::max<int32_t>(1, llama_n_seq_max(ctx_target));
  const int32_t n_batch = static_cast<int32_t>(llama_n_batch(ctx_target));
  const int32_t sched_depth = spec_depth;

  static bool spec_disabled_logged = false;
  static bool spec_seen = false;
  static bool spec_engines_logged = false;
  static const llama_context * last_ctx = nullptr;
  if (last_ctx != ctx_target) {
    spec_disabled_logged = false;
    spec_seen = false;
    spec_engines_logged = false;
    last_ctx = ctx_target;
  }
  auto log_spec_disabled = [&](int reason_code, int engines, int pressure) {
    if (spec_seen || spec_disabled_logged) {
      return;
    }
    std::fprintf(stderr,
                 "[SPEC_DISABLED] reason=%s sched_depth=%d plan_depth=%d lanes=%d "
                 "n_seq_max=%d engines=%d n_batch=%d pressure=%d\n",
                 korith::core::spec_disable_reason_name(reason_code),
                 static_cast<int>(sched_depth),
                 static_cast<int>(plan.depth),
                 static_cast<int>(plan.lanes),
                 n_seq_max,
                 engines,
                 n_batch,
                 pressure);
    (void) std::fflush(stderr);
    spec_disabled_logged = true;
  };

  const bool spec_v3 = spec_v3_enabled();
  if (spec_v3) {
    (void) gpu_accept_scan_enabled();
    korith::core::cuda_graph_log_config_once();
    static bool fused_verify_logged = false;
    if (!fused_verify_logged) {
      std::fprintf(stderr, "[SPEC_FUSED_VERIFY] enabled=%d\n", spec_fused_verify_enabled() ? 1 : 0);
      (void) std::fflush(stderr);
      fused_verify_logged = true;
    }
  }
  const int32_t spec_depth_cap = spec_v3 ? 16 : 8;
  const int32_t max_depth_ctx = std::min<int32_t>(spec_depth_cap, max_spec_depth(ctx_target));
  int32_t spec_v3_min_block = 0;
  int32_t spec_v3_max_block = 0;
  if (spec_v3 && max_depth_ctx >= 2) {
    spec_v3_min_block = std::clamp(spec_v3_block_min(), 2, max_depth_ctx);
    spec_v3_max_block = std::clamp(spec_v3_block_max(), spec_v3_min_block, max_depth_ctx);
    static bool spec_v3_config_logged = false;
    if (!spec_v3_config_logged) {
      std::fprintf(stderr,
                   "[SPEC_V3] enabled=1 block_min=%d block_max=%d depth_cap=%d\n",
                   spec_v3_min_block,
                   spec_v3_max_block,
                   max_depth_ctx);
      (void) std::fflush(stderr);
      spec_v3_config_logged = true;
    }
  }

  // Spec depth is controlled by the Rust scheduler; clamp to what the context can support.
  spec_depth = std::clamp(spec_depth, 1, max_depth_ctx);
  spec_elapsed_ns = 0;
  if (spec_window_limit < 0) {
    spec_window_limit = 0;
  }

  // Benchmark / sustained baseline mode:
  // - In normal mode, an EOG/EOS token stops generation immediately.
  // - When a minimum token threshold is configured, we suppress EOG/EOS termination
  //   until *printed_total* reaches that threshold. This enables sustained streaming
  //   so TPS windows have enough samples to converge (benchmark + n-predict runs).
  //
  // Audit rule: token accounting remains ONLY at print sites (see count_printed()).
  auto allow_eog_now = [&]() -> bool {
    if (min_tokens_before_eog == 0) {
      return true;
    }
    return printed_total >= min_tokens_before_eog;
  };

  llama_memory_t mem_target = llama_get_memory(ctx_target);
  const bool has_seq1 = llama_n_seq_max(ctx_target) >= 2;

  static HoloKVStoreState holo_kv{};
  if (holo_kv.ctx != ctx_target) {
    holo_kv = HoloKVStoreState{};
    holo_kv.ctx = ctx_target;
  }

  static SpecPersistState spec_state{};
  if (spec_state.ctx != ctx_target || spec_state.n_vocab != n_vocab) {
    spec_state = SpecPersistState{};
    spec_state.ctx = ctx_target;
    spec_state.n_vocab = n_vocab;
    spec_state.seq_id = 1;
    if (n_vocab > 0) {
      spec_state.logits.assign(static_cast<std::size_t>(n_vocab) * kMaxSpecTokens, 0.0f);
    }
    const int32_t seq_cap = std::max<int32_t>(1, llama_n_seq_max(ctx_target));
    spec_state.conf_ema.assign(static_cast<std::size_t>(seq_cap), 0.0f);
    spec_state.conf_valid.assign(static_cast<std::size_t>(seq_cap), 0);
  } else {
    const int32_t seq_cap = std::max<int32_t>(1, llama_n_seq_max(ctx_target));
    if (spec_state.conf_ema.size() != static_cast<std::size_t>(seq_cap)) {
      spec_state.conf_ema.assign(static_cast<std::size_t>(seq_cap), 0.0f);
      spec_state.conf_valid.assign(static_cast<std::size_t>(seq_cap), 0);
    }
  }

  constexpr int kDecisionCommit = 0;
  constexpr int kDecisionRollback = 1;
  constexpr int kDecisionFallback = 2;

  auto log_holo_kv = [&]() {
    std::fprintf(stderr,
                 "[HOLO_KV] hit=%llu miss=%llu evicted=%llu\n",
                 static_cast<unsigned long long>(kv_hit_count),
                 static_cast<unsigned long long>(kv_miss_count),
                 static_cast<unsigned long long>(kv_evict_count));
  };

  // Evict stale holographic KV state only when KV reuse is not active for this step.
  // This frees KV slots without immediately paying a re-sync cost.
  if (holo_kv.valid && has_seq1 && mem_target && !(plan.allow_exec && reuse_kv)) {
    const std::uint64_t step_id = (pos_target > 0) ? static_cast<std::uint64_t>(pos_target) : 0u;
    const std::uint64_t age = (step_id >= holo_kv.last_used_step) ? (step_id - holo_kv.last_used_step) : 0u;
    const float entropy_now = normalized_entropy_from_logits(logits_target, n_vocab, vocab_target, allow_eog_now());

    constexpr std::uint64_t kEvictAgeSteps = 512u;
    constexpr float kEvictEntropyThreshold = 0.70f;

    if (age > kEvictAgeSteps && std::isfinite(entropy_now) && entropy_now > kEvictEntropyThreshold) {
      if (llama_memory_seq_rm(mem_target, /* seq_id = */ 1, /* p0 = */ -1, /* p1 = */ -1)) {
        holo_kv.valid = false;
        holo_kv.prefix_hash = 0;
        kv_evict_count += 1;
        log_holo_kv();
      } else {
        holo_kv.valid = false;
      }
    }
  }

  auto decision_name = [&](int decision) -> const char * {
    switch (decision) {
      case kDecisionCommit:
        return "COMMIT";
      case kDecisionRollback:
        return "ROLLBACK";
      case kDecisionFallback:
        return "FALLBACK";
      default:
        return "FALLBACK";
    }
  };

  auto log_holo_commit = [&](std::uint64_t step_id, int decision) {
    // Avoid log spam: emit on decision transitions and periodically (benchmark mode only).
    if (!benchmark_mode) {
      return;
    }
    static int last_decision = -1;
    static std::uint64_t last_periodic = 0;

    const bool changed = (decision != last_decision);
    const bool periodic = ((step_id % 1000u) == 0u) && (step_id != last_periodic);
    if (!changed && !periodic) {
      return;
    }

    last_decision = decision;
    if (periodic) {
      last_periodic = step_id;
    }

    std::fprintf(stderr, "[HOLO_COMMIT] decision=%s buffer=%zu\n", decision_name(decision), holo_size());
  };

  auto bench_report_if_needed = [&]() {
    if (!benchmark_mode) {
      return;
    }

    // Print occasionally to avoid spamming stderr; engine still prints its own TPS line.
    constexpr std::uint64_t kReportEveryTokens = 4096;
    const std::uint64_t total = printed_total;
    if (total == 0 || (total % kReportEveryTokens) != 0) {
      return;
    }

    static std::uint64_t last_report_total = 0;
    if (last_report_total == total) {
      return;
    }
    last_report_total = total;

    const std::uint64_t committed = g_tokens_committed_without_decode.load(std::memory_order_relaxed);
    const std::uint64_t decoded = g_tokens_decoded_normally.load(std::memory_order_relaxed);
    const double fill_pct = 100.0 * (static_cast<double>(holo_size()) / static_cast<double>(kHoloCapacity));
    std::fprintf(stderr, "\nbench: committed_wo_decode=%llu decoded_normally=%llu\n",
                 static_cast<unsigned long long>(committed),
                 static_cast<unsigned long long>(decoded));
    std::fprintf(stderr, "bench: holo_fill=%.1f%%\n", fill_pct);
    (void) std::fflush(stderr);
  };

  auto ensure_seq1_synced = [&](llama_pos pos_next, int32_t depth_hint, float entropy_hint) -> bool {
    (void) entropy_hint;
    if (!has_seq1 || !mem_target) {
      holo_kv.valid = false;
      return false;
    }

    const llama_pos want_max = (pos_next > 0) ? (pos_next - 1) : -1;
    const llama_pos max0 = llama_memory_seq_pos_max(mem_target, /* seq_id = */ 0);
    if (max0 != want_max) {
      holo_kv.valid = false;
      return false;
    }

    const llama_pos max1 = llama_memory_seq_pos_max(mem_target, /* seq_id = */ 1);
    if (max1 == want_max && (!holo_kv.valid || holo_kv.prefix_hash == token_prefix_hash)) {
      kv_hit_count += 1;
      holo_kv.valid = true;
      holo_kv.prefix_hash = token_prefix_hash;
      holo_kv.last_used_step = (pos_next > 0) ? static_cast<std::uint64_t>(pos_next) : 0u;
      holo_kv.last_depth = depth_hint;
      log_holo_kv();
      return true;
    }

    kv_miss_count += 1;

    // Safe re-sync: clear seq 1 then copy seq 0 prefix up to pos_next.
    if (!llama_memory_seq_rm(mem_target, /* seq_id = */ 1, /* p0 = */ -1, /* p1 = */ -1)) {
      holo_kv.valid = false;
      return false;
    }
    llama_memory_seq_cp(mem_target, /* src = */ 0, /* dst = */ 1, /* p0 = */ 0, /* p1 = */ pos_next);

    holo_kv.valid = true;
    holo_kv.prefix_hash = token_prefix_hash;
    holo_kv.last_used_step = (pos_next > 0) ? static_cast<std::uint64_t>(pos_next) : 0u;
    holo_kv.last_depth = depth_hint;
    log_holo_kv();
    return true;
  };

  auto ensure_seq_synced = [&](llama_seq_id seq_id, llama_pos pos_next, int32_t depth_hint, float entropy_hint) -> bool {
    if (seq_id == 1) {
      return ensure_seq1_synced(pos_next, depth_hint, entropy_hint);
    }
    if (!mem_target) {
      return false;
    }
    if (!llama_memory_seq_rm(mem_target, /* seq_id = */ seq_id, /* p0 = */ -1, /* p1 = */ -1)) {
      return false;
    }
    llama_memory_seq_cp(mem_target, /* src = */ 0, /* dst = */ seq_id, /* p0 = */ 0, /* p1 = */ pos_next);
    return true;
  };

  auto spec_index = [&](std::size_t offset) -> std::size_t {
    return (spec_state.head + offset) % kMaxSpecTokens;
  };
  auto spec_meta_at = [&](std::size_t offset) -> TokenMeta & {
    return spec_state.meta[spec_index(offset)];
  };
  auto spec_logits_ptr = [&](std::size_t idx) -> float * {
    return spec_state.logits.data() + idx * static_cast<std::size_t>(spec_state.n_vocab);
  };
  auto reset_spec_state = [&](bool clear_seq, bool preserve_profit_state = false) {
    if (clear_seq && mem_target) {
      (void) llama_memory_seq_rm(mem_target, /* seq_id = */ spec_state.seq_id, /* p0 = */ -1, /* p1 = */ -1);
    }
    if (clear_seq && spec_state.seq_id >= 0 &&
        static_cast<std::size_t>(spec_state.seq_id) < spec_state.conf_ema.size()) {
      spec_state.conf_ema[static_cast<std::size_t>(spec_state.seq_id)] = 0.0f;
      spec_state.conf_valid[static_cast<std::size_t>(spec_state.seq_id)] = 0;
    }
    spec_state.valid = false;
    spec_state.finish_pending = false;
    spec_state.flush_pending = false;
    spec_state.flush_reason = FlushReason::kNone;
    spec_state.head = 0;
    spec_state.count = 0;
    spec_state.step_id = 0;
    spec_state.last_entropy = 0.0f;
    spec_state.last_accept = 0.0;
    spec_state.accept_gate_ema = std::numeric_limits<double>::quiet_NaN();
    spec_state.have_last_entropy = false;
    spec_state.have_last_accept = false;
    spec_state.entropy_hot_streak = 0;
    if (!preserve_profit_state) {
      spec_state.deferred_draft_tokens.clear();
    }
    if (!preserve_profit_state) {
      spec_state.profit_window.fill(0.0);
      spec_state.profit_head = 0;
      spec_state.profit_count = 0;
      spec_state.profit_sum = 0.0;
      spec_state.baseline_decode_ms_ema = std::numeric_limits<double>::quiet_NaN();
      spec_state.negative_effective_streak = 0;
      spec_state.profitability_cooldown_steps = 0;
      spec_state.profitability_probe_steps = 1;
      spec_state.profitability_retry_count = 0;
      spec_state.profitability_hard_disabled = false;
    }
  };

  auto update_baseline_decode_ms = [&](double sample_ms) {
    if (!(sample_ms > 0.0) || !std::isfinite(sample_ms)) {
      return;
    }
    constexpr double kBaselineAlpha = 0.2;
    if (!std::isfinite(spec_state.baseline_decode_ms_ema)) {
      spec_state.baseline_decode_ms_ema = sample_ms;
    } else {
      spec_state.baseline_decode_ms_ema +=
          kBaselineAlpha * (sample_ms - spec_state.baseline_decode_ms_ema);
    }
  };

  struct SpecEvent {
    bool accepted = false;
    int32_t depth = 2;
    int32_t speculative_cost = 0;
  };

  auto emit_spec_event = [&](const SpecEvent & ev) {
    (void) ev;
  };

  auto sample_greedy_with_eog_control = [&](const float * logits_row, bool allow_eog) -> llama_token {
    if (!logits_row || n_vocab <= 0) {
      return LLAMA_TOKEN_NULL;
    }

    // Fast path: normal mode (or after the benchmark threshold) uses plain greedy.
    if (allow_eog) {
      return sample_greedy_from_logits(logits_row, n_vocab);
    }

    // Benchmark mode below threshold: choose the highest-logit token that is NOT EOG/EOS.
    //
    // To avoid calling llama_vocab_is_eog() for every vocab entry, track a small top-K
    // (K=4) set, then choose the first non-EOG candidate.
    struct Candidate {
      int32_t id;
      float logit;
    };
    Candidate top[4] = {
        Candidate{-1, -std::numeric_limits<float>::infinity()},
        Candidate{-1, -std::numeric_limits<float>::infinity()},
        Candidate{-1, -std::numeric_limits<float>::infinity()},
        Candidate{-1, -std::numeric_limits<float>::infinity()},
    };

    for (int32_t i = 0; i < n_vocab; ++i) {
      const float v = logits_row[i];
      if (v > top[0].logit) {
        top[3] = top[2];
        top[2] = top[1];
        top[1] = top[0];
        top[0] = Candidate{i, v};
      } else if (v > top[1].logit) {
        top[3] = top[2];
        top[2] = top[1];
        top[1] = Candidate{i, v};
      } else if (v > top[2].logit) {
        top[3] = top[2];
        top[2] = Candidate{i, v};
      } else if (v > top[3].logit) {
        top[3] = Candidate{i, v};
      }
    }

    for (const Candidate & c : top) {
      if (c.id < 0) {
        continue;
      }
      const llama_token tok = static_cast<llama_token>(c.id);
      if (!llama_vocab_is_eog(vocab_target, tok)) {
        return tok;
      }
    }

    // Last-resort scan: pick the best non-EOG token from the full vocab.
    int32_t best_id = -1;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n_vocab; ++i) {
      const llama_token tok = static_cast<llama_token>(i);
      if (llama_vocab_is_eog(vocab_target, tok)) {
        continue;
      }
      const float v = logits_row[i];
      if (v > best_logit) {
        best_logit = v;
        best_id = i;
      }
    }

    return (best_id < 0) ? LLAMA_TOKEN_NULL : static_cast<llama_token>(best_id);
  };

  auto count_printed = [&](llama_token token) -> bool {
    if (benchmark_mode) {
      // Benchmark hygiene: suppress token output while still advancing the shared counter.
      printed_this_call += 1;
      printed_total += 1;
      (void) korith_tokens_printed_total.fetch_add(1, std::memory_order_relaxed);
      return true;
    }

    const int32_t written = write_token_piece(vocab_target, token);
    if (written < 0) {
      return false;
    }
    // Count progress even when a token renders to an empty string so TPS/predict limits
    // reflect decoded tokens, not just visible output bytes.
    printed_this_call += 1;
    printed_total += 1;
    (void) korith_tokens_printed_total.fetch_add(1, std::memory_order_relaxed);
    return true;
  };

  auto step_one_target = [&](bool sync_draft = true) -> bool {
    if (n_ctx > 0 && static_cast<std::uint64_t>(pos_target) >= n_ctx) {
      if (step_debug_enabled) {
        std::fprintf(stderr,
                     "[STEP_TARGET_EXIT] reason=CTX_LIMIT pos=%lld n_ctx=%u\n",
                     static_cast<long long>(pos_target),
                     n_ctx);
        (void) std::fflush(stderr);
      }
      finished = true;
      return false;
    }

    const llama_pos pos0 = pos_target;
    const bool allow_eog = allow_eog_now();
    static bool step_target_enter_logged = false;
    if (step_debug_enabled && !step_target_enter_logged) {
      std::fprintf(stderr,
                   "[STEP_TARGET] enter pos=%lld allow_eog=%d logits=%p printed=%llu\n",
                   static_cast<long long>(pos0),
                   allow_eog ? 1 : 0,
                   static_cast<const void *>(logits_target),
                   static_cast<unsigned long long>(printed_total));
      (void) std::fflush(stderr);
      step_target_enter_logged = true;
    }
    const float entropy_at_commit = normalized_entropy_from_logits(logits_target, n_vocab, vocab_target, allow_eog);
    const llama_token token = sample_greedy_with_eog_control(logits_target, allow_eog);
    if (token == LLAMA_TOKEN_NULL) {
      if (step_debug_enabled) {
        std::fprintf(stderr, "[STEP_TARGET_EXIT] reason=TOKEN_NULL\n");
        (void) std::fflush(stderr);
      }
      finished = true;
      return false;
    }
	    if (llama_vocab_is_eog(vocab_target, token)) {
	      if (step_debug_enabled) {
	        std::fprintf(stderr, "[STEP_TARGET_EXIT] reason=EOG token=%d\n", token);
	        (void) std::fflush(stderr);
	      }
	      // Normal mode (or after the benchmark threshold): stop at EOG.
	      // Benchmark mode below threshold should never select EOG here.
	      finished = true;
	      return false;
	    }

	    const int32_t sched_depth = spec_depth;
	    const int decision = plan.allow_commit ? kDecisionCommit : kDecisionFallback;

      const auto baseline_decode_t0 = std::chrono::steady_clock::now();
	    if (!decode_one_token(ctx_target, batch_target, /* seq_id = */ 0, pos0, token, /* want_logits = */ true)) {
		      if (step_debug_enabled) {
		        std::fprintf(stderr,
		                     "[STEP_TARGET_EXIT] reason=DECODE_ONE_TOKEN_FALSE pos=%lld token=%d\n",
		                     static_cast<long long>(pos0),
		                     token);
		        (void) std::fflush(stderr);
		      }
		      finished = true;
		      return false;
		    }
      const auto baseline_decode_t1 = std::chrono::steady_clock::now();
      const double baseline_decode_ms = static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(baseline_decode_t1 - baseline_decode_t0).count()) * 1e-6;
      update_baseline_decode_ms(baseline_decode_ms);
	    pos_target = pos0 + 1;
	    token_prefix_hash = hash_token_prefix_step(token_prefix_hash, token);
	    logits_target = llama_get_logits(ctx_target);
	    if (!logits_target) {
	      if (step_debug_enabled) {
	        std::fprintf(stderr, "[STEP_TARGET_EXIT] reason=LOGITS_NULL\n");
	        (void) std::fflush(stderr);
	      }
	      finished = true;
	      return false;
	    }

    if (!count_printed(token)) {
      if (step_debug_enabled) {
        std::fprintf(stderr, "[STEP_TARGET_EXIT] reason=PRINT_FAIL\n");
        (void) std::fflush(stderr);
      }
      finished = true;
      return false;
    }
    (void) g_tokens_decoded_normally.fetch_add(1, std::memory_order_relaxed);
    bench_report_if_needed();
    generated_this_call += 1;

	    HoloEvent he{};
	    he.step_id = static_cast<std::uint64_t>(pos0);
	    he.token_id = static_cast<std::int32_t>(token);
	    he.thermo_depth_at_commit = static_cast<std::uint8_t>(std::clamp(sched_depth, 1, 255));
	    he.accepted = false;
	    he.acceptance_rate_at_commit = static_cast<float>(accept_ema);
	    he.entropy_at_commit = entropy_at_commit;
	    he.speculative_cost = 0.0f;
	    he.tps_snapshot = holo_tps_snapshot();
	    holo_record(he);
	    log_holo_commit(he.step_id, decision);

    // Keep draft in lock-step when we are forced to do target-only steps.
    if (spec_enabled && ctx_draft) {
      if (!sync_draft) {
        if (!spec_state.profitability_hard_disabled) {
          spec_state.deferred_draft_tokens.push_back(token);
        }
      } else if (pos_draft == pos0) {
        if (!decode_one_token(ctx_draft, batch_draft, /* seq_id = */ 0, pos_draft, token, /* want_logits = */ true)) {
          finished = true;
          return false;
        }
        pos_draft += 1;
        logits_draft = llama_get_logits(ctx_draft);
        if (!logits_draft) {
          finished = true;
          return false;
        }
      }
    }

    return true;
  };

  auto step_one_soft_spec = [&]() -> bool {
    // Same stopping rules as the normal path.
    if (n_ctx > 0 && static_cast<std::uint64_t>(pos_target) >= n_ctx) {
      if (step_debug_enabled) {
        std::fprintf(stderr,
                     "[STEP_SPEC_EXIT] reason=CTX_LIMIT pos=%lld n_ctx=%u\n",
                     static_cast<long long>(pos_target),
                     n_ctx);
        (void) std::fflush(stderr);
      }
      finished = true;
      return false;
    }

    const llama_pos pos0 = pos_target;
    const bool allow_eog = allow_eog_now();
    static bool step_spec_enter_logged = false;
    if (step_debug_enabled && !step_spec_enter_logged) {
      std::fprintf(stderr,
                   "[STEP_SPEC] enter pos=%lld allow_eog=%d logits=%p printed=%llu\n",
                   static_cast<long long>(pos0),
                   allow_eog ? 1 : 0,
                   static_cast<const void *>(logits_target),
                   static_cast<unsigned long long>(printed_total));
      (void) std::fflush(stderr);
      step_spec_enter_logged = true;
    }
    const float entropy_at_commit = normalized_entropy_from_logits(logits_target, n_vocab, vocab_target, allow_eog);

    // "Normal decode": choose the canonical next token deterministically.
    const llama_token main_tok = sample_greedy_with_eog_control(logits_target, allow_eog);
    if (main_tok == LLAMA_TOKEN_NULL) {
      if (step_debug_enabled) {
        std::fprintf(stderr, "[STEP_SPEC_EXIT] reason=TOKEN_NULL\n");
        (void) std::fflush(stderr);
      }
      finished = true;
      return false;
    }
    if (llama_vocab_is_eog(vocab_target, main_tok)) {
      if (step_debug_enabled) {
        std::fprintf(stderr, "[STEP_SPEC_EXIT] reason=EOG token=%d\n", main_tok);
        (void) std::fflush(stderr);
      }
      finished = true;
      return false;
    }

    const int32_t sched_depth = spec_depth;
    const int decision = plan.allow_commit ? kDecisionCommit : kDecisionFallback;
    const bool allow_skip = plan.allow_commit && reuse_kv;

    bool accepted = false;
    bool committed_via_spec = false;
    bool seq1_synced = false;

    SpecEvent ev{};
    ev.accepted = false;
    ev.depth = 2;
    ev.speculative_cost = 0;

    if (allow_skip) {
      seq1_synced = ensure_seq1_synced(pos0, /* depth_hint = */ 1, entropy_at_commit);

      // Two speculative candidates per step with different sampler seeds (single-token depth).
      const uint32_t base_seed = 0x6a09e667u ^ static_cast<uint32_t>(pos0 + 1);
      const llama_token cand1 = sample_dist_from_logits(
          logits_target, n_vocab, base_seed ^ 0xbb67ae85u, vocab_target, /* allow_eog = */ allow_eog);
      const llama_token cand2 = sample_dist_from_logits(
          logits_target, n_vocab, base_seed ^ 0x3c6ef372u, vocab_target, /* allow_eog = */ allow_eog);

      const bool match1 = (cand1 == main_tok);
      const bool match2 = (cand2 == main_tok);
      accepted = match1 || match2;

      // Speculative telemetry: 1-token proposals, accepted iff at least one candidate matched.
      spec_proposed += 1;
      if (accepted) {
        spec_accepted += 1;
      }

      ev.accepted = accepted;

      // Skip-on-accept:
      // If a speculative token is accepted AND allow_skip is true:
      // - Commit speculative token (seq 1 -> seq 0)
      // - Advance KV/cache state without running main decode (seq 0 decode) for this token.
      if (accepted && seq1_synced) {
        const llama_token spec_tok = match1 ? cand1 : cand2;
        ev.speculative_cost = 1;

        // Decode the speculative token in seq 1 and commit its KV to seq 0 on accept.
        if (decode_one_token(ctx_target, batch_target, /* seq_id = */ 1, pos0, spec_tok, /* want_logits = */ true)) {
          llama_memory_seq_cp(mem_target, /* src = */ 1, /* dst = */ 0, /* p0 = */ pos0, /* p1 = */ pos0 + 1);

          pos_target = pos0 + 1;
          logits_target = llama_get_logits(ctx_target);
          if (!logits_target) {
            finished = true;
            return false;
          }
          committed_via_spec = true;
        } else {
          // Speculative decode failed; try to roll back seq 1 to keep it in sync for future steps.
          (void) llama_memory_seq_rm(mem_target, /* seq_id = */ 1, /* p0 = */ pos0, /* p1 = */ pos0 + 1);
          (void) ensure_seq1_synced(pos0, /* depth_hint = */ 1, entropy_at_commit);
        }
      }
    }

    emit_spec_event(ev);

    if (!committed_via_spec) {
      // Deterministic path: decode the verified token in the committed sequence (seq 0).
      const auto baseline_decode_t0 = std::chrono::steady_clock::now();
      if (!decode_one_token(ctx_target, batch_target, /* seq_id = */ 0, pos0, main_tok, /* want_logits = */ true)) {
        if (step_debug_enabled) {
          std::fprintf(stderr,
                       "[STEP_SPEC_EXIT] reason=DECODE_ONE_TOKEN_FALSE pos=%lld token=%d\n",
                       static_cast<long long>(pos0),
                       main_tok);
          (void) std::fflush(stderr);
        }
        finished = true;
        return false;
      }
      const auto baseline_decode_t1 = std::chrono::steady_clock::now();
      const double baseline_decode_ms = static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(baseline_decode_t1 - baseline_decode_t0).count()) * 1e-6;
      update_baseline_decode_ms(baseline_decode_ms);
      pos_target = pos0 + 1;
      logits_target = llama_get_logits(ctx_target);
      if (!logits_target) {
        if (step_debug_enabled) {
          std::fprintf(stderr, "[STEP_SPEC_EXIT] reason=LOGITS_NULL\n");
          (void) std::fflush(stderr);
        }
        finished = true;
        return false;
      }

      // Keep seq 1 mirrored to seq 0 so future speculative commits don't require a full copy.
      if (seq1_synced) {
        llama_memory_seq_cp(mem_target, /* src = */ 0, /* dst = */ 1, /* p0 = */ pos0, /* p1 = */ pos0 + 1);
      }
	    }

	    token_prefix_hash = hash_token_prefix_step(token_prefix_hash, main_tok);
	    if (seq1_synced) {
	      holo_kv.valid = true;
	      holo_kv.prefix_hash = token_prefix_hash;
	      holo_kv.last_used_step = static_cast<std::uint64_t>(pos_target);
	      holo_kv.last_depth = 2;
	    }

	    // Metrics: count printed tokens only (no speculative accounting here).
	    if (!count_printed(main_tok)) {
	      finished = true;
	      return false;
	    }
    if (committed_via_spec) {
      (void) g_tokens_committed_without_decode.fetch_add(1, std::memory_order_relaxed);
    } else {
      (void) g_tokens_decoded_normally.fetch_add(1, std::memory_order_relaxed);
    }
    bench_report_if_needed();
    generated_this_call += 1;

    HoloEvent he{};
    he.step_id = static_cast<std::uint64_t>(pos0);
    he.token_id = static_cast<std::int32_t>(main_tok);
    he.thermo_depth_at_commit = static_cast<std::uint8_t>(std::clamp(sched_depth, 1, 255));
    he.accepted = accepted;
    he.acceptance_rate_at_commit = static_cast<float>(accept_ema);
    he.entropy_at_commit = entropy_at_commit;
    he.speculative_cost = static_cast<float>(ev.speculative_cost);
    he.tps_snapshot = holo_tps_snapshot();
    holo_record(he);
    log_holo_commit(he.step_id, decision);
    const std::uint64_t step_id = he.step_id;
    if ((step_id % 1000) == 0) {
      std::fprintf(stderr,
          "[HOLO] steps=%lu buffer=%zu\n",
          static_cast<unsigned long>(step_id),
          holo_size());
      (void) holo_is_future_safe(/* steps_ahead = */ 1, /* required_confidence = */ 0.0f);
    }

    // Keep the optional draft context in lock-step, but never let it affect correctness.
    if (ctx_draft && pos_draft == pos0) {
      if (!decode_one_token(ctx_draft, batch_draft, /* seq_id = */ 0, pos_draft, main_tok, /* want_logits = */ true)) {
        finished = true;
        return false;
      }
      pos_draft += 1;
      logits_draft = llama_get_logits(ctx_draft);
      if (!logits_draft) {
        finished = true;
        return false;
      }
    }

    return true;
  };

  auto adapt_depth = [&](int32_t proposed, int32_t accepted) {
    if (proposed <= 0) {
      return;
    }

    spec_proposed += static_cast<std::uint64_t>(proposed);
    spec_accepted += static_cast<std::uint64_t>(accepted);
  };

  auto ensure_draft_pending_synced = [&](llama_pos base_pos, std::size_t pending_count) -> bool {
    if (!ctx_draft) {
      return true;
    }
    llama_memory_t mem_d = llama_get_memory(ctx_draft);
    if (!mem_d) {
      return false;
    }
    const llama_pos want = base_pos + static_cast<llama_pos>(pending_count);
    if (pos_draft == want) {
      return true;
    }
    if (!llama_memory_seq_rm(mem_d, /* seq_id = */ 0, base_pos, /* p1 = */ -1)) {
      return false;
    }
    pos_draft = base_pos;
    for (std::size_t i = 0; i < pending_count; ++i) {
      const llama_token tok = spec_meta_at(i).token_id;
      if (!decode_one_token(ctx_draft, batch_draft, /* seq_id = */ 0, pos_draft, tok, /* want_logits = */ true)) {
        return false;
      }
      pos_draft += 1;
      logits_draft = llama_get_logits(ctx_draft);
      if (!logits_draft) {
        return false;
      }
    }
    return true;
  };

  bool commit_blocked_logged = false;

  while (!finished && generated_this_call < batch_tokens) {
    if ((spec_state.adaptive_request_disabled ||
         spec_state.profitability_hard_disabled ||
         spec_state.profitability_cooldown_steps > 0) &&
        spec_state.count > 0) {
      // Preserve correctness: flush already-verified pending tokens before forcing
      // profitability cooldown/hard-disable baseline mode.
      spec_state.flush_pending = true;
      spec_state.flush_reason = FlushReason::kPressure;
    }

    const bool commit_blocked =
        plan.allow_exec && !plan.allow_commit && !spec_shadow_when_commit_blocked();
    if (!commit_blocked) {
      commit_blocked_logged = false;
    }
    if (force_normal_decode || !plan.allow_exec || commit_blocked) {
      int reason_code = plan.reason_code;
      if (force_normal_decode) {
        reason_code = plan.enabled ? korith::core::SPEC_DISABLE_FAILSAFE : plan.reason_code;
      } else if (commit_blocked) {
        reason_code = korith::core::SPEC_DISABLE_CP_GATE_OFF;
        if (!commit_blocked_logged) {
          std::fprintf(stderr, "[SPEC_SKIP] disabled reason=commit_blocked_baseline_decode\n");
          (void) std::fflush(stderr);
          commit_blocked_logged = true;
        }
      }
      log_spec_disabled(reason_code, /* engines = */ 0, /* pressure = */ 0);
      if (spec_state.valid) {
        reset_spec_state(/* clear_seq = */ true);
      }
      if (!step_one_target(/* sync_draft = */ false)) {
        break;
      }
      continue;
    }

    if (spec_state.adaptive_request_disabled && spec_state.count == 0) {
      if (spec_state.valid) {
        reset_spec_state(/* clear_seq = */ true, /* preserve_profit_state = */ true);
      }
      if (!step_one_target(/* sync_draft = */ false)) {
        break;
      }
      continue;
    }

    if (spec_state.profitability_hard_disabled && spec_state.count == 0) {
      if (spec_state.valid) {
        reset_spec_state(/* clear_seq = */ true, /* preserve_profit_state = */ true);
      }
      if (!step_one_target(/* sync_draft = */ false)) {
        break;
      }
      continue;
    }

    if (spec_state.profitability_cooldown_steps > 0 && spec_state.count == 0) {
      const int32_t remaining_before = spec_state.profitability_cooldown_steps;
      if (spec_state.valid) {
        // Drop speculative pending state while forcing normal decode cooldown.
        reset_spec_state(/* clear_seq = */ true, /* preserve_profit_state = */ true);
      }
      spec_state.profitability_cooldown_steps =
          std::max<int32_t>(0, spec_state.profitability_cooldown_steps - 1);
      std::fprintf(stderr,
                   "[SPEC_PROFIT] gate=cooldown remaining=%d\n",
                   remaining_before);
      (void) std::fflush(stderr);
      if (!step_one_target(/* sync_draft = */ false)) {
        break;
      }
      if (spec_state.profitability_cooldown_steps == 0) {
        spec_state.profitability_probe_steps = 1;
        spec_state.profitability_retry_count += 1;
        std::fprintf(stderr, "[SPEC_PROFIT] gate=retry\n");
        (void) std::fflush(stderr);
      }
      continue;
    }

    const bool allow_speculation = plan.enabled && (force_speculation || spec_depth > 1);
    if (!spec_enabled || !reuse_kv || !allow_speculation) {
      int reason_code = plan.reason_code;
      if (plan.enabled) {
        if (!allow_speculation || spec_depth < 2) {
          reason_code = (n_seq_max < 2) ? korith::core::SPEC_DISABLE_CONTEXT_LIMIT
                                        : korith::core::SPEC_DISABLE_DEPTH_LT_2;
        } else if (plan.lanes < 2) {
          reason_code = korith::core::SPEC_DISABLE_LANES_LT_2;
        } else if (!reuse_kv) {
          reason_code = korith::core::SPEC_DISABLE_CP_GATE_OFF;
        } else if (!spec_enabled) {
          reason_code = korith::core::SPEC_DISABLE_CONTEXT_LIMIT;
        }
      }
      log_spec_disabled(reason_code, /* engines = */ 0, /* pressure = */ 0);
      if (spec_state.valid) {
        reset_spec_state(/* clear_seq = */ true);
      }
      if (!step_one_soft_spec()) {
        break;
      }
      continue;
    }

    llama_pos base_pos = pos_target;
    std::size_t pending_count = spec_state.count;
    llama_pos pending_pos = base_pos + static_cast<llama_pos>(pending_count);

    if (pending_count > 0 && !spec_state.valid) {
      reset_spec_state(/* clear_seq = */ true);
      pending_count = 0;
      pending_pos = base_pos;
    }
    if (ctx_draft && !spec_state.deferred_draft_tokens.empty()) {
      const int32_t rc_replay = decode_batch_tokens(
          ctx_draft,
          batch_draft,
          /* seq_id = */ 0,
          pos_draft,
          spec_state.deferred_draft_tokens,
          /* logits_all = */ false);
      if (rc_replay != 0) {
        reset_spec_state(/* clear_seq = */ true, /* preserve_profit_state = */ true);
        if (!step_one_target(/* sync_draft = */ false)) {
          break;
        }
        continue;
      }
      pos_draft += static_cast<llama_pos>(spec_state.deferred_draft_tokens.size());
      logits_draft = llama_get_logits(ctx_draft);
      spec_state.deferred_draft_tokens.clear();
      if (!logits_draft) {
        finished = true;
        break;
      }
    }
    if (!ensure_draft_pending_synced(base_pos, pending_count)) {
      reset_spec_state(/* clear_seq = */ true);
      if (!step_one_target()) {
        break;
      }
      continue;
    }

    double accept_gate = 0.0;
    if (std::isfinite(spec_state.accept_gate_ema)) {
      accept_gate = spec_state.accept_gate_ema;
    } else if (std::isfinite(accept_ema)) {
      accept_gate = accept_ema;
    }
    const double delta_gate = std::isfinite(tps_delta) ? tps_delta : 0.0;

    if (pending_count > 0 && mem_target) {
      const llama_pos want_max = base_pos + static_cast<llama_pos>(pending_count) - 1;
      const llama_pos seq_max = llama_memory_seq_pos_max(mem_target, /* seq_id = */ spec_state.seq_id);
      if (seq_max != want_max) {
        reset_spec_state(/* clear_seq = */ true);
        pending_count = 0;
        pending_pos = base_pos;
      }
    }

    spec_state.step_id += 1;
    const bool allow_eog_base = allow_eog_now();
    const float * gate_logits =
        (pending_count > 0)
            ? spec_logits_ptr(spec_index(pending_count - 1))
            : logits_target;
    if (!gate_logits) {
      finished = true;
      break;
    }
    const float base_entropy =
        normalized_entropy_from_logits(gate_logits, n_vocab, vocab_target, allow_eog_base);
    const float entropy_delta = spec_state.have_last_entropy
        ? (base_entropy - spec_state.last_entropy)
        : 0.0f;
    const bool entropy_increase = spec_state.have_last_entropy && entropy_delta > 0.1f;
    const bool accept_drop = spec_state.have_last_accept && (accept_gate + 1e-6 < spec_state.last_accept);
    const bool age_decay = entropy_increase || accept_drop;
    const bool had_accept_history = spec_state.have_last_accept;
    spec_state.last_entropy = base_entropy;
    spec_state.have_last_entropy = true;
    const float safety_entropy_max = spec_safety_entropy_max();
    const double safety_min_accept = spec_safety_min_accept();
    const double safety_relax_accept = spec_safety_relax_accept();
    const int entropy_streak_min = spec_safety_entropy_streak();
    // Avoid forcing a safety override before we have acceptance history.
    const bool low_accept = had_accept_history && (accept_gate < safety_min_accept);
    const bool entropy_hot = (base_entropy > safety_entropy_max);
    if (entropy_hot) {
      spec_state.entropy_hot_streak = static_cast<std::uint16_t>(
          std::min<int>(static_cast<int>(spec_state.entropy_hot_streak) + 1, 65535));
    } else {
      spec_state.entropy_hot_streak = 0;
    }
    // High acceptance is strong evidence that speculative tokens are stable; do
    // not repeatedly reset ages from entropy alone in that case.
    const bool entropy_needs_guard = !had_accept_history || (accept_gate < safety_relax_accept);
    const bool entropy_override =
        entropy_needs_guard && (static_cast<int>(spec_state.entropy_hot_streak) >= entropy_streak_min);
    const bool safety_override = low_accept || entropy_override;

    std::size_t skipped_tokens = 0;
    std::size_t fast_verified_tokens = 0;
    std::size_t aged_tokens = 0;
    if (pending_count > 0) {
      std::size_t commit_count = 0;
      auto has_pending_verified_speculation = [&]() -> bool {
        return pending_count > 0;
      };
      auto has_aged_speculative_tokens = [&]() -> bool {
        return commit_count > 0;
      };

      bool flush_pending = spec_state.flush_pending;
      FlushReason flush_reason = spec_state.flush_reason;
      if (spec_state.finish_pending) {
        flush_pending = true;
        flush_reason = FlushReason::kEos;
      } else if (pending_count >= kMaxSpecTokens) {
        flush_pending = true;
        flush_reason = FlushReason::kPressure;
      }

      float conf_ema = 0.0f;
      bool conf_valid = false;
      if (spec_state.seq_id >= 0 &&
          static_cast<std::size_t>(spec_state.seq_id) < spec_state.conf_ema.size()) {
        conf_ema = spec_state.conf_ema[static_cast<std::size_t>(spec_state.seq_id)];
        conf_valid = spec_state.conf_valid[static_cast<std::size_t>(spec_state.seq_id)] != 0;
      }
      std::uint16_t age_inc = 1;
      if (conf_valid) {
        if (conf_ema >= kConfEmaBoost) {
          age_inc = 2;
        } else if (conf_ema < kConfEmaFreeze) {
          age_inc = 0;
        }
      }
      std::uint16_t age_min = 0;
      std::uint16_t age_max = 0;
      double age_sum = 0.0;
      for (std::size_t i = 0; i < pending_count; ++i) {
        TokenMeta & meta = spec_meta_at(i);
        if (flush_pending || safety_override) {
          meta.age = 0;
          meta.last_verified_step = spec_state.step_id;
        } else {
          if (meta.last_verified_step != spec_state.step_id) {
            if (meta.age < kAgeSkip) {
              const std::uint16_t next_age = static_cast<std::uint16_t>(meta.age + age_inc);
              const std::uint16_t new_age = static_cast<std::uint16_t>(std::min<std::uint16_t>(kAgeSkip, next_age));
              if (new_age > meta.age) {
                aged_tokens += 1;
              }
              meta.age = new_age;
            }
          }
          if (age_decay && meta.age > 0) {
            meta.age = static_cast<std::uint16_t>(meta.age - 1);
          }
          meta.last_verified_step = spec_state.step_id;
          if (meta.age >= kAgeFastVerify && meta.age < kAgeSkip) {
            const float * row = spec_logits_ptr(spec_index(i));
            if (!row) {
              finished = true;
              return -1;
            }
            const llama_token predicted = sample_greedy_from_logits(row, n_vocab);
            if (predicted != meta.token_id) {
              meta.age = 0;
            } else {
              fast_verified_tokens += 1;
            }
          }
        }
        if (i == 0) {
          age_min = meta.age;
          age_max = meta.age;
        } else {
          age_min = std::min<std::uint16_t>(age_min, meta.age);
          age_max = std::max<std::uint16_t>(age_max, meta.age);
        }
        age_sum += static_cast<double>(meta.age);
      }

      if (safety_override) {
        const char * detail = low_accept ? "low_accept" : "entropy";
        std::fprintf(stderr,
                     "[SPEC_SKIP] disabled reason=safety_override detail=%s entropy=%.3f "
                     "accept=%.3f streak=%u\n",
                     detail,
                     static_cast<double>(base_entropy),
                     accept_gate,
                     static_cast<unsigned>(spec_state.entropy_hot_streak));
        (void) std::fflush(stderr);
      }
      const double age_avg = age_sum / static_cast<double>(pending_count);
      std::fprintf(stderr,
                   "[SPEC_AGE] min=%u avg=%.2f max=%u\n",
                   static_cast<unsigned>(age_min),
                   age_avg,
                   static_cast<unsigned>(age_max));
      (void) std::fflush(stderr);

      if (!flush_pending && !safety_override) {
        while (commit_count < pending_count) {
          if (spec_meta_at(commit_count).age >= kAgeSkip) {
            commit_count += 1;
            continue;
          }
          break;
        }
      }
      if (flush_pending) {
        commit_count = pending_count;
      }
      if (!plan.allow_commit) {
        // Commit is blocked by control-plane policy. Do not keep speculative
        // tokens resident, otherwise we can spin on the pending buffer without
        // making target-sequence progress.
        commit_count = 0;
        spec_state.head = spec_index(pending_count);
        spec_state.count = 0;
        spec_state.flush_pending = false;
        spec_state.flush_reason = FlushReason::kNone;
        pending_count = 0;
        pending_pos = base_pos;
        std::fprintf(stderr, "[SPEC_SKIP] disabled reason=commit_blocked_drop_pending\n");
        (void) std::fflush(stderr);
        if (!step_one_target()) {
          break;
        }
        continue;
      }

      const bool hold_pending =
          has_pending_verified_speculation() && !has_aged_speculative_tokens() && !flush_pending && !safety_override;
      if (hold_pending) {
        // Delayed-commit override: keep verified tokens resident so they can age.
      }

      const std::size_t remaining_call_budget =
          (generated_this_call < batch_tokens)
              ? static_cast<std::size_t>(batch_tokens - generated_this_call)
              : 0u;
      if (commit_count > remaining_call_budget) {
        commit_count = remaining_call_budget;
      }

      if (commit_count > 0) {
        static bool determinism_checked = false;
        static bool determinism_enabled = false;
        if (!determinism_checked) {
          determinism_checked = true;
          if (const char * env = std::getenv("KORITH_DETERMINISM_CHECK")) {
            determinism_enabled = (env[0] != '\0') && (env[0] != '0');
          }
        }
        if (determinism_enabled) {
          for (std::size_t i = 0; i < commit_count; ++i) {
            const std::size_t idx = spec_index(i);
            const float * row = spec_logits_ptr(idx);
            if (!row) {
              finished = true;
              return -1;
            }
            const llama_token predicted = sample_greedy_from_logits(row, n_vocab);
            if (predicted != spec_state.meta[idx].token_id) {
              std::fprintf(stderr,
                           "[DETERMINISM] mismatch token=%d predicted=%d\n",
                           static_cast<int>(spec_state.meta[idx].token_id),
                           static_cast<int>(predicted));
              (void) std::fflush(stderr);
              finished = true;
              return -1;
            }
          }
        }
        const std::size_t last_commit_idx = spec_index(commit_count - 1);
        g_best_logits_buffer.resize(static_cast<std::size_t>(n_vocab));
        std::memcpy(g_best_logits_buffer.data(),
                    spec_logits_ptr(last_commit_idx),
                    static_cast<std::size_t>(n_vocab) * sizeof(float));
        logits_target = g_best_logits_buffer.data();

        const llama_pos commit_end = base_pos + static_cast<llama_pos>(commit_count);
        const llama_pos commit_start = base_pos;
        llama_memory_seq_cp(mem_target, /* src = */ spec_state.seq_id, /* dst = */ 0, /* p0 = */ base_pos, /* p1 = */ commit_end);

        const std::uint8_t depth_u8 = static_cast<std::uint8_t>(std::clamp(spec_depth, 1, 255));
        for (std::size_t i = 0; i < commit_count; ++i) {
          const std::size_t idx = spec_index(i);
          const llama_token tok = spec_state.meta[idx].token_id;
          if (!count_printed(tok)) {
            finished = true;
            return -1;
          }
          token_prefix_hash = hash_token_prefix_step(token_prefix_hash, tok);

          HoloEvent he{};
          he.step_id = static_cast<std::uint64_t>(base_pos + static_cast<llama_pos>(i));
          he.token_id = static_cast<std::int32_t>(tok);
          he.thermo_depth_at_commit = depth_u8;
          he.accepted = spec_state.speculative[idx] != 0;
          he.acceptance_rate_at_commit = static_cast<float>(accept_ema);
          he.entropy_at_commit = std::clamp(spec_state.entropy[idx], 0.0f, 1.0f);
          he.speculative_cost = 0.0f;
          he.tps_snapshot = holo_tps_snapshot();
          holo_record(he);
        }

        skipped_tokens = commit_count;
        pos_target = commit_end;
        base_pos = pos_target;
        generated_this_call += static_cast<int32_t>(commit_count);
        (void) g_tokens_committed_without_decode.fetch_add(
            static_cast<std::uint64_t>(commit_count), std::memory_order_relaxed);
        (void) g_tokens_decoded_normally.fetch_add(
            static_cast<std::uint64_t>(commit_count), std::memory_order_relaxed);
        bench_report_if_needed();
        log_holo_commit(static_cast<std::uint64_t>(commit_start), kDecisionCommit);
        holo_kv.valid = true;
        holo_kv.prefix_hash = token_prefix_hash;
        holo_kv.last_used_step = static_cast<std::uint64_t>(pos_target);
        holo_kv.last_depth = spec_depth;

        spec_state.head = spec_index(commit_count);
        spec_state.count -= commit_count;
        pending_count = spec_state.count;
        pending_pos = pos_target + static_cast<llama_pos>(pending_count);
      }

      const char * reason = "age";
      if (flush_pending) {
        reason = (flush_reason == FlushReason::kEos) ? "eos" : "pressure";
      }
      std::fprintf(stderr,
                   "[SPEC_SKIP] skipped=%zu aged=%zu total_spec=%zu reason=%s\n",
                   skipped_tokens,
                   aged_tokens,
                   pending_count + skipped_tokens,
                   reason);
      (void) std::fflush(stderr);
      spec_skipped_tokens += static_cast<std::uint64_t>(skipped_tokens);
      spec_fast_verified_tokens += static_cast<std::uint64_t>(fast_verified_tokens);

      if (!has_pending_verified_speculation() || commit_count > 0) {
        spec_state.flush_pending = false;
        spec_state.flush_reason = FlushReason::kNone;
      }

      if (spec_state.finish_pending) {
        if (pending_count == 0) {
          finished = true;
        }
        break;
      }
    }

    if (spec_state.finish_pending && pending_count == 0) {
      finished = true;
      break;
    }

    if (generated_this_call >= batch_tokens) {
      break;
    }

    if ((spec_state.profitability_hard_disabled || spec_state.profitability_cooldown_steps > 0) &&
        pending_count == 0) {
      // Pending speculative tokens were flushed; next loop iteration enters
      // baseline cooldown/hard-disable path without proposing new draft tokens.
      continue;
    }

    const int32_t remaining_call = batch_tokens - generated_this_call;
    int32_t spec_window = spec_depth;
    bool window_expanded = false;
    if (spec_window >= 2) {
      if (spec_v3 && spec_v3_max_block >= 2) {
        spec_window = std::clamp(spec_window, spec_v3_min_block, spec_v3_max_block);
        if (force_speculation && spec_window_limit > 0) {
          spec_window = std::clamp(spec_window_limit, spec_v3_min_block, spec_v3_max_block);
        }
        if (delta_gate <= 0.0 && spec_window > spec_v3_min_block) {
          spec_window = spec_v3_min_block;
          std::fprintf(stderr, "[SPEC_V3_WINDOW] size=%d reason=negative_delta\n", spec_window);
          (void) std::fflush(stderr);
        } else if (spec_window > spec_v3_min_block) {
          window_expanded = true;
          std::fprintf(stderr,
                       "[SPEC_V3_WINDOW] size=%d range=%d..%d\n",
                       spec_window,
                       spec_v3_min_block,
                       spec_v3_max_block);
          (void) std::fflush(stderr);
        }
      } else {
        const bool allow_expand = (spec_depth >= 3) && (accept_gate >= 0.7) && (delta_gate > 0.0);
        const int32_t max_window = allow_expand ? 6 : 4;
        spec_window = std::clamp(spec_window, 2, max_window);
        if (force_speculation && spec_window_limit > spec_window) {
          spec_window = spec_window_limit;
        }
        if (delta_gate <= 0.0 && spec_window > 2) {
          spec_window = 2;
          std::fprintf(stderr, "[SPEC_WINDOW] size=2 reason=negative_delta\n");
          (void) std::fflush(stderr);
        } else if (allow_expand && spec_window > 4) {
          window_expanded = true;
          std::fprintf(stderr,
                       "[SPEC_WINDOW] size=%d reason=accept_ema+tps_delta\n",
                       spec_window);
          (void) std::fflush(stderr);
        }
      }
    }
    int32_t k = std::min<int32_t>(spec_window, remaining_call);
    k = std::min<int32_t>(k, max_spec_depth(ctx_target));
    const int32_t max_pending_room = static_cast<int32_t>(kMaxSpecTokens - pending_count);
    if (max_pending_room <= 0) {
      if (pending_count > 0) {
        spec_state.flush_pending = true;
        spec_state.flush_reason = FlushReason::kPressure;
        continue;
      }
      if (!step_one_target()) {
        break;
      }
      continue;
    }
    k = std::min<int32_t>(k, max_pending_room);
    if (k <= 1) {
      if (pending_count > 0) {
        spec_state.flush_pending = true;
        spec_state.flush_reason = FlushReason::kPressure;
        continue;
      }
      if (!step_one_target()) {
        break;
      }
      continue;
    }

    // Clamp by context limit.
    if (n_ctx > 0) {
      const std::int64_t avail = static_cast<std::int64_t>(n_ctx) - static_cast<std::int64_t>(pending_pos);
      if (avail <= 0) {
        if (pending_count > 0) {
          spec_state.finish_pending = true;
          spec_state.flush_pending = true;
          spec_state.flush_reason = FlushReason::kPressure;
          continue;
        }
        finished = true;
        break;
      }
      if (k > avail) {
        k = static_cast<int32_t>(avail);
      }
      if (k <= 1) {
        if (pending_count > 0) {
          spec_state.flush_pending = true;
          spec_state.flush_reason = FlushReason::kPressure;
          continue;
        }
        if (!step_one_target()) {
          break;
        }
        continue;
      }
    }

	    const llama_pos base_pos_draft = pending_pos;
	    const llama_pos verify_base_pos = pending_pos;

	    const int32_t sched_depth = spec_depth;

	    const float * verify_base_logits =
	        (pending_count > 0)
	            ? spec_logits_ptr(spec_index(pending_count - 1))
	            : logits_target;
	    if (!verify_base_logits) {
	      finished = true;
	      break;
	    }
	    const float verify_entropy =
	        normalized_entropy_from_logits(verify_base_logits, n_vocab, vocab_target, allow_eog_base);

	    // Sample the first target token from the current (base) logits before overwriting them.
	    const llama_token target_tok0 = sample_greedy_with_eog_control(verify_base_logits, allow_eog_base);
	    if (target_tok0 == LLAMA_TOKEN_NULL) {
	      finished = true;
	      break;
	    }

	    // If the target predicts termination at the current position, stop immediately after pending commits.
	    if (llama_vocab_is_eog(vocab_target, target_tok0)) {
	      if (pending_count == 0) {
	        finished = true;
	        break;
	      }
	      spec_state.finish_pending = true;
	      spec_state.flush_pending = true;
	      spec_state.flush_reason = FlushReason::kEos;
	      continue;
	    }

    if (spec_v3 && spec_v3_max_block >= 2) {
      const int32_t v3_depth_cap = std::max<int32_t>(sched_depth, spec_v3_max_block);
      k = std::min<int32_t>(k, v3_depth_cap);
    } else if (k > sched_depth) {
      k = sched_depth;
    }

    // 1) Draft proposes up to k tokens (not including EOG).
    std::vector<llama_token> draft_tokens;
    std::vector<float> draft_logits_rows;
    draft_tokens.reserve(static_cast<std::size_t>(k));
    draft_logits_rows.reserve(static_cast<std::size_t>(k) * static_cast<std::size_t>(n_vocab));

    bool draft_proposed_eog = false;
    bool draft_failed = false;
    const bool allow_eog_spec = allow_eog_now();
    const auto draft_step_t0 = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < k; ++i) {
      const float * row_logits = logits_draft;
      if (!row_logits) {
        draft_failed = true;
        break;
      }
      const llama_token tok = sample_greedy_with_eog_control(row_logits, allow_eog_spec);
      if (tok == LLAMA_TOKEN_NULL) {
        draft_failed = true;
        break;
      }
      if (llama_vocab_is_eog(vocab_target, tok)) {
        // In normal mode, the draft can propose EOG to terminate speculation early.
        // In benchmark mode below threshold, EOG is suppressed by the sampler above.
        draft_proposed_eog = true;
        break;
      }

      draft_logits_rows.insert(draft_logits_rows.end(), row_logits, row_logits + n_vocab);
      if (!decode_one_token(
              ctx_draft,
              batch_draft,
              /* seq_id = */ 0,
              pos_draft,
              tok,
              /* want_logits = */ true,
              /* allow_cuda_graph = */ spec_v3)) {
        draft_failed = true;
        break;
      }
      pos_draft += 1;
      logits_draft = llama_get_logits(ctx_draft);
      if (!logits_draft) {
        draft_failed = true;
        break;
      }
      draft_tokens.push_back(tok);
    }
    const auto draft_step_t1 = std::chrono::steady_clock::now();
    const std::uint64_t draft_elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(draft_step_t1 - draft_step_t0).count());

    if (draft_failed) {
      if (!rollback_draft_to(ctx_draft, pos_draft, base_pos_draft)) {
        finished = true;
        return -1;
      }
      if (pending_count > 0) {
        reset_spec_state(/* clear_seq = */ true);
        if (!rollback_draft_to(ctx_draft, pos_draft, base_pos)) {
          finished = true;
          return -1;
        }
      }
      if (!step_one_target()) {
        break;
      }
      continue;
    }

    // If draft failed to propose anything, fall back to target-only.
    if (draft_tokens.empty() && !draft_proposed_eog) {
      if (!rollback_draft_to(ctx_draft, pos_draft, base_pos_draft)) {
        finished = true;
        return -1;
      }
      if (pending_count > 0) {
        reset_spec_state(/* clear_seq = */ true);
        if (!rollback_draft_to(ctx_draft, pos_draft, base_pos)) {
          finished = true;
          return -1;
        }
      }
      if (!step_one_target()) {
        break;
      }
      continue;
    }

    if (draft_tokens.empty() && draft_proposed_eog) {
      if (!rollback_draft_to(ctx_draft, pos_draft, base_pos_draft)) {
        finished = true;
        return -1;
      }
      if (pending_count > 0) {
        reset_spec_state(/* clear_seq = */ true);
        if (!rollback_draft_to(ctx_draft, pos_draft, base_pos)) {
          finished = true;
          return -1;
        }
      }
      if (!step_one_target()) {
        break;
      }
      continue;
    }

    // 2) Target verifies in temporary speculative sequences (seq_id >= 1).
    llama_memory_t mem_t = mem_target;
    if (!mem_t) {
      finished = true;
      return -1;
    }

    const int32_t max_engines = std::max<int32_t>(0, llama_n_seq_max(ctx_target) - 1);
    const int32_t requested_engines = std::max<int32_t>(0, plan.lanes - 1);
    int32_t engine_count = std::min<int32_t>(max_engines, requested_engines);
    const int32_t env_engine_cap = spec_max_verify_engines();
    if (env_engine_cap > 0) {
      engine_count = std::min<int32_t>(engine_count, env_engine_cap);
    } else if (spec_v3) {
      // Blockwise verify is most efficient with a single verify lane unless explicitly overridden.
      engine_count = std::min<int32_t>(engine_count, 1);
    }
    if (engine_count > 1) {
      if (accept_gate <= 0.6 || delta_gate < 0.0) {
        engine_count = 1;
      }
    }
    if (window_expanded && engine_count > 1) {
      engine_count = 1;
    }
    if (pending_count > 0) {
      engine_count = 1;
    }
    if (engine_count_cap > 0 && engine_count > engine_count_cap) {
      engine_count = engine_count_cap;
    }
    if (engine_count > 0 && !spec_engines_logged) {
      std::fprintf(stderr,
                   "[SPEC_ENGINES] created=%d lanes=%d n_seq_max=%d\n",
                   engine_count,
                   plan.lanes,
                   n_seq_max);
      (void) std::fflush(stderr);
      spec_engines_logged = true;
    }
    if (engine_count <= 0) {
      const int pressure = (pending_count >= kMaxSpecTokens) ? 1 : 0;
      const int reason_code = pressure ? korith::core::SPEC_DISABLE_MEMORY_PRESSURE
                                       : korith::core::SPEC_DISABLE_ENGINES_0;
      log_spec_disabled(reason_code, /* engines = */ 0, pressure);
      if (!rollback_draft_to(ctx_draft, pos_draft, base_pos_draft)) {
        finished = true;
        return -1;
      }
      if (pending_count > 0) {
        reset_spec_state(/* clear_seq = */ true);
        if (!rollback_draft_to(ctx_draft, pos_draft, base_pos)) {
          finished = true;
          return -1;
        }
      }
      if (!step_one_target()) {
        break;
      }
      continue;
    }
    spec_seen = true;

    struct BestEngine {
      int32_t index = -1;
      llama_seq_id seq_id = 1;
      int32_t proposed = 0;
      int32_t accepted = 0;
      double score = 0.0;
      double confidence = 0.0;
      std::vector<llama_token> out_tokens;
      std::vector<std::uint8_t> out_speculative;
      std::vector<float> out_logits_rows;
      std::array<float, 33> out_entropies{};
      bool have_out_entropies = false;
      bool decode_mismatch_in_spec = false;
      llama_token mismatch_token = LLAMA_TOKEN_NULL;
      llama_pos mismatch_pos = 0;
      bool finished = false;
      std::uint64_t elapsed_ns = 0;
      std::uint64_t verify_ns = 0;
      std::uint64_t accept_scan_ns = 0;
    };

    BestEngine best{};
    double best_score = -1.0;

    const std::size_t row_stride = static_cast<std::size_t>(n_vocab);
    for (int32_t engine_idx = 0; engine_idx < engine_count; ++engine_idx) {
      const llama_seq_id seq_id =
          (pending_count > 0 && engine_count == 1)
              ? spec_state.seq_id
              : static_cast<llama_seq_id>(engine_idx + 1);
      std::vector<llama_token> engine_tokens;
      const std::vector<llama_token> * verify_tokens = &draft_tokens;
      bool engine_ok = true;

      if (engine_idx > 0) {
        engine_tokens.reserve(draft_tokens.size());
        if (draft_logits_rows.size() < draft_tokens.size() * row_stride) {
          engine_ok = false;
        } else {
          const uint32_t base_seed =
              0x9e3779b9u ^ static_cast<uint32_t>(verify_base_pos + 1) ^ static_cast<uint32_t>(engine_idx + 1);
          for (std::size_t i = 0; i < draft_tokens.size(); ++i) {
            const float * row_logits = draft_logits_rows.data() + i * row_stride;
            const uint32_t seed = base_seed
                ^ static_cast<uint32_t>((i + 1) * 0x85ebca6bu)
                ^ static_cast<uint32_t>((engine_idx + 1) * 0xc2b2ae35u);
            llama_token tok = sample_dist_from_logits(row_logits, n_vocab, seed, vocab_target, allow_eog_spec);
            if (tok == LLAMA_TOKEN_NULL) {
              engine_ok = false;
              break;
            }
            engine_tokens.push_back(tok);
          }
        }

        if (engine_ok && engine_tokens.size() == draft_tokens.size()) {
          bool identical = true;
          for (std::size_t i = 0; i < engine_tokens.size(); ++i) {
            if (engine_tokens[i] != draft_tokens[i]) {
              identical = false;
              break;
            }
          }
          if (identical && !engine_tokens.empty()) {
            for (std::size_t i = engine_tokens.size(); i-- > 0;) {
              const float * row_logits = draft_logits_rows.data() + i * row_stride;
              const llama_token alt = pick_alternate_token(
                  row_logits, n_vocab, vocab_target, allow_eog_spec, draft_tokens[i]);
              if (alt != LLAMA_TOKEN_NULL && alt != draft_tokens[i]) {
                engine_tokens[i] = alt;
                break;
              }
            }
          }
        }
        verify_tokens = &engine_tokens;
      }

      const int32_t proposed = static_cast<int32_t>(verify_tokens->size());
      int32_t accepted = 0;
      std::vector<llama_token> out_tokens;
      std::vector<std::uint8_t> out_speculative;
      std::vector<float> out_logits_rows;
      out_tokens.reserve(static_cast<std::size_t>(proposed + 1));
      out_speculative.reserve(static_cast<std::size_t>(proposed + 1));
      out_logits_rows.reserve(static_cast<std::size_t>(proposed) * static_cast<std::size_t>(n_vocab));
      float out_entropies[33]{};
      bool have_out_entropies = false;
      const float * verify_logits = nullptr;
      bool decode_mismatch_in_spec = false;
      llama_token mismatch_token = LLAMA_TOKEN_NULL;
      llama_pos mismatch_pos = verify_base_pos;
      bool engine_finished = false;
      std::uint64_t verify_forward_ns = 0;
      std::uint64_t accept_scan_ns = 0;
      const auto t0 = std::chrono::steady_clock::now();

      if (!engine_ok) {
        (void) llama_memory_seq_rm(mem_t, /* seq_id = */ seq_id, /* p0 = */ -1, /* p1 = */ -1);
      } else if (pending_count == 0) {
        if (!ensure_seq_synced(seq_id, base_pos, /* depth_hint = */ k, verify_entropy)) {
          engine_ok = false;
        }
      } else if (seq_id != spec_state.seq_id) {
        engine_ok = false;
      }

      if (engine_ok) {
        const auto verify_forward_t0 = std::chrono::steady_clock::now();
        const int32_t rc =
            decode_batch_tokens(ctx_target, batch_target, /* seq_id = */ seq_id, verify_base_pos, *verify_tokens, /* logits_all = */ true);
        const auto verify_forward_t1 = std::chrono::steady_clock::now();
        verify_forward_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(verify_forward_t1 - verify_forward_t0).count());
        if (rc != 0) {
          engine_ok = false;
        }
      }

      if (!engine_ok) {
        (void) llama_memory_seq_rm(mem_t, /* seq_id = */ seq_id, /* p0 = */ -1, /* p1 = */ -1);
      } else {
        verify_logits = llama_get_logits(ctx_target);
        if (!verify_logits) {
          (void) llama_memory_seq_rm(mem_t, /* seq_id = */ seq_id, /* p0 = */ -1, /* p1 = */ -1);
          (void) rollback_draft_to(ctx_draft, pos_draft, base_pos_draft);
          finished = true;
          return -1;
        }

        bool mismatch = false;
        if (proposed > 0) {
          bool gpu_verify_ok = false;
          bool allow_eog_verify = allow_eog_now();
          int32_t mismatch_idx = -1;
          llama_token mismatch_tok = LLAMA_TOKEN_NULL;

          if (spec_v3 &&
              gpu_accept_scan_enabled() &&
              gpu_spec_verify_available() &&
              spec_fused_verify_enabled() &&
              allow_eog_verify) {
            korith::core::GpuSpecVerifyResult verify_result{};
            const auto accept_scan_t0 = std::chrono::steady_clock::now();
            gpu_verify_ok = korith::core::gpu_spec_verify_greedy_segmented_from_host(
                verify_base_logits,
                proposed > 1 ? verify_logits : nullptr,
                proposed,
                n_vocab,
                reinterpret_cast<const int32_t *>(verify_tokens->data()),
                &verify_result);
            const auto accept_scan_t1 = std::chrono::steady_clock::now();
            accept_scan_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(accept_scan_t1 - accept_scan_t0).count());

            if (gpu_verify_ok) {
              accepted = std::clamp(verify_result.accepted_count, 0, proposed);
              mismatch_idx = verify_result.first_mismatch;
              mismatch_tok = static_cast<llama_token>(verify_result.mismatch_token);
            }
          }

          if (!gpu_verify_ok) {
            std::array<llama_token, kMaxSpecTokens> target_tokens{};
            int32_t target_tokens_count = 0;
            bool saw_target_eog = false;
            const auto accept_scan_t0 = std::chrono::steady_clock::now();
            for (int32_t i = 0; i < proposed; ++i) {
              llama_token target_tok = LLAMA_TOKEN_NULL;
              if (i == 0) {
                target_tok = target_tok0;
              } else {
                // After token i-1, logits row index is (i-1).
                target_tok = sample_greedy_with_eog_control(
                    verify_logits + static_cast<std::ptrdiff_t>(i - 1) * n_vocab, allow_eog_now());
              }

              if (target_tok == LLAMA_TOKEN_NULL) {
                (void) llama_memory_seq_rm(mem_t, /* seq_id = */ seq_id, /* p0 = */ -1, /* p1 = */ -1);
                (void) rollback_draft_to(ctx_draft, pos_draft, base_pos_draft);
                finished = true;
                return -1;
              }

              if (llama_vocab_is_eog(vocab_target, target_tok)) {
                // Target ends here; do not commit EOG itself.
                if (allow_eog_now()) {
                  saw_target_eog = true;
                }
                mismatch_idx = target_tokens_count;
                mismatch_tok = target_tok;
                break;
              }

              if (target_tokens_count >= static_cast<int32_t>(target_tokens.size())) {
                (void) llama_memory_seq_rm(mem_t, /* seq_id = */ seq_id, /* p0 = */ -1, /* p1 = */ -1);
                (void) rollback_draft_to(ctx_draft, pos_draft, base_pos_draft);
                finished = true;
                return -1;
              }
              target_tokens[static_cast<std::size_t>(target_tokens_count)] = target_tok;
              target_tokens_count += 1;
            }

            if (target_tokens_count > 0) {
              mismatch_idx = first_mismatch_accept_scan(
                  target_tokens.data(),
                  verify_tokens->data(),
                  target_tokens_count);
              if (mismatch_idx < 0) {
                accepted = target_tokens_count;
              } else {
                accepted = mismatch_idx;
                mismatch_tok = target_tokens[static_cast<std::size_t>(mismatch_idx)];
              }
            } else {
              accepted = 0;
            }
            if (saw_target_eog && mismatch_idx < 0) {
              engine_finished = true;
            }
            const auto accept_scan_t1 = std::chrono::steady_clock::now();
            accept_scan_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(accept_scan_t1 - accept_scan_t0).count());
          }

          accepted = std::clamp(accepted, 0, proposed);
          for (int32_t i = 0; i < accepted; ++i) {
            out_tokens.push_back((*verify_tokens)[static_cast<std::size_t>(i)]);
            out_speculative.push_back(1);
          }

          if (mismatch_idx >= 0 && mismatch_idx < proposed) {
            if (mismatch_tok == LLAMA_TOKEN_NULL) {
              const float * mismatch_row =
                  (mismatch_idx == 0)
                      ? verify_base_logits
                      : (verify_logits + static_cast<std::ptrdiff_t>(mismatch_idx - 1) * n_vocab);
              mismatch_tok = sample_greedy_with_eog_control(mismatch_row, allow_eog_now());
            }
            if (llama_vocab_is_eog(vocab_target, mismatch_tok)) {
              if (allow_eog_now()) {
                engine_finished = true;
              }
            } else {
              out_tokens.push_back(mismatch_tok);
              out_speculative.push_back(0);
              mismatch = true;
            }
          }
        }

        // Capture entropy for each verified token now, before any extra decode invalidates `verify_logits`.
        if (!out_tokens.empty()) {
          have_out_entropies = true;
          out_entropies[0] = verify_entropy;
          for (std::size_t j = 1; j < out_tokens.size(); ++j) {
            const std::ptrdiff_t row = static_cast<std::ptrdiff_t>(j - 1);
            const float * row_logits = verify_logits + row * n_vocab;
            out_entropies[j] = normalized_entropy_from_logits(row_logits, n_vocab, vocab_target, allow_eog_base);
          }
        }

        if (accepted > 0) {
          out_logits_rows.resize(static_cast<std::size_t>(accepted) * static_cast<std::size_t>(n_vocab));
          for (int32_t j = 0; j < accepted; ++j) {
            std::memcpy(out_logits_rows.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(n_vocab),
                        verify_logits + static_cast<std::ptrdiff_t>(j) * n_vocab,
                        static_cast<std::size_t>(n_vocab) * sizeof(float));
          }
        }

        // Roll back seq to the accepted prefix.
        const llama_pos rollback_start = verify_base_pos + static_cast<llama_pos>(accepted);
        const llama_pos rollback_end = verify_base_pos + static_cast<llama_pos>(proposed);
        if (rollback_start < rollback_end) {
          if (!llama_memory_seq_rm(mem_t, /* seq_id = */ seq_id, rollback_start, rollback_end)) {
            (void) llama_memory_seq_rm(mem_t, /* seq_id = */ seq_id, /* p0 = */ -1, /* p1 = */ -1);
            (void) rollback_draft_to(ctx_draft, pos_draft, base_pos_draft);
            finished = true;
            return -1;
          }
        }

        if (mismatch && !out_tokens.empty() && !engine_finished) {
          decode_mismatch_in_spec = true;
          mismatch_token = out_tokens.back();
          mismatch_pos = verify_base_pos + static_cast<llama_pos>(accepted);
        }
      }

      const auto t1 = std::chrono::steady_clock::now();
      const std::uint64_t elapsed_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
      const double elapsed_s = (elapsed_ns > 0) ? (static_cast<double>(elapsed_ns) * 1e-9) : 0.0;
      const double accept_ratio =
          (proposed > 0) ? (static_cast<double>(accepted) / static_cast<double>(proposed)) : 0.0;
      const double confidence = accept_ratio;
      const double spec_tps = (elapsed_s > 1e-9) ? (static_cast<double>(accepted) / elapsed_s) : 0.0;
      const double score = spec_tps * confidence;
      double delta = 0.0;
      if (baseline_tps_ema > 0.0 && std::isfinite(baseline_tps_ema)) {
        delta = (spec_tps - baseline_tps_ema) / baseline_tps_ema;
      }

      std::fprintf(stderr,
                   "[SPEC_METRIC] engine=%d committed=%d speculative=%d accept=%.3f spec_tps=%.2f delta=%.3f\n",
                   engine_idx,
                   accepted,
                   proposed,
                   accept_ratio,
                   spec_tps,
                   delta);
      (void) std::fflush(stderr);
      if (engine_ok && proposed > 0 && seq_id >= 0 &&
          static_cast<std::size_t>(seq_id) < spec_state.conf_ema.size()) {
        const std::size_t conf_idx = static_cast<std::size_t>(seq_id);
        const float prev = spec_state.conf_ema[conf_idx];
        const float next = (spec_state.conf_valid[conf_idx] != 0)
            ? static_cast<float>((1.0f - kConfEmaAlpha) * prev + kConfEmaAlpha * accept_ratio)
            : static_cast<float>(accept_ratio);
        spec_state.conf_ema[conf_idx] = next;
        spec_state.conf_valid[conf_idx] = 1;
        std::fprintf(stderr, "[SPEC_CONF] engine=%d conf_ema=%.3f\n", engine_idx, next);
        (void) std::fflush(stderr);
      }

      if (engine_ok && confidence >= 0.6 && score > best_score) {
        best_score = score;
        best.index = engine_idx;
        best.seq_id = seq_id;
        best.proposed = proposed;
        best.accepted = accepted;
        best.score = score;
        best.confidence = confidence;
        best.out_tokens = std::move(out_tokens);
        best.out_speculative = std::move(out_speculative);
        best.out_logits_rows = std::move(out_logits_rows);
        std::copy(std::begin(out_entropies), std::end(out_entropies), best.out_entropies.begin());
        best.have_out_entropies = have_out_entropies;
        best.decode_mismatch_in_spec = decode_mismatch_in_spec;
        best.mismatch_token = mismatch_token;
        best.mismatch_pos = mismatch_pos;
        best.finished = engine_finished;
        best.elapsed_ns = elapsed_ns;
        best.verify_ns = verify_forward_ns;
        best.accept_scan_ns = accept_scan_ns;
      }
    }

    if (best.index < 0) {
      for (int32_t engine_idx = 0; engine_idx < engine_count; ++engine_idx) {
        const llama_seq_id seq_id = static_cast<llama_seq_id>(engine_idx + 1);
        (void) llama_memory_seq_rm(mem_t, /* seq_id = */ seq_id, /* p0 = */ -1, /* p1 = */ -1);
      }
      if (!rollback_draft_to(ctx_draft, pos_draft, base_pos_draft)) {
        finished = true;
        return -1;
      }
      if (!step_one_target()) {
        break;
      }
      continue;
    }

    std::fprintf(stderr,
                 "[SPEC_SELECT] engine=%d score=%.3f committed=%d confidence=%.3f\n",
                 best.index,
                 best.score,
                 best.accepted,
                 best.confidence);
    (void) std::fflush(stderr);

    int32_t proposed = best.proposed;
    int32_t accepted = best.accepted;
    std::vector<llama_token> out_tokens = std::move(best.out_tokens);
    std::vector<std::uint8_t> out_speculative = std::move(best.out_speculative);
    std::vector<float> out_logits_rows = std::move(best.out_logits_rows);
    float out_entropies[33]{};
    std::copy(best.out_entropies.begin(), best.out_entropies.end(), out_entropies);
    bool have_out_entropies = best.have_out_entropies;
    bool decode_mismatch_in_spec = best.decode_mismatch_in_spec;
    llama_token mismatch_token = best.mismatch_token;
    llama_pos mismatch_pos = best.mismatch_pos;
    bool finished_now = best.finished;
    spec_elapsed_ns += best.elapsed_ns;
    spec_draft_ns += draft_elapsed_ns;
    spec_verify_ns += best.verify_ns;
    spec_accept_scan_ns += best.accept_scan_ns;

    const llama_seq_id selected_seq = best.seq_id;
    if (pending_count > 0 && selected_seq != spec_state.seq_id) {
      reset_spec_state(/* clear_seq = */ true);
      if (!rollback_draft_to(ctx_draft, pos_draft, base_pos)) {
        finished = true;
        return -1;
      }
      if (!step_one_target()) {
        break;
      }
      continue;
    }

    if (pending_count == 0) {
      spec_state.seq_id = selected_seq;
    }
    spec_state.valid = true;

    const std::size_t pending_before = pending_count;
    const llama_pos pending_pos_before = base_pos + static_cast<llama_pos>(pending_before);
    if (pending_before + out_tokens.size() > kMaxSpecTokens) {
      reset_spec_state(/* clear_seq = */ true);
      if (!rollback_draft_to(ctx_draft, pos_draft, base_pos)) {
        finished = true;
        return -1;
      }
      if (!step_one_target()) {
        break;
      }
      continue;
    }

    if (decode_mismatch_in_spec && mismatch_token != LLAMA_TOKEN_NULL && !finished_now) {
      if (!decode_one_token(ctx_target, batch_target, /* seq_id = */ selected_seq, mismatch_pos, mismatch_token, /* want_logits = */ true)) {
        (void) llama_memory_seq_rm(mem_t, /* seq_id = */ selected_seq, /* p0 = */ -1, /* p1 = */ -1);
        (void) rollback_draft_to(ctx_draft, pos_draft, base_pos_draft);
        finished = true;
        return -1;
      }
      const float * mismatch_logits = llama_get_logits(ctx_target);
      if (!mismatch_logits) {
        (void) llama_memory_seq_rm(mem_t, /* seq_id = */ selected_seq, /* p0 = */ -1, /* p1 = */ -1);
        (void) rollback_draft_to(ctx_draft, pos_draft, base_pos_draft);
        finished = true;
        return -1;
      }
      g_best_logits_buffer.resize(static_cast<std::size_t>(n_vocab));
      std::memcpy(g_best_logits_buffer.data(),
                  mismatch_logits,
                  static_cast<std::size_t>(n_vocab) * sizeof(float));
    }

    for (std::size_t i = 0; i < out_tokens.size(); ++i) {
      const std::size_t idx = spec_index(pending_before + i);
      TokenMeta & meta = spec_state.meta[idx];
      meta.token_id = out_tokens[i];
      meta.age = 0;
      meta.last_verified_step = spec_state.step_id;
      const std::uint8_t spec_flag =
          (i < out_speculative.size()) ? out_speculative[i] : 0;
      spec_state.speculative[idx] = spec_flag;
      if (have_out_entropies && i < sizeof(out_entropies) / sizeof(out_entropies[0])) {
        spec_state.entropy[idx] = out_entropies[i];
      } else {
        spec_state.entropy[idx] = verify_entropy;
      }

      if (i < static_cast<std::size_t>(accepted)) {
        std::memcpy(spec_logits_ptr(idx),
                    out_logits_rows.data() + i * static_cast<std::size_t>(n_vocab),
                    static_cast<std::size_t>(n_vocab) * sizeof(float));
      } else if (decode_mismatch_in_spec && i == static_cast<std::size_t>(accepted)) {
        std::memcpy(spec_logits_ptr(idx),
                    g_best_logits_buffer.data(),
                    static_cast<std::size_t>(n_vocab) * sizeof(float));
      }
    }

    spec_state.count += out_tokens.size();
    pending_count = spec_state.count;
    pending_pos = base_pos + static_cast<llama_pos>(pending_count);
    if (finished_now) {
      spec_state.finish_pending = true;
    }

    for (int32_t engine_idx = 0; engine_idx < engine_count; ++engine_idx) {
      const llama_seq_id seq_id = static_cast<llama_seq_id>(engine_idx + 1);
      if (seq_id == selected_seq) {
        continue;
      }
      (void) llama_memory_seq_rm(mem_t, /* seq_id = */ seq_id, /* p0 = */ -1, /* p1 = */ -1);
    }

    if (!out_tokens.empty()) {
      // Speculative verification stats:
      // - committed = accepted speculative tokens (draft == target)
      // - rejected  = proposed - accepted
      //
      // Note: `out_tokens` may include the first mismatching target token, which is
      // not counted as "committed" here.
      const int32_t rejected = std::max<int32_t>(0, proposed - accepted);
      std::fprintf(stderr, "[HOLO_COLLAPSE] committed=%d rejected=%d\n", accepted, rejected);
    }
    if (proposed > 0) {
      const double step_accept =
          std::clamp(static_cast<double>(accepted) / static_cast<double>(proposed), 0.0, 1.0);
      constexpr double kAcceptGateAlpha = 0.2;
      if (!std::isfinite(spec_state.accept_gate_ema)) {
        spec_state.accept_gate_ema = step_accept;
      } else {
        spec_state.accept_gate_ema += kAcceptGateAlpha * (step_accept - spec_state.accept_gate_ema);
      }
      // Safety gating should track realized speculative acceptance, not a
      // placeholder value from the pre-step gate.
      spec_state.last_accept = step_accept;
      spec_state.have_last_accept = true;

      const double draft_forward_ms = static_cast<double>(draft_elapsed_ns) * 1e-6;
      const double target_verify_ms = static_cast<double>(best.verify_ns) * 1e-6;
      const double accept_scan_ms = static_cast<double>(best.accept_scan_ns) * 1e-6;
      const double total_spec_overhead_ms = draft_forward_ms + target_verify_ms + accept_scan_ms;
      double baseline_decode_ms = std::numeric_limits<double>::quiet_NaN();
      if (std::isfinite(baseline_tps_ema) && baseline_tps_ema > 1e-6) {
        baseline_decode_ms = 1000.0 / baseline_tps_ema;
      } else if (std::isfinite(spec_state.baseline_decode_ms_ema) &&
                 spec_state.baseline_decode_ms_ema > 0.0) {
        baseline_decode_ms = spec_state.baseline_decode_ms_ema;
      }
      const double spec_effective_now_ms =
          (std::isfinite(baseline_decode_ms))
              ? (static_cast<double>(accepted) * baseline_decode_ms) - total_spec_overhead_ms
              : std::numeric_limits<double>::quiet_NaN();
      spec_effective_ms = spec_effective_now_ms;

      double rolling_effective_ms = std::numeric_limits<double>::quiet_NaN();
      if (std::isfinite(spec_effective_now_ms)) {
        const std::size_t window = static_cast<std::size_t>(spec_profit_window());
        if (spec_state.profit_head >= window) {
          spec_state.profit_head = 0;
        }
        const std::size_t idx = spec_state.profit_head;
        if (spec_state.profit_count >= window) {
          spec_state.profit_sum -= spec_state.profit_window[idx];
        } else {
          spec_state.profit_count += 1;
        }
        spec_state.profit_window[idx] = spec_effective_now_ms;
        spec_state.profit_sum += spec_effective_now_ms;
        spec_state.profit_head = (spec_state.profit_head + 1) % window;
        const std::size_t denom = std::min<std::size_t>(window, spec_state.profit_count);
        if (denom > 0) {
          rolling_effective_ms = spec_state.profit_sum / static_cast<double>(denom);
        }
      }

      if (spec_adaptive_enabled() && !spec_state.adaptive_request_disabled &&
          best.elapsed_ns > 0) {
        constexpr std::uint64_t kAdaptiveWindowSteps = 10;
        constexpr int32_t kAdaptiveBadChecksLimit = 3;
        constexpr double kAdaptiveKeepMargin = 1.05;
        const std::uint64_t produced_tokens =
            static_cast<std::uint64_t>(std::max<std::size_t>(1, out_tokens.size()));
        spec_state.adaptive_window_tokens += produced_tokens;
        spec_state.adaptive_window_ns += best.elapsed_ns;
        if (std::isfinite(spec_effective_now_ms)) {
          spec_state.adaptive_window_effective_ms += spec_effective_now_ms;
        }

        if (spec_state.adaptive_window_tokens >= kAdaptiveWindowSteps &&
            spec_state.adaptive_window_ns > 0) {
          double baseline_decode_ms = std::numeric_limits<double>::quiet_NaN();
          if (std::isfinite(spec_state.baseline_decode_ms_ema) &&
              spec_state.baseline_decode_ms_ema > 0.0) {
            baseline_decode_ms = spec_state.baseline_decode_ms_ema;
          } else if (std::isfinite(baseline_tps_ema) && baseline_tps_ema > 1e-6) {
            baseline_decode_ms = 1000.0 / baseline_tps_ema;
          }

          const double baseline_tps =
              (std::isfinite(baseline_decode_ms) && baseline_decode_ms > 1e-9)
                  ? (1000.0 / baseline_decode_ms)
                  : std::numeric_limits<double>::quiet_NaN();
          const double baseline_window_ms =
              (std::isfinite(baseline_decode_ms) && baseline_decode_ms > 0.0)
                  ? (baseline_decode_ms * static_cast<double>(spec_state.adaptive_window_tokens))
                  : std::numeric_limits<double>::quiet_NaN();
          const double spec_window_ms =
              (std::isfinite(baseline_window_ms))
                  ? (baseline_window_ms - spec_state.adaptive_window_effective_ms)
                  : std::numeric_limits<double>::quiet_NaN();
          const double spec_tps =
              (std::isfinite(spec_window_ms) && spec_window_ms > 1e-9)
                  ? (static_cast<double>(spec_state.adaptive_window_tokens) / (spec_window_ms * 1e-3))
                  : std::numeric_limits<double>::quiet_NaN();

          if (std::isfinite(spec_tps) && std::isfinite(baseline_tps) && spec_tps < baseline_tps) {
            spec_state.adaptive_bad_checks += 1;
            if (spec_state.adaptive_bad_checks >= kAdaptiveBadChecksLimit) {
              spec_state.adaptive_request_disabled = true;
              spec_state.profitability_cooldown_steps = 0;
              spec_state.profitability_probe_steps = 0;
              spec_state.deferred_draft_tokens.clear();
              std::fprintf(stderr,
                           "[SPEC_ADAPTIVE] disabled for request (spec=%.2f TPS < baseline=%.2f TPS)\n",
                           spec_tps,
                           baseline_tps);
              (void) std::fflush(stderr);
            }
          } else if (std::isfinite(spec_tps) && std::isfinite(baseline_tps) &&
                     spec_tps > (baseline_tps * kAdaptiveKeepMargin)) {
            spec_state.adaptive_bad_checks = 0;
          } else if (std::isfinite(spec_tps) && std::isfinite(baseline_tps) &&
                     spec_tps >= baseline_tps) {
            spec_state.adaptive_bad_checks = 0;
          }
          spec_state.adaptive_window_tokens = 0;
          spec_state.adaptive_window_ns = 0;
          spec_state.adaptive_window_effective_ms = 0.0;
        }
      }

      if (!spec_adaptive_enabled()) {
        if (std::isfinite(spec_effective_now_ms) && spec_effective_now_ms < 0.0) {
          spec_state.negative_effective_streak += 1;
        } else {
          spec_state.negative_effective_streak = 0;
        }

        if (spec_state.negative_effective_streak >= spec_profit_bad_streak_limit()) {
          spec_state.profitability_cooldown_steps = spec_profit_cooldown_steps();
          spec_state.profitability_probe_steps = 0;
          spec_state.negative_effective_streak = 0;
          std::fprintf(stderr,
                       "[SPEC_PROFIT] gate=disable cooldown_steps=%d reason=negative_effective\n",
                       spec_state.profitability_cooldown_steps);
          (void) std::fflush(stderr);
        }

        if (spec_state.profitability_probe_steps > 0) {
          spec_state.profitability_probe_steps -= 1;
          if (std::isfinite(spec_effective_now_ms) && spec_effective_now_ms < 0.0) {
            spec_state.profitability_probe_steps = 0;
            spec_state.negative_effective_streak = 0;
            if (spec_state.profitability_retry_count >= 1) {
              spec_state.profitability_hard_disabled = true;
              spec_state.profitability_cooldown_steps = 0;
              spec_state.deferred_draft_tokens.clear();
              std::fprintf(stderr,
                           "[SPEC_PROFIT] gate=disable reason=probe_negative_hard\n");
            } else {
              spec_state.profitability_cooldown_steps = spec_profit_cooldown_steps();
              std::fprintf(stderr,
                           "[SPEC_PROFIT] gate=disable cooldown_steps=%d reason=probe_negative\n",
                           spec_state.profitability_cooldown_steps);
            }
            (void) std::fflush(stderr);
          } else {
            std::fprintf(stderr,
                         "[SPEC_PROFIT] gate=probe_success effective_ms=%.3f\n",
                         spec_effective_now_ms);
            (void) std::fflush(stderr);
          }
        }
      } else {
        spec_state.negative_effective_streak = 0;
      }

      std::fprintf(stderr,
                   "[SPEC_TIMING] draft_forward_ms=%.3f target_verify_ms=%.3f accept_scan_ms=%.3f "
                   "total_spec_overhead_ms=%.3f baseline_decode_ms=%.3f accepted_tokens=%d "
                   "spec_effective_ms=%.3f rolling_effective_ms=%.3f neg_streak=%d\n",
                   draft_forward_ms,
                   target_verify_ms,
                   accept_scan_ms,
                   total_spec_overhead_ms,
                   baseline_decode_ms,
                   accepted,
                   spec_effective_now_ms,
                   rolling_effective_ms,
                   spec_state.negative_effective_streak);
      (void) std::fflush(stderr);
    }

    // Sync draft state to the speculative sequence (pending prefix + new verified tokens).
    if (ctx_draft) {
      llama_memory_t mem_d = llama_get_memory(ctx_draft);
      if (!mem_d) {
        finished = true;
        return -1;
      }

      const llama_pos draft_keep_end = pending_pos_before + static_cast<llama_pos>(draft_tokens.size());
      if (pos_draft != draft_keep_end) {
        finished = true;
        return -1;
      }

      bool draft_prefix_matches = true;
      if (out_tokens.size() > draft_tokens.size()) {
        draft_prefix_matches = false;
      } else {
        for (std::size_t i = 0; i < out_tokens.size(); ++i) {
          if (out_tokens[i] != draft_tokens[i]) {
            draft_prefix_matches = false;
            break;
          }
        }
      }

      if (!draft_prefix_matches) {
        if (!llama_memory_seq_rm(mem_d, /* seq_id = */ 0, pending_pos_before, /* p1 = */ -1)) {
          finished = true;
          return -1;
        }
        pos_draft = pending_pos_before;
        for (llama_token tok : out_tokens) {
          if (!decode_one_token(ctx_draft, batch_draft, /* seq_id = */ 0, pos_draft, tok, /* want_logits = */ true)) {
            finished = true;
            return -1;
          }
          pos_draft += 1;
          logits_draft = llama_get_logits(ctx_draft);
          if (!logits_draft) {
            finished = true;
            return -1;
          }
        }
      } else {
        const llama_pos draft_commit_end = pending_pos_before + static_cast<llama_pos>(out_tokens.size());
        if (draft_commit_end < draft_keep_end) {
          if (!llama_memory_seq_rm(mem_d, /* seq_id = */ 0, draft_commit_end, draft_keep_end)) {
            finished = true;
            return -1;
          }
          pos_draft = draft_commit_end;
        }
      }
    }

    adapt_depth(/* proposed = */ proposed, /* accepted = */ accepted);
  }

  return printed_this_call;
}

// Backwards-compatible entrypoint (current engine ABI).
//
// The engine does not yet plumb benchmark controls through its call graph; to enable
// throughput-mode runs without modifying engine.cpp, this wrapper reads:
// - KORITH_BENCHMARK_MODE=1 to enable benchmark behavior
// - KORITH_MIN_TOKENS_TO_GENERATE=<u64> (default: 4096; also used as baseline EOG guard)
int32_t batch_executor_step(
    llama_context * ctx_target,
    llama_batch & batch_target,
    const llama_vocab * vocab_target,
    int32_t n_vocab,
    llama_context * ctx_draft,
    llama_batch & batch_draft,
    llama_pos & pos_target,
    llama_pos & pos_draft,
    const float *& logits_target,
    const float *& logits_draft,
    bool & finished,
    int32_t batch_tokens,
    bool force_normal_decode,
    bool reuse_kv,
    bool force_speculation,
    int32_t spec_window_limit,
    int32_t & spec_depth,
    const korith::core::SpecPlan & plan,
    double & accept_ema,
    std::uint64_t & spec_proposed,
    std::uint64_t & spec_accepted,
    std::uint64_t & printed_total,
    std::uint64_t & token_prefix_hash,
    std::uint64_t & kv_hit_count,
    std::uint64_t & kv_miss_count,
    std::uint64_t & kv_evict_count,
    double baseline_tps_ema,
    double tps_delta,
    std::uint64_t & spec_elapsed_ns,
    std::uint64_t & spec_draft_ns,
    std::uint64_t & spec_verify_ns,
    std::uint64_t & spec_accept_scan_ns,
    double & spec_effective_ms,
    std::uint64_t & spec_skipped_tokens,
    std::uint64_t & spec_fast_verified_tokens,
    int32_t engine_count_cap,
    std::uint64_t max_tokens) {
  if (max_tokens > 0) {
    if (printed_total >= max_tokens) {
      return 0;
    }
    const std::uint64_t remaining_u64 = max_tokens - printed_total;
    const std::uint64_t i32_cap_u64 =
        static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max());
    const int32_t remaining_tokens = static_cast<int32_t>(
        std::min<std::uint64_t>(remaining_u64, i32_cap_u64));
    batch_tokens = std::min<int32_t>(batch_tokens, remaining_tokens);
    if (batch_tokens <= 0) {
      return 0;
    }
  }

  const char * env_bench = std::getenv("KORITH_BENCHMARK_MODE");
  const bool benchmark_mode = (env_bench != nullptr) && (env_bench[0] != '\0') && (env_bench[0] != '0');

  std::uint64_t min_tokens_to_generate = 0;
  if (const char * env_min = std::getenv("KORITH_MIN_TOKENS_TO_GENERATE")) {
    if (env_min[0] != '\0') {
      char * end = nullptr;
      const unsigned long long parsed = std::strtoull(env_min, &end, 10);
      if (end && end != env_min && *end == '\0') {
        min_tokens_to_generate = static_cast<std::uint64_t>(parsed);
      }
    }
  }

  std::uint64_t min_tokens_before_eog = 0;
  if (benchmark_mode) {
    if (min_tokens_to_generate == 0) {
      min_tokens_to_generate = 4096;
    }
    min_tokens_before_eog = min_tokens_to_generate;
  } else if (max_tokens > 1) {
    if (min_tokens_to_generate == 0) {
      min_tokens_to_generate = std::min<std::uint64_t>(max_tokens, 32u);
    } else {
      min_tokens_to_generate = std::min<std::uint64_t>(max_tokens, min_tokens_to_generate);
    }
    min_tokens_before_eog = min_tokens_to_generate;
  }

  return batch_executor_step(
      ctx_target,
      batch_target,
      vocab_target,
      n_vocab,
      ctx_draft,
      batch_draft,
      pos_target,
      pos_draft,
      logits_target,
      logits_draft,
      finished,
      batch_tokens,
      force_normal_decode,
      reuse_kv,
      force_speculation,
      spec_window_limit,
      spec_depth,
      plan,
      accept_ema,
      spec_proposed,
      spec_accepted,
      printed_total,
      token_prefix_hash,
      kv_hit_count,
      kv_miss_count,
      kv_evict_count,
      baseline_tps_ema,
      tps_delta,
      spec_elapsed_ns,
      spec_draft_ns,
      spec_verify_ns,
      spec_accept_scan_ns,
      spec_effective_ms,
      spec_skipped_tokens,
      spec_fast_verified_tokens,
      engine_count_cap,
      benchmark_mode,
      min_tokens_before_eog);
}

}  // namespace korith::core
