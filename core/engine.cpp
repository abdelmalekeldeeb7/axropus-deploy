// Assumptions / scope (v0):
// - Built and linked against llama.cpp's public C API (`llama.h`) and `libllama`.
// - `libllama` is built with CUDA support; if GPU offload is unavailable at runtime, this falls back to CPU.
// - Single-sequence, greedy decoding only. No prompt ingestion yet: generation starts from the model's BOS token.
// - Sampling is greedy argmax over the cached logits buffer to avoid llama.cpp's sampler helpers (which call
//   `llama_get_logits*_()` and can force a GPU synchronize per sampled token).
// - Optional speculative decoding is enabled when a draft model is provided either via:
//     - `engine_init("target.gguf|draft.gguf")`, or
//     - the `KORITH_DRAFT_MODEL` environment variable set to a draft model path.
// - Target-side verification runs in batches (one `llama_decode()` call over the draft token block) using a temporary
//   sequence id inside the target context. Accepted tokens are committed via KV copy (no re-decoding).
// - Tokens are printed to stdout as detokenized pieces; the return value is the number of decoded tokens
//   advanced this step (including tokens that render to an empty string).

#include "amf_direct_kv.h"
#include "amf_store.h"
#include "bindings.h"
#include "cuda_graph_decode.h"
#include "collapse_controller.h"
#include "engine_pool.h"
#include "memory_field.h"
#include "spec_plan.h"
#include "spec_v2.h"
#include "thermo_controller.h"

#include <llama.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <dlfcn.h>
#include <deque>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "../holographic/memory.h"

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
    std::uint64_t max_tokens);

}  // namespace korith::core

namespace {

// ------------------------ Rust scheduler C ABI -------------------------
//
// This is intentionally defined locally to avoid introducing new headers.
// The layout must match `controller/ffi.rs` (`#[repr(C)]`).

// Engine invariants:
// - AMF replay only allowed under deterministic sampling.
// - ROI must only be computed when baseline is stable.
// - Sustained negative ROI disables replay.
// - Replay remains disabled until MF cooldown expires.
// - AMF fails closed on corruption or uncertainty.
// - MF outputs are bounded/monotonic and must not oscillate replay enable/disable.
// - MF never overrides AMF hard disables.
// - Baseline EMAs must converge before ROI use; ROI suppressed if baseline unstable.

struct KorithSchedulerState {
  std::int32_t speculative_depth;
  std::int32_t batch_size;
  std::int32_t baseline_ready;
};

struct KorithScheduleInput {
  float acceptance;
  float entropy;
  float current_tps;
  float engine_costs;
  float amf_reuse_score;
  float amf_avg_prefix_length;
  float amf_accept_rate;
  float amf_restore_cost_ms;
  std::int32_t baseline_ready;
};

struct KorithScheduleOutput {
  std::int32_t desired_depth;
  float stability_score;
  std::int32_t throttle_flag;
};

struct RustSchedulerApi {
  using StepFn = KorithScheduleOutput (*)(KorithSchedulerState * state, const KorithScheduleInput * input);

  void * handle = nullptr;
  StepFn step = nullptr;
  bool active = false;
};

const RustSchedulerApi & rust_scheduler_api() {
  static const RustSchedulerApi api = []() -> RustSchedulerApi {
    RustSchedulerApi out{};

    auto scheduler_from_exe = []() -> std::string {
      char buf[PATH_MAX];
      const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
      if (n <= 0) {
        return {};
      }
      buf[n] = '\0';
      const char * slash = std::strrchr(buf, '/');
      if (!slash) {
        return {};
      }
      std::string dir(buf, static_cast<std::size_t>(slash - buf));
      dir += "/lib/libkorith_scheduler.so";
      return dir;
    };

    auto try_dlopen = [](const char * path) -> void * {
      if (!path || path[0] == '\0') {
        return nullptr;
      }
      (void) dlerror();
      void * handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
      if (!handle) {
        const char * err = dlerror();
        std::fprintf(stderr,
                     "[SCHEDULER] warning: dlopen(%s) failed: %s\n",
                     path,
                     err ? err : "unknown error");
      }
      return handle;
    };

    void * h = nullptr;
    const std::string exe_path = scheduler_from_exe();
    if (!exe_path.empty()) {
      h = try_dlopen(exe_path.c_str());
    }

    if (!h) {
      const char * env_path = std::getenv("KORITH_SCHEDULER_LIB");
      if (env_path && env_path[0] != '\0') {
        h = try_dlopen(env_path);
      }
    }

    if (!h) {
      std::fprintf(stderr,
                   "[SCHEDULER] warning: scheduler library not loaded; using static fallback policy "
                   "(depth=4, throttle=2). Set KORITH_SCHEDULER_LIB to suppress this warning.\n");
      (void) std::fflush(stderr);
      return out;  // out.active == false; callers must check and use fallback.
    }

    (void) dlerror();
    auto step = reinterpret_cast<RustSchedulerApi::StepFn>(dlsym(h, "korith_schedule_step"));
    const char * err_step = dlerror();

    if (!step || err_step) {
      std::fprintf(stderr,
                   "[SCHEDULER] warning: missing symbol 'korith_schedule_step' in scheduler library; "
                   "using static fallback policy.\n");
      (void) dlclose(h);
      (void) std::fflush(stderr);
      return out;  // out.active == false; callers must check and use fallback.
    }

    std::fprintf(stderr, "[SCHEDULER] loaded libkorith_scheduler.so\n");
    (void) std::fflush(stderr);
    out.handle = h;
    out.step = step;
    out.active = true;
    return out;
  }();
  return api;
}

std::mutex g_engines_mu;
std::vector<korith::core::KorithEngine *> g_registered_engines;

void collect_engine_signals(std::vector<korith::core::EngineSignal> & out, korith::core::Context & ctx) {
  std::lock_guard<std::mutex> lock(g_engines_mu);
  out.clear();
  out.reserve(g_registered_engines.size());
  for (korith::core::KorithEngine * engine : g_registered_engines) {
    if (!engine) {
      continue;
    }
    out.push_back(engine->evaluate(ctx));
  }
}

std::uint64_t now_epoch_ms() {
  const auto now = std::chrono::system_clock::now();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
}

std::string g_engine_events_path;

std::string json_escape(const std::string & s) {
  std::ostringstream out;
  for (char c : s) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c; break;
    }
  }
  return out.str();
}

bool env_truthy(const char * name, bool default_value = false) {
  const char * raw = std::getenv(name);
  if (!raw || raw[0] == '\0') {
    return default_value;
  }
  const std::string v(raw);
  return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on";
}

std::string collapse_whitespace_for_amf(const std::string & input, bool * changed) {
  std::string out;
  out.reserve(input.size());
  bool in_space = false;
  for (unsigned char ch : input) {
    if (std::isspace(ch) != 0) {
      in_space = true;
      continue;
    }
    if (in_space && !out.empty()) {
      out.push_back(' ');
    }
    in_space = false;
    out.push_back(static_cast<char>(ch));
  }
  if (changed) {
    *changed = (out != input);
  }
  return out;
}

std::string replace_all(std::string input,
                        const std::string & needle,
                        const std::string & value,
                        bool * changed) {
  if (needle.empty()) {
    return input;
  }
  std::size_t pos = 0;
  while ((pos = input.find(needle, pos)) != std::string::npos) {
    input.replace(pos, needle.size(), value);
    pos += value.size();
    if (changed) {
      *changed = true;
    }
  }
  return input;
}

std::string normalize_unicode_nfc_for_amf(const std::string & input, bool * changed) {
  std::string out = input;
  if (changed) {
    *changed = false;
  }
  // Best-effort UTF-8 NFC collapsing for common accent combinations.
  out = replace_all(out, "A\xCC\x81", "\xC3\x81", changed);  // A + acute
  out = replace_all(out, "a\xCC\x81", "\xC3\xA1", changed);  // a + acute
  out = replace_all(out, "E\xCC\x81", "\xC3\x89", changed);  // E + acute
  out = replace_all(out, "e\xCC\x81", "\xC3\xA9", changed);  // e + acute
  out = replace_all(out, "I\xCC\x81", "\xC3\x8D", changed);  // I + acute
  out = replace_all(out, "i\xCC\x81", "\xC3\xAD", changed);  // i + acute
  out = replace_all(out, "O\xCC\x81", "\xC3\x93", changed);  // O + acute
  out = replace_all(out, "o\xCC\x81", "\xC3\xB3", changed);  // o + acute
  out = replace_all(out, "U\xCC\x81", "\xC3\x9A", changed);  // U + acute
  out = replace_all(out, "u\xCC\x81", "\xC3\xBA", changed);  // u + acute
  out = replace_all(out, "N\xCC\x83", "\xC3\x91", changed);  // N + tilde
  out = replace_all(out, "n\xCC\x83", "\xC3\xB1", changed);  // n + tilde
  out = replace_all(out, "C\xCC\xA7", "\xC3\x87", changed);  // C + cedilla
  out = replace_all(out, "c\xCC\xA7", "\xC3\xA7", changed);  // c + cedilla
  return out;
}

struct PromptCanonicalizationResult {
  std::string text;
  bool whitespace_changed = false;
  bool unicode_changed = false;
};

PromptCanonicalizationResult canonicalize_prompt_for_amf(const std::string & input) {
  PromptCanonicalizationResult out{};
  out.text = input;

  if (env_truthy("KORITH_AMF_NORMALIZE_WHITESPACE", true)) {
    out.text = collapse_whitespace_for_amf(out.text, &out.whitespace_changed);
  }
  if (env_truthy("KORITH_AMF_NORMALIZE_UNICODE", true)) {
    out.text = normalize_unicode_nfc_for_amf(out.text, &out.unicode_changed);
  }
  return out;
}

std::string sanitize_tenant_id_for_amf(std::string tenant_id) {
  if (tenant_id.empty()) {
    return "default";
  }
  for (char & ch : tenant_id) {
    const bool ok =
        (ch >= 'a' && ch <= 'z') ||
        (ch >= 'A' && ch <= 'Z') ||
        (ch >= '0' && ch <= '9') ||
        ch == '-' || ch == '_' || ch == '.';
    if (!ok) {
      ch = '_';
    }
  }
  if (tenant_id.size() > 128) {
    tenant_id.resize(128);
  }
  return tenant_id;
}

std::vector<std::string> split_csv_patterns(const std::string & csv) {
  std::vector<std::string> out;
  std::string current;
  for (char ch : csv) {
    if (ch == ',') {
      if (!current.empty()) {
        out.push_back(current);
      }
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    out.push_back(current);
  }
  return out;
}

bool prompt_matches_shared_patterns(const std::string & prompt_text) {
  const char * raw = std::getenv("KORITH_AMF_SHARED_PREFIX_PATTERNS");
  if (!raw || raw[0] == '\0') {
    return false;
  }
  const std::vector<std::string> patterns = split_csv_patterns(raw);
  for (const std::string & pattern : patterns) {
    if (pattern.empty()) {
      continue;
    }
    if (prompt_text.rfind(pattern, 0) == 0) {
      return true;
    }
  }
  return false;
}

std::size_t find_last_subsequence(const std::vector<llama_token> & haystack,
                                  const std::vector<llama_token> & needle) {
  if (needle.empty() || haystack.size() < needle.size()) {
    return static_cast<std::size_t>(-1);
  }
  std::size_t i = haystack.size() - needle.size();
  while (true) {
    if (std::equal(needle.begin(), needle.end(), haystack.begin() + static_cast<std::ptrdiff_t>(i))) {
      return i;
    }
    if (i == 0) {
      break;
    }
    --i;
  }
  return static_cast<std::size_t>(-1);
}

void emit_engine_event(const char * type, const std::string & payload_json) {
  if (!type || !type[0]) {
    return;
  }
  if (g_engine_events_path.empty()) {
    return;
  }
  std::ofstream out(g_engine_events_path, std::ios::app);
  if (!out) {
    return;
  }
  out << "{\"type\":\"" << json_escape(type) << "\","
      << "\"ts\":" << static_cast<unsigned long long>(now_epoch_ms()) << ","
      << "\"payload\":" << payload_json << "}\n";
}

bool load_text_file(const std::string & path, std::string & out) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}

bool write_text_file(const std::string & path, const std::string & data) {
  std::ofstream out(path);
  if (!out) {
    return false;
  }
  out << data;
  return true;
}

static inline int32_t max_spec_depth(llama_context * ctx_target) {
  if (!ctx_target) {
    return 1;
  }
  const int32_t n_seq = std::max<int32_t>(1, llama_n_seq_max(ctx_target));
  if (n_seq < 2) {
    return 1;
  }
  return 32;
}

enum class SpecMode : std::uint8_t {
  kOff = 0,
  kV2 = 1,
  kV3 = 2,
};

static inline bool env_flag_on(const char * name) {
  const char * env = std::getenv(name);
  return env && env[0] != '\0' && env[0] != '0';
}

static inline SpecMode resolve_spec_mode_from_env() {
  // V3 has strict precedence over V2.
  if (env_flag_on("KORITH_SPEC_V3")) {
    return SpecMode::kV3;
  }
  if (env_flag_on("KORITH_SPEC_V2")) {
    return SpecMode::kV2;
  }
  return SpecMode::kOff;
}

static inline const char * spec_mode_name(SpecMode mode) {
  switch (mode) {
    case SpecMode::kV3:
      return "v3";
    case SpecMode::kV2:
      return "v2";
    case SpecMode::kOff:
    default:
      return "off";
  }
}

static inline const char * ggml_type_name_short(enum ggml_type t) {
  switch (t) {
    case GGML_TYPE_F16:  return "f16";
    case GGML_TYPE_BF16: return "bf16";
    case GGML_TYPE_Q8_0: return "q8_0";
    case GGML_TYPE_Q8_K: return "q8_k";
    case GGML_TYPE_Q6_K: return "q6_k";
    case GGML_TYPE_Q5_K: return "q5_k";
    case GGML_TYPE_Q4_K: return "q4_k";
    default:             return "default";
  }
}

static inline bool parse_ggml_type_env_token(const std::string & token, enum ggml_type & out) {
  if (token == "f16") {
    out = GGML_TYPE_F16;
    return true;
  }
  if (token == "bf16") {
    out = GGML_TYPE_BF16;
    return true;
  }
  if (token == "q8_0" || token == "q8") {
    out = GGML_TYPE_Q8_0;
    return true;
  }
  if (token == "q8_k") {
    out = GGML_TYPE_Q8_K;
    return true;
  }
  if (token == "q6_k") {
    out = GGML_TYPE_Q6_K;
    return true;
  }
  if (token == "q5_k") {
    out = GGML_TYPE_Q5_K;
    return true;
  }
  if (token == "q4_k") {
    out = GGML_TYPE_Q4_K;
    return true;
  }
  return false;
}

static inline bool apply_kv_compression_env(
    llama_context_params & cparams,
    std::string & mode_label) {
  bool changed = false;
  mode_label = "off";

  if (env_flag_on("KORITH_KV_FP8")) {
    // llama.cpp does not expose a native FP8 KV type yet; Q8_0 is the closest
    // 8-bit KV option and still cuts KV bandwidth materially versus f16.
    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_Q8_0;
    mode_label = "fp8_compat_q8_0";
    changed = true;
  }

  auto apply_named = [&](const char * env_name, enum ggml_type & dst) {
    if (const char * env = std::getenv(env_name)) {
      if (env[0] != '\0') {
        std::string v = env;
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
          return static_cast<char>(std::tolower(c));
        });
        enum ggml_type parsed{};
        if (parse_ggml_type_env_token(v, parsed)) {
          dst = parsed;
          changed = true;
          mode_label = "custom";
        }
      }
    }
  };

  apply_named("KORITH_KV_TYPE_K", cparams.type_k);
  apply_named("KORITH_KV_TYPE_V", cparams.type_v);
  return changed;
}

static inline bool spec_v3_enabled() {
  static bool initialized = false;
  static bool enabled = false;
  if (!initialized) {
    initialized = true;
    enabled = (resolve_spec_mode_from_env() == SpecMode::kV3);
  }
  return enabled;
}

static inline float logit_variance_entropy(const float * logits, int32_t n_vocab) {
  if (!logits || n_vocab <= 0) {
    return 1.0f;
  }

  constexpr int kTopK = 16;
  std::array<float, kTopK> top{};
  int count = 0;
  float min_val = std::numeric_limits<float>::infinity();
  int min_idx = -1;

  for (int32_t i = 0; i < n_vocab; ++i) {
    const float v = logits[i];
    if (!std::isfinite(v)) {
      continue;
    }
    if (count < kTopK) {
      top[count] = v;
      if (v < min_val) {
        min_val = v;
        min_idx = count;
      }
      count += 1;
      continue;
    }
    if (v <= min_val) {
      continue;
    }
    top[min_idx] = v;
    min_val = top[0];
    min_idx = 0;
    for (int j = 1; j < kTopK; ++j) {
      if (top[j] < min_val) {
        min_val = top[j];
        min_idx = j;
      }
    }
  }

  if (count <= 1) {
    return 0.0f;
  }

  double sum = 0.0;
  double sumsq = 0.0;
  for (int i = 0; i < count; ++i) {
    const double v = static_cast<double>(top[i]);
    sum += v;
    sumsq += v * v;
  }
  const double mean = sum / static_cast<double>(count);
  const double var = std::max(0.0, (sumsq / static_cast<double>(count)) - (mean * mean));
  const double stddev = std::sqrt(var);
  const double norm = stddev / (stddev + 1.0);
  return std::clamp(static_cast<float>(norm), 0.0f, 1.0f);
}

static inline std::uint64_t hash_token_prefix_step(std::uint64_t h, llama_token tok) noexcept {
  // FNV-1a over token ids (deterministic, fast).
  constexpr std::uint64_t kPrime = 1099511628211ull;
  const std::uint32_t v = static_cast<std::uint32_t>(tok);
  h ^= static_cast<std::uint64_t>(v);
  h *= kPrime;
  return h;
}

static inline std::uint64_t hash_u64_step(std::uint64_t h, std::uint64_t v) noexcept {
  constexpr std::uint64_t kPrime = 1099511628211ull;
  h ^= v;
  h *= kPrime;
  return h;
}

static std::uint64_t amf_sampling_hash() {
  // Stable hash of the effective sampling configuration (greedy + benchmark gates).
  constexpr std::uint64_t kOffset = 1469598103934665603ull;
  std::uint64_t h = kOffset;
  h = hash_u64_step(h, 1u);  // sampling_scheme = greedy

  const char * bench_env = std::getenv("KORITH_BENCHMARK_MODE");
  const bool bench_mode = bench_env && bench_env[0] != '\0' && bench_env[0] != '0';
  h = hash_u64_step(h, bench_mode ? 1u : 0u);

  std::uint64_t bench_min = 0;
  if (const char * env = std::getenv("KORITH_BENCHMARK_MIN_TOKENS")) {
    if (env[0] != '\0') {
      char * end = nullptr;
      const unsigned long long parsed = std::strtoull(env, &end, 10);
      if (end && end != env && *end == '\0') {
        bench_min = static_cast<std::uint64_t>(parsed);
      }
    }
  }
  h = hash_u64_step(h, bench_min);
  return h;
}

static bool amf_sampling_is_deterministic() {
  // Engine sampling is greedy; if this changes, enforce stricter gates here.
  return true;
}

static std::uint64_t amf_rng_hash() {
  // Deterministic, stateless sampling (no RNG state to restore).
  return 0x4b4f524954485f30ull;  // "KORITH_0"
}

static bool amf_restore_rng_state() {
  // Stateless sampling: only allow replay under deterministic sampling.
  return amf_sampling_is_deterministic();
}

constexpr int kAmfNegativeRoiMax = 3;

static double update_ema(double current, double sample, double alpha) {
  if (!(alpha > 0.0 && alpha <= 1.0) || !std::isfinite(sample)) {
    return current;
  }
  if (!(current > 0.0) || !std::isfinite(current)) {
    return sample;
  }
  return (current * (1.0 - alpha)) + (sample * alpha);
}

struct EngineState {
  bool backend_inited = false;

  // Target model/context (accurate).
  llama_model * model_target = nullptr;
  llama_context * ctx_target = nullptr;
  const llama_vocab * vocab_target = nullptr;
  int32_t n_vocab = 0;
  const float * logits_target = nullptr;  // last logits row (ready for sampling)

  // Draft model/context (fast proposer). Optional.
  llama_model * model_draft = nullptr;
  llama_context * ctx_draft = nullptr;
  const llama_vocab * vocab_draft = nullptr;
  const float * logits_draft = nullptr;  // last logits row (ready for sampling)

  llama_batch batch_target{};
  bool batch_target_inited = false;

  llama_batch batch_draft{};
  bool batch_draft_inited = false;

  llama_pos pos_target = 0;  // next position to be written (seq 0)
  llama_pos pos_draft = 0;
  bool ready = false;       // logits available
  bool finished = true;     // reached EOG or ctx limit

  // Speculative decoding state.
  int32_t spec_depth = 1;
  double accept_ema = 0.0;
  float entropy_ema = 1.0f;
  std::uint64_t spec_proposed = 0;
  std::uint64_t spec_accepted = 0;
  std::uint64_t spec_elapsed_ns_total = 0;
  std::uint64_t spec_draft_ns_total = 0;
  std::uint64_t spec_verify_ns_total = 0;
  std::uint64_t spec_accept_scan_ns_total = 0;
  std::uint64_t spec_skipped_tokens_total = 0;
  std::uint64_t spec_fast_verified_tokens_total = 0;
  // Decode savings metrics (accumulated across steps).
  double decode_ms_saved_total = 0.0;    // estimated decode ms saved by speculation
  double spec_overhead_ms_total = 0.0;   // total overhead ms of running draft model
  double spec_effective_ema = std::numeric_limits<double>::quiet_NaN();
  bool last_spec_allow_exec = false;
  bool last_spec_allow_commit = false;
  int32_t last_spec_plan_depth = 1;
  std::string last_spec_disable_reason = "unavailable";
  std::string last_spec_draft_init_reason = "unknown";
  bool cuda_graph_path_valid = false;
  bool cuda_graph_spec_path = false;
  SpecMode spec_mode = SpecMode::kOff;
  // --- Speculation ---
  korith::core::SpecPlan plan;

  // Caller-owned state for the Rust scheduler FFI (deterministic across calls).
  KorithSchedulerState rust_sched_state{};

  // Controller stats (token-weighted).
  std::array<std::uint64_t, 33> speculative_depth_histogram{};
  std::uint64_t control_tokens_total = 0;
  std::uint64_t control_tokens_unscheduled = 0;

  std::uint64_t printed_total = 0;
  std::uint64_t max_tokens = 0;
  std::uint64_t token_prefix_hash = 1469598103934665603ull;  // FNV offset basis
  std::uint64_t kv_hits = 0;
  std::uint64_t kv_misses = 0;
  std::uint64_t kv_evicted = 0;
  bool fallback_used = false;
  bool run_summary_logged = false;

  // Platform integration: paths and per-run cached values.
  std::string engine_metrics_path;
  std::string engine_events_path;
  std::string mf_snapshot_in_path;
  std::string mf_snapshot_out_path;
  std::string job_id;

  std::string last_amf_decision = "unavailable";
  std::uint32_t last_prefix_len = 0;
  std::uint32_t last_skipped_tokens = 0;
  double last_skip_ratio = 0.0;
  double last_restore_ms = 0.0;
  double last_baseline_prefix_ms = 0.0;
  double last_saved_ms = 0.0;
  double last_roi = 0.0;
  double last_prompt_decode_ms = 0.0;
  std::size_t last_prompt_tokens = 0;

  // Timing / throughput (computed from printed tokens).
  std::chrono::steady_clock::time_point started_at{};
  std::chrono::steady_clock::time_point last_step_at{};
  std::chrono::nanoseconds last_step_time{0};
  double tps_rolling = 0.0;
  std::chrono::nanoseconds tps_window = std::chrono::milliseconds(750);
  std::chrono::steady_clock::time_point tps_last_at{};
  std::uint64_t tps_last_tokens = 0;
  bool tps_last_valid = false;
  std::uint64_t tps_sample_count = 0;
  double baseline_tps_ema = 0.0;
  double baseline_prompt_ms_ema = 0.0;
  std::uint64_t baseline_samples = 0;
  std::uint64_t baseline_prompt_samples = 0;
  std::chrono::steady_clock::time_point baseline_last_update{};
  std::chrono::steady_clock::time_point baseline_last_log{};
  double tps_delta = 0.0;
  double thermo_flops_saved = 0.0;
  double thermo_kv_reuse_ratio = 0.0;
  double thermo_entropy_gain = 0.0;
  double thermo_cost = 0.0;
  double thermo_gain_ema = 0.0;
  double last_step_accept = 1.0;
  std::uint64_t last_step_spec_tokens = 0;
  std::uint64_t last_depth_change_tokens = 0;
  std::uint64_t last_probe_tokens = 0;
  std::uint64_t probe_attempts = 0;
  // --- AMF ---
  korith::core::AmfStore amf;
  korith::core::AmfContext amf_ctx{};
  bool amf_ready = false;
  // Direct-GPU KV path (KORITH_AMF_DIRECT_GPU=1).
  korith::core::AmfDirectKvCtx * amf_direct_kv_ctx = nullptr;
  bool amf_direct_gpu_enabled = false;
  float amf_reuse_score = 0.0f;
  float amf_avg_prefix_len = 0.0f;
  float amf_accept_rate = 0.0f;
  float amf_restore_cost_ms = 0.0f;
  int amf_negative_roi_hits = 0;
  bool amf_roi_disabled = false;
  std::uint64_t amf_prompt_tokens_total = 0;
  std::uint64_t amf_skipped_tokens_total = 0;
  double amf_roi_sum = 0.0;
  std::uint64_t amf_roi_count = 0;
  std::uint64_t amf_disable_count = 0;
  std::uint64_t amf_enable_count = 0;
  std::uint64_t amf_eviction_events = 0;
  std::uint64_t mf_policy_updates = 0;
  double amf_roi_slope_ema = std::numeric_limits<double>::quiet_NaN();
  double amf_last_avg_roi_ema = std::numeric_limits<double>::quiet_NaN();
  korith::core::MemoryFieldState memory_field{};
  std::uint64_t amf_replay_disabled_until_ms = 0;
  std::uint32_t amf_replay_disable_mask = 0;
  struct TpsSample {
    std::chrono::steady_clock::time_point t;
    std::uint64_t tokens = 0;
  };
  std::deque<TpsSample> tps_samples;
};

struct CollapseExecState {
  korith::core::CollapseMetrics metrics{};
  EngineState * engine = nullptr;
  int32_t batch_size = 1;
  bool reuse_kv = false;
  int32_t scheduler_depth = 1;
  korith::core::SpecPlan plan{};
  int32_t printed_this_call = 0;
  bool force_speculation = false;
  int32_t spec_window_limit = 0;
  std::uint64_t spec_proposed_before = 0;
  std::uint64_t spec_accepted_before = 0;
  std::uint64_t spec_elapsed_ns = 0;
  std::uint64_t spec_draft_ns = 0;
  std::uint64_t spec_verify_ns = 0;
  std::uint64_t spec_accept_scan_ns = 0;
  double spec_effective_ms = std::numeric_limits<double>::quiet_NaN();
  std::uint64_t spec_skipped_tokens = 0;
  std::uint64_t spec_fast_verified_tokens = 0;
  int32_t engine_count_cap = 0;
  bool executed = false;
};

static_assert(offsetof(CollapseExecState, metrics) == 0, "CollapseMetrics must be the first field");

constexpr std::uint64_t kBaselineMinSamplesDefault = 32;
constexpr std::uint64_t kBaselinePromptMinSamples = 2;
static bool baseline_ready = false;

std::uint64_t baseline_min_samples() {
  static bool initialized = false;
  static std::uint64_t value = kBaselineMinSamplesDefault;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_BASELINE_MIN_SAMPLES")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end && end != env && *end == '\0') {
          value = std::clamp<std::uint64_t>(static_cast<std::uint64_t>(parsed), 1ull, 4096ull);
        }
      }
    }
  }
  return value;
}

EngineState g;
std::mutex g_mu;

void reset_runtime_state_unlocked() {
  const auto now = std::chrono::steady_clock::now();
  g.pos_target = 0;
  g.pos_draft = 0;
  g.ready = false;
  g.finished = true;
  g.logits_target = nullptr;
  g.logits_draft = nullptr;
  g.spec_depth = 1;
  g.accept_ema = 0.0;
  g.entropy_ema = 1.0f;
  g.spec_proposed = 0;
  g.spec_accepted = 0;
  g.spec_elapsed_ns_total = 0;
  g.spec_draft_ns_total = 0;
  g.spec_verify_ns_total = 0;
  g.spec_accept_scan_ns_total = 0;
  g.spec_skipped_tokens_total = 0;
  g.spec_fast_verified_tokens_total = 0;
  g.decode_ms_saved_total = 0.0;
  g.spec_overhead_ms_total = 0.0;
  g.spec_effective_ema = std::numeric_limits<double>::quiet_NaN();
  g.last_spec_allow_exec = false;
  g.last_spec_allow_commit = false;
  g.last_spec_plan_depth = 1;
  g.last_spec_disable_reason = "unavailable";
  g.cuda_graph_path_valid = false;
  g.cuda_graph_spec_path = false;
  g.spec_mode = SpecMode::kOff;
  g.rust_sched_state.speculative_depth = 1;
  g.rust_sched_state.batch_size = 1;
  g.rust_sched_state.baseline_ready = 0;
  g.speculative_depth_histogram.fill(0);
  g.control_tokens_total = 0;
  g.control_tokens_unscheduled = 0;
  g.printed_total = 0;
  g.last_amf_decision = "unavailable";
  g.last_prefix_len = 0;
  g.last_skipped_tokens = 0;
  g.last_skip_ratio = 0.0;
  g.last_restore_ms = 0.0;
  g.last_baseline_prefix_ms = 0.0;
  g.last_saved_ms = 0.0;
  g.last_roi = 0.0;
  g.last_prompt_decode_ms = 0.0;
  g.last_prompt_tokens = 0;
  g.max_tokens = 0;
  g.token_prefix_hash = 1469598103934665603ull;
  g.kv_hits = 0;
  g.kv_misses = 0;
  g.kv_evicted = 0;
  g.fallback_used = false;
  g.run_summary_logged = false;
  g.started_at = now;
  g.last_step_at = now;
  g.last_step_time = std::chrono::nanoseconds(0);
  g.tps_rolling = 0.0;
  g.tps_window = std::chrono::milliseconds(750);
  g.tps_last_at = now;
  g.tps_last_tokens = 0;
  g.tps_last_valid = false;
  g.tps_sample_count = 0;
  g.baseline_tps_ema = 0.0;
  g.baseline_prompt_ms_ema = 0.0;
  g.baseline_samples = 0;
  g.baseline_prompt_samples = 0;
  g.baseline_last_update = now;
  g.baseline_last_log = now;
  baseline_ready = false;
  g.tps_delta = 0.0;
  g.thermo_flops_saved = 0.0;
  g.thermo_kv_reuse_ratio = 0.0;
  g.thermo_entropy_gain = 0.0;
  g.thermo_cost = 0.0;
  g.thermo_gain_ema = 0.0;
  g.last_step_accept = 1.0;
  g.last_step_spec_tokens = 0;
  g.last_depth_change_tokens = 0;
  g.last_probe_tokens = 0;
  g.probe_attempts = 0;
  g.tps_samples.clear();
  g.amf_reuse_score = 0.0f;
  g.amf_avg_prefix_len = 0.0f;
  g.amf_accept_rate = 0.0f;
  g.amf_restore_cost_ms = 0.0f;
  g.amf_negative_roi_hits = 0;
  g.amf_roi_disabled = false;
  g.amf_prompt_tokens_total = 0;
  g.amf_skipped_tokens_total = 0;
  g.amf_roi_sum = 0.0;
  g.amf_roi_count = 0;
  g.amf_disable_count = 0;
  g.amf_enable_count = 0;
  g.amf_eviction_events = 0;
  g.mf_policy_updates = 0;
  g.amf_roi_slope_ema = std::numeric_limits<double>::quiet_NaN();
  g.amf_last_avg_roi_ema = std::numeric_limits<double>::quiet_NaN();
  g.memory_field = {};
  g.amf_replay_disabled_until_ms = 0;
  g.amf_replay_disable_mask = 0;

  korith::core::thermo_reset();
}

void commit_speculative_cb(korith::core::Context & ctx, std::int32_t depth) {
  auto * exec = static_cast<CollapseExecState *>(ctx.opaque);
  if (!exec || !exec->engine) {
    if (exec) {
      exec->printed_this_call = -1;
      exec->executed = true;
    }
    std::fprintf(stderr, "[DECODE] committed=true depth=%d\n", depth);
    (void) std::fflush(stderr);
    return;
  }

  EngineState & g = *exec->engine;
  const int32_t commit_depth = std::max<int32_t>(1, depth);
  g.spec_depth = commit_depth;
  g.rust_sched_state.speculative_depth = g.spec_depth;

  std::fprintf(stderr,
               "[DECODE] committed=%s depth=%d\n",
               exec->plan.allow_commit ? "true" : "false",
               commit_depth);
  (void) std::fflush(stderr);

  exec->printed_this_call = korith::core::batch_executor_step(
      g.ctx_target,
      g.batch_target,
      g.vocab_target,
      g.n_vocab,
      g.ctx_draft,
      g.batch_draft,
      g.pos_target,
      g.pos_draft,
      g.logits_target,
      g.logits_draft,
      g.finished,
      exec->batch_size,
      /* force_normal_decode = */ false,
      exec->reuse_kv,
      exec->force_speculation,
      exec->spec_window_limit,
      g.spec_depth,
      exec->plan,
      g.accept_ema,
      g.spec_proposed,
      g.spec_accepted,
      g.printed_total,
      g.token_prefix_hash,
      g.kv_hits,
      g.kv_misses,
      g.kv_evicted,
      g.baseline_tps_ema,
      g.tps_delta,
      exec->spec_elapsed_ns,
      exec->spec_draft_ns,
      exec->spec_verify_ns,
      exec->spec_accept_scan_ns,
      exec->spec_effective_ms,
      exec->spec_skipped_tokens,
      exec->spec_fast_verified_tokens,
      exec->engine_count_cap,
      g.max_tokens);
  exec->metrics.speculative_tokens =
      (g.spec_proposed >= exec->spec_proposed_before) ? (g.spec_proposed - exec->spec_proposed_before) : 0;
  exec->metrics.accepted_tokens =
      (g.spec_accepted >= exec->spec_accepted_before) ? (g.spec_accepted - exec->spec_accepted_before) : 0;
  exec->executed = true;
}

void fallback_decode_cb(korith::core::Context & ctx) {
  auto * exec = static_cast<CollapseExecState *>(ctx.opaque);
  if (!exec || !exec->engine) {
    if (exec) {
      exec->printed_this_call = -1;
      exec->executed = true;
    }
    std::fprintf(stderr, "[DECODE] committed=false depth=1\n");
    (void) std::fflush(stderr);
    return;
  }

  EngineState & g = *exec->engine;
  g.fallback_used = true;
  const int32_t log_depth = std::max<int32_t>(1, g.spec_depth);
  std::fprintf(stderr, "[DECODE] committed=false depth=%d\n", log_depth);
  (void) std::fflush(stderr);

  exec->printed_this_call = korith::core::batch_executor_step(
      g.ctx_target,
      g.batch_target,
      g.vocab_target,
      g.n_vocab,
      g.ctx_draft,
      g.batch_draft,
      g.pos_target,
      g.pos_draft,
      g.logits_target,
      g.logits_draft,
      g.finished,
      exec->batch_size,
      /* force_normal_decode = */ false,
      /* reuse_kv = */ exec->reuse_kv,
      /* force_speculation = */ exec->force_speculation,
      /* spec_window_limit = */ exec->spec_window_limit,
      g.spec_depth,
      exec->plan,
      g.accept_ema,
      g.spec_proposed,
      g.spec_accepted,
      g.printed_total,
      g.token_prefix_hash,
      g.kv_hits,
      g.kv_misses,
      g.kv_evicted,
      g.baseline_tps_ema,
      g.tps_delta,
      exec->spec_elapsed_ns,
      exec->spec_draft_ns,
      exec->spec_verify_ns,
      exec->spec_accept_scan_ns,
      exec->spec_effective_ms,
      exec->spec_skipped_tokens,
      exec->spec_fast_verified_tokens,
      exec->engine_count_cap,
      g.max_tokens);
  exec->metrics.speculative_tokens =
      (g.spec_proposed >= exec->spec_proposed_before) ? (g.spec_proposed - exec->spec_proposed_before) : 0;
  exec->metrics.accepted_tokens =
      (g.spec_accepted >= exec->spec_accepted_before) ? (g.spec_accepted - exec->spec_accepted_before) : 0;
  exec->executed = true;
}

void shutdown_unlocked() {
  g.amf.shutdown();
  if (g.amf_direct_kv_ctx) {
    korith::core::amf_direct_kv_free(g.amf_direct_kv_ctx);
    g.amf_direct_kv_ctx = nullptr;
    g.amf_direct_gpu_enabled = false;
  }
  korith::core::cuda_graph_decode_invalidate_all();

  if (g.batch_draft_inited) {
    llama_batch_free(g.batch_draft);
    g.batch_draft = {};
    g.batch_draft_inited = false;
  }

  if (g.batch_target_inited) {
    llama_batch_free(g.batch_target);
    g.batch_target = {};
    g.batch_target_inited = false;
  }

  if (g.ctx_draft) {
    llama_free(g.ctx_draft);
    g.ctx_draft = nullptr;
  }

  if (g.ctx_target) {
    llama_free(g.ctx_target);
    g.ctx_target = nullptr;
  }

  if (g.model_draft) {
    llama_model_free(g.model_draft);
    g.model_draft = nullptr;
  }

  if (g.model_target) {
    llama_model_free(g.model_target);
    g.model_target = nullptr;
  }

  g.vocab_target = nullptr;
  g.vocab_draft = nullptr;
  g.n_vocab = 0;
  reset_runtime_state_unlocked();

  if (g.backend_inited) {
    llama_backend_free();
    g.backend_inited = false;
  }
}

bool decode_one_token(
    llama_context * ctx,
    llama_batch & batch,
    llama_seq_id seq_id,
    llama_pos pos,
    llama_token token,
    bool want_logits) {
  if (!ctx) {
    return false;
  }

  batch.n_tokens = 1;
  batch.token[0] = token;
  batch.pos[0] = pos;
  batch.n_seq_id[0] = 1;
  batch.seq_id[0][0] = seq_id;
  batch.logits[0] = want_logits ? 1 : 0;

  const int32_t rc = llama_decode(ctx, batch);
  return rc == 0;
}

enum class PepTier : std::uint8_t {
  kExact = 1,
  kProof = 2,
};

struct PepPlan {
  PepTier tier = PepTier::kExact;
  std::size_t total_tokens = 0;
  int32_t n_batch = 0;
  std::size_t chunks = 0;
};

PepTier pep_selected_tier() {
  static bool checked = false;
  static PepTier cached = PepTier::kExact;
  if (checked) {
    return cached;
  }
  checked = true;
  int tier = 1;
  if (const char * env = std::getenv("KORITH_PEP_TIER")) {
    if (env[0] != '\0') {
      tier = std::atoi(env);
    }
  }
  if (tier >= static_cast<int>(PepTier::kProof)) {
    std::fprintf(stderr, "[PEP] tier=%d unavailable; forcing tier=1\n", tier);
    (void) std::fflush(stderr);
    cached = PepTier::kExact;
    return cached;
  }
  cached = PepTier::kExact;
  return cached;
}

PepPlan build_pep_plan(std::size_t token_count, int32_t n_batch) {
  PepPlan plan{};
  plan.tier = pep_selected_tier();
  plan.total_tokens = token_count;
  plan.n_batch = n_batch;
  if (token_count == 0 || n_batch <= 0) {
    plan.chunks = 0;
    return plan;
  }
  const std::size_t batch = static_cast<std::size_t>(n_batch);
  plan.chunks = (token_count + batch - 1) / batch;
  return plan;
}

bool execute_pep_plan(
    llama_context * ctx,
    llama_batch & batch,
    llama_pos & pos,
    const std::vector<llama_token> & tokens,
    const PepPlan & plan) {
  if (!ctx) {
    return false;
  }
  if (tokens.empty()) {
    return true;
  }
  if (plan.tier != PepTier::kExact) {
    return false;
  }
  if (plan.n_batch <= 0) {
    return false;
  }

  std::size_t idx = 0;
  while (idx < tokens.size()) {
    const std::size_t remaining = tokens.size() - idx;
    const int32_t chunk = std::min<int32_t>(plan.n_batch, static_cast<int32_t>(remaining));
    batch.n_tokens = chunk;
    for (int32_t i = 0; i < chunk; ++i) {
      const std::size_t t = idx + static_cast<std::size_t>(i);
      batch.token[i] = tokens[t];
      batch.pos[i] = pos + i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = (t + 1 == tokens.size()) ? 1 : 0;
    }
    const int32_t rc = llama_decode(ctx, batch);
    if (rc != 0) {
      return false;
    }
    pos += chunk;
    idx += static_cast<std::size_t>(chunk);
  }
  return true;
}

std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const char * prompt) {
  if (!vocab || !prompt) {
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
  int32_t n = llama_tokenize(vocab,
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
    n = llama_tokenize(vocab,
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

bool decode_tokens(
    llama_context * ctx,
    llama_batch & batch,
    llama_pos & pos,
    const std::vector<llama_token> & tokens) {
  if (!ctx) {
    return false;
  }
  if (tokens.empty()) {
    return true;
  }

  const uint32_t n_ctx = llama_n_ctx(ctx);
  if (n_ctx > 0 && (static_cast<std::uint64_t>(pos) + tokens.size()) >= n_ctx) {
    return false;
  }

  const uint32_t n_batch_u32 = llama_n_batch(ctx);
  if (n_batch_u32 == 0 || n_batch_u32 > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  const int32_t n_batch = static_cast<int32_t>(n_batch_u32);
  const PepPlan plan = build_pep_plan(tokens.size(), n_batch);
  return execute_pep_plan(ctx, batch, pos, tokens, plan);
}

struct ModelPaths {
  std::string target;
  std::string draft;
  bool draft_explicit = false;
};

ModelPaths parse_model_paths(const char * model_path) {
  ModelPaths out;
  if (!model_path) {
    return out;
  }

  const std::string s(model_path);
  const std::size_t sep = s.find('|');
  if (sep != std::string::npos) {
    out.target = s.substr(0, sep);
    out.draft = s.substr(sep + 1);
    out.draft_explicit = true;
    return out;
  }

  out.target = s;
  if (const char * env = std::getenv("KORITH_DRAFT_MODEL")) {
    if (env[0] != '\0') {
      out.draft = std::string(env);
    }
  }
  if (out.draft.empty()) {
    if (const char * env = std::getenv("KORITH_DRAFT_MODEL_PATH")) {
      if (env[0] != '\0') {
        out.draft = std::string(env);
      }
    }
  }
  return out;
}

bool init_batch_for_ctx(llama_context * ctx, llama_batch & batch, bool & inited, int32_t n_seq_max) {
  if (!ctx) {
    return false;
  }
  const uint32_t ctx_n_batch_u32 = llama_n_batch(ctx);
  if (ctx_n_batch_u32 == 0 || ctx_n_batch_u32 > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }

  const int32_t ctx_n_batch = static_cast<int32_t>(ctx_n_batch_u32);
  batch = llama_batch_init(ctx_n_batch, 0, n_seq_max);
  inited = true;

  if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
    return false;
  }

  for (int32_t i = 0; i < ctx_n_batch; ++i) {
    if (!batch.seq_id[i]) {
      return false;
    }
  }
  if (batch.seq_id[ctx_n_batch] != nullptr) {
    return false;
  }
  return true;
}

void free_draft_unlocked() {
  if (g.batch_draft_inited) {
    llama_batch_free(g.batch_draft);
    g.batch_draft = {};
    g.batch_draft_inited = false;
  }

  if (g.ctx_draft) {
    llama_free(g.ctx_draft);
    g.ctx_draft = nullptr;
  }

  if (g.model_draft) {
    llama_model_free(g.model_draft);
    g.model_draft = nullptr;
  }

  g.vocab_draft = nullptr;
  g.logits_draft = nullptr;
}

bool vocabs_compatible(const llama_vocab * a, const llama_vocab * b, std::string * detail = nullptr) {
  auto set_detail = [&](const std::string & s) {
    if (detail) {
      *detail = s;
    }
  };
  if (!a || !b) {
    set_detail("null_vocab");
    return false;
  }
  if (llama_vocab_type(a) != llama_vocab_type(b)) {
    set_detail("vocab_type_mismatch");
    return false;
  }
  if (llama_vocab_n_tokens(a) != llama_vocab_n_tokens(b)) {
    set_detail("token_count_mismatch");
    return false;
  }

  const llama_token a_bos = llama_vocab_bos(a);
  const llama_token a_eos = llama_vocab_eos(a);
  if (a_bos == LLAMA_TOKEN_NULL || a_eos == LLAMA_TOKEN_NULL) {
    set_detail("target_missing_bos_or_eos");
    return false;
  }

  const llama_token b_bos = llama_vocab_bos(b);
  const llama_token b_eos = llama_vocab_eos(b);
  if (b_bos == LLAMA_TOKEN_NULL || b_eos == LLAMA_TOKEN_NULL) {
    set_detail("draft_missing_bos_or_eos");
    return false;
  }

  if (a_bos != b_bos || a_eos != b_eos) {
    set_detail("bos_eos_id_mismatch");
    return false;
  }

  // Ensure common special tokens match (the draft must share the exact token-id mapping).
  if (llama_vocab_eot(a) != llama_vocab_eot(b) || llama_vocab_sep(a) != llama_vocab_sep(b) ||
      llama_vocab_nl(a) != llama_vocab_nl(b) || llama_vocab_pad(a) != llama_vocab_pad(b) ||
      llama_vocab_mask(a) != llama_vocab_mask(b)) {
    set_detail("special_token_id_mismatch");
    return false;
  }

  if (llama_vocab_fim_pre(a) != llama_vocab_fim_pre(b) || llama_vocab_fim_suf(a) != llama_vocab_fim_suf(b) ||
      llama_vocab_fim_mid(a) != llama_vocab_fim_mid(b) || llama_vocab_fim_pad(a) != llama_vocab_fim_pad(b) ||
      llama_vocab_fim_rep(a) != llama_vocab_fim_rep(b) || llama_vocab_fim_sep(a) != llama_vocab_fim_sep(b)) {
    set_detail("fim_token_id_mismatch");
    return false;
  }

  if (llama_vocab_get_add_bos(a) != llama_vocab_get_add_bos(b) || llama_vocab_get_add_eos(a) != llama_vocab_get_add_eos(b) ||
      llama_vocab_get_add_sep(a) != llama_vocab_get_add_sep(b)) {
    set_detail("add_token_policy_mismatch");
    return false;
  }

  // Spot-check a few token texts to guard against "same size but different mapping" cases.
  const int32_t n = llama_vocab_n_tokens(a);
  const int32_t idxs[] = {0, 1, 2, 3, 10, 100, 1000, n > 0 ? (n - 1) : 0};
  for (int32_t idx : idxs) {
    if (idx < 0 || idx >= n) {
      continue;
    }
    const char * ta = llama_vocab_get_text(a, static_cast<llama_token>(idx));
    const char * tb = llama_vocab_get_text(b, static_cast<llama_token>(idx));
    if (!ta || !tb) {
      set_detail("token_text_missing");
      return false;
    }
    if (std::strcmp(ta, tb) != 0) {
      std::ostringstream os;
      os << "token_text_mismatch_idx_" << idx;
      set_detail(os.str());
      return false;
    }
  }

  set_detail("ok");
  return true;
}

enum class DraftQuantMode : std::uint8_t {
  kOff = 0,
  kInt8 = 1,
  kInt4 = 2,
};

DraftQuantMode draft_quant_mode() {
  static bool initialized = false;
  static DraftQuantMode mode = DraftQuantMode::kOff;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_DRAFT_QUANTIZE")) {
      std::string v = env;
      std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (v == "1" || v == "true" || v == "int8" || v == "q8") {
        mode = DraftQuantMode::kInt8;
      } else if (v == "int4" || v == "q4") {
        mode = DraftQuantMode::kInt4;
      } else {
        mode = DraftQuantMode::kOff;
      }
    }
  }
  return mode;
}

bool draft_model_is_unquantized(const llama_model * model) {
  if (!model) {
    return false;
  }
  char buf[64];
  const int32_t n = llama_model_meta_val_str(model, "general.file_type", buf, sizeof(buf));
  if (n <= 0) {
    return false;
  }
  char * end = nullptr;
  const long ftype = std::strtol(buf, &end, 10);
  if (!end || end == buf) {
    return false;
  }
  return ftype == static_cast<long>(LLAMA_FTYPE_ALL_F32) ||
      ftype == static_cast<long>(LLAMA_FTYPE_MOSTLY_F16) ||
      ftype == static_cast<long>(LLAMA_FTYPE_MOSTLY_BF16) ||
      ftype == static_cast<long>(LLAMA_FTYPE_GUESSED);
}

llama_ftype draft_quant_target_ftype(DraftQuantMode mode) {
  switch (mode) {
    case DraftQuantMode::kInt4:
      return LLAMA_FTYPE_MOSTLY_Q4_K_M;
    case DraftQuantMode::kInt8:
      return LLAMA_FTYPE_MOSTLY_Q8_0;
    case DraftQuantMode::kOff:
    default:
      return LLAMA_FTYPE_MOSTLY_Q8_0;
  }
}

const char * draft_quant_label(DraftQuantMode mode) {
  switch (mode) {
    case DraftQuantMode::kInt4:
      return "q4_k_m";
    case DraftQuantMode::kInt8:
      return "q8_0";
    case DraftQuantMode::kOff:
    default:
      return "off";
  }
}

int32_t draft_quant_threads() {
  static bool initialized = false;
  static int32_t value = 0;
  if (!initialized) {
    initialized = true;
    if (const char * env = std::getenv("KORITH_SPEC_DRAFT_QUANT_THREADS")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end && end != env && *end == '\0') {
          value = std::max<int32_t>(0, static_cast<int32_t>(parsed));
        }
      }
    }
  }
  return value;
}

std::string draft_quantized_output_path(const std::string & input_path, DraftQuantMode mode) {
  if (const char * env = std::getenv("KORITH_SPEC_DRAFT_QUANT_PATH")) {
    if (env[0] != '\0') {
      return std::string(env);
    }
  }
  const std::size_t h = std::hash<std::string>{}(input_path);
  std::ostringstream out;
  out << "/tmp/korith_draft_" << std::hex << h << "_" << draft_quant_label(mode) << ".gguf";
  return out.str();
}

bool quantize_draft_file(
    const std::string & input_path,
    DraftQuantMode mode,
    std::string * output_path) {
  if (mode == DraftQuantMode::kOff || input_path.empty()) {
    return false;
  }
  const std::string out_path = draft_quantized_output_path(input_path, mode);
  if (std::filesystem::exists(out_path)) {
    if (output_path) {
      *output_path = out_path;
    }
    std::fprintf(stderr,
                 "[SPEC_DRAFT_QUANT] reuse path=%s mode=%s\n",
                 out_path.c_str(),
                 draft_quant_label(mode));
    (void) std::fflush(stderr);
    return true;
  }

  llama_model_quantize_params qparams = llama_model_quantize_default_params();
  qparams.ftype = draft_quant_target_ftype(mode);
  qparams.nthread = draft_quant_threads();
  qparams.quantize_output_tensor = true;
  qparams.allow_requantize = false;

  std::fprintf(stderr,
               "[SPEC_DRAFT_QUANT] start input=%s output=%s mode=%s\n",
               input_path.c_str(),
               out_path.c_str(),
               draft_quant_label(mode));
  (void) std::fflush(stderr);

  const uint32_t rc = llama_model_quantize(input_path.c_str(), out_path.c_str(), &qparams);
  if (rc != 0) {
    std::fprintf(stderr,
                 "[SPEC_DRAFT_QUANT] failed rc=%u input=%s mode=%s\n",
                 rc,
                 input_path.c_str(),
                 draft_quant_label(mode));
    (void) std::fflush(stderr);
    return false;
  }

  if (output_path) {
    *output_path = out_path;
  }
  std::fprintf(stderr,
               "[SPEC_DRAFT_QUANT] ready output=%s mode=%s\n",
               out_path.c_str(),
               draft_quant_label(mode));
  (void) std::fflush(stderr);
  return true;
}

bool draft_is_sufficiently_smaller(const llama_model * target, const llama_model * draft) {
  if (!target || !draft) {
    return false;
  }

  double max_ratio = 0.75;
  if (const char * env = std::getenv("KORITH_SPEC_MAX_DRAFT_SIZE_RATIO")) {
    if (env[0] != '\0') {
      char * end = nullptr;
      const double parsed = std::strtod(env, &end);
      if (end && end != env && *end == '\0' && std::isfinite(parsed)) {
        max_ratio = std::clamp(parsed, 0.10, 1.0);
      }
    }
  }

  const std::uint64_t target_size = llama_model_size(target);
  const std::uint64_t draft_size = llama_model_size(draft);
  const std::uint64_t target_params = llama_model_n_params(target);
  const std::uint64_t draft_params = llama_model_n_params(draft);
  const bool allow_size_only =
      (std::getenv("KORITH_SPEC_ALLOW_SIZE_ONLY") != nullptr) &&
      (std::strcmp(std::getenv("KORITH_SPEC_ALLOW_SIZE_ONLY"), "0") != 0);

  bool size_ok = false;
  double size_ratio = 0.0;
  if (target_size > 0 && draft_size > 0) {
    size_ratio = static_cast<double>(draft_size) / static_cast<double>(target_size);
    size_ok = size_ratio <= max_ratio;
  }

  bool params_ok = false;
  double params_ratio = 0.0;
  if (target_params > 0 && draft_params > 0) {
    params_ratio = static_cast<double>(draft_params) / static_cast<double>(target_params);
    params_ok = params_ratio <= max_ratio;
  }

  bool keep = false;
  if (target_params > 0 && draft_params > 0) {
    // Default policy: draft must be materially smaller in parameter count.
    // Size-only eligibility can be enabled for experiments.
    keep = params_ok || (allow_size_only && size_ok);
  } else {
    // If param metadata is unavailable, fall back to file-size ratio.
    keep = size_ok;
  }

  std::fprintf(
      stderr,
      "[SPEC_DRAFT_CHECK] target_bytes=%llu draft_bytes=%llu size_ratio=%.3f "
      "target_params=%llu draft_params=%llu params_ratio=%.3f max_ratio=%.3f "
      "allow_size_only=%d keep=%d\n",
      static_cast<unsigned long long>(target_size),
      static_cast<unsigned long long>(draft_size),
      size_ratio,
      static_cast<unsigned long long>(target_params),
      static_cast<unsigned long long>(draft_params),
      params_ratio,
      max_ratio,
      allow_size_only ? 1 : 0,
      keep ? 1 : 0);
  (void) std::fflush(stderr);

  return keep;
}

void update_tps_unlocked() {
  const auto now = std::chrono::steady_clock::now();
  const std::uint64_t tokens = korith_tokens_printed_total.load(std::memory_order_relaxed);

  if (!g.tps_last_valid) {
    g.tps_last_valid = true;
    g.tps_last_at = now;
    g.tps_last_tokens = tokens;
    g.tps_rolling = 0.0;
    if (tokens > 0) {
      const double dt_s = std::chrono::duration<double>(now - g.started_at).count();
      if (dt_s > 1e-9) {
        g.tps_rolling = static_cast<double>(tokens) / dt_s;
        g.tps_sample_count = 1;
        std::fprintf(stderr, "[TPS] instant=%.2f ema=%.2f\n", g.tps_rolling, g.tps_rolling);
        (void) std::fflush(stderr);
      }
    }
    return;
  }

  // Counter reset or wrap: drop history.
  if (tokens < g.tps_last_tokens) {
    g.tps_last_at = now;
    g.tps_last_tokens = tokens;
    g.tps_rolling = 0.0;
    return;
  }

  const double dt_s = std::chrono::duration<double>(now - g.tps_last_at).count();
  if (dt_s <= 1e-9) {
    return;
  }

  const std::uint64_t d_tok = tokens - g.tps_last_tokens;
  const double instant = (d_tok == 0) ? 0.0 : (static_cast<double>(d_tok) / dt_s);
  const double tau_s = std::max(0.5, std::min(1.0, std::chrono::duration<double>(g.tps_window).count()));
  const double alpha = std::clamp(1.0 - std::exp(-dt_s / tau_s), 0.0, 1.0);
  g.tps_rolling = g.tps_rolling + alpha * (instant - g.tps_rolling);
  g.tps_sample_count += 1;

  g.tps_last_at = now;
  g.tps_last_tokens = tokens;

  std::fprintf(stderr, "[TPS] instant=%.2f ema=%.2f\n", instant, g.tps_rolling);
  (void) std::fflush(stderr);
}

void log_run_summary_unlocked() {
  if (g.run_summary_logged) {
    return;
  }

  g.run_summary_logged = true;
  const auto now = std::chrono::steady_clock::now();
  const double dt_s = std::chrono::duration<double>(now - g.started_at).count();
  const double avg_tps =
      (dt_s > 1e-9) ? (static_cast<double>(g.printed_total) / dt_s) : 0.0;
  double avg_roi = 0.0;
  int amf_hit = 0;
  if (g.amf_ready) {
    const korith::core::AmfStats & stats = g.amf.stats();
    amf_hit = (stats.hits > 0) ? 1 : 0;
    if (g.amf_roi_count > 0) {
      avg_roi = g.amf_roi_sum / static_cast<double>(g.amf_roi_count);
    }
  }
  double avg_skip_ratio = 0.0;
  if (g.amf_prompt_tokens_total > 0) {
    avg_skip_ratio = static_cast<double>(g.amf_skipped_tokens_total) /
        static_cast<double>(g.amf_prompt_tokens_total);
  }
  avg_skip_ratio = std::clamp(avg_skip_ratio, 0.0, 1.0);

  std::fprintf(stderr,
               "[KORITH_RUN_SUMMARY] tokens_generated=%llu avg_tps=%.2f amf_hit=%d avg_skip_ratio=%.3f "
               "avg_roi=%.2f fallback_used=%d\n",
               static_cast<unsigned long long>(g.printed_total),
               avg_tps,
               amf_hit,
               avg_skip_ratio,
               avg_roi,
               g.fallback_used ? 1 : 0);
  (void) std::fflush(stderr);

  const korith::core::AmfStats & stats = g.amf.stats();
  const double denom = static_cast<double>(stats.hits + stats.misses);
  const double hit_rate = (denom > 0.0) ? (static_cast<double>(stats.hits) / denom) : 0.0;
  const double warm_ratio = g.amf.warm_ratio();
  const bool prewarm_complete = g.amf.prewarm_complete();
  const bool amf_ready_now = prewarm_complete || (hit_rate > 0.5);
  double avg_roi_ema = std::numeric_limits<double>::quiet_NaN();
  if (std::isfinite(stats.restore_ms_ema) && stats.restore_ms_ema > 0.0 &&
      std::isfinite(stats.saved_ms_ema)) {
    avg_roi_ema = stats.saved_ms_ema / stats.restore_ms_ema;
    if (!std::isfinite(avg_roi_ema)) {
      avg_roi_ema = std::numeric_limits<double>::quiet_NaN();
    }
  }
  const double avg_saved_tokens_per_hit = (stats.hits > 0)
      ? (static_cast<double>(stats.tokens_saved) / static_cast<double>(stats.hits))
      : 0.0;
  std::fprintf(stderr,
               "[AMF_SUMMARY] hits=%llu misses=%llu hit_rate=%.3f avg_roi_ema=%.3f "
               "avg_saved_tokens_per_hit=%.2f bytes=%llu evictions=%llu\n",
               static_cast<unsigned long long>(stats.hits),
               static_cast<unsigned long long>(stats.misses),
               hit_rate,
               avg_roi_ema,
               avg_saved_tokens_per_hit,
               static_cast<unsigned long long>(stats.bytes),
               static_cast<unsigned long long>(stats.evictions));
  (void) std::fflush(stderr);

  std::fprintf(stderr,
               "[KORITH_HEALTH] hit_rate=%.3f avg_skip_ratio=%.3f avg_roi_ema=%.3f "
               "warm_ratio=%.3f prewarm_complete=%d ready=%d "
               "roi_slope=%.4f evictions=%llu mf_updates=%llu replay_disables=%llu\n",
               hit_rate,
               avg_skip_ratio,
               avg_roi_ema,
               warm_ratio,
               prewarm_complete ? 1 : 0,
               amf_ready_now ? 1 : 0,
               g.amf_roi_slope_ema,
               static_cast<unsigned long long>(g.amf_eviction_events),
               static_cast<unsigned long long>(g.mf_policy_updates),
               static_cast<unsigned long long>(g.amf_disable_count));
  (void) std::fflush(stderr);

  if (!g.engine_metrics_path.empty()) {
    auto finite_or_zero = [](double v) -> double {
      return std::isfinite(v) ? v : 0.0;
    };
    const double total_ms = dt_s * 1000.0;
    const double prefill_ms = finite_or_zero(g.last_prompt_decode_ms);
    const double decode_ms = std::max(0.0, total_ms - prefill_ms);
    const bool spec_supported = (g.ctx_draft != nullptr);
    const bool spec_requested =
        spec_supported && g.last_spec_allow_exec && (g.last_spec_plan_depth > 1);
    const bool spec_has_activity =
        (g.spec_proposed > 0 || g.spec_elapsed_ns_total > 0 ||
         g.spec_skipped_tokens_total > 0 || g.spec_fast_verified_tokens_total > 0);
    // Report speculative mode as enabled only when commit is allowed. This avoids
    // "enabled=true" metrics during control-plane commit gating (e.g. baseline_gate).
    const bool spec_active =
        spec_requested && g.last_spec_allow_commit && spec_has_activity;
    const double acceptance_rate = (g.spec_proposed > 0)
        ? (static_cast<double>(g.spec_accepted) / static_cast<double>(g.spec_proposed))
        : 0.0;
    const double draft_ms = static_cast<double>(g.spec_draft_ns_total) * 1e-6;
    const double verify_ms = static_cast<double>(g.spec_verify_ns_total) * 1e-6;
    const double accept_scan_ms = static_cast<double>(g.spec_accept_scan_ns_total) * 1e-6;
    const double overhead_ms = draft_ms + verify_ms + accept_scan_ms;
    double baseline_total_ms = 0.0;
    if (spec_active && std::isfinite(g.baseline_tps_ema) && g.baseline_tps_ema > 1e-6 && g.printed_total > 0) {
      baseline_total_ms = (static_cast<double>(g.printed_total) / g.baseline_tps_ema) * 1000.0;
    }
    const double net_saved_ms = (baseline_total_ms > 0.0) ? (baseline_total_ms - total_ms) : 0.0;
    const double saved_ms = std::max(0.0, net_saved_ms);
    const double roi = (overhead_ms > 1e-9) ? (net_saved_ms / overhead_ms) : 0.0;
    const double speedup_est =
        (baseline_total_ms > 0.0 && total_ms > 1e-9) ? (baseline_total_ms / total_ms) : 0.0;
    const bool baseline_ready_now = baseline_ready && (g.baseline_samples >= baseline_min_samples());
    const double prefill_saved_ms = std::max(0.0, static_cast<double>(stats.ms_saved));
    const double spec_saved_ms = std::max(0.0, finite_or_zero(g.decode_ms_saved_total));
    const double kernels_saved_ms = 0.0;
    const double savings_total_ms = prefill_saved_ms + spec_saved_ms + kernels_saved_ms;
    const double baseline_prefill_ms = std::max(0.0, prefill_ms + prefill_saved_ms);
    const double baseline_decode_ms = std::max(0.0, decode_ms + spec_saved_ms + kernels_saved_ms);
    const double baseline_components_total_ms = std::max(0.0, baseline_prefill_ms + baseline_decode_ms);
    const bool spec_comparable = spec_active && baseline_ready_now && (g.spec_proposed > 0);
    const bool kernels_comparable = false;
    const bool savings_comparable = baseline_ready_now && (g.printed_total > 0);
    std::string spec_disable_reason;
    if (!spec_active) {
      if (!spec_supported) {
        if (!g.last_spec_draft_init_reason.empty() &&
            g.last_spec_draft_init_reason != "draft_ready") {
          spec_disable_reason = g.last_spec_draft_init_reason;
        } else {
          spec_disable_reason = "draft_unavailable";
        }
      } else if (!spec_requested) {
        if (!g.last_spec_disable_reason.empty() &&
            g.last_spec_disable_reason != "ok") {
          spec_disable_reason = g.last_spec_disable_reason;
        } else if (g.last_spec_plan_depth <= 1) {
          spec_disable_reason = "depth_lt_2";
        } else {
          spec_disable_reason = "disabled";
        }
      } else if (!g.last_spec_disable_reason.empty() &&
                 g.last_spec_disable_reason != "ok") {
        spec_disable_reason = g.last_spec_disable_reason;
      } else {
        spec_disable_reason = "no_spec_activity";
      }
    }
    std::ostringstream out;
    out << "{";
    out << "\"amf\":{"
        << "\"supported\":" << (g.amf_ready ? "true" : "false") << ","
        << "\"decision\":\"" << json_escape(g.last_amf_decision) << "\","
        << "\"cache_entries\":" << static_cast<unsigned long long>(stats.entries) << ","
        << "\"hit_rate\":" << finite_or_zero(hit_rate) << ","
        << "\"warm_ratio\":" << finite_or_zero(warm_ratio) << ","
        << "\"prewarm_complete\":" << (prewarm_complete ? "true" : "false") << ","
        << "\"ready\":" << (amf_ready_now ? "true" : "false") << ","
        << "\"prefix_len\":" << g.last_prefix_len << ","
        << "\"skipped_tokens\":" << g.last_skipped_tokens << ","
        << "\"skip_ratio\":" << finite_or_zero(g.last_skip_ratio) << ","
        << "\"restore_ms\":" << finite_or_zero(g.last_restore_ms) << ","
        << "\"baseline_prefix_ms\":" << finite_or_zero(g.last_baseline_prefix_ms) << ","
        << "\"saved_ms\":" << finite_or_zero(g.last_saved_ms) << ","
        << "\"roi\":" << finite_or_zero(g.last_roi)
        << "},";
    out << "\"mf\":{"
        << "\"supported\":" << (g.amf_ready ? "true" : "false") << ","
        << "\"min_admit_roi\":" << finite_or_zero(g.memory_field.last.min_admit_roi) << ","
        << "\"eviction_pressure\":" << finite_or_zero(g.memory_field.last.eviction_pressure) << ","
        << "\"replay_disable_mask\":" << g.memory_field.last.replay_disable_mask << ","
        << "\"cooldown_ms\":" << g.memory_field.last.cooldown_ms << ","
        << "\"snapshot_id\":\"" << json_escape(g.job_id) << "\""
        << "},";
    out << "\"spec\":{"
        << "\"supported\":" << (spec_supported ? "true" : "false") << ","
        << "\"enabled\":" << (spec_active ? "true" : "false") << ","
        << "\"comparable\":" << (spec_comparable ? "true" : "false") << ","
        << "\"k\":" << std::max<int32_t>(1, g.last_spec_plan_depth) << ","
        << "\"proposed_tokens\":" << static_cast<unsigned long long>(g.spec_proposed) << ","
        << "\"accepted_tokens\":" << static_cast<unsigned long long>(g.spec_accepted) << ","
        << "\"acceptance_rate\":" << finite_or_zero(acceptance_rate) << ","
        << "\"verify_ms\":" << finite_or_zero(verify_ms) << ","
        << "\"draft_ms\":" << finite_or_zero(draft_ms) << ","
        << "\"accept_scan_ms\":" << finite_or_zero(accept_scan_ms) << ","
        << "\"overhead_ms\":" << finite_or_zero(overhead_ms) << ","
        << "\"decode_ms_saved\":" << finite_or_zero(g.decode_ms_saved_total) << ","
        << "\"spec_overhead_ms\":" << finite_or_zero(g.spec_overhead_ms_total) << ","
        << "\"baseline_total_ms\":" << finite_or_zero(baseline_total_ms) << ","
        << "\"net_saved_ms\":" << finite_or_zero(net_saved_ms) << ","
        << "\"saved_ms\":" << finite_or_zero(saved_ms) << ","
        << "\"roi\":" << finite_or_zero(roi) << ","
        << "\"speedup_est\":" << finite_or_zero(speedup_est) << ","
        << "\"cache_hit\":false,"
        << "\"cache_ms\":0.0,"
        << "\"cache_only\":false,"
        << "\"draft_init_reason\":\"" << json_escape(g.last_spec_draft_init_reason) << "\","
        << "\"baseline_ready\":" << (baseline_ready_now ? "true" : "false") << ","
        << "\"disable_reason\":\"" << json_escape(spec_disable_reason) << "\""
        << "},";
    out << "\"kernels\":{"
        << "\"enabled\":false,"
        << "\"backend\":\"none\","
        << "\"kernels_applied\":false,"
        << "\"comparable\":" << (kernels_comparable ? "true" : "false") << ","
        << "\"decode_ms_actual\":" << finite_or_zero(decode_ms) << ","
        << "\"decode_ms_baseline_est\":" << finite_or_zero(decode_ms + kernels_saved_ms) << ","
        << "\"ms_saved\":" << finite_or_zero(kernels_saved_ms)
        << "},";
    out << "\"savings\":{"
        << "\"prefill_saved_ms\":" << finite_or_zero(prefill_saved_ms) << ","
        << "\"spec_saved_ms\":" << finite_or_zero(spec_saved_ms) << ","
        << "\"kernels_saved_ms\":" << finite_or_zero(kernels_saved_ms) << ","
        << "\"total_saved_ms\":" << finite_or_zero(savings_total_ms) << ","
        << "\"baseline_prefill_ms\":" << finite_or_zero(baseline_prefill_ms) << ","
        << "\"baseline_decode_ms\":" << finite_or_zero(baseline_decode_ms) << ","
        << "\"baseline_total_ms\":" << finite_or_zero(baseline_components_total_ms) << ","
        << "\"comparable\":" << (savings_comparable ? "true" : "false")
        << "},";
    out << "\"perf\":{"
        << "\"tokens_out\":" << g.printed_total << ","
        << "\"total_ms\":" << finite_or_zero(total_ms) << ","
        << "\"prefill_ms\":" << prefill_ms << ","
        << "\"decode_ms\":" << finite_or_zero(decode_ms) << ","
        << "\"avg_tps\":" << finite_or_zero(avg_tps)
        << "},";
    out << "\"measurement\":{"
        << "\"baseline_ready\":" << (baseline_ready_now ? "true" : "false") << ","
        << "\"spec_comparable\":" << (spec_comparable ? "true" : "false") << ","
        << "\"kernels_comparable\":" << (kernels_comparable ? "true" : "false") << ","
        << "\"savings_comparable\":" << (savings_comparable ? "true" : "false")
        << "},";
    out << "\"input\":{"
        << "\"prompt_tokens\":" << g.last_prompt_tokens << ","
        << "\"sampling_hash\":\"" << g.amf_ctx.sampling_hash << "\""
        << "},";
    out << "\"health\":{"
        << "\"hit_rate\":" << finite_or_zero(hit_rate) << ","
        << "\"avg_skip_ratio\":" << finite_or_zero(avg_skip_ratio) << ","
        << "\"avg_roi_ema\":" << finite_or_zero(avg_roi_ema)
        << "}";
    out << "}";
    (void) write_text_file(g.engine_metrics_path, out.str());
  }
}

}  // namespace

extern "C" {

bool engine_init(const char * model_path) {
  std::lock_guard<std::mutex> lock(g_mu);
  shutdown_unlocked();

  if (!model_path || model_path[0] == '\0') {
    return false;
  }

  try {
    korith_tokens_printed_total.store(0, std::memory_order_relaxed);

    if (const char * env = std::getenv("KORITH_ENGINE_METRICS_PATH")) {
      g.engine_metrics_path = env;
    } else {
      g.engine_metrics_path.clear();
    }
    if (const char * env = std::getenv("KORITH_ENGINE_EVENTS_PATH")) {
      g.engine_events_path = env;
    } else {
      g.engine_events_path.clear();
    }
    g_engine_events_path = g.engine_events_path;
    if (const char * env = std::getenv("KORITH_MF_SNAPSHOT_IN")) {
      g.mf_snapshot_in_path = env;
    } else {
      g.mf_snapshot_in_path.clear();
    }
    if (const char * env = std::getenv("KORITH_MF_SNAPSHOT_OUT")) {
      g.mf_snapshot_out_path = env;
    } else {
      g.mf_snapshot_out_path.clear();
    }
    if (const char * env = std::getenv("KORITH_JOB_ID")) {
      g.job_id = env;
    } else {
      g.job_id.clear();
    }

    const ModelPaths paths = parse_model_paths(model_path);
    if (paths.target.empty()) {
      return false;
    }
    g.spec_mode = resolve_spec_mode_from_env();
    if (g.spec_mode == SpecMode::kV3) {
      std::fprintf(stderr, "[SPEC] mode=v3 (v2 disabled)\n");
    } else if (g.spec_mode == SpecMode::kV2) {
      std::fprintf(stderr, "[SPEC] mode=v2 (v3 disabled)\n");
    } else {
      std::fprintf(stderr, "[SPEC] mode=off\n");
    }
    (void) std::fflush(stderr);
    g.last_spec_draft_init_reason = paths.draft.empty() ? "draft_model_missing" : "draft_model_present";
    const bool draft_optional = !paths.draft_explicit;
    const bool draft_same_as_target = !paths.draft.empty() && paths.draft == paths.target;
    if (draft_same_as_target) {
      g.last_spec_draft_init_reason = "draft_model_same_as_target";
      // A draft identical to the target is never a throughput multiplier.
      // Treat an explicit request as a hard error, but ignore an env-provided draft.
      if (!draft_optional) {
        return false;
      }
    }

    korith::core::cuda_graph_apply_backend_env_policy();
    llama_backend_init();
    g.backend_inited = true;

    llama_model_params mparams = llama_model_default_params();
    if (llama_supports_gpu_offload()) {
      mparams.n_gpu_layers = 999;  // clamped internally
      const char* main_gpu_env = std::getenv("KORITH_MAIN_GPU");
      mparams.main_gpu = main_gpu_env ? std::atoi(main_gpu_env) : 0;
    } else {
      mparams.n_gpu_layers = 0;
    }
    static std::vector<float> s_tensor_split;
    if (const char* ts_env = std::getenv("KORITH_TENSOR_SPLIT")) {
      s_tensor_split.clear();
      std::istringstream ss(ts_env);
      std::string tok;
      while (std::getline(ss, tok, ','))
        s_tensor_split.push_back(std::stof(tok));
      mparams.tensor_split = s_tensor_split.data();
    }

    g.model_target = llama_model_load_from_file(paths.target.c_str(), mparams);
    if (!g.model_target) {
      shutdown_unlocked();
      return false;
    }
    g.vocab_target = llama_model_get_vocab(g.model_target);
    if (!g.vocab_target) {
      shutdown_unlocked();
      return false;
    }
    g.n_vocab = llama_vocab_n_tokens(g.vocab_target);
    if (g.n_vocab <= 0) {
      shutdown_unlocked();
      return false;
    }

    struct RuntimeCfg {
      int32_t n_seq_max = 8;
      int32_t n_batch = 512;
    };
    const RuntimeCfg cfg = []() {
      RuntimeCfg out{};
      if (const char * env = std::getenv("KORITH_N_SEQ_MAX")) {
        if (env[0] != '\0') {
          char * end = nullptr;
          const long parsed = std::strtol(env, &end, 10);
          if (end && end != env && *end == '\0' && parsed > 0 && parsed <= INT32_MAX) {
            out.n_seq_max = static_cast<int32_t>(parsed);
          }
        }
      }
      if (const char * env = std::getenv("KORITH_N_BATCH")) {
        if (env[0] != '\0') {
          char * end = nullptr;
          const long parsed = std::strtol(env, &end, 10);
          if (end && end != env && *end == '\0' && parsed > 0 && parsed <= INT32_MAX) {
            out.n_batch = static_cast<int32_t>(parsed);
          }
        }
      }
      return out;
    }();
    // Fixed defaults for now (no external config surface yet).
    // These will become scheduler/controller inputs once the rest of the engine is in place.
    llama_context_params cparams = llama_context_default_params();
    // --- Korith sequence overrides ---
    if (const char * env = std::getenv("KORITH_N_SEQ_MAX")) {
        const int v = std::atoi(env);
        if (v > 0) cparams.n_seq_max = v;
    }

    if (const char * env = std::getenv("KORITH_N_BATCH")) {
        const int v = std::atoi(env);
        if (v > 0) {
            cparams.n_batch  = v;
            cparams.n_ubatch = v;
        }
    }
    // === KORITH FIX: forward runtime params ===
    cparams.n_seq_max = cfg.n_seq_max;   // <-- REQUIRED for speculation
    cparams.n_batch   = cfg.n_batch;
    cparams.n_ubatch  = cfg.n_batch;
    int32_t n_ctx_override = 0;
    if (const char * env = std::getenv("KORITH_N_CTX")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end && end != env && *end == '\0' && parsed > 0 && parsed <= INT32_MAX) {
          n_ctx_override = static_cast<int32_t>(parsed);
        }
      }
    }
    cparams.n_ctx = (n_ctx_override > 0) ? n_ctx_override : 8192;
    const int32_t n_ctx_train = static_cast<int32_t>(llama_model_n_ctx_train(g.model_target));
    if (n_ctx_train > 0 && static_cast<int64_t>(cparams.n_ctx) > n_ctx_train) {
      cparams.n_ctx = static_cast<uint32_t>(n_ctx_train);
    }
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;
    cparams.kv_unified = true;
    const enum ggml_type default_type_k = cparams.type_k;
    const enum ggml_type default_type_v = cparams.type_v;
    std::string kv_mode_label = "off";
    const bool kv_override_applied = apply_kv_compression_env(cparams, kv_mode_label);
    if (kv_override_applied) {
      std::fprintf(stderr,
                   "[KV_COMPRESS] enabled=1 mode=%s type_k=%s type_v=%s\n",
                   kv_mode_label.c_str(),
                   ggml_type_name_short(cparams.type_k),
                   ggml_type_name_short(cparams.type_v));
      (void) std::fflush(stderr);
    } else {
      std::fprintf(stderr,
                   "[KV_COMPRESS] enabled=0 type_k=%s type_v=%s\n",
                   ggml_type_name_short(cparams.type_k),
                   ggml_type_name_short(cparams.type_v));
      (void) std::fflush(stderr);
    }
    bool allow_n_ctx_fallback = false;
    if (const char * env = std::getenv("KORITH_ALLOW_N_CTX_FALLBACK")) {
      allow_n_ctx_fallback = (env[0] != '\0' && env[0] != '0');
    }
    const uint32_t requested_n_ctx = cparams.n_ctx;

    g.ctx_target = llama_init_from_model(g.model_target, cparams);
    if (!g.ctx_target && kv_override_applied) {
      std::fprintf(stderr,
                   "[KV_COMPRESS] fallback reason=ctx_init_fail restore_defaults type_k=%s type_v=%s\n",
                   ggml_type_name_short(default_type_k),
                   ggml_type_name_short(default_type_v));
      (void) std::fflush(stderr);
      cparams.type_k = default_type_k;
      cparams.type_v = default_type_v;
      g.ctx_target = llama_init_from_model(g.model_target, cparams);
    }
    if (!g.ctx_target && allow_n_ctx_fallback && cparams.n_ctx > 4096u) {
      std::fprintf(stderr, "[CTX_FALLBACK] requested_n_ctx=%u fallback_n_ctx=4096\n", requested_n_ctx);
      (void) std::fflush(stderr);
      cparams.n_ctx = 4096;
      g.ctx_target = llama_init_from_model(g.model_target, cparams);
    }
    if (!g.ctx_target) {
      std::fprintf(stderr,
                   "[CTX_INIT_FAIL] requested_n_ctx=%u fallback_allowed=%d\n",
                   requested_n_ctx,
                   allow_n_ctx_fallback ? 1 : 0);
      (void) std::fflush(stderr);
      shutdown_unlocked();
      return false;
    }
    if (!init_batch_for_ctx(g.ctx_target,
                            g.batch_target,
                            g.batch_target_inited,
                            /* n_seq_max = */ cparams.n_seq_max)) {
      shutdown_unlocked();
      return false;
    }

    if (!paths.draft.empty() && !draft_same_as_target) {
      std::string active_draft_path = paths.draft;
      g.last_spec_draft_init_reason = "draft_init_started";
      g.model_draft = llama_model_load_from_file(paths.draft.c_str(), mparams);
      if (!g.model_draft) {
        g.last_spec_draft_init_reason = "draft_model_load_failed";
        std::fprintf(stderr, "[SPEC_DRAFT_INIT] reason=%s path=%s optional=%d\n",
                     g.last_spec_draft_init_reason.c_str(),
                     paths.draft.c_str(),
                     draft_optional ? 1 : 0);
        (void) std::fflush(stderr);
        if (!draft_optional) {
          shutdown_unlocked();
          return false;
        }
        free_draft_unlocked();
      }

      if (g.model_draft) {
        const DraftQuantMode qmode = draft_quant_mode();
        if (qmode != DraftQuantMode::kOff && draft_model_is_unquantized(g.model_draft)) {
          std::string quantized_path;
          if (quantize_draft_file(paths.draft, qmode, &quantized_path)) {
            llama_model * quantized_model = llama_model_load_from_file(quantized_path.c_str(), mparams);
            if (quantized_model) {
              std::fprintf(stderr,
                           "[SPEC_DRAFT_QUANT] loaded path=%s\n",
                           quantized_path.c_str());
              (void) std::fflush(stderr);
              llama_model_free(g.model_draft);
              g.model_draft = quantized_model;
              active_draft_path = quantized_path;
            } else {
              std::fprintf(stderr,
                           "[SPEC_DRAFT_QUANT] load_failed path=%s; continuing with original draft\n",
                           quantized_path.c_str());
              (void) std::fflush(stderr);
            }
          }
        }
      }

      if (g.model_draft) {
        // Require a meaningfully smaller/faster draft. This is the throughput multiplier.
        if (!draft_is_sufficiently_smaller(g.model_target, g.model_draft)) {
          g.last_spec_draft_init_reason = "draft_model_too_large";
          std::fprintf(stderr, "[SPEC_DRAFT_INIT] reason=%s path=%s optional=%d\n",
                       g.last_spec_draft_init_reason.c_str(),
                       active_draft_path.c_str(),
                       draft_optional ? 1 : 0);
          (void) std::fflush(stderr);
          if (!draft_optional) {
            shutdown_unlocked();
            return false;
          }
          free_draft_unlocked();
        }
      }

      if (g.model_draft) {
        g.vocab_draft = llama_model_get_vocab(g.model_draft);
        if (!g.vocab_draft) {
          g.last_spec_draft_init_reason = "draft_vocab_missing";
          std::fprintf(stderr, "[SPEC_DRAFT_INIT] reason=%s path=%s optional=%d\n",
                       g.last_spec_draft_init_reason.c_str(),
                       paths.draft.c_str(),
                       draft_optional ? 1 : 0);
          (void) std::fflush(stderr);
          if (!draft_optional) {
            shutdown_unlocked();
            return false;
          }
          free_draft_unlocked();
        }
      }

      if (g.vocab_draft) {
        std::string vocab_detail;
        if (!vocabs_compatible(g.vocab_target, g.vocab_draft, &vocab_detail)) {
          g.last_spec_draft_init_reason = "draft_vocab_incompatible";
          std::fprintf(stderr, "[SPEC_DRAFT_INIT] reason=%s detail=%s path=%s optional=%d\n",
                       g.last_spec_draft_init_reason.c_str(),
                       vocab_detail.c_str(),
                       paths.draft.c_str(),
                       draft_optional ? 1 : 0);
          (void) std::fflush(stderr);
          if (!draft_optional) {
            shutdown_unlocked();
            return false;
          }
          free_draft_unlocked();
        }
      }

      if (g.model_draft) {
        llama_context_params dparams = cparams;
        dparams.n_seq_max = 1;

        g.ctx_draft = llama_init_from_model(g.model_draft, dparams);
        if (!g.ctx_draft) {
          g.last_spec_draft_init_reason = "draft_ctx_init_failed";
          std::fprintf(stderr, "[SPEC_DRAFT_INIT] reason=%s path=%s optional=%d\n",
                       g.last_spec_draft_init_reason.c_str(),
                       paths.draft.c_str(),
                       draft_optional ? 1 : 0);
          (void) std::fflush(stderr);
          if (!draft_optional) {
            shutdown_unlocked();
            return false;
          }
          free_draft_unlocked();
        }
      }

      if (g.ctx_draft) {
        if (!init_batch_for_ctx(g.ctx_draft, g.batch_draft, g.batch_draft_inited, /* n_seq_max = */ 1)) {
          g.last_spec_draft_init_reason = "draft_batch_init_failed";
          std::fprintf(stderr, "[SPEC_DRAFT_INIT] reason=%s path=%s optional=%d\n",
                       g.last_spec_draft_init_reason.c_str(),
                       paths.draft.c_str(),
                       draft_optional ? 1 : 0);
          (void) std::fflush(stderr);
          if (!draft_optional) {
            shutdown_unlocked();
            return false;
          }
          free_draft_unlocked();
        }
      }

      if (g.ctx_draft) {
        g.last_spec_draft_init_reason = "draft_ready";
        // Wire the draft model into the speculative engine pool so that
        // SpeculativeEngine::run_step() uses real draft proposals.
        korith::core::engine_pool_attach_draft(
            g.ctx_draft, &g.batch_draft, g.vocab_draft, g.n_vocab);
      }

    }

    {
      const int draft_loaded = (g.model_draft != nullptr) ? 1 : 0;
      const int total_models_loaded = 1 + draft_loaded;
      std::fprintf(stderr,
                   "[SPEC_MODELS] mode=%s target_models=1 draft_models=%d total_models=%d\n",
                   spec_mode_name(g.spec_mode),
                   draft_loaded,
                   total_models_loaded);
      (void) std::fflush(stderr);
    }

    g.amf_ctx = {};
    g.amf_ctx.model_hash = korith::core::amf_hash_file(paths.target);
    g.amf_ctx.n_ctx = llama_n_ctx(g.ctx_target);
    g.amf_ctx.kv_version = 1;
    g.amf_ctx.rope_base_bits = korith::core::amf_float_bits(0.0f);
    g.amf_ctx.rope_scale_bits =
        korith::core::amf_float_bits(llama_model_rope_freq_scale_train(g.model_target));
    g.amf_ctx.sampling_hash = amf_sampling_hash();
    g.amf_ctx.rng_hash = amf_rng_hash();
    g.amf_ctx.tenant_hash = 0;
    if (env_truthy("KORITH_AMF_TENANT_ISOLATION", false)) {
      std::string tenant_id = "default";
      if (const char * env = std::getenv("KORITH_TENANT_ID")) {
        if (env[0] != '\0') {
          tenant_id = sanitize_tenant_id_for_amf(std::string(env));
        }
      }
      g.amf_ctx.tenant_hash = korith::core::amf_hash_tenant_id(tenant_id);
      std::fprintf(stderr,
                   "[AMF_TENANT] tenant_id=%s tenant_hash=%016llx\n",
                   tenant_id.c_str(),
                   static_cast<unsigned long long>(g.amf_ctx.tenant_hash));
      (void) std::fflush(stderr);
    }
    bool amf_flag = false;
    if (const char * env = std::getenv("KORITH_ENABLE_AMF")) {
      amf_flag = (env[0] != '\0') && (env[0] != '0');
    }
    g.amf_ready = amf_flag && g.amf.init_from_env();
    if (!g.amf_ready || g.amf_ctx.model_hash == 0) {
      g.amf_ready = false;
      if (!amf_flag) {
        std::fprintf(stderr, "[AMF_DISABLED] reason=flag_off\n");
        (void) std::fflush(stderr);
      } else {
        g.amf.log_disabled_once();
      }
    }

    // Direct-GPU KV path — enabled via KORITH_AMF_DIRECT_GPU=1.
    {
      const char * dkv_env = std::getenv("KORITH_AMF_DIRECT_GPU");
      const bool want_direct = dkv_env && dkv_env[0] == '1';
      if (want_direct && g.amf_ready) {
        // Allocate a 32 GB pinned buffer by default; resize on first save if
        // needed (ensure_pinned will grow it).  Zero means allocate-on-demand.
        constexpr std::size_t kDefaultPinnedBytes = 0;
        g.amf_direct_kv_ctx =
            korith::core::amf_direct_kv_init(kDefaultPinnedBytes);
        g.amf_direct_gpu_enabled = (g.amf_direct_kv_ctx != nullptr);
        std::fprintf(stderr, "[AMF_DIRECT_GPU] %s\n",
                     g.amf_direct_gpu_enabled ? "enabled" : "init_failed");
        (void) std::fflush(stderr);
      }
    }

    reset_runtime_state_unlocked();
    korith::core::cuda_graph_log_config_once();
    g.memory_field.last.min_admit_roi = g.amf.min_admit_roi();
    g.memory_field.last.eviction_pressure = g.amf.eviction_pressure();
    if (!g.mf_snapshot_in_path.empty()) {
      std::string snapshot_json;
      if (load_text_file(g.mf_snapshot_in_path, snapshot_json)) {
        korith::core::MemoryFieldState restored{};
        if (korith::core::memory_field_state_from_json(snapshot_json, &restored)) {
          g.memory_field = restored;
          g.amf.set_min_admit_roi(g.memory_field.last.min_admit_roi);
          g.amf_replay_disable_mask = g.memory_field.last.replay_disable_mask;
          if ((g.amf_replay_disable_mask & 0x1u) != 0u) {
            const std::uint64_t now_ms = now_epoch_ms();
            g.amf_replay_disabled_until_ms = now_ms + g.memory_field.last.cooldown_ms;
          }
          emit_engine_event("MF_RESTORE", std::string("{\"job_id\":\"") +
                json_escape(g.job_id) + "\",\"snapshot_path\":\"" +
                json_escape(g.mf_snapshot_in_path) + "\"}");
        } else {
          emit_engine_event("MF_RESTORE_FAILED", std::string("{\"job_id\":\"") +
                json_escape(g.job_id) + "\",\"reason\":\"parse_failed\",\"snapshot_path\":\"" +
                json_escape(g.mf_snapshot_in_path) + "\"}");
        }
      } else {
        emit_engine_event("MF_RESTORE_FAILED", std::string("{\"job_id\":\"") +
              json_escape(g.job_id) + "\",\"reason\":\"read_failed\",\"snapshot_path\":\"" +
              json_escape(g.mf_snapshot_in_path) + "\"}");
      }
    }
    if (const char * env = std::getenv("KORITH_MAX_TOKENS")) {
      if (env[0] != '\0') {
        char * end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end && end != env && *end == '\0' && parsed > 0) {
          g.max_tokens = static_cast<std::uint64_t>(parsed);
        }
      }
    }

    const char * prompt_file_env = std::getenv("KORITH_PROMPT_FILE");
    const char * prompt_env = std::getenv("KORITH_PROMPT");
    std::vector<llama_token> prompt_tokens;
    std::string prompt_file_text;
    std::string prompt_text;
    std::string canonical_prompt_text;
    bool prompt_loaded = false;
    if (prompt_file_env && prompt_file_env[0] != '\0') {
      if (load_text_file(prompt_file_env, prompt_file_text)) {
        prompt_text = prompt_file_text;
        prompt_loaded = true;
      } else {
        std::fprintf(stderr, "[PROMPT_LOAD_FAILED] path=%s reason=read_failed\n", prompt_file_env);
        (void) std::fflush(stderr);
        emit_engine_event("PROMPT_LOAD_FAILED", std::string("{\"path\":\"") +
              json_escape(prompt_file_env) + "\",\"reason\":\"read_failed\"}");
      }
    }
    if (!prompt_loaded && prompt_env && prompt_env[0] != '\0') {
      prompt_text = prompt_env;
      prompt_loaded = true;
    }
    if (prompt_loaded && !prompt_text.empty()) {
      const PromptCanonicalizationResult canon = canonicalize_prompt_for_amf(prompt_text);
      canonical_prompt_text = canon.text;
      if (canon.whitespace_changed || canon.unicode_changed) {
        std::string changes;
        if (canon.whitespace_changed) {
          changes = "whitespace";
        }
        if (canon.unicode_changed) {
          if (!changes.empty()) {
            changes += ",";
          }
          changes += "unicode";
        }
        std::fprintf(stderr,
                     "[AMF_CANONICALIZE] changes=%s original_len=%zu canonical_len=%zu\n",
                     changes.c_str(),
                     prompt_text.size(),
                     canon.text.size());
        (void) std::fflush(stderr);
      }
      prompt_tokens = tokenize_prompt(g.vocab_target, canon.text.c_str());
    }

    if (!prompt_tokens.empty()) {
      korith::core::AmfContext amf_ctx_request = g.amf_ctx;
      std::vector<llama_token> amf_lookup_tokens = prompt_tokens;
      bool amf_prefix_extracted = false;
      if (env_truthy("KORITH_AMF_SHARED_PREFIXES", false) &&
          prompt_matches_shared_patterns(canonical_prompt_text)) {
        amf_ctx_request.tenant_hash = korith::core::amf_hash_tenant_id("__shared__");
      }
      if (env_truthy("KORITH_AMF_PREFIX_EXTRACTION", false)) {
        const char * boundary_raw = std::getenv("KORITH_AMF_PREFIX_BOUNDARY_TOKEN");
        std::string boundary = boundary_raw ? boundary_raw : "";
        if (boundary.empty()) {
          boundary = "<|user|>";
        }
        const std::vector<llama_token> boundary_tokens =
            tokenize_prompt(g.vocab_target, boundary.c_str());
        const std::size_t boundary_at = find_last_subsequence(prompt_tokens, boundary_tokens);
        if (boundary_at != static_cast<std::size_t>(-1)) {
          amf_prefix_extracted = true;
          amf_lookup_tokens.assign(
              prompt_tokens.begin(),
              prompt_tokens.begin() + static_cast<std::ptrdiff_t>(boundary_at));
          std::fprintf(stderr,
                       "[AMF_PREFIX] total_tokens=%zu prefix_tokens=%zu boundary=\"%s\"\n",
                       prompt_tokens.size(),
                       amf_lookup_tokens.size(),
                       boundary.c_str());
          (void) std::fflush(stderr);
        }
      }
      g.last_prompt_tokens = prompt_tokens.size();
      g.last_amf_decision = g.amf_ready ? "miss" : "unavailable";
      if (!g.amf_ready) {
        emit_engine_event("AMF_BLOCK", "{\"reason\":\"unavailable\"}");
      }
      llama_memory_t mem_target = llama_get_memory(g.ctx_target);
      if (!mem_target) {
        shutdown_unlocked();
        return false;
      }
      llama_memory_clear(mem_target, /* data = */ true);

      if (g.ctx_draft) {
        if (llama_memory_t mem_draft = llama_get_memory(g.ctx_draft)) {
          llama_memory_clear(mem_draft, /* data = */ true);
        }
      }

      g.pos_target = 0;
      g.pos_draft = 0;

      bool used_amf = false;
      bool amf_fallback = false;
      bool hit_attempted = false;
      korith::core::AmfLookupResult lookup = korith::core::AmfLookupResult::kMissLoadFailed;
      korith::core::AmfEntry hit_entry{};
      std::vector<llama_token> hit_tokens;
      const char * miss_reason = nullptr;
      double prompt_decode_ms = 0.0;
      std::uint64_t admit_saved_ms = 0;
      const std::uint64_t now_ms = now_epoch_ms();

      if (g.amf_replay_disabled_until_ms > 0 && now_ms >= g.amf_replay_disabled_until_ms) {
        if ((g.amf_replay_disable_mask & 0x1u) != 0u) {
          g.amf_enable_count += 1;
        }
        g.amf_replay_disabled_until_ms = 0;
        g.amf_replay_disable_mask = 0;
      }

      if (g.amf_ready) {
        g.amf.evict_expired_entries(now_ms);
        bool amf_disable_restore = false;
        if (const char * env = std::getenv("KORITH_AMF_DISABLE_RESTORE")) {
          amf_disable_restore = (env[0] != '\0') && (env[0] != '0');
        }
        if ((g.amf_replay_disable_mask & 0x1u) != 0u &&
            (g.amf_replay_disabled_until_ms == 0 || now_ms < g.amf_replay_disabled_until_ms)) {
          g.amf.note_block();
          std::fprintf(stderr, "[AMF_REPLAY_BLOCKED] reason=memory_field_cooldown\n");
          (void) std::fflush(stderr);
          miss_reason = "memory_field_cooldown";
          g.last_amf_decision = "blocked";
          emit_engine_event("AMF_BLOCK", "{\"reason\":\"memory_field_cooldown\"}");
          goto amf_skip_lookup;
        }
        if (g.amf_roi_disabled) {
          g.amf.note_block();
          static bool roi_block_logged = false;
          if (!roi_block_logged) {
            std::fprintf(stderr, "[AMF_REPLAY_BLOCKED] reason=negative_roi\n");
            (void) std::fflush(stderr);
            roi_block_logged = true;
          }
          miss_reason = "negative_roi";
          g.last_amf_decision = "blocked";
          emit_engine_event("AMF_BLOCK", "{\"reason\":\"negative_roi\"}");
          goto amf_skip_lookup;
        }
        if (!amf_sampling_is_deterministic()) {
          g.amf.note_miss();
          g.amf.note_block();
          std::fprintf(stderr, "[AMF_REPLAY_BLOCKED] reason=non_deterministic\n");
          (void) std::fflush(stderr);
          miss_reason = "non_deterministic";
          g.last_amf_decision = "blocked";
          emit_engine_event("AMF_BLOCK", "{\"reason\":\"non_deterministic\"}");
        } else {
          const std::uint64_t prompt_hash = korith::core::amf_hash_tokens(amf_lookup_tokens);
          lookup = g.amf.find_longest_prefix(amf_ctx_request, amf_lookup_tokens, &hit_entry, &hit_tokens);
          std::fprintf(stderr,
                       "[AMF_KEY] result=%s model_hash=%llx tenant_hash=%llx prompt_hash=%llx n_ctx=%u n_batch=%u "
                       "rope_base=0x%08x rope_scale=0x%08x sampling_hash=%llx seed=%llx\n",
                       korith::core::amf_lookup_reason_name(lookup),
                       static_cast<unsigned long long>(g.amf_ctx.model_hash),
                       static_cast<unsigned long long>(amf_ctx_request.tenant_hash),
                       static_cast<unsigned long long>(prompt_hash),
                       g.amf_ctx.n_ctx,
                       static_cast<unsigned int>(llama_n_batch(g.ctx_target)),
                       g.amf_ctx.rope_base_bits,
                       g.amf_ctx.rope_scale_bits,
                       static_cast<unsigned long long>(g.amf_ctx.sampling_hash),
                       static_cast<unsigned long long>(g.amf_ctx.rng_hash));
          (void) std::fflush(stderr);
          emit_engine_event("AMF_LOOKUP", std::string("{\"result\":\"") +
                json_escape(korith::core::amf_lookup_reason_name(lookup)) + "\"}");
          if (amf_disable_restore) {
            g.amf.note_miss();
            g.amf.note_block();
            std::fprintf(stderr, "[AMF_REPLAY_BLOCKED] reason=env_restore_disable\n");
            (void) std::fflush(stderr);
            miss_reason = "env_restore_disable";
            g.last_amf_decision = "blocked";
            emit_engine_event("AMF_BLOCK", "{\"reason\":\"env_restore_disable\"}");
            goto amf_skip_lookup;
          }
          if (lookup == korith::core::AmfLookupResult::kHit) {
            hit_attempted = true;
            const double min_roi = g.amf.min_roi();
            if (min_roi > 0.0 && std::isfinite(hit_entry.ema_roi)) {
              if (hit_entry.ema_roi < min_roi) {
                g.amf.note_miss();
                g.amf.note_block(hit_entry);
                std::fprintf(stderr, "[AMF_REPLAY_BLOCKED] reason=low_roi\n");
                (void) std::fflush(stderr);
                miss_reason = "low_roi";
                g.last_amf_decision = "blocked";
                emit_engine_event("AMF_BLOCK", "{\"reason\":\"low_roi\"}");
                goto amf_skip_lookup;
              }
            }
            const auto load_t0 = std::chrono::steady_clock::now();
            double suffix_ms = 0.0;
            // ── Restore path: direct-GPU (fast) or legacy llama_state_set_data ──
            bool restore_ok = false;
            if (g.amf_direct_gpu_enabled && g.amf_direct_kv_ctx) {
              // Peek at the blob header to decide which format it is.
              std::vector<std::uint8_t> peek;
              if (g.amf.load_kv(hit_entry, &peek)) {
                if (korith::core::amf_direct_kv_is_direct_format(peek.data(), peek.size())) {
                  // Direct-GPU path: write blob back to a temp entry, then restore.
                  // Since load_kv already read the data, we pass it via a temporary
                  // store operation.  For the direct path the blob IS the snapshot.
                  restore_ok = korith::core::amf_direct_kv_restore(
                      g.amf_direct_kv_ctx, g.ctx_target, hit_entry, g.amf);
                } else {
                  // Legacy blob — fall through to llama_state_set_data below.
                  const std::size_t got =
                      llama_state_set_data(g.ctx_target, peek.data(), peek.size());
                  restore_ok = (got == peek.size());
                }
              }
            } else {
              std::vector<std::uint8_t> kv_blob;
              if (g.amf.load_kv(hit_entry, &kv_blob)) {
                const std::size_t got =
                    llama_state_set_data(g.ctx_target, kv_blob.data(), kv_blob.size());
                restore_ok = (got == kv_blob.size());
              }
            }
            if (restore_ok) {
                const auto load_t1 = std::chrono::steady_clock::now();
                const double load_ms =
                    std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();
                g.pos_target = static_cast<llama_pos>(hit_entry.prefix_len);
                const bool rng_ok = amf_restore_rng_state();
                g.logits_target = llama_get_logits(g.ctx_target);
                if (g.logits_target && rng_ok) {
                  used_amf = true;
                  const std::size_t prefix_len = static_cast<std::size_t>(hit_entry.prefix_len);
                  if (prefix_len < prompt_tokens.size()) {
                    const std::vector<llama_token> suffix(prompt_tokens.begin() + prefix_len,
                                                          prompt_tokens.end());
                    const auto suffix_t0 = std::chrono::steady_clock::now();
                    if (!decode_tokens(g.ctx_target, g.batch_target, g.pos_target, suffix)) {
                      used_amf = false;
                      amf_fallback = true;
                    }
                    const auto suffix_t1 = std::chrono::steady_clock::now();
                    suffix_ms = std::chrono::duration<double, std::milli>(suffix_t1 - suffix_t0).count();
                  }
                  if (used_amf) {
                    const double restore_ms = load_ms + suffix_ms;
                    const bool baseline_stable =
                        (g.baseline_prompt_samples >= kBaselinePromptMinSamples);
                    double baseline_prompt_tps = 0.0;
                    if (baseline_stable && g.baseline_prompt_ms_ema > 0.0 &&
                        std::isfinite(g.baseline_prompt_ms_ema) &&
                        !prompt_tokens.empty()) {
                      baseline_prompt_tps = static_cast<double>(prompt_tokens.size()) /
                          g.baseline_prompt_ms_ema;
                    }
                    double baseline_prefix_ms = 0.0;
                    if (baseline_prompt_tps > 0.0) {
                      baseline_prefix_ms = static_cast<double>(hit_entry.prefix_len) /
                          baseline_prompt_tps;
                    }
                    double saved_ms = 0.0;
                    if (baseline_stable && baseline_prefix_ms > restore_ms &&
                        std::isfinite(baseline_prefix_ms) && std::isfinite(restore_ms)) {
                      saved_ms = baseline_prefix_ms - restore_ms;
                    }
                    g.amf.note_hit(
                        hit_entry,
                        hit_entry.prefix_len,
                        static_cast<std::uint64_t>(restore_ms),
                        static_cast<std::uint64_t>(saved_ms));
                    if (g.amf_ready) {
                      const korith::core::AmfStats & stats = g.amf.stats();
                      double avg_roi_ema = std::numeric_limits<double>::quiet_NaN();
                      if (std::isfinite(stats.restore_ms_ema) && stats.restore_ms_ema > 0.0 &&
                          std::isfinite(stats.saved_ms_ema)) {
                        avg_roi_ema = stats.saved_ms_ema / stats.restore_ms_ema;
                        if (!std::isfinite(avg_roi_ema)) {
                          avg_roi_ema = std::numeric_limits<double>::quiet_NaN();
                        }
                      }
                      if (std::isfinite(avg_roi_ema)) {
                        if (std::isfinite(g.amf_last_avg_roi_ema)) {
                          const double delta = avg_roi_ema - g.amf_last_avg_roi_ema;
                          g.amf_roi_slope_ema = update_ema(g.amf_roi_slope_ema, delta, 0.1);
                        }
                        g.amf_last_avg_roi_ema = avg_roi_ema;
                      }
                    }
                    double roi = 0.0;
                    if (baseline_stable && restore_ms > 0.0 && saved_ms > 0.0) {
                      roi = saved_ms / restore_ms;
                    }
                    const std::size_t total_prompt_tokens = prompt_tokens.size();
                    const std::size_t skipped_tokens = static_cast<std::size_t>(hit_entry.prefix_len);
                    double skip_ratio = 0.0;
                    if (total_prompt_tokens > 0) {
                      skip_ratio = static_cast<double>(skipped_tokens) /
                          static_cast<double>(total_prompt_tokens);
                    }
                    skip_ratio = std::clamp(skip_ratio, 0.0, 1.0);
                    g.last_amf_decision = "hit";
                    g.last_prefix_len = hit_entry.prefix_len;
                    g.last_skipped_tokens = static_cast<std::uint32_t>(skipped_tokens);
                    g.last_skip_ratio = skip_ratio;
                    g.last_restore_ms = restore_ms;
                    g.last_baseline_prefix_ms = baseline_prefix_ms;
                    g.last_saved_ms = saved_ms;
                    g.last_roi = roi;
                    admit_saved_ms = static_cast<std::uint64_t>(std::max(0.0, saved_ms));
                    g.amf_prompt_tokens_total += static_cast<std::uint64_t>(total_prompt_tokens);
                    g.amf_skipped_tokens_total += static_cast<std::uint64_t>(skipped_tokens);
                    if (baseline_stable && restore_ms > 0.0) {
                      g.amf_roi_sum += roi;
                      g.amf_roi_count += 1;
                    }
                    if (baseline_stable) {
                      if (roi < 1.0) {
                        g.amf_negative_roi_hits += 1;
                        if (g.amf_negative_roi_hits >= kAmfNegativeRoiMax) {
                          if (!g.amf_roi_disabled) {
                            g.amf_roi_disabled = true;
                            g.amf_disable_count += 1;
                          }
                          std::fprintf(stderr, "[AMF_REPLAY_BLOCKED] reason=negative_roi\n");
                          (void) std::fflush(stderr);
                        }
                      } else {
                        g.amf_negative_roi_hits = 0;
                      }
                    }
                    if (!baseline_stable) {
                      std::fprintf(stderr,
                                   "[AMF_SKIP] reason=baseline_unstable prompt_tokens=%u "
                                   "skipped_tokens=%u skip_ratio=%.3f restore_ms=%.2f saved_ms=%.2f\n",
                                   static_cast<unsigned>(total_prompt_tokens),
                                   static_cast<unsigned>(skipped_tokens),
                                   skip_ratio,
                                   restore_ms,
                                   saved_ms);
                    } else if (restore_ms > 0.0) {
                      std::fprintf(stderr,
                                   "[AMF_SKIP] prompt_tokens=%u skipped_tokens=%u skip_ratio=%.3f "
                                   "restore_ms=%.2f saved_ms=%.2f roi=%.2f\n",
                                   static_cast<unsigned>(total_prompt_tokens),
                                   static_cast<unsigned>(skipped_tokens),
                                   skip_ratio,
                                   restore_ms,
                                   saved_ms,
                                   roi);
                    } else {
                      std::fprintf(stderr,
                                   "[AMF_SKIP] prompt_tokens=%u skipped_tokens=%u skip_ratio=%.3f "
                                   "restore_ms=%.2f saved_ms=%.2f\n",
                                   static_cast<unsigned>(total_prompt_tokens),
                                   static_cast<unsigned>(skipped_tokens),
                                   skip_ratio,
                                   restore_ms,
                                   saved_ms);
                    }
                    if (amf_prefix_extracted &&
                        static_cast<std::size_t>(hit_entry.prefix_len) < amf_lookup_tokens.size()) {
                      const std::size_t cached_prefix = static_cast<std::size_t>(hit_entry.prefix_len);
                      const std::size_t full_prefix = amf_lookup_tokens.size();
                      const std::size_t compute_tokens = full_prefix - cached_prefix;
                      std::fprintf(stderr,
                                   "[AMF_PARTIAL_HIT] full_prefix=%zu cached_prefix=%zu compute_tokens=%zu\n",
                                   full_prefix,
                                   cached_prefix,
                                   compute_tokens);
                    }
                    std::fprintf(stderr,
                                 "[AMF_HIT] prefix_tokens=%u skipped_tokens=%u baseline_ms=%.2f "
                                 "restore_ms=%.2f saved_ms=%.2f roi=%.2f\n",
                                 hit_entry.prefix_len,
                                 hit_entry.prefix_len,
                                 baseline_prefix_ms,
                                 restore_ms,
                                 saved_ms,
                                 roi);
                    std::fprintf(stderr, "[AMF_REPLAY_OK] prefix_len=%u\n", hit_entry.prefix_len);
                    (void) std::fflush(stderr);
                    std::ostringstream payload;
                    payload << "{"
                            << "\"prefix_len\":" << hit_entry.prefix_len << ","
                            << "\"skipped_tokens\":" << skipped_tokens << ","
                            << "\"skip_ratio\":" << skip_ratio << ","
                            << "\"restore_ms\":" << restore_ms << ","
                            << "\"saved_ms\":" << saved_ms << ","
                            << "\"roi\":" << roi
                            << "}";
                    emit_engine_event("AMF_HIT", payload.str());
                  }
                } else {
                  if (!rng_ok) {
                    std::fprintf(stderr, "[AMF_REPLAY_BLOCKED] reason=rng_restore_failed\n");
                    (void) std::fflush(stderr);
                    g.last_amf_decision = "blocked";
                    emit_engine_event("AMF_BLOCK", "{\"reason\":\"rng_restore_failed\"}");
                  }
                  amf_fallback = true;
                }
              } else {
                amf_fallback = true;
              }
          } else {
            g.amf.note_miss();
            const char * reason = korith::core::amf_lookup_reason_name(lookup);
            miss_reason = reason;
            g.last_amf_decision = "miss";
            std::string payload = std::string("{\"reason\":\"") + json_escape(reason) + "\"}";
            emit_engine_event("AMF_MISS", payload);
          }
        }
      }
amf_skip_lookup:

      if (amf_fallback) {
        llama_memory_clear(mem_target, /* data = */ true);
        g.pos_target = 0;
        g.logits_target = nullptr;
        std::fprintf(stderr, "[AMF_FALLBACK] reason=deserialize_fail\n");
        (void) std::fflush(stderr);
        if (hit_attempted) {
          g.amf.note_failure(hit_entry);
          g.last_amf_decision = "blocked";
          emit_engine_event("AMF_BLOCK", "{\"reason\":\"deserialize_fail\"}");
        }
      }

      if (!used_amf) {
        const auto decode_t0 = std::chrono::steady_clock::now();
        if (!decode_tokens(g.ctx_target, g.batch_target, g.pos_target, prompt_tokens)) {
          shutdown_unlocked();
          return false;
        }
        const auto decode_t1 = std::chrono::steady_clock::now();
        prompt_decode_ms = std::chrono::duration<double, std::milli>(decode_t1 - decode_t0).count();
        admit_saved_ms = static_cast<std::uint64_t>(std::max(0.0, prompt_decode_ms));
        g.baseline_prompt_ms_ema = update_ema(g.baseline_prompt_ms_ema, prompt_decode_ms, 0.2);
        g.baseline_prompt_samples += 1;
        g.last_prompt_decode_ms = prompt_decode_ms;
        if (miss_reason) {
          const double tps = (prompt_decode_ms > 0.0)
              ? (static_cast<double>(prompt_tokens.size()) / (prompt_decode_ms / 1000.0))
              : 0.0;
          std::fprintf(stderr,
                       "[AMF_MISS] reason=%s prompt_ms=%.2f tokens=%zu base_prompt_tps=%.2f\n",
                       miss_reason,
                       prompt_decode_ms,
                       prompt_tokens.size(),
                       tps);
          (void) std::fflush(stderr);
          std::string payload = std::string("{\"reason\":\"") + json_escape(miss_reason) +
                                "\",\"prompt_ms\":" + std::to_string(prompt_decode_ms) + "}";
          emit_engine_event("AMF_MISS", payload);
        }
      }

      g.logits_target = llama_get_logits(g.ctx_target);
      if (!g.logits_target) {
        shutdown_unlocked();
        return false;
      }

      if (g.ctx_draft) {
        if (!decode_tokens(g.ctx_draft, g.batch_draft, g.pos_draft, prompt_tokens)) {
          free_draft_unlocked();
        } else {
          g.logits_draft = llama_get_logits(g.ctx_draft);
        }
      }

      g.token_prefix_hash = 1469598103934665603ull;
      for (llama_token tok : prompt_tokens) {
        g.token_prefix_hash = hash_token_prefix_step(g.token_prefix_hash, tok);
      }

      const std::vector<llama_token> & amf_store_tokens =
          amf_prefix_extracted ? amf_lookup_tokens : prompt_tokens;
      const bool amf_prefix_extended =
          used_amf &&
          hit_entry.prefix_len > 0 &&
          static_cast<std::size_t>(hit_entry.prefix_len) < amf_store_tokens.size();
      if (!g.amf_ready) {
        std::fprintf(stderr, "[AMF_SKIP] reason=not_ready\n");
        (void) std::fflush(stderr);
      } else if (amf_store_tokens.size() < g.amf.min_tokens()) {
        std::fprintf(stderr,
                     "[AMF_SKIP] reason=too_short tokens=%zu min=%zu "
                     "(set KORITH_AMF_MIN_TOKENS to lower threshold)\n",
                     amf_store_tokens.size(), g.amf.min_tokens());
        (void) std::fflush(stderr);
      } else if (used_amf && !amf_prefix_extended) {
        std::fprintf(stderr, "[AMF_SKIP] reason=full_hit_no_extension\n");
        (void) std::fflush(stderr);
      }
      if (g.amf_ready &&
          amf_store_tokens.size() >= g.amf.min_tokens() &&
          (!used_amf || amf_prefix_extended)) {
        bool stored = false;
        if (g.amf_direct_gpu_enabled && g.amf_direct_kv_ctx) {
          // Direct-GPU save path: bypass llama_state_get_data serialization.
          stored = korith::core::amf_direct_kv_save(
              g.amf_direct_kv_ctx,
              g.ctx_target,
              amf_ctx_request,
              amf_store_tokens,
              g.amf,
              admit_saved_ms);
          if (stored) {
            g.amf.note_store(admit_saved_ms);
          }
        } else {
          // Legacy path: serialize via llama_state_get_data.
          const std::size_t size = llama_state_get_size(g.ctx_target);
          if (size > 0) {
            std::vector<std::uint8_t> kv_blob(size);
            const std::size_t got =
                llama_state_get_data(g.ctx_target, kv_blob.data(), kv_blob.size());
            if (got == kv_blob.size()) {
              stored = g.amf.store_entry(
                  amf_ctx_request,
                  amf_store_tokens,
                  kv_blob.data(),
                  kv_blob.size(),
                  admit_saved_ms);
              if (stored) {
                g.amf.note_store(admit_saved_ms);
              }
            }
          }
        }
        if (stored && amf_prefix_extended) {
          std::fprintf(stderr,
                       "[AMF_ADMIT_UPGRADE] old_prefix=%u new_prefix=%zu\n",
                       hit_entry.prefix_len,
                       amf_store_tokens.size());
          (void) std::fflush(stderr);
        }
      }

      const korith::core::AmfStats & stats = g.amf.stats();
      if (g.amf_ready || amf_flag) {
        g.amf_eviction_events = stats.evictions;
        std::fprintf(stderr,
                     "[AMF_STATS] saved_ms=%llu saved_tokens=%llu hits=%llu misses=%llu blocks=%llu failures=%llu "
                     "admit=%llu reject=%llu admit_bytes=%llu evict=%llu evict_bytes=%llu store_fail=%llu "
                     "entries=%llu bytes=%llu\n",
                     static_cast<unsigned long long>(stats.ms_saved),
                     static_cast<unsigned long long>(stats.tokens_saved),
                     static_cast<unsigned long long>(stats.hits),
                     static_cast<unsigned long long>(stats.misses),
                     static_cast<unsigned long long>(stats.blocks),
                     static_cast<unsigned long long>(stats.failures),
                     static_cast<unsigned long long>(stats.admissions),
                     static_cast<unsigned long long>(stats.admission_rejects),
                     static_cast<unsigned long long>(stats.admission_bytes),
                     static_cast<unsigned long long>(stats.evictions),
                     static_cast<unsigned long long>(stats.evicted_bytes),
                     static_cast<unsigned long long>(stats.store_failures),
                     static_cast<unsigned long long>(stats.entries),
                     static_cast<unsigned long long>(stats.bytes));
        (void) std::fflush(stderr);
        const korith::core::AmfStorageStats storage = g.amf.storage_stats(now_ms);
        std::fprintf(stderr,
                     "[AMF_STORAGE] total_bytes=%llu budget_bytes=%llu utilization=%.2f%% hot=%llu warm=%llu cold=%llu "
                     "disk_free_bytes=%llu disk_capacity_bytes=%llu disk_free_pct=%.2f%% low_disk=%d\n",
                     static_cast<unsigned long long>(storage.total_bytes),
                     static_cast<unsigned long long>(storage.budget_bytes),
                     storage.utilization * 100.0,
                     static_cast<unsigned long long>(storage.hot_entries),
                     static_cast<unsigned long long>(storage.warm_entries),
                     static_cast<unsigned long long>(storage.cold_entries),
                     static_cast<unsigned long long>(storage.disk_free_bytes),
                     static_cast<unsigned long long>(storage.disk_capacity_bytes),
                     storage.disk_free_ratio * 100.0,
                     storage.low_disk ? 1 : 0);
        (void) std::fflush(stderr);
        if (storage.low_disk) {
          std::fprintf(stderr,
                       "[AMF_ALERT] type=disk_pressure free_bytes=%llu free_pct=%.2f%%\n",
                       static_cast<unsigned long long>(storage.disk_free_bytes),
                       storage.disk_free_ratio * 100.0);
          (void) std::fflush(stderr);
        }
      }

      if (g.amf_ready) {
        const std::uint32_t prefix_len =
            (lookup == korith::core::AmfLookupResult::kHit) ? hit_entry.prefix_len : 0u;
        const korith::core::AmfStore::AmfSignals signals =
            g.amf.signals_for_prefix(prompt_tokens.size(), prefix_len);
        g.amf_reuse_score = signals.reuse_score;
        g.amf_avg_prefix_len = signals.avg_prefix_length;
        g.amf_accept_rate = signals.historical_accept_rate;
        g.amf_restore_cost_ms = signals.restore_cost_ms;
        std::fprintf(stderr,
                     "[AMF_SIGNAL] reuse=%.3f accept=%.3f restore_ms=%.2f\n",
                     g.amf_reuse_score,
                     g.amf_accept_rate,
                     g.amf_restore_cost_ms);
        (void) std::fflush(stderr);
      }

      if (g.amf_ready) {
        korith::core::MemoryFieldInput mf_in{};
        mf_in.now_ms = now_ms;
        const korith::core::AmfStats & stats = g.amf.stats();
        mf_in.hits = stats.hits;
        mf_in.misses = stats.misses;
        mf_in.blocks = stats.blocks;
        mf_in.failures = stats.failures;
        mf_in.tokens_saved = stats.tokens_saved;
        mf_in.ms_saved = stats.ms_saved;
        mf_in.restore_ms = stats.restore_ms;
        mf_in.admissions = stats.admissions;
        mf_in.admission_rejects = stats.admission_rejects;
        mf_in.evictions = stats.evictions;
        mf_in.evicted_bytes = stats.evicted_bytes;
        mf_in.entries = stats.entries;
        const korith::core::AmfStorageStats storage = g.amf.storage_stats(now_ms);
        mf_in.bytes = storage.total_bytes;
        mf_in.max_bytes = storage.budget_bytes;
        mf_in.oldest_entry_age_ms = g.amf.oldest_entry_age_ms(now_ms);
        mf_in.negative_roi_streak = static_cast<std::uint64_t>(g.amf_negative_roi_hits);

        const std::uint64_t mf_prev_update_ms = g.memory_field.last_update_ms;
        const korith::core::MemoryFieldOutput mf_out =
            korith::core::memory_field_update(&g.memory_field, mf_in);
        const bool mf_updated = (g.memory_field.last_update_ms != mf_prev_update_ms);
        if (mf_updated) {
          g.mf_policy_updates += 1;
          const std::uint32_t prev_mask = g.amf_replay_disable_mask;
          g.amf.set_min_admit_roi(mf_out.min_admit_roi);

          if ((mf_out.replay_disable_mask & 0x1u) != 0u) {
            g.amf_replay_disable_mask = mf_out.replay_disable_mask;
            g.amf_replay_disabled_until_ms = now_ms + mf_out.cooldown_ms;
          }
          if ((prev_mask & 0x1u) == 0u &&
              (g.amf_replay_disable_mask & 0x1u) != 0u) {
            g.amf_disable_count += 1;
          }

          std::fprintf(stderr,
                       "[MF_APPLY] min_admit_roi=%.2f evict_pressure=%.2f "
                       "replay_mask=0x%02x cooldown_ms=%llu\n",
                       mf_out.min_admit_roi,
                       mf_out.eviction_pressure,
                       static_cast<unsigned>(mf_out.replay_disable_mask),
                       static_cast<unsigned long long>(mf_out.cooldown_ms));
          (void) std::fflush(stderr);
          std::ostringstream mf_payload;
          mf_payload << "{"
                     << "\"min_admit_roi\":" << mf_out.min_admit_roi << ","
                     << "\"eviction_pressure\":" << mf_out.eviction_pressure << ","
                     << "\"replay_disable_mask\":" << mf_out.replay_disable_mask << ","
                     << "\"cooldown_ms\":" << static_cast<unsigned long long>(mf_out.cooldown_ms)
                     << "}";
          emit_engine_event("MF_APPLY", mf_payload.str());
          if (!g.mf_snapshot_out_path.empty()) {
            const std::string snap = korith::core::memory_field_state_to_json(g.memory_field);
            if (write_text_file(g.mf_snapshot_out_path, snap)) {
              emit_engine_event("MF_SNAPSHOT", std::string("{\"snapshot_path\":\"") +
                  json_escape(g.mf_snapshot_out_path) + "\"}");
            }
          }
        }
      }

      g.ready = true;
      g.finished = false;
      return true;
    }

    // Prime contexts by evaluating BOS so logits are available for the first sampling step.
    const llama_token bos = llama_vocab_bos(g.vocab_target);
    if (bos == LLAMA_TOKEN_NULL) {
      shutdown_unlocked();
      return false;
    }
    if (!decode_one_token(
            g.ctx_target, g.batch_target, /* seq_id = */ 0, /* pos = */ g.pos_target, bos, /* want_logits = */ true)) {
      shutdown_unlocked();
      return false;
    }
    g.pos_target += 1;
    g.token_prefix_hash = hash_token_prefix_step(g.token_prefix_hash, bos);
    g.logits_target = llama_get_logits(g.ctx_target);
    if (!g.logits_target) {
      shutdown_unlocked();
      return false;
    }

    if (g.ctx_draft) {
      if (!decode_one_token(
              g.ctx_draft, g.batch_draft, /* seq_id = */ 0, /* pos = */ g.pos_draft, bos, /* want_logits = */ true)) {
        shutdown_unlocked();
        return false;
      }
      g.pos_draft += 1;
      g.logits_draft = llama_get_logits(g.ctx_draft);
      if (!g.logits_draft) {
        shutdown_unlocked();
        return false;
      }
    }

    g.ready = true;
    g.finished = false;
    return true;
  } catch (...) {
    shutdown_unlocked();
    return false;
  }
}

int32_t engine_step(int32_t batch_tokens) {
  std::lock_guard<std::mutex> lock(g_mu);
  try {
    if (!g.ctx_target || !g.vocab_target || !g.batch_target_inited || !g.ready || !g.logits_target) {
      return -1;
    }

    if (g.finished || batch_tokens <= 0) {
      if (g.finished) {
        log_run_summary_unlocked();
      }
      return 0;
    }
    if (g.max_tokens > 0 && g.printed_total >= g.max_tokens) {
      g.finished = true;
      log_run_summary_unlocked();
      return 0;
    }

    const std::uint64_t spec_proposed_before = g.spec_proposed;
    const std::uint64_t spec_accepted_before = g.spec_accepted;
    const std::uint64_t kv_hits_before = g.kv_hits;
    const std::uint64_t kv_misses_before = g.kv_misses;
    float entropy_before = 0.0f;

    const auto step_t0 = std::chrono::steady_clock::now();
    if (!g.batch_draft_inited) {
      g.logits_draft = nullptr;
    }

    // ── SPEC V2 BYPASS ─────────────────────────────────────────────────────
    // V2 and V3 are mutually exclusive:
    // - mode=v3: V2 is always disabled
    // - mode=v2: run the legacy clean draft-verify path
    {
      if (g.spec_mode == SpecMode::kV2 && g.ctx_draft && g.logits_draft) {
        SpecV2Result v2_res{};
        const int32_t printed = spec_v2_step(
            g.ctx_target, g.batch_target, g.vocab_target, g.n_vocab,
            g.ctx_draft, g.batch_draft,
            g.pos_target, g.pos_draft,
            g.logits_target, g.logits_draft,
            g.finished,
            batch_tokens, g.printed_total, g.max_tokens,
            v2_res);

        g.spec_proposed += v2_res.proposed;
        g.spec_accepted += v2_res.accepted;

        const auto step_t1 = std::chrono::steady_clock::now();
        g.last_step_time = std::chrono::duration_cast<std::chrono::nanoseconds>(step_t1 - step_t0);
        g.last_step_at = step_t1;
        update_tps_unlocked();

        if (g.max_tokens > 0 && g.printed_total >= g.max_tokens) {
          g.finished = true;
        }
        if (g.finished) {
          spec_v2_print_summary();
          log_run_summary_unlocked();
        }
        if (printed > 0) {
          (void) std::fflush(stdout);
        }
        return printed;
      }
    }

    {
      korith::core::Metrics m{};
      m.accept_ema = static_cast<float>(g.accept_ema);
      m.logits = g.logits_target;
      m.n_vocab = g.n_vocab;

      (void) korith::core::thermo_next_depth(m);

      // Continuous scheduler signals.
      float entropy_now = logit_variance_entropy(g.logits_target, g.n_vocab);
      {
        static bool forced_entropy_checked = false;
        static bool forced_entropy_enabled = false;
        static float forced_entropy_value = 0.0f;
        if (!forced_entropy_checked) {
          forced_entropy_checked = true;
          if (const char * env = std::getenv("KORITH_FORCE_ENTROPY")) {
            if (env[0] != '\0') {
              char * end = nullptr;
              const double parsed = std::strtod(env, &end);
              if (end && end != env && std::isfinite(parsed)) {
                forced_entropy_value = static_cast<float>(std::clamp(parsed, 0.0, 1.0));
                forced_entropy_enabled = true;
              }
            }
          }
        }
        if (forced_entropy_enabled) {
          entropy_now = forced_entropy_value;
        }
      }
      entropy_before = entropy_now;
      const float entropy_prev = g.entropy_ema;
      constexpr float kEntropyEmaAlpha = 0.05f;
      if (!std::isfinite(g.entropy_ema) || g.printed_total == 0) {
        g.entropy_ema = entropy_now;
      } else {
        g.entropy_ema = (1.0f - kEntropyEmaAlpha) * g.entropy_ema + kEntropyEmaAlpha * entropy_now;
      }
      std::fprintf(stderr, "[ENTROPY] value=%.4f\n", entropy_now);
      (void) std::fflush(stderr);

      float accept_ratio = static_cast<float>(g.accept_ema);
      if (!std::isfinite(accept_ratio)) {
        accept_ratio = 0.0f;
      }
      accept_ratio = std::clamp(accept_ratio, 0.0f, 1.0f);
      const bool baseline_warmup = (g.baseline_samples < baseline_min_samples());
      const bool baseline_ready_now = !baseline_warmup;
      if (baseline_warmup) {
        accept_ratio = 1.0f;
      }
      korith::core::Context eval_ctx{};
      eval_ctx.opaque = &g;
      eval_ctx.baseline_ready = baseline_ready_now;
      std::vector<korith::core::EngineSignal> engine_signals;
      collect_engine_signals(engine_signals, eval_ctx);

      float engine_costs = 0.0f;
      for (const korith::core::EngineSignal & sig : engine_signals) {
        if (std::isfinite(sig.cost_estimate) && sig.cost_estimate > 0.0f) {
          engine_costs += sig.cost_estimate;
        }
      }

      const RustSchedulerApi & sched_api = rust_scheduler_api();
      const bool scheduler_active = sched_api.active && (sched_api.step != nullptr);

      const int32_t prev_depth = g.spec_depth;
      int32_t desired_depth = g.spec_depth;
      int32_t final_depth = g.spec_depth;
      int32_t batch_size = batch_tokens;
      int32_t throttle_profile = 0;
      bool reuse_kv = false;
      const bool sched_require_baseline_ready = []() {
        const char * env = std::getenv("KORITH_SCHED_REQUIRE_BASELINE_READY");
        return env && env[0] != '\0' && env[0] != '0';
      }();
      const bool sched_baseline_ready = !sched_require_baseline_ready || baseline_ready_now;
      if (scheduler_active) {
        g.rust_sched_state.baseline_ready = sched_baseline_ready ? 1 : 0;
        KorithScheduleInput in{};
        in.acceptance = accept_ratio;
        in.entropy = entropy_now;
        in.current_tps = baseline_warmup ? 0.0f : static_cast<float>(g.tps_delta);
        in.engine_costs = engine_costs;
        in.amf_reuse_score = g.amf_reuse_score;
        in.amf_avg_prefix_length = g.amf_avg_prefix_len;
        in.amf_accept_rate = g.amf_accept_rate;
        in.amf_restore_cost_ms = g.amf_restore_cost_ms;
        in.baseline_ready = sched_baseline_ready ? 1 : 0;

        const KorithScheduleOutput rust_decision = sched_api.step(&g.rust_sched_state, &in);
        desired_depth = rust_decision.desired_depth;
        std::fprintf(stderr, "[SCHED_CALL] returned_depth=%u\n", static_cast<unsigned>(desired_depth));
        (void) std::fflush(stderr);
        final_depth = desired_depth;
        throttle_profile = rust_decision.throttle_flag;
        if (throttle_profile != 0) {
          // Budget profiles from Rust scheduler:
          //   1 = MISS lane (conservative)
          //   2 = HIT lane (balanced)
          //   3 = SPEC_HIT lane (aggressive)
          if (throttle_profile == 1) {
            batch_size = std::max<int32_t>(1, std::min<int32_t>(batch_size, 8));
          } else if (throttle_profile == 2) {
            batch_size = std::max<int32_t>(1, std::min<int32_t>(batch_size, 32));
          } else if (throttle_profile >= 3) {
            batch_size = std::max<int32_t>(1, std::min<int32_t>(batch_size, 96));
          } else {
            batch_size = 1;
          }
          std::fprintf(stderr,
                       "[SCHED_BUDGET] profile=%d batch_size=%d\n",
                       throttle_profile,
                       batch_size);
          (void) std::fflush(stderr);
        }
      }

      const int32_t depth_cap = spec_v3_enabled() ? 16 : 12;
      const int32_t max_depth = std::min<int32_t>(depth_cap, max_spec_depth(g.ctx_target));
      desired_depth = std::clamp(desired_depth, 2, max_depth);
      final_depth = desired_depth;
      const std::uint64_t tokens_now = g.printed_total;
      bool probe_active = false;
      int32_t probe_window = 0;
      constexpr std::uint64_t kProbeInterval = 64;
      constexpr int32_t kProbeDepth = 2;
      constexpr int32_t kProbeWindow = 4;
      if (baseline_ready_now && final_depth <= 1 && max_depth >= kProbeDepth) {
        bool run_probe = (g.probe_attempts == 0 && g.spec_proposed == 0);
        const std::uint64_t since_probe = tokens_now - g.last_probe_tokens;
        if (!run_probe && since_probe >= kProbeInterval) {
          run_probe = true;
        }
        if (run_probe) {
          probe_active = true;
          final_depth = kProbeDepth;
          probe_window = std::min<int32_t>(kProbeWindow, max_depth);
          g.last_probe_tokens = tokens_now;
          g.probe_attempts += 1;
          std::fprintf(stderr, "[SCHED_PROBE] depth=2\n");
          std::fprintf(stderr, "[SPEC] spawned engine window=%d\n", probe_window);
          (void) std::fflush(stderr);
        }
      }
      constexpr std::uint64_t kHysteresisTokens = 32;
      const bool has_spec_data = (g.spec_proposed > 0) || (g.probe_attempts > 0);
      if (!probe_active && final_depth != g.spec_depth) {
        if (final_depth < g.spec_depth) {
          g.last_depth_change_tokens = tokens_now;
        } else {
          if (!has_spec_data) {
            // Bootstrap only: allow the first depth increase before any speculative metrics exist.
            if (g.probe_attempts == 0) {
              g.probe_attempts = 1;
              std::fprintf(stderr, "[SCHED_PROBE] depth=%d\n", final_depth);
              (void) std::fflush(stderr);
            }
            g.last_depth_change_tokens = tokens_now;
          } else {
            const std::uint64_t delta = tokens_now - g.last_depth_change_tokens;
            if (delta < kHysteresisTokens) {
              final_depth = g.spec_depth;
              std::fprintf(stderr, "[SCHED] hysteresis_blocked\n");
              (void) std::fflush(stderr);
            } else {
              g.last_depth_change_tokens = tokens_now;
            }
          }
        }
      } else if (probe_active && final_depth != g.spec_depth) {
        g.last_depth_change_tokens = tokens_now;
      }

      int32_t engine_count_cap = 0;
      const double gain_ema = std::isfinite(g.thermo_gain_ema) ? g.thermo_gain_ema : 0.0;
      if (gain_ema < 0.0) {
        probe_active = false;
        probe_window = 0;
        g.last_depth_change_tokens = tokens_now;
      }
      if (final_depth > prev_depth) {
        engine_count_cap = std::max<int32_t>(1, prev_depth - 1);
      }
      if (final_depth != prev_depth) {
        std::fprintf(stderr,
                     "[THERMO_DEPTH] depth=%d gain_ema=%.3f\n",
                     final_depth,
                     gain_ema);
        (void) std::fflush(stderr);
      }

      const bool divergence = (g.last_step_spec_tokens > 0) && (g.last_step_accept < 0.30);
      const bool entropy_spike = (entropy_now > 0.80f) || ((entropy_now - entropy_prev) > 0.20f);
      const bool entropy_failsafe = !baseline_ready_now && entropy_spike;
      const bool conf_collapse = (g.spec_proposed > 0) && (g.accept_ema < 0.50);
      bool force_collapse = false;
      {
        static bool forced_collapse_checked = false;
        static bool forced_collapse_enabled = false;
        if (!forced_collapse_checked) {
          forced_collapse_checked = true;
          if (const char * env = std::getenv("KORITH_FORCE_COLLAPSE")) {
            forced_collapse_enabled = (env[0] != '\0') && (env[0] != '0');
          }
        }
        force_collapse = forced_collapse_enabled;
      }

      if (divergence || entropy_failsafe || conf_collapse || force_collapse) {
        probe_active = false;
        probe_window = 0;
        g.last_depth_change_tokens = tokens_now;
        std::fprintf(stderr,
                     "[FAILSAFE] divergence=%d entropy_spike=%d conf_collapse=%d forced=%d\n",
                     divergence ? 1 : 0,
                     entropy_spike ? 1 : 0,
                     conf_collapse ? 1 : 0,
                     force_collapse ? 1 : 0);
        (void) std::fflush(stderr);
      }

      korith::core::SpecPlan sched_plan{};
      sched_plan.depth = final_depth;
      int32_t lane_quota = std::max<int32_t>(1, sched_plan.depth);
      const bool replay_local = (g.amf_reuse_score >= 0.70f);
      if (throttle_profile <= 1) {
        lane_quota = std::min<int32_t>(2, std::max<int32_t>(1, lane_quota));
      } else if (throttle_profile == 2) {
        lane_quota = std::min<int32_t>(4, std::max<int32_t>(2, lane_quota));
      } else if (throttle_profile >= 3) {
        lane_quota = std::min<int32_t>(8, std::max<int32_t>(4, lane_quota));
      }
      if (replay_local && lane_quota >= 2 && lane_quota < 8) {
        lane_quota += 1;  // replay-local bonus
      }
      sched_plan.lanes = lane_quota;
      std::fprintf(stderr,
                   "[SCHED_LANES] lanes=%d depth=%d replay_local=%d profile=%d\n",
                   sched_plan.lanes,
                   sched_plan.depth,
                   replay_local ? 1 : 0,
                   throttle_profile);
      (void) std::fflush(stderr);
      sched_plan.accept_min = 0.6f;
      sched_plan.enabled = true;
      sched_plan.reason_code = korith::core::SPEC_DISABLE_UNKNOWN;
      if (max_depth < 2) {
        sched_plan.enabled = false;
        sched_plan.reason_code = korith::core::SPEC_DISABLE_CONTEXT_LIMIT;
      }
      if (sched_plan.depth < 2) {
        sched_plan.enabled = false;
        sched_plan.reason_code = korith::core::SPEC_DISABLE_DEPTH_LT_2;
      } else if (sched_plan.lanes < 2) {
        sched_plan.enabled = false;
        sched_plan.reason_code = korith::core::SPEC_DISABLE_LANES_LT_2;
      }
      std::fprintf(stderr, "[SCHED_PLAN] depth=%d\n", sched_plan.depth);
      (void) std::fflush(stderr);

      korith::core::SpecControlDecision cp_decision{};
      const bool require_baseline_ready = []() {
        const char * env = std::getenv("KORITH_SPEC_REQUIRE_BASELINE_READY");
        return env && env[0] != '\0' && env[0] != '0';
      }();
      if (!g.ctx_target || !g.vocab_target) {
        cp_decision.hard_disable = true;
      }
      cp_decision.allow_exec = !cp_decision.hard_disable;
      cp_decision.allow_commit = !cp_decision.hard_disable;
      if (require_baseline_ready && !baseline_ready_now) {
        cp_decision.allow_commit = false;
      }
      if (divergence || entropy_failsafe || conf_collapse || force_collapse) {
        cp_decision.allow_commit = false;
      }
      if (gain_ema < 0.0) {
        cp_decision.allow_commit = false;
      }

      auto load_env_overrides = []() -> korith::core::SpecEnvOverrides {
        static bool checked = false;
        static korith::core::SpecEnvOverrides overrides{};
        if (checked) {
          return overrides;
        }
        checked = true;
        if (const char * env = std::getenv("KORITH_SPEC_DEPTH")) {
          if (env[0] != '\0') {
            char * end = nullptr;
            const long parsed = std::strtol(env, &end, 10);
            if (end && end != env && *end == '\0') {
              overrides.has_depth = true;
              overrides.depth = static_cast<int>(parsed);
            }
          }
        }
        if (const char * env = std::getenv("KORITH_SPEC_ALLOW_EXEC")) {
          overrides.has_allow_exec = true;
          overrides.allow_exec = (env[0] != '\0') && (env[0] != '0');
        }
        if (const char * env = std::getenv("KORITH_SPEC_ALLOW_COMMIT")) {
          overrides.has_allow_commit = true;
          overrides.allow_commit = (env[0] != '\0') && (env[0] != '0');
        }
        return overrides;
      };

      korith::core::SpecEnvOverrides env_overrides = load_env_overrides();
      if (env_overrides.has_depth) {
        env_overrides.depth = std::clamp(env_overrides.depth, 1, max_depth);
      }
      korith::core::SpecPlan plan = korith::core::resolve_spec_plan(sched_plan, cp_decision, env_overrides);
      const bool spec_paused = []() {
        const char * env = std::getenv("KORITH_SPEC_PAUSE");
        return env && env[0] != '\0' && env[0] != '0';
      }();
      if (spec_paused) {
        plan.enabled = false;
        plan.allow_exec = false;
        plan.allow_commit = false;
        plan.depth = 1;
        plan.lanes = 1;
        plan.reason_code = korith::core::SPEC_DISABLE_CP_GATE_OFF;
        static bool logged = false;
        if (!logged) {
          std::fprintf(stderr, "[SPEC_PAUSED] reason=env\n");
          (void) std::fflush(stderr);
          logged = true;
        }
      }
      if (!g.ctx_draft) {
        plan.enabled = false;
        plan.allow_exec = false;
        plan.allow_commit = false;
        plan.depth = 1;
        plan.lanes = 1;
        plan.reason_code = korith::core::SPEC_DISABLE_BASELINE_ONLY;
      }

      std::fprintf(stderr,
                   "[SPEC_PLAN_RESOLVED] depth=%d exec=%d commit=%d\n",
                   plan.depth,
                   plan.allow_exec ? 1 : 0,
                   plan.allow_commit ? 1 : 0);
      (void) std::fflush(stderr);
      std::fprintf(stderr, "[SCHED_DECISION] depth=%d lanes=%d\n", plan.depth, plan.lanes);
      (void) std::fflush(stderr);

      const char * cp_reason = "ok";
      if (plan.hard_disable) {
        cp_reason = "invalid_state";
      } else if (!plan.allow_exec) {
        cp_reason = korith::core::spec_disable_reason_name(plan.reason_code);
      } else if (!plan.allow_commit) {
        if (divergence || entropy_failsafe || conf_collapse || force_collapse) {
          cp_reason = "failsafe";
        } else if (gain_ema < 0.0) {
          cp_reason = "thermo_gain";
        } else if (require_baseline_ready && !baseline_ready_now) {
          cp_reason = "baseline_gate";
        } else {
          cp_reason = "commit_blocked";
        }
      }
      std::fprintf(stderr,
                   "[CP_DECISION] allow_exec=%d allow_commit=%d hard_disable=%d reason=%s\n",
                   plan.allow_exec ? 1 : 0,
                   plan.allow_commit ? 1 : 0,
                   plan.hard_disable ? 1 : 0,
                   cp_reason);
      (void) std::fflush(stderr);
      g.last_spec_allow_exec = plan.allow_exec;
      g.last_spec_allow_commit = plan.allow_commit;
      g.last_spec_plan_depth = std::max<int32_t>(1, plan.depth);
      g.last_spec_disable_reason = cp_reason ? std::string(cp_reason) : std::string("unknown");
      const bool spec_path_now = plan.allow_exec && (g.ctx_draft != nullptr) && (plan.depth > 1);
      if (!g.cuda_graph_path_valid || g.cuda_graph_spec_path != spec_path_now) {
        korith::core::cuda_graph_decode_invalidate_context(g.ctx_target);
        korith::core::cuda_graph_decode_invalidate_context(g.ctx_draft);
        g.cuda_graph_spec_path = spec_path_now;
        g.cuda_graph_path_valid = true;
        std::fprintf(stderr,
                     "[CUDA_GRAPH] invalidate reason=path_switch spec_path=%d\n",
                     spec_path_now ? 1 : 0);
        (void) std::fflush(stderr);
      }
      {
        std::ostringstream cp_payload;
        cp_payload << "{"
                   << "\"allow_exec\":" << (plan.allow_exec ? 1 : 0) << ","
                   << "\"allow_commit\":" << (plan.allow_commit ? 1 : 0) << ","
                   << "\"hard_disable\":" << (plan.hard_disable ? 1 : 0) << ","
                   << "\"reason\":\"" << json_escape(cp_reason) << "\""
                   << "}";
        emit_engine_event("CP_DECISION", cp_payload.str());
      }

      g.spec_depth = std::max<int32_t>(1, plan.depth);
      g.rust_sched_state.speculative_depth = g.spec_depth;

      const bool kv_allowed = (llama_n_seq_max(g.ctx_target) >= 2) && (llama_get_memory(g.ctx_target) != nullptr);
      reuse_kv = kv_allowed && (g.spec_depth > 1 || probe_active);

      CollapseExecState exec{};
      exec.engine = &g;
      exec.batch_size = batch_size;
      exec.reuse_kv = reuse_kv;
      exec.scheduler_depth = plan.depth;
      exec.plan = plan;
      exec.force_speculation = probe_active;
      exec.spec_window_limit = probe_window;
      exec.engine_count_cap = engine_count_cap;
      exec.spec_proposed_before = spec_proposed_before;
      exec.spec_accepted_before = spec_accepted_before;

      korith::core::Context exec_ctx{};
      exec_ctx.opaque = &exec;
      exec_ctx.baseline_ready = baseline_ready_now;

      korith::core::CollapseCallbacks callbacks{};
      callbacks.commit_speculative = commit_speculative_cb;
      callbacks.fallback_decode = fallback_decode_cb;

      korith::core::CollapseController collapse;
      collapse.execute(exec_ctx, engine_signals.data(), engine_signals.size(), plan.depth, callbacks);

      if (!exec.executed) {
        std::fprintf(stderr, "[DECODE] committed=false depth=1\n");
        (void) std::fflush(stderr);
        g.finished = true;
        return -1;
      }

      const int32_t printed_this_call = exec.printed_this_call;
      if (printed_this_call < 0) {
        g.finished = true;
        return -1;
      }

      if (printed_this_call > 0) {
        (void) std::fflush(stdout);
      }
      if (g.max_tokens > 0 && g.printed_total >= g.max_tokens) {
        g.finished = true;
      }

      if (printed_this_call > 0) {
        const std::size_t bucket = static_cast<std::size_t>(std::clamp<int32_t>(g.spec_depth, 0, 32));
        g.speculative_depth_histogram[bucket] += static_cast<std::uint64_t>(printed_this_call);
        g.control_tokens_total += static_cast<std::uint64_t>(printed_this_call);
        (void) scheduler_active;
      }

      const auto step_t1 = std::chrono::steady_clock::now();
      g.last_step_time = std::chrono::duration_cast<std::chrono::nanoseconds>(step_t1 - step_t0);
      g.last_step_at = step_t1;
      update_tps_unlocked();

      const std::uint64_t spec_proposed_delta = exec.metrics.speculative_tokens;
      const std::uint64_t spec_accepted_delta = exec.metrics.accepted_tokens;
      g.spec_elapsed_ns_total += exec.spec_elapsed_ns;
      g.spec_draft_ns_total += exec.spec_draft_ns;
      g.spec_verify_ns_total += exec.spec_verify_ns;
      g.spec_accept_scan_ns_total += exec.spec_accept_scan_ns;
      g.spec_skipped_tokens_total += exec.spec_skipped_tokens;
      g.spec_fast_verified_tokens_total += exec.spec_fast_verified_tokens;

      // Compute per-step decode savings and overhead metrics.
      // spec_overhead_ms: measured speculation overhead for this step.
      const double step_draft_ms = static_cast<double>(exec.spec_draft_ns) * 1e-6;
      const double step_verify_ms = static_cast<double>(exec.spec_verify_ns) * 1e-6;
      const double step_accept_scan_ms = static_cast<double>(exec.spec_accept_scan_ns) * 1e-6;
      const double step_spec_overhead_ms = step_draft_ms + step_verify_ms + step_accept_scan_ms;
      g.spec_overhead_ms_total += step_spec_overhead_ms;
      if (std::isfinite(exec.spec_effective_ms)) {
        constexpr double kSpecEffectiveAlpha = 0.2;
        if (!std::isfinite(g.spec_effective_ema)) {
          g.spec_effective_ema = exec.spec_effective_ms;
        } else {
          g.spec_effective_ema =
              g.spec_effective_ema + kSpecEffectiveAlpha * (exec.spec_effective_ms - g.spec_effective_ema);
        }
      }

      if (spec_proposed_delta > 0) {
        const double baseline_decode_ms =
            (std::isfinite(g.baseline_tps_ema) && g.baseline_tps_ema > 1e-6)
                ? (1000.0 / g.baseline_tps_ema)
                : std::numeric_limits<double>::quiet_NaN();
        std::fprintf(stderr,
                     "[SPEC_OVERHEAD] draft_ms=%.3f verify_ms=%.3f accept_scan_ms=%.3f overhead_ms=%.3f "
                     "baseline_decode_ms=%.3f accepted=%llu effective_ms=%.3f effective_ema=%.3f\n",
                     step_draft_ms,
                     step_verify_ms,
                     step_accept_scan_ms,
                     step_spec_overhead_ms,
                     baseline_decode_ms,
                     static_cast<unsigned long long>(spec_accepted_delta),
                     exec.spec_effective_ms,
                     g.spec_effective_ema);
        (void) std::fflush(stderr);
      }
      // decode_ms_saved: estimated decode ms saved = (accepted_tokens - 1) * baseline_ms_per_token.
      // Count this only after baseline warmup has converged to keep savings comparable.
      if (baseline_ready && spec_accepted_delta > 0 && std::isfinite(g.baseline_tps_ema) && g.baseline_tps_ema > 1e-6) {
        const double baseline_ms_per_token = 1000.0 / g.baseline_tps_ema;
        // Accepted tokens beyond the first one are "free" — the first token would
        // have been generated by a normal decode step anyway.
        const double tokens_saved = static_cast<double>(spec_accepted_delta > 1 ? spec_accepted_delta - 1 : 0);
        const double step_decode_ms_saved = tokens_saved * baseline_ms_per_token - step_spec_overhead_ms;
        g.decode_ms_saved_total += std::max(0.0, step_decode_ms_saved);
      }

      const std::uint64_t kv_hits_delta =
          (g.kv_hits >= kv_hits_before) ? (g.kv_hits - kv_hits_before) : 0;
      const std::uint64_t kv_misses_delta =
          (g.kv_misses >= kv_misses_before) ? (g.kv_misses - kv_misses_before) : 0;
      const std::uint64_t kv_total = kv_hits_delta + kv_misses_delta;
      const double kv_reuse_ratio =
          (kv_total > 0) ? (static_cast<double>(kv_hits_delta) / static_cast<double>(kv_total)) : 0.0;
      const double flops_saved = static_cast<double>(exec.spec_skipped_tokens);
      const double verification_cost = static_cast<double>(spec_proposed_delta);
      const double thermo_cost = flops_saved - verification_cost;
      float entropy_after = entropy_before;
      if (g.logits_target && g.n_vocab > 0) {
        entropy_after = logit_variance_entropy(g.logits_target, g.n_vocab);
      }
      const double entropy_gain =
          (printed_this_call > 0)
              ? (static_cast<double>(entropy_before - entropy_after) / static_cast<double>(printed_this_call))
              : 0.0;
      g.thermo_flops_saved = flops_saved;
      g.thermo_kv_reuse_ratio = kv_reuse_ratio;
      g.thermo_entropy_gain = entropy_gain;
      g.thermo_cost = thermo_cost;
      constexpr double kThermoGainAlpha = 0.2;
      if (!std::isfinite(g.thermo_gain_ema)) {
        g.thermo_gain_ema = thermo_cost;
      } else {
        g.thermo_gain_ema = g.thermo_gain_ema + kThermoGainAlpha * (thermo_cost - g.thermo_gain_ema);
      }
      std::fprintf(stderr, "[THERMO] saved=%.2f cost=%.2f gain=%.4f\n", flops_saved, thermo_cost, entropy_gain);
      (void) std::fflush(stderr);

      if (probe_active && spec_proposed_delta == 0) {
        std::fprintf(stderr, "[SPEC] ERROR: speculative engine produced no tokens\n");
        (void) std::fflush(stderr);
      }

      if (spec_proposed_delta == 0) {
        if (g.tps_sample_count < 2) {
          g.baseline_last_update = g.last_step_at;
        } else {
        const double dt_s = std::chrono::duration<double>(g.last_step_at - g.baseline_last_update).count();
        if (dt_s > 1e-6) {
          const double tau_s = 1.0;
          const double alpha = std::clamp(1.0 - std::exp(-dt_s / tau_s), 0.0, 1.0);
          if (!std::isfinite(g.baseline_tps_ema)) {
            g.baseline_tps_ema = g.tps_rolling;
          } else {
            g.baseline_tps_ema = g.baseline_tps_ema + alpha * (g.tps_rolling - g.baseline_tps_ema);
          }
          g.baseline_last_update = g.last_step_at;
          g.baseline_samples += 1;
        }

        const double log_dt = std::chrono::duration<double>(g.last_step_at - g.baseline_last_log).count();
        if (log_dt >= 1.0) {
          if (!baseline_ready) {
            const std::uint64_t baseline_target = baseline_min_samples();
            std::fprintf(stderr,
                         "[BASELINE_WARMUP] samples=%llu target=%llu tps_ema=%.2f\n",
                         static_cast<unsigned long long>(g.baseline_samples),
                         static_cast<unsigned long long>(baseline_target),
                         g.baseline_tps_ema);
          } else {
            std::fprintf(stderr, "[BASELINE] tps_ema=%.2f\n", g.baseline_tps_ema);
          }
          (void) std::fflush(stderr);
          g.baseline_last_log = g.last_step_at;
        }
        }
      }

      const std::uint64_t baseline_target = baseline_min_samples();
      if (!baseline_ready && g.baseline_samples >= baseline_target) {
        baseline_ready = true;
        std::fprintf(stderr, "[BASELINE_READY] tps_ema=%.2f\n", g.baseline_tps_ema);
        (void) std::fflush(stderr);
      }

      double tps_delta = 0.0;
      double committed_tps = 0.0;
      if (spec_proposed_delta > 0) {
        const double step_s =
            (exec.spec_elapsed_ns > 0)
                ? (static_cast<double>(exec.spec_elapsed_ns) * 1e-9)
                : std::chrono::duration<double>(g.last_step_time).count();
        committed_tps =
            (step_s > 1e-9) ? (static_cast<double>(spec_accepted_delta) / step_s) : 0.0;
        if (g.baseline_tps_ema > 0.0 && std::isfinite(g.baseline_tps_ema)) {
          tps_delta = (committed_tps - g.baseline_tps_ema) / g.baseline_tps_ema;
        }
        if (exec.spec_skipped_tokens == 0 && tps_delta > 0.0) {
          tps_delta = 0.0;
        }

        double accept_ratio = static_cast<double>(spec_accepted_delta) / static_cast<double>(spec_proposed_delta);
        if (!std::isfinite(accept_ratio)) {
          accept_ratio = 0.0;
        }
        accept_ratio = std::clamp(accept_ratio, 0.0, 1.0);
        constexpr double kAcceptAlpha = 0.2;
        if (!std::isfinite(g.accept_ema)) {
          g.accept_ema = accept_ratio;
        } else {
          g.accept_ema = g.accept_ema + kAcceptAlpha * (accept_ratio - g.accept_ema);
        }
        g.last_step_accept = accept_ratio;
        g.last_step_spec_tokens = spec_proposed_delta;
      } else {
        committed_tps = g.tps_rolling;
        g.last_step_accept = 1.0;
        g.last_step_spec_tokens = 0;
      }
      g.tps_delta = std::isfinite(tps_delta) ? tps_delta : 0.0;
      std::fprintf(stderr,
                   "[KORITH] tps_base=%.2f tps_commit=%.2f delta=%.3f skip=%llu thermo=%.3f\n",
                   g.baseline_tps_ema,
                   committed_tps,
                   g.tps_delta,
                   static_cast<unsigned long long>(exec.spec_skipped_tokens),
                   g.thermo_gain_ema);
      (void) std::fflush(stderr);

      if (printed_this_call > 0 || g.finished) {
        const double accept_ratio = (g.spec_proposed == 0)
                                        ? 0.0
                                        : (static_cast<double>(g.spec_accepted) / static_cast<double>(g.spec_proposed));
        std::fprintf(stderr,
                     "\rTPS %.1f | accept %.2f (ema %.2f) | depth %d   ",
                     g.tps_rolling,
                     accept_ratio,
                     g.accept_ema,
                     g.spec_depth);
        if (g.finished) {
          std::fputc('\n', stderr);
        }
        (void) std::fflush(stderr);
      }

      if (g.finished) {
        log_run_summary_unlocked();
      }

      return printed_this_call;
    }
  } catch (...) {
    g.finished = true;
    return -1;
  }
}

bool engine_set_spec_depth(int32_t depth) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g.ctx_target) {
    return false;
  }

  const uint32_t n_batch_u32 = llama_n_batch(g.ctx_target);
  const int32_t n_batch = static_cast<int32_t>(std::min<uint32_t>(n_batch_u32, 1024u));
  int32_t max_env = spec_v3_enabled() ? 16 : 8;
  if (const char * v = std::getenv("KORITH_MAX_DEPTH")) {
    if (v[0] != '\0') {
      char * end = nullptr;
      const long parsed = std::strtol(v, &end, 10);
      if (end && end != v && *end == '\0' && parsed > 0 && parsed <= INT32_MAX) {
        max_env = static_cast<int32_t>(parsed);
      }
    }
  }
  max_env = std::clamp(max_env, 1, 32);

  const int32_t max_depth = std::min<int32_t>(max_env, std::max<int32_t>(1, std::min<int32_t>(32, n_batch)));

  g.spec_depth = std::clamp(depth, 1, max_depth);
  return true;
}

bool engine_get_spec_stats(engine_spec_stats * out) {
  if (!out) {
    return false;
  }

  std::lock_guard<std::mutex> lock(g_mu);
  if (!g.ctx_target) {
    return false;
  }

  const double ratio = (g.spec_proposed == 0)
                           ? 0.0
                           : (static_cast<double>(g.spec_accepted) / static_cast<double>(g.spec_proposed));
  out->spec_depth = g.spec_depth;
  out->_pad0 = 0;
  out->accept_ratio = ratio;
  out->accept_ema = g.accept_ema;
  out->proposed = g.spec_proposed;
  out->accepted = g.spec_accepted;
  out->decode_ms_saved = g.decode_ms_saved_total;
  out->spec_overhead_ms = g.spec_overhead_ms_total;
  return true;
}

const float * engine_get_logits(void) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g.ctx_target) {
    return nullptr;
  }
  return g.logits_target;
}

void engine_shutdown(void) {
  std::lock_guard<std::mutex> lock(g_mu);
  shutdown_unlocked();
}

}  // extern "C"
