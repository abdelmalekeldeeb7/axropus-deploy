#include "amf_store.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <thread>
#include <unistd.h>

namespace korith::core {

namespace {

// AMF invariants:
// - Replay only allowed under deterministic sampling.
// - ROI must only be computed when baseline is stable.
// - Sustained negative ROI disables replay.
// - Replay remains disabled until MF cooldown expires.
// - AMF must fail closed on corruption or uncertainty.

constexpr std::uint64_t kFNVOffset = 1469598103934665603ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

constexpr std::uint32_t kIndexMagic = 0x31464d41u;  // "AMF1"
constexpr std::uint32_t kIndexVersion = 9u;
constexpr std::uint32_t kIndexVersionV8 = 8u;
constexpr std::uint32_t kRoiDeclineWarn = 3u;
constexpr std::uint32_t kRoiDeclineStop = 5u;

struct IndexHeader {
  std::uint32_t magic = kIndexMagic;
  std::uint32_t version = kIndexVersion;
  std::uint64_t count = 0;
  std::uint64_t checksum = 0;
};

struct IndexRecord {
  AmfKey key{};
  std::uint64_t model_hash = 0;
  std::uint32_t prefix_len = 0;
  std::uint32_t reserved = 0;
  std::uint64_t seen_count = 0;
  std::uint64_t restore_count = 0;
  double ema_saved_ms = std::numeric_limits<double>::quiet_NaN();
  double ema_restore_ms = std::numeric_limits<double>::quiet_NaN();
  double ema_roi = std::numeric_limits<double>::quiet_NaN();
  std::uint64_t failure_count = 0;
  std::uint64_t block_count = 0;
  std::uint64_t saved_ms = 0;
  std::uint64_t hit_count = 0;
  std::uint64_t last_used = 0;
  std::uint64_t last_hit_ms = 0;
  std::uint32_t low_roi_hits = 0;
  std::uint32_t reserved2 = 0;
  std::uint64_t kv_size = 0;
  std::uint64_t last_access_time_ms = 0;
  std::uint64_t access_count = 0;
  std::uint64_t stored_at_ms = 0;
  std::uint64_t expires_at_ms = 0;
  std::uint64_t size_bytes = 0;
  std::uint64_t kv_checksum = 0;
  std::uint8_t state = 0;
  std::uint8_t reserved3[7]{};
};

struct LegacyAmfKeyV8 {
  std::uint64_t model_hash = 0;
  std::uint64_t prefix_hash = 0;
  std::uint32_t n_ctx = 0;
  std::uint32_t kv_version = 0;
  std::uint32_t rope_base_bits = 0;
  std::uint32_t rope_scale_bits = 0;
  std::uint64_t sampling_hash = 0;
  std::uint64_t rng_hash = 0;
};

struct LegacyIndexRecordV8 {
  LegacyAmfKeyV8 key{};
  std::uint64_t model_hash = 0;
  std::uint32_t prefix_len = 0;
  std::uint32_t reserved = 0;
  std::uint64_t seen_count = 0;
  std::uint64_t restore_count = 0;
  double ema_saved_ms = std::numeric_limits<double>::quiet_NaN();
  double ema_restore_ms = std::numeric_limits<double>::quiet_NaN();
  double ema_roi = std::numeric_limits<double>::quiet_NaN();
  std::uint64_t failure_count = 0;
  std::uint64_t block_count = 0;
  std::uint64_t saved_ms = 0;
  std::uint64_t hit_count = 0;
  std::uint64_t last_used = 0;
  std::uint64_t last_hit_ms = 0;
  std::uint32_t low_roi_hits = 0;
  std::uint32_t reserved2 = 0;
  std::uint64_t kv_size = 0;
  std::uint64_t last_access_time_ms = 0;
  std::uint64_t access_count = 0;
  std::uint64_t stored_at_ms = 0;
  std::uint64_t expires_at_ms = 0;
  std::uint64_t size_bytes = 0;
  std::uint64_t kv_checksum = 0;
  std::uint8_t state = 0;
  std::uint8_t reserved3[7]{};
};

std::uint64_t now_epoch_ms() {
  const auto now = std::chrono::system_clock::now();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
}

void hash_bytes(std::uint64_t & h, const std::uint8_t * data, std::size_t len) {
  for (std::size_t i = 0; i < len; ++i) {
    h ^= static_cast<std::uint64_t>(data[i]);
    h *= kFNVPrime;
  }
}

std::uint64_t hash_token_prefix_step(std::uint64_t h, llama_token tok) noexcept {
  h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(tok));
  h *= kFNVPrime;
  return h;
}

bool parse_env_u64(const char * name, std::uint64_t & out) {
  const char * v = std::getenv(name);
  if (!v || v[0] == '\0') {
    return false;
  }
  char * end = nullptr;
  const unsigned long long parsed = std::strtoull(v, &end, 10);
  if (!end || end == v || *end != '\0') {
    return false;
  }
  out = static_cast<std::uint64_t>(parsed);
  return true;
}

bool parse_env_double(const char * name, double & out) {
  const char * v = std::getenv(name);
  if (!v || v[0] == '\0') {
    return false;
  }
  char * end = nullptr;
  const double parsed = std::strtod(v, &end);
  if (!end || end == v || *end != '\0' || !std::isfinite(parsed)) {
    return false;
  }
  out = parsed;
  return true;
}

bool parse_env_bool(const char * name, bool default_value) {
  const char * v = std::getenv(name);
  if (!v || v[0] == '\0') {
    return default_value;
  }
  const std::string s(v);
  if (s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "on") {
    return true;
  }
  if (s == "0" || s == "false" || s == "FALSE" || s == "no" || s == "off") {
    return false;
  }
  return default_value;
}

double update_ema(double current, double sample, double alpha) {
  if (!(alpha > 0.0 && alpha <= 1.0) || !std::isfinite(sample)) {
    return current;
  }
  if (!std::isfinite(current)) {
    return sample;
  }
  return (current * (1.0 - alpha)) + (sample * alpha);
}

}  // namespace

const char * amf_disable_reason_name(AmfDisableReason r) {
  switch (r) {
    case AmfDisableReason::kEnvOff:
      return "env_off";
    case AmfDisableReason::kNoPath:
      return "no_path";
    case AmfDisableReason::kIoError:
      return "io_error";
    case AmfDisableReason::kStateApiMissing:
      return "state_api_missing";
    case AmfDisableReason::kNone:
    default:
      return "none";
  }
}

const char * amf_lookup_reason_name(AmfLookupResult r) {
  switch (r) {
    case AmfLookupResult::kHit:
      return "hit";
    case AmfLookupResult::kMissFirstRun:
      return "first_run";
    case AmfLookupResult::kMissShortPrefix:
      return "short_prefix";
    case AmfLookupResult::kMissHashMismatch:
      return "hash_mismatch";
    case AmfLookupResult::kMissTokenMismatch:
      return "token_mismatch";
    case AmfLookupResult::kMissLoadFailed:
      return "load_failed";
    default:
      return "unknown";
  }
}

std::uint64_t amf_hash_file(const std::string & path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return 0;
  }

  std::uint64_t h = kFNVOffset;
  std::vector<std::uint8_t> buf(1 << 20);
  while (in) {
    in.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
    const std::streamsize got = in.gcount();
    if (got <= 0) {
      break;
    }
    hash_bytes(h, buf.data(), static_cast<std::size_t>(got));
  }
  return h;
}

std::uint64_t amf_hash_tenant_id(const std::string & tenant_id) {
  std::uint64_t h = kFNVOffset;
  if (tenant_id.empty()) {
    return h;
  }
  hash_bytes(h,
             reinterpret_cast<const std::uint8_t *>(tenant_id.data()),
             tenant_id.size());
  return h;
}

std::uint64_t amf_hash_tokens(const std::vector<llama_token> & tokens) {
  return amf_hash_tokens_prefix(tokens, tokens.size());
}

std::uint64_t amf_hash_tokens_prefix(const std::vector<llama_token> & tokens, std::size_t len) {
  const std::size_t max_len = std::min<std::size_t>(len, tokens.size());
  std::uint64_t h = kFNVOffset;
  for (std::size_t i = 0; i < max_len; ++i) {
    h = hash_token_prefix_step(h, tokens[i]);
  }
  return h;
}

std::uint32_t amf_float_bits(float v) {
  std::uint32_t out = 0;
  static_assert(sizeof(out) == sizeof(v), "float size mismatch");
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

AmfKey amf_make_key(const AmfContext & ctx, std::uint64_t prefix_hash) {
  AmfKey key{};
  key.model_hash = ctx.model_hash;
  key.tenant_hash = ctx.tenant_hash;
  key.prefix_hash = prefix_hash;
  key.n_ctx = ctx.n_ctx;
  key.kv_version = ctx.kv_version;
  key.rope_base_bits = ctx.rope_base_bits;
  key.rope_scale_bits = ctx.rope_scale_bits;
  key.sampling_hash = ctx.sampling_hash;
  key.rng_hash = ctx.rng_hash;
  return key;
}

std::size_t AmfStore::KeyHash::operator()(const AmfKey & k) const noexcept {
  std::uint64_t h = kFNVOffset;
  auto mix = [&](std::uint64_t v) {
    h ^= v;
    h *= kFNVPrime;
  };
  mix(k.model_hash);
  mix(k.tenant_hash);
  mix(k.prefix_hash);
  mix(static_cast<std::uint64_t>(k.n_ctx));
  mix(static_cast<std::uint64_t>(k.kv_version));
  mix(static_cast<std::uint64_t>(k.rope_base_bits));
  mix(static_cast<std::uint64_t>(k.rope_scale_bits));
  mix(k.sampling_hash);
  mix(k.rng_hash);
  return static_cast<std::size_t>(h);
}

AmfStore::~AmfStore() {
  shutdown();
}

bool AmfStore::init_from_env() {
  const char * enable = std::getenv("KORITH_ENABLE_AMF");
  if (!enable || enable[0] == '\0' || enable[0] == '0') {
    enabled_ = false;
    disable_reason_ = AmfDisableReason::kEnvOff;
    return false;
  }

  const char * path = std::getenv("KORITH_AMF_PATH");
  std::string resolved = path ? path : "";
  if (resolved.empty()) {
    resolved = "amf";
  }

  std::uint64_t min_tokens = 0;
  if (parse_env_u64("KORITH_AMF_MIN_TOKENS", min_tokens) && min_tokens > 0) {
    min_tokens_ = static_cast<std::size_t>(min_tokens);
  }
  std::uint64_t max_bytes = 0;
  if (parse_env_u64("KORITH_AMF_MAX_BYTES", max_bytes) && max_bytes > 0) {
    max_bytes_ = max_bytes;
    storage_budget_bytes_ = max_bytes;
  }
  double storage_budget_gb = 0.0;
  if (parse_env_double("KORITH_AMF_STORAGE_BUDGET_GB", storage_budget_gb) && storage_budget_gb > 0.0) {
    const long double bytes = static_cast<long double>(storage_budget_gb) * 1024.0L * 1024.0L * 1024.0L;
    if (bytes > 0.0L && bytes <= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
      storage_budget_bytes_ = static_cast<std::uint64_t>(bytes);
      max_bytes_ = storage_budget_bytes_;
    }
  }
  double eviction_watermark = 0.0;
  if (parse_env_double("KORITH_AMF_EVICTION_WATERMARK", eviction_watermark) &&
      eviction_watermark > 0.0 && eviction_watermark <= 1.0) {
    eviction_watermark_ = eviction_watermark;
  }
  std::uint64_t hot_ttl_s = 0;
  if (parse_env_u64("KORITH_AMF_HOT_TTL_S", hot_ttl_s) && hot_ttl_s > 0) {
    hot_ttl_ms_ = hot_ttl_s * 1000ull;
  }
  std::uint64_t warm_ttl_s = 0;
  if (parse_env_u64("KORITH_AMF_WARM_TTL_S", warm_ttl_s) && warm_ttl_s > 0) {
    warm_ttl_ms_ = warm_ttl_s * 1000ull;
  }
  if (hot_ttl_ms_ > warm_ttl_ms_) {
    hot_ttl_ms_ = warm_ttl_ms_;
  }
  if (const char * policy = std::getenv("KORITH_AMF_EVICTION_POLICY")) {
    const std::string p(policy);
    if (p == "lru") {
      eviction_policy_ = AmfEvictionPolicy::kLru;
    } else if (p == "lfu") {
      eviction_policy_ = AmfEvictionPolicy::kLfu;
    } else if (p == "roi") {
      eviction_policy_ = AmfEvictionPolicy::kRoi;
    }
  }
  double min_prompt_ms = 0.0;
  if (parse_env_double("KORITH_AMF_MIN_PROMPT_MS", min_prompt_ms) && min_prompt_ms > 0.0) {
    min_prompt_ms_ = min_prompt_ms;
  }
  double min_roi = 0.0;
  if (parse_env_double("KORITH_AMF_MIN_ROI", min_roi) && min_roi >= 0.0) {
    min_roi_ = min_roi;
  }
  double min_admit_roi = 0.0;
  if (parse_env_double("KORITH_AMF_MIN_ADMIT_ROI", min_admit_roi) && min_admit_roi >= 0.0) {
    min_admit_roi_ = min_admit_roi;
  }
  std::uint64_t max_age_ms = 0;
  if (parse_env_u64("KORITH_AMF_MAX_AGE_MS", max_age_ms)) {
    max_age_ms_ = max_age_ms;
  }
  std::uint64_t default_ttl_s = 0;
  if (parse_env_u64("KORITH_AMF_DEFAULT_TTL_S", default_ttl_s) && default_ttl_s > 0) {
    default_ttl_ms_ = default_ttl_s * 1000ull;
  }
  std::uint64_t max_ttl_s = 0;
  if (parse_env_u64("KORITH_AMF_MAX_TTL_S", max_ttl_s) && max_ttl_s > 0) {
    max_ttl_ms_ = max_ttl_s * 1000ull;
  }
  if (max_ttl_ms_ < default_ttl_ms_) {
    max_ttl_ms_ = default_ttl_ms_;
  }
  model_version_check_ = parse_env_bool("KORITH_AMF_MODEL_VERSION_CHECK", true);
  std::uint64_t low_roi_hits = 0;
  if (parse_env_u64("KORITH_AMF_LOW_ROI_HITS", low_roi_hits) && low_roi_hits > 0) {
    low_roi_hit_limit_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(low_roi_hits, UINT32_MAX));
  }
  prewarm_enabled_ = parse_env_bool("KORITH_AMF_PREWARM", false);
  std::uint64_t prewarm_top_k = 0;
  if (parse_env_u64("KORITH_AMF_PREWARM_TOP_K", prewarm_top_k) && prewarm_top_k > 0) {
    prewarm_top_k_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(prewarm_top_k, UINT32_MAX));
  }
  std::uint64_t prewarm_timeout_s = 0;
  if (parse_env_u64("KORITH_AMF_PREWARM_TIMEOUT_S", prewarm_timeout_s)) {
    prewarm_timeout_ms_ = prewarm_timeout_s * 1000ull;
  }
  disk_guard_enabled_ = parse_env_bool("KORITH_AMF_DISK_GUARD", true);
  std::uint64_t min_free_disk_bytes = 0;
  if (parse_env_u64("KORITH_AMF_MIN_FREE_DISK_BYTES", min_free_disk_bytes)) {
    min_free_disk_bytes_ = min_free_disk_bytes;
  }
  double min_free_disk_pct = 0.0;
  if (parse_env_double("KORITH_AMF_MIN_FREE_DISK_PCT", min_free_disk_pct) &&
      min_free_disk_pct >= 0.0 && min_free_disk_pct <= 1.0) {
    min_free_disk_ratio_ = min_free_disk_pct;
  }

  if (storage_budget_bytes_ == 0) {
    storage_budget_bytes_ = max_bytes_;
  }
  if (max_bytes_ == 0) {
    max_bytes_ = storage_budget_bytes_;
  }
  return init(resolved);
}

bool AmfStore::init(const std::string & path) {
  dir_ = path;
  if (dir_.empty()) {
    enabled_ = false;
    disable_reason_ = AmfDisableReason::kNoPath;
    return false;
  }

  std::error_code ec;
  std::filesystem::create_directories(dir_, ec);
  if (ec) {
    enabled_ = false;
    disable_reason_ = AmfDisableReason::kIoError;
    return false;
  }

  index_path_ = (std::filesystem::path(dir_) / "amf_index.bin").string();
  enabled_ = load_index();
  if (!enabled_) {
    return false;
  }
  disable_reason_ = AmfDisableReason::kNone;
  std::fprintf(stderr,
               "[AMF_PATH] dir=%s index=%s entries=%llu bytes=%llu\n",
               dir_.c_str(),
               index_path_.c_str(),
               static_cast<unsigned long long>(stats_.entries),
               static_cast<unsigned long long>(stats_.bytes));
  (void) std::fflush(stderr);
  if (prewarm_enabled_) {
    start_prewarm_async();
  }
  return true;
}

void AmfStore::log_disabled_once() const {
  if (enabled_ || disabled_logged_) {
    return;
  }
  std::fprintf(stderr, "[AMF_DISABLED] reason=%s\n", amf_disable_reason_name(disable_reason_));
  (void) std::fflush(stderr);
  disabled_logged_ = true;
}

bool AmfStore::load_index() {
  entries_.clear();
  index_.clear();
  total_bytes_ = 0;
  stats_.entries = 0;
  stats_.bytes = 0;
  refresh_storage_pressure();
  roi_decline_streak_ = 0;
  admissions_disabled_ = false;
  prewarm_running_.store(false);
  prewarm_complete_.store(!prewarm_enabled_);
  prewarm_cancel_.store(false);
  prewarm_loaded_.store(0);
  prewarm_failed_.store(0);
  {
    std::lock_guard<std::mutex> lock(prewarm_mu_);
    prewarmed_keys_.clear();
  }

  std::ifstream in(index_path_, std::ios::binary);
  if (!in) {
    enabled_ = true;
    return true;
  }

  IndexHeader hdr{};
  in.read(reinterpret_cast<char *>(&hdr), static_cast<std::streamsize>(sizeof(hdr)));
  if (!in || hdr.magic != kIndexMagic) {
    enabled_ = false;
    disable_reason_ = AmfDisableReason::kIoError;
    return false;
  }
  const bool is_v9 = (hdr.version == kIndexVersion);
  const bool is_v8 = (hdr.version == kIndexVersionV8);
  if (!is_v9 && !is_v8) {
    // Index layout changed across versions. Reset to cold-start instead of
    // hard-disabling AMF so replay can re-bootstrap on the new format.
    in.close();
    std::error_code ec;
    std::filesystem::remove(index_path_, ec);
    if (ec) {
      enabled_ = false;
      disable_reason_ = AmfDisableReason::kIoError;
      return false;
    }
    std::fprintf(stderr,
                 "[AMF_INDEX_RESET] reason=version_mismatch found=%u expected=%u\n",
                 hdr.version,
                 kIndexVersion);
    (void) std::fflush(stderr);
    enabled_ = true;
    disable_reason_ = AmfDisableReason::kNone;
    return true;
  }

  if (hdr.count > (1ull << 20)) {
    enabled_ = false;
    disable_reason_ = AmfDisableReason::kIoError;
    return false;
  }

  std::uint64_t checksum = kFNVOffset;
  IndexHeader hdr_check = hdr;
  hdr_check.checksum = 0;
  hash_bytes(checksum,
             reinterpret_cast<const std::uint8_t *>(&hdr_check),
             sizeof(hdr_check));

  for (std::uint64_t i = 0; i < hdr.count; ++i) {
    AmfEntry entry{};
    if (is_v9) {
      IndexRecord rec{};
      in.read(reinterpret_cast<char *>(&rec), static_cast<std::streamsize>(sizeof(rec)));
      if (!in) {
        enabled_ = false;
        disable_reason_ = AmfDisableReason::kIoError;
        return false;
      }
      hash_bytes(checksum,
                 reinterpret_cast<const std::uint8_t *>(&rec),
                 sizeof(rec));
      entry.key = rec.key;
      entry.model_hash = rec.model_hash;
      if (entry.model_hash == 0) {
        entry.model_hash = entry.key.model_hash;
      }
      entry.prefix_len = rec.prefix_len;
      entry.seen_count = rec.seen_count;
      entry.restore_count = rec.restore_count;
      entry.ema_saved_ms = rec.ema_saved_ms;
      entry.ema_restore_ms = rec.ema_restore_ms;
      entry.ema_roi = rec.ema_roi;
      entry.failure_count = rec.failure_count;
      entry.block_count = rec.block_count;
      entry.saved_ms = rec.saved_ms;
      entry.hit_count = rec.hit_count;
      entry.last_used = rec.last_used;
      entry.last_hit_ms = rec.last_hit_ms;
      entry.low_roi_hits = rec.low_roi_hits;
      entry.kv_size = rec.kv_size;
      entry.last_access_time_ms = rec.last_access_time_ms;
      entry.access_count = rec.access_count;
      entry.stored_at_ms = rec.stored_at_ms;
      entry.expires_at_ms = rec.expires_at_ms;
      entry.size_bytes = rec.size_bytes;
      entry.kv_checksum = rec.kv_checksum;
      entry.state = rec.state;
    } else {
      LegacyIndexRecordV8 rec{};
      in.read(reinterpret_cast<char *>(&rec), static_cast<std::streamsize>(sizeof(rec)));
      if (!in) {
        enabled_ = false;
        disable_reason_ = AmfDisableReason::kIoError;
        return false;
      }
      hash_bytes(checksum,
                 reinterpret_cast<const std::uint8_t *>(&rec),
                 sizeof(rec));
      entry.key.model_hash = rec.key.model_hash;
      entry.key.tenant_hash = 0;
      entry.key.prefix_hash = rec.key.prefix_hash;
      entry.key.n_ctx = rec.key.n_ctx;
      entry.key.kv_version = rec.key.kv_version;
      entry.key.rope_base_bits = rec.key.rope_base_bits;
      entry.key.rope_scale_bits = rec.key.rope_scale_bits;
      entry.key.sampling_hash = rec.key.sampling_hash;
      entry.key.rng_hash = rec.key.rng_hash;
      entry.model_hash = rec.model_hash;
      if (entry.model_hash == 0) {
        entry.model_hash = entry.key.model_hash;
      }
      entry.prefix_len = rec.prefix_len;
      entry.seen_count = rec.seen_count;
      entry.restore_count = rec.restore_count;
      entry.ema_saved_ms = rec.ema_saved_ms;
      entry.ema_restore_ms = rec.ema_restore_ms;
      entry.ema_roi = rec.ema_roi;
      entry.failure_count = rec.failure_count;
      entry.block_count = rec.block_count;
      entry.saved_ms = rec.saved_ms;
      entry.hit_count = rec.hit_count;
      entry.last_used = rec.last_used;
      entry.last_hit_ms = rec.last_hit_ms;
      entry.low_roi_hits = rec.low_roi_hits;
      entry.kv_size = rec.kv_size;
      entry.last_access_time_ms = rec.last_access_time_ms;
      entry.access_count = rec.access_count;
      entry.stored_at_ms = rec.stored_at_ms;
      entry.expires_at_ms = rec.expires_at_ms;
      entry.size_bytes = rec.size_bytes;
      entry.kv_checksum = rec.kv_checksum;
      entry.state = rec.state;
    }
    if (entry.size_bytes == 0) {
      entry.size_bytes = entry.kv_size;
    }
    if (entry.last_access_time_ms == 0) {
      entry.last_access_time_ms = entry.last_used;
    }
    if (entry.expires_at_ms == 0 && default_ttl_ms_ > 0) {
      const std::uint64_t base_ms = (entry.last_access_time_ms > 0) ? entry.last_access_time_ms : now_epoch_ms();
      entry.expires_at_ms = base_ms + effective_ttl_ms(entry.access_count);
    }
    if (prewarm_enabled_) {
      entry.state = static_cast<std::uint8_t>(AmfEntryState::kWarmPending);
    } else {
      const AmfEntryTier tier = classify_tier(entry, now_epoch_ms());
      entry.state = (tier == AmfEntryTier::kHot)
          ? static_cast<std::uint8_t>(AmfEntryState::kHot)
          : static_cast<std::uint8_t>(AmfEntryState::kCold);
    }
    const std::size_t idx = entries_.size();
    entries_.push_back(entry);
    index_[entry.key] = idx;
    total_bytes_ += entry.size_bytes;
  }

  if (checksum != hdr.checksum) {
    enabled_ = false;
    disable_reason_ = AmfDisableReason::kIoError;
    return false;
  }

  stats_.entries = static_cast<std::uint64_t>(entries_.size());
  stats_.bytes = total_bytes_;
  refresh_storage_pressure();
  enabled_ = true;
  return true;
}

bool AmfStore::flush_index() const {
  const std::string tmp_path =
      (std::filesystem::path(dir_) / "amf_index.tmp").string();
  errno = 0;
  std::FILE * fp = std::fopen(tmp_path.c_str(), "wb");
  if (!fp) {
    log_store_fail("flush_index_open", "open_failed", tmp_path.c_str());
    return false;
  }

  IndexHeader hdr{};
  hdr.count = static_cast<std::uint64_t>(entries_.size());
  hdr.checksum = 0;

  std::uint64_t checksum = kFNVOffset;
  IndexHeader hdr_check = hdr;
  hash_bytes(checksum,
             reinterpret_cast<const std::uint8_t *>(&hdr_check),
             sizeof(hdr_check));

  for (const AmfEntry & entry : entries_) {
    IndexRecord rec{};
    rec.key = entry.key;
    rec.model_hash = entry.model_hash;
    rec.prefix_len = entry.prefix_len;
    rec.seen_count = entry.seen_count;
    rec.restore_count = entry.restore_count;
    rec.ema_saved_ms = entry.ema_saved_ms;
    rec.ema_restore_ms = entry.ema_restore_ms;
    rec.ema_roi = entry.ema_roi;
    rec.failure_count = entry.failure_count;
    rec.block_count = entry.block_count;
    rec.saved_ms = entry.saved_ms;
    rec.hit_count = entry.hit_count;
    rec.last_used = entry.last_used;
    rec.last_hit_ms = entry.last_hit_ms;
    rec.low_roi_hits = entry.low_roi_hits;
    rec.kv_size = entry.kv_size;
    rec.last_access_time_ms = entry.last_access_time_ms;
    rec.access_count = entry.access_count;
    rec.stored_at_ms = entry.stored_at_ms;
    rec.expires_at_ms = entry.expires_at_ms;
    rec.size_bytes = entry.size_bytes;
    rec.kv_checksum = entry.kv_checksum;
    rec.state = entry.state;
    hash_bytes(checksum,
               reinterpret_cast<const std::uint8_t *>(&rec),
               sizeof(rec));
  }

  hdr.checksum = checksum;
  if (std::fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
    std::fclose(fp);
    log_store_fail("flush_index_header", "write_failed", tmp_path.c_str());
    return false;
  }

  for (const AmfEntry & entry : entries_) {
    IndexRecord rec{};
    rec.key = entry.key;
    rec.model_hash = entry.model_hash;
    rec.prefix_len = entry.prefix_len;
    rec.seen_count = entry.seen_count;
    rec.restore_count = entry.restore_count;
    rec.ema_saved_ms = entry.ema_saved_ms;
    rec.ema_restore_ms = entry.ema_restore_ms;
    rec.ema_roi = entry.ema_roi;
    rec.failure_count = entry.failure_count;
    rec.block_count = entry.block_count;
    rec.saved_ms = entry.saved_ms;
    rec.hit_count = entry.hit_count;
    rec.last_used = entry.last_used;
    rec.last_hit_ms = entry.last_hit_ms;
    rec.low_roi_hits = entry.low_roi_hits;
    rec.kv_size = entry.kv_size;
    rec.last_access_time_ms = entry.last_access_time_ms;
    rec.access_count = entry.access_count;
    rec.stored_at_ms = entry.stored_at_ms;
    rec.expires_at_ms = entry.expires_at_ms;
    rec.size_bytes = entry.size_bytes;
    rec.kv_checksum = entry.kv_checksum;
    rec.state = entry.state;
    if (std::fwrite(&rec, sizeof(rec), 1, fp) != 1) {
      std::fclose(fp);
      log_store_fail("flush_index_record", "write_failed", tmp_path.c_str());
      return false;
    }
  }

  if (std::fflush(fp) != 0) {
    std::fclose(fp);
    log_store_fail("flush_index_fflush", "flush_failed", tmp_path.c_str());
    return false;
  }
  if (::fsync(::fileno(fp)) != 0) {
    std::fclose(fp);
    log_store_fail("flush_index_fsync", "fsync_failed", tmp_path.c_str());
    return false;
  }
  if (std::fclose(fp) != 0) {
    log_store_fail("flush_index_close", "close_failed", tmp_path.c_str());
    return false;
  }

  std::error_code ec;
  std::filesystem::rename(tmp_path, index_path_, ec);
  if (ec) {
    std::fprintf(stderr,
                 "[AMF_STORE_FAIL] stage=flush_index_rename reason=rename_failed path=%s ec=%d msg=%s\n",
                 index_path_.c_str(),
                 ec.value(),
                 ec.message().c_str());
    (void) std::fflush(stderr);
    std::filesystem::remove(tmp_path, ec);
    return false;
  }
  return true;
}

std::string AmfStore::entry_basename(const AmfEntry & entry) const {
  char buf[256];
  std::snprintf(buf,
                sizeof(buf),
                "amf_%016llx_%016llx_%016llx_%08x_%08x_%08x_%08x_%016llx_%016llx",
                static_cast<unsigned long long>(entry.key.model_hash),
                static_cast<unsigned long long>(entry.key.tenant_hash),
                static_cast<unsigned long long>(entry.key.prefix_hash),
                entry.key.n_ctx,
                entry.key.kv_version,
                entry.key.rope_base_bits,
                entry.key.rope_scale_bits,
                static_cast<unsigned long long>(entry.key.sampling_hash),
                static_cast<unsigned long long>(entry.key.rng_hash));
  return std::string(buf);
}

std::string AmfStore::entry_kv_path(const AmfEntry & entry) const {
  return (std::filesystem::path(dir_) / (entry_basename(entry) + ".kv")).string();
}

std::string AmfStore::entry_tok_path(const AmfEntry & entry) const {
  return (std::filesystem::path(dir_) / (entry_basename(entry) + ".tok")).string();
}

bool AmfStore::write_tokens_file(const AmfEntry & entry, const std::vector<llama_token> & tokens) const {
  const std::string path = entry_tok_path(entry);
  errno = 0;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    log_store_fail("write_tokens_open", "open_failed", path.c_str());
    return false;
  }
  if (!tokens.empty()) {
    out.write(reinterpret_cast<const char *>(tokens.data()),
              static_cast<std::streamsize>(tokens.size() * sizeof(llama_token)));
  }
  if (!out) {
    log_store_fail("write_tokens_write", "write_failed", path.c_str());
    return false;
  }
  return true;
}

bool AmfStore::read_tokens_file(const AmfEntry & entry, std::vector<llama_token> * out) const {
  out->clear();
  const std::string path = entry_tok_path(entry);
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamsize size = in.tellg();
  if (size <= 0 || size % static_cast<std::streamsize>(sizeof(llama_token)) != 0) {
    return false;
  }
  in.seekg(0, std::ios::beg);
  const std::size_t count = static_cast<std::size_t>(size / sizeof(llama_token));
  out->resize(count);
  in.read(reinterpret_cast<char *>(out->data()), size);
  return static_cast<bool>(in);
}

bool AmfStore::write_kv_file(const AmfEntry & entry, const std::uint8_t * data, std::size_t size) const {
  const std::string path = entry_kv_path(entry);
  errno = 0;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    log_store_fail("write_kv_open", "open_failed", path.c_str());
    return false;
  }
  if (size > 0) {
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
  }
  if (!out) {
    log_store_fail("write_kv_write", "write_failed", path.c_str());
    return false;
  }
  return true;
}

bool AmfStore::read_kv_file(const AmfEntry & entry, std::vector<std::uint8_t> * out) const {
  out->clear();
  const std::string path = entry_kv_path(entry);
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamsize size = in.tellg();
  if (size <= 0) {
    return false;
  }
  in.seekg(0, std::ios::beg);
  out->resize(static_cast<std::size_t>(size));
  in.read(reinterpret_cast<char *>(out->data()), size);
  return static_cast<bool>(in);
}

void AmfStore::set_min_prompt_ms(double v) {
  if (std::isfinite(v) && v >= 0.0) {
    min_prompt_ms_ = v;
  }
}

void AmfStore::set_min_admit_roi(double v) {
  if (std::isfinite(v) && v >= 0.0) {
    min_admit_roi_ = v;
  }
}

void AmfStore::set_eviction_pressure(double v) {
  if (std::isfinite(v) && v >= 0.0) {
    eviction_pressure_ = v;
  }
}

void AmfStore::refresh_storage_pressure() {
  const std::uint64_t budget = (storage_budget_bytes_ > 0) ? storage_budget_bytes_ : max_bytes_;
  if (budget == 0) {
    eviction_pressure_ = 0.0;
    return;
  }
  const double pressure = static_cast<double>(total_bytes_) / static_cast<double>(budget);
  eviction_pressure_ = std::isfinite(pressure) ? std::max(0.0, pressure) : 0.0;
}

bool AmfStore::disk_guard_allows(std::size_t incoming_bytes,
                                 std::uint64_t * out_available,
                                 std::uint64_t * out_capacity,
                                 double * out_free_ratio) const {
  if (out_available) {
    *out_available = 0;
  }
  if (out_capacity) {
    *out_capacity = 0;
  }
  if (out_free_ratio) {
    *out_free_ratio = 0.0;
  }
  if (!disk_guard_enabled_) {
    return true;
  }
  std::error_code ec;
  const std::filesystem::space_info info = std::filesystem::space(dir_, ec);
  if (ec) {
    return true;
  }
  const std::uint64_t available = static_cast<std::uint64_t>(info.available);
  const std::uint64_t capacity = static_cast<std::uint64_t>(info.capacity);
  if (out_available) {
    *out_available = available;
  }
  if (out_capacity) {
    *out_capacity = capacity;
  }
  if (out_free_ratio && capacity > 0) {
    *out_free_ratio = static_cast<double>(available) / static_cast<double>(capacity);
  }
  if (min_free_disk_bytes_ > 0 && available <= min_free_disk_bytes_) {
    return false;
  }
  if (min_free_disk_ratio_ > 0.0 && capacity > 0) {
    const double free_ratio = static_cast<double>(available) / static_cast<double>(capacity);
    if (free_ratio <= min_free_disk_ratio_) {
      return false;
    }
  }
  if (incoming_bytes > 0 && available <= static_cast<std::uint64_t>(incoming_bytes)) {
    return false;
  }
  return true;
}

void AmfStore::log_store_fail(const char * stage, const char * reason, const char * path) const {
  std::fprintf(stderr,
               "[AMF_STORE_FAIL] stage=%s reason=%s path=%s errno=%d err=%s\n",
               stage ? stage : "unknown",
               reason ? reason : "unknown",
               path ? path : "-",
               errno,
               std::strerror(errno));
  (void) std::fflush(stderr);
}

AmfEntryTier AmfStore::classify_tier(const AmfEntry & entry, std::uint64_t now_ms) const {
  const std::uint64_t last =
      (entry.last_access_time_ms > 0) ? entry.last_access_time_ms : entry.last_used;
  if (last == 0 || now_ms == 0) {
    return AmfEntryTier::kCold;
  }
  if (now_ms < last) {
    return AmfEntryTier::kCold;
  }
  const std::uint64_t age_ms = now_ms - last;
  if (age_ms <= hot_ttl_ms_) {
    return AmfEntryTier::kHot;
  }
  if (age_ms <= warm_ttl_ms_) {
    return AmfEntryTier::kWarm;
  }
  return AmfEntryTier::kCold;
}

const char * AmfStore::tier_name(AmfEntryTier tier) const {
  switch (tier) {
    case AmfEntryTier::kHot:
      return "hot";
    case AmfEntryTier::kWarm:
      return "warm";
    case AmfEntryTier::kCold:
    default:
      return "cold";
  }
}

const char * AmfStore::eviction_policy_name() const {
  switch (eviction_policy_) {
    case AmfEvictionPolicy::kLru:
      return "lru";
    case AmfEvictionPolicy::kLfu:
      return "lfu";
    case AmfEvictionPolicy::kRoi:
      return "roi";
    default:
      return "unknown";
  }
}

std::uint64_t AmfStore::effective_ttl_ms(std::uint64_t access_count) const {
  if (default_ttl_ms_ == 0) {
    return 0;
  }
  const std::uint64_t capped_access = std::max<std::uint64_t>(1, access_count);
  const double boost = 1.0 + std::log2(static_cast<double>(capped_access));
  double ttl = static_cast<double>(default_ttl_ms_) * boost;
  if (!std::isfinite(ttl) || ttl < static_cast<double>(default_ttl_ms_)) {
    ttl = static_cast<double>(default_ttl_ms_);
  }
  if (max_ttl_ms_ > 0) {
    ttl = std::min(ttl, static_cast<double>(max_ttl_ms_));
  }
  if (ttl > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
    ttl = static_cast<double>(std::numeric_limits<std::uint64_t>::max());
  }
  return static_cast<std::uint64_t>(ttl);
}

const char * AmfStore::entry_state_name(AmfEntryState state) const {
  switch (state) {
    case AmfEntryState::kHot:
      return "hot";
    case AmfEntryState::kWarmPending:
      return "warm_pending";
    case AmfEntryState::kCold:
    default:
      return "cold";
  }
}

bool AmfStore::is_prewarmed_key(const AmfKey & key) const {
  std::lock_guard<std::mutex> lock(prewarm_mu_);
  const auto it = prewarmed_keys_.find(key);
  return it != prewarmed_keys_.end() && it->second;
}

void AmfStore::mark_prewarmed_key(const AmfKey & key) {
  std::lock_guard<std::mutex> lock(prewarm_mu_);
  prewarmed_keys_[key] = true;
}

bool AmfStore::should_wait_warm_pending(const AmfEntry & entry) const {
  if (!prewarm_enabled_) {
    return false;
  }
  const AmfEntryState state = static_cast<AmfEntryState>(entry.state);
  if (state != AmfEntryState::kWarmPending) {
    return false;
  }
  if (is_prewarmed_key(entry.key)) {
    return false;
  }
  return prewarm_running_.load();
}

void AmfStore::start_prewarm_async() {
  if (!enabled_ || !prewarm_enabled_ || entries_.empty()) {
    prewarm_running_.store(false);
    prewarm_complete_.store(true);
    return;
  }
  if (prewarm_thread_.joinable()) {
    prewarm_cancel_.store(true);
    prewarm_thread_.join();
    prewarm_cancel_.store(false);
  }
  prewarm_loaded_.store(0);
  prewarm_failed_.store(0);
  prewarm_running_.store(true);
  prewarm_complete_.store(false);
  std::vector<AmfEntry> ranked = entries_;
  prewarm_thread_ = std::thread(&AmfStore::run_prewarm, this, std::move(ranked));
}

void AmfStore::run_prewarm(std::vector<AmfEntry> ranked) {
  const auto t0 = std::chrono::steady_clock::now();
  std::sort(ranked.begin(), ranked.end(), [](const AmfEntry & a, const AmfEntry & b) {
    if (a.access_count != b.access_count) {
      return a.access_count > b.access_count;
    }
    const std::uint64_t a_last = (a.last_access_time_ms > 0) ? a.last_access_time_ms : a.last_used;
    const std::uint64_t b_last = (b.last_access_time_ms > 0) ? b.last_access_time_ms : b.last_used;
    return a_last > b_last;
  });
  if (ranked.size() > static_cast<std::size_t>(prewarm_top_k_)) {
    ranked.resize(static_cast<std::size_t>(prewarm_top_k_));
  }

  std::uint64_t loaded = 0;
  std::uint64_t failed = 0;
  for (const AmfEntry & entry : ranked) {
    if (prewarm_cancel_.load()) {
      break;
    }
    if (prewarm_timeout_ms_ == 0) {
      break;
    }
    const auto now = std::chrono::steady_clock::now();
    const std::uint64_t elapsed_ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count());
    if (elapsed_ms >= prewarm_timeout_ms_) {
      break;
    }

    std::vector<std::uint8_t> blob;
    bool ok = read_kv_file(entry, &blob);
    if (ok && entry.kv_checksum > 0) {
      std::uint64_t checksum = kFNVOffset;
      hash_bytes(checksum, blob.data(), blob.size());
      ok = checksum == entry.kv_checksum;
    }
    if (ok) {
      mark_prewarmed_key(entry.key);
      loaded += 1;
    } else {
      failed += 1;
    }
  }

  prewarm_loaded_.store(loaded);
  prewarm_failed_.store(failed);
  prewarm_running_.store(false);
  prewarm_complete_.store(true);
  const auto t1 = std::chrono::steady_clock::now();
  const std::uint64_t total_ms = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  std::fprintf(stderr,
               "[AMF_PREWARM] loaded=%llu failed=%llu total_ms=%llu\n",
               static_cast<unsigned long long>(loaded),
               static_cast<unsigned long long>(failed),
               static_cast<unsigned long long>(total_ms));
  (void) std::fflush(stderr);
}

double AmfStore::warm_ratio() const {
  if (entries_.empty()) {
    return 1.0;
  }
  std::uint64_t warm = 0;
  for (const AmfEntry & entry : entries_) {
    const AmfEntryState state = static_cast<AmfEntryState>(entry.state);
    if (state == AmfEntryState::kHot) {
      warm += 1;
      continue;
    }
    if (state == AmfEntryState::kWarmPending && is_prewarmed_key(entry.key)) {
      warm += 1;
    }
  }
  return static_cast<double>(warm) / static_cast<double>(entries_.size());
}

void AmfStore::shutdown() {
  prewarm_cancel_.store(true);
  if (prewarm_thread_.joinable()) {
    prewarm_thread_.join();
  }
  prewarm_running_.store(false);
  prewarm_complete_.store(true);
  prewarm_cancel_.store(false);
  if (!enabled_) {
    return;
  }
  (void) flush_index();
  std::fprintf(stderr,
               "[AMF_SHUTDOWN] entries_persisted=%llu bytes=%llu\n",
               static_cast<unsigned long long>(entries_.size()),
               static_cast<unsigned long long>(total_bytes_));
  (void) std::fflush(stderr);
}

std::uint64_t AmfStore::oldest_entry_age_ms(std::uint64_t now_ms) const {
  if (entries_.empty() || now_ms == 0) {
    return 0;
  }
  std::uint64_t oldest = 0;
  for (const AmfEntry & e : entries_) {
    const std::uint64_t last =
        (e.last_access_time_ms > 0) ? e.last_access_time_ms : e.last_used;
    if (last == 0) {
      continue;
    }
    if (oldest == 0 || last < oldest) {
      oldest = last;
    }
  }
  if (oldest == 0 || now_ms <= oldest) {
    return 0;
  }
  return now_ms - oldest;
}

void AmfStore::evict_entry_at(std::size_t idx, const char * reason) {
  if (idx >= entries_.size()) {
    return;
  }

  const AmfEntry entry = entries_[idx];
  const std::uint64_t now_ms = now_epoch_ms();
  const AmfEntryTier tier = classify_tier(entry, now_ms);
  std::error_code ec;
  std::filesystem::remove(entry_kv_path(entry), ec);
  std::filesystem::remove(entry_tok_path(entry), ec);

  const std::uint64_t entry_bytes = (entry.size_bytes > 0) ? entry.size_bytes : entry.kv_size;
  if (entry_bytes <= total_bytes_) {
    total_bytes_ -= entry_bytes;
  } else {
    total_bytes_ = 0;
  }

  stats_.evictions += 1;
  stats_.evicted_bytes += entry_bytes;

  entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(idx));
  index_.erase(entry.key);
  {
    std::lock_guard<std::mutex> lock(prewarm_mu_);
    prewarmed_keys_.erase(entry.key);
  }
  index_.clear();
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    index_[entries_[i].key] = i;
  }

  stats_.entries = static_cast<std::uint64_t>(entries_.size());
  stats_.bytes = total_bytes_;
  refresh_storage_pressure();

  std::fprintf(stderr,
               "[AMF_EVICT] hash=%016llx size=%llu tier=%s policy=%s reason=%s\n",
               static_cast<unsigned long long>(entry.key.prefix_hash),
               static_cast<unsigned long long>(entry_bytes),
               tier_name(tier),
               eviction_policy_name(),
               reason ? reason : "unknown");
  (void) std::fflush(stderr);
  (void) flush_index();
}

std::size_t AmfStore::pick_budget_eviction_candidate(AmfEntryTier tier, std::uint64_t now_ms) const {
  std::size_t best_idx = kInvalidIndex;

  auto roi_score = [](const AmfEntry & e) -> double {
    if (std::isfinite(e.ema_roi)) {
      return e.ema_roi;
    }
    if (std::isfinite(e.ema_restore_ms) && e.ema_restore_ms > 0.0 &&
        std::isfinite(e.ema_saved_ms)) {
      const double roi = e.ema_saved_ms / e.ema_restore_ms;
      return std::isfinite(roi) ? roi : 0.0;
    }
    return 0.0;
  };

  auto is_better = [&](const AmfEntry & cand, const AmfEntry & current) -> bool {
    const std::uint64_t cand_last =
        (cand.last_access_time_ms > 0) ? cand.last_access_time_ms : cand.last_used;
    const std::uint64_t cur_last =
        (current.last_access_time_ms > 0) ? current.last_access_time_ms : current.last_used;
    if (eviction_policy_ == AmfEvictionPolicy::kLru) {
      if (cand_last != cur_last) {
        return cand_last < cur_last;
      }
      return cand.access_count < current.access_count;
    }
    if (eviction_policy_ == AmfEvictionPolicy::kRoi) {
      const double cand_roi = roi_score(cand);
      const double cur_roi = roi_score(current);
      if (cand_roi != cur_roi) {
        return cand_roi < cur_roi;
      }
      if (cand.access_count != current.access_count) {
        return cand.access_count < current.access_count;
      }
      return cand_last < cur_last;
    }
    if (cand.access_count != current.access_count) {
      return cand.access_count < current.access_count;
    }
    return cand_last < cur_last;
  };

  for (std::size_t i = 0; i < entries_.size(); ++i) {
    const AmfEntry & entry = entries_[i];
    if (classify_tier(entry, now_ms) != tier) {
      continue;
    }
    if (best_idx == kInvalidIndex || is_better(entry, entries_[best_idx])) {
      best_idx = i;
    }
  }
  return best_idx;
}

void AmfStore::evict_if_needed(std::size_t incoming_bytes) {
  const std::uint64_t budget_bytes = (storage_budget_bytes_ > 0) ? storage_budget_bytes_ : max_bytes_;
  if (budget_bytes == 0) {
    return;
  }
  if (incoming_bytes > budget_bytes) {
    return;
  }
  const long double projected_bytes_ld = static_cast<long double>(total_bytes_) +
      static_cast<long double>(incoming_bytes);
  const long double watermark_bytes_ld =
      static_cast<long double>(budget_bytes) * static_cast<long double>(eviction_watermark_);
  if (projected_bytes_ld <= watermark_bytes_ld) {
    return;
  }

  const long double critical_bytes_ld =
      static_cast<long double>(budget_bytes) * static_cast<long double>(critical_watermark_);

  while ((static_cast<long double>(total_bytes_) + static_cast<long double>(incoming_bytes)) >
             watermark_bytes_ld &&
         !entries_.empty()) {
    const std::uint64_t now_ms = now_epoch_ms();
    std::size_t idx = pick_budget_eviction_candidate(AmfEntryTier::kCold, now_ms);
    if (idx == kInvalidIndex) {
      idx = pick_budget_eviction_candidate(AmfEntryTier::kWarm, now_ms);
    }
    if (idx == kInvalidIndex &&
        (static_cast<long double>(total_bytes_) + static_cast<long double>(incoming_bytes)) > critical_bytes_ld) {
      idx = pick_budget_eviction_candidate(AmfEntryTier::kHot, now_ms);
    }
    if (idx == kInvalidIndex) {
      break;
    }
    evict_entry_at(idx, "budget");
  }
  refresh_storage_pressure();
}

AmfLookupResult AmfStore::find_longest_prefix(const AmfContext & ctx,
                                              const std::vector<llama_token> & tokens,
                                              AmfEntry * out_entry,
                                              std::vector<llama_token> * out_tokens) {
  if (!enabled_) {
    return AmfLookupResult::kMissLoadFailed;
  }
  if (tokens.size() < min_tokens_) {
    return AmfLookupResult::kMissShortPrefix;
  }
  if (entries_.empty()) {
    return AmfLookupResult::kMissFirstRun;
  }

  std::vector<std::uint64_t> prefix_hashes(tokens.size());
  const std::uint64_t now_ms = now_epoch_ms();
  std::uint64_t h = kFNVOffset;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    h = hash_token_prefix_step(h, tokens[i]);
    prefix_hashes[i] = h;
  }

  for (std::size_t len = tokens.size(); len >= min_tokens_; --len) {
    const AmfKey key = amf_make_key(ctx, prefix_hashes[len - 1]);
    auto it = index_.find(key);
    if (it == index_.end()) {
      if (len == min_tokens_) {
        break;
      }
      continue;
    }

    const AmfEntry entry = entries_[it->second];
    if (entry.prefix_len != len) {
      continue;
    }
    if (model_version_check_ && entry.model_hash != 0 && entry.model_hash != ctx.model_hash) {
      std::fprintf(stderr,
                   "[AMF_MODEL_MISMATCH] hash=%016llx\n",
                   static_cast<unsigned long long>(entry.key.prefix_hash));
      (void) std::fflush(stderr);
      evict_entry_at(it->second, "model_mismatch");
      if (len == min_tokens_) {
        break;
      }
      continue;
    }
    if (entry.expires_at_ms > 0 && now_ms > entry.expires_at_ms) {
      const std::uint64_t age_ms = (now_ms > entry.stored_at_ms) ? (now_ms - entry.stored_at_ms) : 0;
      std::fprintf(stderr,
                   "[AMF_EXPIRED] hash=%016llx age_s=%llu\n",
                   static_cast<unsigned long long>(entry.key.prefix_hash),
                   static_cast<unsigned long long>(age_ms / 1000ull));
      (void) std::fflush(stderr);
      evict_entry_at(it->second, "ttl");
      if (len == min_tokens_) {
        break;
      }
      continue;
    }
    if (should_wait_warm_pending(entry)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(prewarm_wait_ms_));
      if (!is_prewarmed_key(entry.key)) {
        if (len == min_tokens_) {
          break;
        }
        continue;
      }
    }
    if (is_prewarmed_key(entry.key)) {
      auto live_it = index_.find(entry.key);
      if (live_it != index_.end()) {
        entries_[live_it->second].state = static_cast<std::uint8_t>(AmfEntryState::kHot);
      }
    }
    if (!read_tokens_file(entry, out_tokens)) {
      return AmfLookupResult::kMissLoadFailed;
    }
    if (out_tokens->size() != len) {
      return AmfLookupResult::kMissTokenMismatch;
    }
    if (!std::equal(out_tokens->begin(), out_tokens->end(), tokens.begin())) {
      return AmfLookupResult::kMissTokenMismatch;
    }
    if (out_entry) {
      *out_entry = entry;
    }
    return AmfLookupResult::kHit;
  }

  return AmfLookupResult::kMissHashMismatch;
}

bool AmfStore::load_kv(const AmfEntry & entry, std::vector<std::uint8_t> * out) const {
  if (!read_kv_file(entry, out)) {
    return false;
  }
  if (entry.kv_checksum > 0) {
    std::uint64_t checksum = kFNVOffset;
    hash_bytes(checksum, out->data(), out->size());
    if (checksum != entry.kv_checksum) {
      return false;
    }
  }
  return true;
}

bool AmfStore::store_entry(const AmfContext & ctx,
                           const std::vector<llama_token> & tokens,
                           const std::uint8_t * kv_data,
                           std::size_t kv_size,
                           std::uint64_t saved_ms) {
  if (!enabled_ || tokens.size() < min_tokens_ || kv_size == 0 || !kv_data) {
    stats_.admission_rejects += 1;
    return false;
  }

  const std::uint64_t budget_bytes = (storage_budget_bytes_ > 0) ? storage_budget_bytes_ : max_bytes_;
  if (budget_bytes > 0 && kv_size > budget_bytes) {
    stats_.admission_rejects += 1;
    return false;
  }

  const AmfKey key = amf_make_key(ctx, amf_hash_tokens(tokens));
  const auto it = index_.find(key);
  const bool has_existing = (it != index_.end());
  std::size_t disk_incoming_bytes = kv_size;
  if (has_existing) {
    const std::uint64_t old_size = entries_[it->second].kv_size;
    if (kv_size > old_size) {
      disk_incoming_bytes = kv_size - static_cast<std::size_t>(old_size);
    } else {
      disk_incoming_bytes = 0;
    }
  }
  std::uint64_t disk_free_bytes = 0;
  std::uint64_t disk_capacity_bytes = 0;
  double disk_free_ratio = 0.0;
  if (!disk_guard_allows(disk_incoming_bytes,
                         &disk_free_bytes,
                         &disk_capacity_bytes,
                         &disk_free_ratio)) {
    stats_.admission_rejects += 1;
    std::fprintf(stderr,
                 "[AMF_ADMIT_REJECT] reason=disk_guard free_bytes=%llu free_pct=%.2f min_free_bytes=%llu min_free_pct=%.2f\n",
                 static_cast<unsigned long long>(disk_free_bytes),
                 disk_free_ratio * 100.0,
                 static_cast<unsigned long long>(min_free_disk_bytes_),
                 min_free_disk_ratio_ * 100.0);
    (void) std::fflush(stderr);
    return false;
  }
  if (admissions_disabled_ && !has_existing) {
    stats_.admission_rejects += 1;
    std::fprintf(stderr, "[AMF_ADMIT_REJECT] reason=admissions_disabled\n");
    (void) std::fflush(stderr);
    return false;
  }
  const AmfEntry * existing_entry = has_existing ? &entries_[it->second] : nullptr;
  const bool ema_ready = existing_entry && std::isfinite(existing_entry->ema_roi);
  // Allow a single bootstrap admission before any restore-derived ROI exists.
  const bool bootstrap_ok = (!has_existing) ||
      (existing_entry->restore_count == 0 && existing_entry->seen_count == 0 && !ema_ready);

  if (min_admit_roi_ > 0.0) {
    if (bootstrap_ok) {
      const std::uint64_t key_id = static_cast<std::uint64_t>(KeyHash{}(key));
      std::fprintf(stderr,
                   "[AMF_ADMIT_BOOTSTRAP] key=%016llx reason=first_restore\n",
                   static_cast<unsigned long long>(key_id));
      (void) std::fflush(stderr);
    } else if (ema_ready) {
      if (existing_entry->ema_roi >= min_admit_roi_) {
        std::fprintf(stderr,
                     "[AMF_ADMIT] reason=roi_pass roi=%.2f min=%.2f\n",
                     existing_entry->ema_roi,
                     min_admit_roi_);
        (void) std::fflush(stderr);
      } else {
        stats_.admission_rejects += 1;
        std::fprintf(stderr,
                     "[AMF_ADMIT_REJECT] reason=ema_roi roi=%.2f min=%.2f\n",
                     existing_entry->ema_roi,
                     min_admit_roi_);
        (void) std::fflush(stderr);
        return false;
      }
    } else {
      stats_.admission_rejects += 1;
      std::fprintf(stderr,
                   "[AMF_ADMIT_REJECT] reason=ema_roi roi=nan min=%.2f\n",
                   min_admit_roi_);
      (void) std::fflush(stderr);
      return false;
    }
  }

  AmfEntry entry{};
  entry.key = key;
  entry.model_hash = ctx.model_hash;
  entry.prefix_len = static_cast<std::uint32_t>(tokens.size());
  entry.seen_count = has_existing ? (entries_[it->second].seen_count + 1) : 1;
  entry.restore_count = has_existing ? entries_[it->second].restore_count : 0;
  entry.ema_saved_ms = has_existing ? entries_[it->second].ema_saved_ms
                                    : std::numeric_limits<double>::quiet_NaN();
  entry.ema_restore_ms = has_existing ? entries_[it->second].ema_restore_ms
                                      : std::numeric_limits<double>::quiet_NaN();
  entry.ema_roi = has_existing ? entries_[it->second].ema_roi
                               : std::numeric_limits<double>::quiet_NaN();
  entry.saved_ms = saved_ms;
  entry.hit_count = 0;
  const std::uint64_t now_ms = now_epoch_ms();
  entry.last_used = now_ms;
  entry.last_access_time_ms = now_ms;
  entry.access_count = 0;
  entry.stored_at_ms = now_ms;
  entry.expires_at_ms = (default_ttl_ms_ > 0) ? (now_ms + effective_ttl_ms(entry.access_count)) : 0;
  entry.kv_size = kv_size;
  entry.size_bytes = kv_size;
  entry.kv_checksum = kFNVOffset;
  hash_bytes(entry.kv_checksum, kv_data, kv_size);
  entry.state = static_cast<std::uint8_t>(AmfEntryState::kHot);

  AmfEntry previous_entry{};
  if (has_existing) {
    previous_entry = entries_[it->second];
  }

  if (has_existing) {
    if (entry.kv_size > previous_entry.kv_size) {
      evict_if_needed(entry.kv_size - previous_entry.kv_size);
    }
  } else {
    evict_if_needed(kv_size);
  }

  if (!write_tokens_file(entry, tokens)) {
    stats_.store_failures += 1;
    return false;
  }
  if (!write_kv_file(entry, kv_data, kv_size)) {
    std::error_code ec;
    std::filesystem::remove(entry_tok_path(entry), ec);
    stats_.store_failures += 1;
    return false;
  }

  if (has_existing) {
    AmfEntry & target = entries_[it->second];
    if (entry.kv_size >= previous_entry.kv_size) {
      total_bytes_ += (entry.kv_size - previous_entry.kv_size);
    } else {
      total_bytes_ -= (previous_entry.kv_size - entry.kv_size);
    }
    entry.failure_count = previous_entry.failure_count;
    entry.block_count = previous_entry.block_count;
    entry.hit_count = previous_entry.hit_count;
    entry.last_hit_ms = previous_entry.last_hit_ms;
    entry.low_roi_hits = previous_entry.low_roi_hits;
    entry.last_access_time_ms = previous_entry.last_access_time_ms;
    entry.access_count = previous_entry.access_count;
    entry.stored_at_ms = previous_entry.stored_at_ms;
    entry.expires_at_ms = previous_entry.expires_at_ms;
    entry.size_bytes = entry.kv_size;
    entry.state = static_cast<std::uint8_t>(AmfEntryState::kHot);
    const std::uint64_t ttl_ms = effective_ttl_ms(entry.access_count);
    if (ttl_ms > 0) {
      entry.expires_at_ms = now_ms + ttl_ms;
    }
    target = entry;
  } else {
    index_[entry.key] = entries_.size();
    entries_.push_back(entry);
    total_bytes_ += kv_size;
  }

  stats_.admissions += 1;
  stats_.admission_bytes += kv_size;
  stats_.entries = static_cast<std::uint64_t>(entries_.size());
  stats_.bytes = total_bytes_;
  refresh_storage_pressure();

  if (!flush_index()) {
    stats_.store_failures += 1;
    return false;
  }
  return true;
}

void AmfStore::note_hit(const AmfEntry & entry,
                        std::uint64_t tokens_saved,
                        std::uint64_t restore_ms,
                        std::uint64_t saved_ms) {
  stats_.hits += 1;
  stats_.tokens_saved += tokens_saved;
  stats_.ms_saved += saved_ms;
  stats_.restore_ms += restore_ms;
  constexpr double kAlpha = 0.2;
  const double saved_ms_f = static_cast<double>(saved_ms);
  const double restore_ms_f = static_cast<double>(restore_ms);
  const double roi = (restore_ms_f > 0.0) ? (saved_ms_f / restore_ms_f) : 0.0;
  stats_.saved_ms_ema = update_ema(stats_.saved_ms_ema, saved_ms_f, kAlpha);
  stats_.restore_ms_ema = update_ema(stats_.restore_ms_ema, restore_ms_f, kAlpha);

  auto it = index_.find(entry.key);
  if (it != index_.end()) {
    AmfEntry & e = entries_[it->second];
    const std::uint64_t now_ms = now_epoch_ms();
    e.hit_count += 1;
    e.access_count += 1;
    e.restore_count += 1;
    e.last_used = now_ms;
    e.last_access_time_ms = now_ms;
    e.last_hit_ms = now_ms;
    e.state = static_cast<std::uint8_t>(AmfEntryState::kHot);
    const std::uint64_t ttl_ms = effective_ttl_ms(e.access_count);
    if (ttl_ms > 0) {
      e.expires_at_ms = now_ms + ttl_ms;
    }
    const double prev_roi = e.ema_roi;
    e.ema_restore_ms = update_ema(e.ema_restore_ms, restore_ms_f, kAlpha);
    e.ema_saved_ms = update_ema(e.ema_saved_ms, saved_ms_f, kAlpha);
    e.ema_roi = update_ema(e.ema_roi, roi, kAlpha);
    if (std::isfinite(prev_roi) && std::isfinite(e.ema_roi) && e.ema_roi < prev_roi) {
      if (roi_decline_streak_ < std::numeric_limits<std::uint32_t>::max()) {
        roi_decline_streak_ += 1;
      }
    } else {
      roi_decline_streak_ = 0;
    }
    if (roi_decline_streak_ == kRoiDeclineWarn) {
      e.failure_count += 1;
    }
    if (roi_decline_streak_ >= kRoiDeclineStop) {
      admissions_disabled_ = true;
    }
    if (min_admit_roi_ > 0.0 && low_roi_hit_limit_ > 0) {
      double roi_ema = e.ema_roi;
      if (!std::isfinite(roi_ema)) {
        roi_ema = 0.0;
      }
      if (roi_ema < min_admit_roi_) {
        if (e.low_roi_hits < std::numeric_limits<std::uint32_t>::max()) {
          e.low_roi_hits += 1;
        }
      } else {
        e.low_roi_hits = 0;
      }
      if (e.low_roi_hits >= low_roi_hit_limit_) {
        evict_entry_at(it->second, "low_roi");
        return;
      }
    }
    (void) flush_index();
  }
}

void AmfStore::note_store(std::uint64_t prompt_ms) {
  (void) prompt_ms;
  // EMAs update only on successful restore to avoid inferred ROI.
}

void AmfStore::note_failure(const AmfEntry & entry) {
  stats_.failures += 1;
  auto it = index_.find(entry.key);
  if (it != index_.end()) {
    AmfEntry & e = entries_[it->second];
    e.failure_count += 1;
    const std::uint64_t now_ms = now_epoch_ms();
    e.last_used = now_ms;
    e.last_access_time_ms = now_ms;
    (void) flush_index();
  }
}

void AmfStore::note_block() {
  stats_.blocks += 1;
}

void AmfStore::note_block(const AmfEntry & entry) {
  stats_.blocks += 1;
  auto it = index_.find(entry.key);
  if (it != index_.end()) {
    AmfEntry & e = entries_[it->second];
    e.block_count += 1;
    const std::uint64_t now_ms = now_epoch_ms();
    e.last_used = now_ms;
    e.last_access_time_ms = now_ms;
    (void) flush_index();
  }
}

AmfStorageStats AmfStore::storage_stats(std::uint64_t now_ms) const {
  AmfStorageStats out{};
  out.total_bytes = total_bytes_;
  out.budget_bytes = (storage_budget_bytes_ > 0) ? storage_budget_bytes_ : max_bytes_;
  if (out.budget_bytes > 0) {
    out.utilization = static_cast<double>(out.total_bytes) / static_cast<double>(out.budget_bytes);
    if (!std::isfinite(out.utilization) || out.utilization < 0.0) {
      out.utilization = 0.0;
    }
  }
  if (now_ms == 0) {
    now_ms = now_epoch_ms();
  }
  for (const AmfEntry & entry : entries_) {
    const AmfEntryTier tier = classify_tier(entry, now_ms);
    if (tier == AmfEntryTier::kHot) {
      out.hot_entries += 1;
    } else if (tier == AmfEntryTier::kWarm) {
      out.warm_entries += 1;
    } else {
      out.cold_entries += 1;
    }
  }
  std::error_code ec;
  const std::filesystem::space_info info = std::filesystem::space(dir_, ec);
  if (!ec) {
    out.disk_free_bytes = static_cast<std::uint64_t>(info.available);
    out.disk_capacity_bytes = static_cast<std::uint64_t>(info.capacity);
    if (out.disk_capacity_bytes > 0) {
      out.disk_free_ratio = static_cast<double>(out.disk_free_bytes) /
          static_cast<double>(out.disk_capacity_bytes);
    }
    if (disk_guard_enabled_) {
      out.low_disk =
          (min_free_disk_bytes_ > 0 && out.disk_free_bytes <= min_free_disk_bytes_) ||
          (min_free_disk_ratio_ > 0.0 && out.disk_free_ratio <= min_free_disk_ratio_);
    }
  }
  return out;
}

void AmfStore::note_miss() {
  stats_.misses += 1;
}

void AmfStore::evict_expired_entries(std::uint64_t now_ms) {
  if (!enabled_ || now_ms == 0 || entries_.empty()) {
    return;
  }

  std::size_t i = 0;
  while (i < entries_.size()) {
    const AmfEntry & e = entries_[i];
    if (e.expires_at_ms > 0 && now_ms > e.expires_at_ms) {
      const std::uint64_t age_ms = (now_ms > e.stored_at_ms) ? (now_ms - e.stored_at_ms) : 0;
      std::fprintf(stderr,
                   "[AMF_EXPIRED] hash=%016llx age_s=%llu\n",
                   static_cast<unsigned long long>(e.key.prefix_hash),
                   static_cast<unsigned long long>(age_ms / 1000ull));
      (void) std::fflush(stderr);
      evict_entry_at(i, "ttl");
      continue;
    }
    if (max_age_ms_ == 0) {
      ++i;
      continue;
    }
    const std::uint64_t last = (e.last_hit_ms > 0) ? e.last_hit_ms : e.last_used;
    if (last == 0 || now_ms <= last) {
      ++i;
      continue;
    }
    const std::uint64_t age_ms = now_ms - last;
    if (age_ms > max_age_ms_) {
      evict_entry_at(i, "age");
      continue;
    }
    ++i;
  }
}

float AmfStore::reuse_score(std::size_t prompt_len, std::uint32_t prefix_len) const {
  if (prompt_len == 0 || prefix_len == 0) {
    return 0.0f;
  }
  const double ratio = static_cast<double>(prefix_len) / static_cast<double>(prompt_len);
  if (!std::isfinite(ratio)) {
    return 0.0f;
  }
  return static_cast<float>(std::clamp(ratio, 0.0, 1.0));
}

float AmfStore::avg_prefix_length() const {
  if (entries_.empty()) {
    return 0.0f;
  }
  std::uint64_t sum = 0;
  for (const AmfEntry & e : entries_) {
    sum += static_cast<std::uint64_t>(e.prefix_len);
  }
  const double avg = static_cast<double>(sum) / static_cast<double>(entries_.size());
  if (!std::isfinite(avg)) {
    return 0.0f;
  }
  return static_cast<float>(avg);
}

float AmfStore::historical_accept_rate() const {
  const double denom = static_cast<double>(stats_.hits + stats_.misses);
  if (denom <= 0.0) {
    return 0.0f;
  }
  const double rate = static_cast<double>(stats_.hits) / denom;
  if (!std::isfinite(rate)) {
    return 0.0f;
  }
  return static_cast<float>(std::clamp(rate, 0.0, 1.0));
}

float AmfStore::restore_cost_ms() const {
  if (stats_.hits == 0) {
    return 0.0f;
  }
  const double avg = static_cast<double>(stats_.restore_ms) /
      static_cast<double>(stats_.hits);
  if (!std::isfinite(avg)) {
    return 0.0f;
  }
  return static_cast<float>(avg);
}

AmfStore::AmfSignals AmfStore::signals_for_prefix(std::size_t prompt_len,
                                                  std::uint32_t prefix_len) const {
  AmfSignals out{};
  out.reuse_score = reuse_score(prompt_len, prefix_len);
  out.avg_prefix_length = avg_prefix_length();
  out.historical_accept_rate = historical_accept_rate();
  out.restore_cost_ms = restore_cost_ms();
  return out;
}

}  // namespace korith::core
