#include "amf_store.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
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
constexpr std::uint32_t kIndexVersion = 6u;
constexpr std::uint32_t kRoiDeclineWarn = 3u;
constexpr std::uint32_t kRoiDeclineStop = 5u;
constexpr double kEvictionPressureBump = 0.1;
constexpr double kEvictionPressureCap = 3.0;

struct IndexHeader {
  std::uint32_t magic = kIndexMagic;
  std::uint32_t version = kIndexVersion;
  std::uint64_t count = 0;
  std::uint64_t checksum = 0;
};

struct IndexRecord {
  AmfKey key{};
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
  mix(k.prefix_hash);
  mix(static_cast<std::uint64_t>(k.n_ctx));
  mix(static_cast<std::uint64_t>(k.kv_version));
  mix(static_cast<std::uint64_t>(k.rope_base_bits));
  mix(static_cast<std::uint64_t>(k.rope_scale_bits));
  mix(k.sampling_hash);
  mix(k.rng_hash);
  return static_cast<std::size_t>(h);
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
  std::uint64_t low_roi_hits = 0;
  if (parse_env_u64("KORITH_AMF_LOW_ROI_HITS", low_roi_hits) && low_roi_hits > 0) {
    low_roi_hit_limit_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(low_roi_hits, UINT32_MAX));
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
  roi_decline_streak_ = 0;
  admissions_disabled_ = false;

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
  if (hdr.version != kIndexVersion) {
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
    AmfEntry entry{};
    entry.key = rec.key;
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
    const std::size_t idx = entries_.size();
    entries_.push_back(entry);
    index_[entry.key] = idx;
    total_bytes_ += entry.kv_size;
  }

  if (checksum != hdr.checksum) {
    enabled_ = false;
    disable_reason_ = AmfDisableReason::kIoError;
    return false;
  }

  stats_.entries = static_cast<std::uint64_t>(entries_.size());
  stats_.bytes = total_bytes_;
  enabled_ = true;
  return true;
}

bool AmfStore::flush_index() const {
  const std::string tmp_path =
      (std::filesystem::path(dir_) / "amf_index.tmp").string();
  std::FILE * fp = std::fopen(tmp_path.c_str(), "wb");
  if (!fp) {
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
    hash_bytes(checksum,
               reinterpret_cast<const std::uint8_t *>(&rec),
               sizeof(rec));
  }

  hdr.checksum = checksum;
  if (std::fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
    std::fclose(fp);
    return false;
  }

  for (const AmfEntry & entry : entries_) {
    IndexRecord rec{};
    rec.key = entry.key;
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
    if (std::fwrite(&rec, sizeof(rec), 1, fp) != 1) {
      std::fclose(fp);
      return false;
    }
  }

  if (std::fflush(fp) != 0) {
    std::fclose(fp);
    return false;
  }
  if (::fsync(::fileno(fp)) != 0) {
    std::fclose(fp);
    return false;
  }
  if (std::fclose(fp) != 0) {
    return false;
  }

  std::error_code ec;
  std::filesystem::rename(tmp_path, index_path_, ec);
  if (ec) {
    std::filesystem::remove(tmp_path);
    return false;
  }
  return true;
}

std::string AmfStore::entry_basename(const AmfEntry & entry) const {
  char buf[256];
  std::snprintf(buf,
                sizeof(buf),
                "amf_%016llx_%016llx_%08x_%08x_%08x_%08x_%016llx_%016llx",
                static_cast<unsigned long long>(entry.key.model_hash),
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
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  if (!tokens.empty()) {
    out.write(reinterpret_cast<const char *>(tokens.data()),
              static_cast<std::streamsize>(tokens.size() * sizeof(llama_token)));
  }
  return static_cast<bool>(out);
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
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  if (size > 0) {
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
  }
  return static_cast<bool>(out);
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
  if (std::isfinite(v) && v > 0.0) {
    eviction_pressure_ = v;
  }
}

std::uint64_t AmfStore::oldest_entry_age_ms(std::uint64_t now_ms) const {
  if (entries_.empty() || now_ms == 0) {
    return 0;
  }
  std::uint64_t oldest = 0;
  for (const AmfEntry & e : entries_) {
    if (e.last_used == 0) {
      continue;
    }
    if (oldest == 0 || e.last_used < oldest) {
      oldest = e.last_used;
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
  std::error_code ec;
  std::filesystem::remove(entry_kv_path(entry), ec);
  std::filesystem::remove(entry_tok_path(entry), ec);

  if (entry.kv_size <= total_bytes_) {
    total_bytes_ -= entry.kv_size;
  } else {
    total_bytes_ = 0;
  }

  stats_.evictions += 1;
  stats_.evicted_bytes += entry.kv_size;

  entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(idx));
  index_.erase(entry.key);
  index_.clear();
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    index_[entries_[i].key] = i;
  }

  stats_.entries = static_cast<std::uint64_t>(entries_.size());
  stats_.bytes = total_bytes_;

  std::fprintf(stderr,
               "[AMF_EVICT] bytes=%llu reason=%s\n",
               static_cast<unsigned long long>(entry.kv_size),
               reason ? reason : "unknown");
  (void) std::fflush(stderr);
  (void) flush_index();
}

void AmfStore::evict_if_needed(std::size_t incoming_bytes) {
  if (max_bytes_ == 0 || total_bytes_ + incoming_bytes <= max_bytes_) {
    return;
  }

  auto entry_roi = [](const AmfEntry & e) -> double {
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

  auto entry_score = [&](const AmfEntry & e) -> double {
    const double roi = entry_roi(e);
    const double pressure = (eviction_pressure_ > 0.0) ? eviction_pressure_ : 1.0;
    const double penalty = 1.0 + pressure * 0.5 * static_cast<double>(e.failure_count + e.block_count);
    const double score = roi / penalty;
    return std::isfinite(score) ? score : 0.0;
  };

  while (total_bytes_ + incoming_bytes > max_bytes_ && !entries_.empty()) {
    auto it = std::min_element(entries_.begin(), entries_.end(), [&](const AmfEntry & a, const AmfEntry & b) {
      const double score_a = entry_score(a);
      const double score_b = entry_score(b);
      if (score_a == score_b) {
        if (a.hit_count == b.hit_count) {
          return a.last_used < b.last_used;
        }
        return a.hit_count < b.hit_count;
      }
      return score_a < score_b;
    });
    if (it == entries_.end()) {
      break;
    }
    const std::size_t idx = static_cast<std::size_t>(it - entries_.begin());
    evict_entry_at(idx, "roi");
  }
}

AmfLookupResult AmfStore::find_longest_prefix(const AmfContext & ctx,
                                              const std::vector<llama_token> & tokens,
                                              AmfEntry * out_entry,
                                              std::vector<llama_token> * out_tokens) const {
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

    const AmfEntry & entry = entries_[it->second];
    if (entry.prefix_len != len) {
      continue;
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
  return read_kv_file(entry, out);
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

  if (max_bytes_ > 0 && kv_size > max_bytes_) {
    stats_.admission_rejects += 1;
    return false;
  }

  const AmfKey key = amf_make_key(ctx, amf_hash_tokens(tokens));
  const auto it = index_.find(key);
  const bool has_existing = (it != index_.end());
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
  entry.last_used = now_epoch_ms();
  entry.kv_size = kv_size;

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

  if (!write_tokens_file(entry, tokens) || !write_kv_file(entry, kv_data, kv_size)) {
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
    e.restore_count += 1;
    e.last_used = now_ms;
    e.last_hit_ms = now_ms;
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
      if (std::isfinite(eviction_pressure_) && eviction_pressure_ < kEvictionPressureCap) {
        eviction_pressure_ = std::min(kEvictionPressureCap,
                                      eviction_pressure_ + kEvictionPressureBump);
      }
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
    e.last_used = now_epoch_ms();
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
    e.last_used = now_epoch_ms();
    (void) flush_index();
  }
}

void AmfStore::note_miss() {
  stats_.misses += 1;
}

void AmfStore::evict_expired_entries(std::uint64_t now_ms) {
  if (!enabled_ || max_age_ms_ == 0 || now_ms == 0 || entries_.empty()) {
    return;
  }

  std::size_t i = 0;
  while (i < entries_.size()) {
    const AmfEntry & e = entries_[i];
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
