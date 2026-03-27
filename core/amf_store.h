#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <llama.h>

namespace korith::core {

// AMF invariants:
// - Replay only allowed under deterministic sampling.
// - ROI must only be computed when baseline is stable.
// - Sustained negative ROI disables replay.
// - Replay remains disabled until MF cooldown expires.
// - AMF must fail closed on corruption or uncertainty.

struct AmfKey {
  std::uint64_t model_hash = 0;
  std::uint64_t tenant_hash = 0;
  std::uint64_t prefix_hash = 0;
  std::uint32_t n_ctx = 0;
  std::uint32_t kv_version = 0;
  std::uint32_t rope_base_bits = 0;
  std::uint32_t rope_scale_bits = 0;
  std::uint64_t sampling_hash = 0;
  std::uint64_t rng_hash = 0;

  bool operator==(const AmfKey & other) const {
    return model_hash == other.model_hash &&
        tenant_hash == other.tenant_hash &&
        prefix_hash == other.prefix_hash &&
        n_ctx == other.n_ctx &&
        kv_version == other.kv_version &&
        rope_base_bits == other.rope_base_bits &&
        rope_scale_bits == other.rope_scale_bits &&
        sampling_hash == other.sampling_hash &&
        rng_hash == other.rng_hash;
  }
};

struct AmfContext {
  std::uint64_t model_hash = 0;
  std::uint64_t tenant_hash = 0;
  std::uint32_t n_ctx = 0;
  std::uint32_t kv_version = 0;
  std::uint32_t rope_base_bits = 0;
  std::uint32_t rope_scale_bits = 0;
  std::uint64_t sampling_hash = 0;
  std::uint64_t rng_hash = 0;
};

struct AmfEntry {
  AmfKey key{};
  std::uint64_t model_hash = 0;
  std::uint32_t prefix_len = 0;
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
  std::uint64_t kv_size = 0;
  std::uint64_t last_access_time_ms = 0;
  std::uint64_t access_count = 0;
  std::uint64_t stored_at_ms = 0;
  std::uint64_t expires_at_ms = 0;
  std::uint64_t size_bytes = 0;
  std::uint64_t kv_checksum = 0;
  std::uint8_t state = 0;
};

struct AmfStats {
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t tokens_saved = 0;
  std::uint64_t ms_saved = 0;
  std::uint64_t restore_ms = 0;
  double saved_ms_ema = std::numeric_limits<double>::quiet_NaN();
  double restore_ms_ema = std::numeric_limits<double>::quiet_NaN();
  std::uint64_t failures = 0;
  std::uint64_t blocks = 0;
  std::uint64_t admissions = 0;
  std::uint64_t admission_rejects = 0;
  std::uint64_t admission_bytes = 0;
  std::uint64_t store_failures = 0;
  std::uint64_t evictions = 0;
  std::uint64_t evicted_bytes = 0;
  std::uint64_t entries = 0;
  std::uint64_t bytes = 0;
};

enum class AmfEntryTier : std::uint8_t {
  kHot = 0,
  kWarm,
  kCold,
};

enum class AmfEvictionPolicy : std::uint8_t {
  kLru = 0,
  kLfu,
  kRoi,
};

enum class AmfEntryState : std::uint8_t {
  kCold = 0,
  kWarmPending,
  kHot,
};

struct AmfStorageStats {
  std::uint64_t total_bytes = 0;
  std::uint64_t budget_bytes = 0;
  double utilization = 0.0;
  std::uint64_t hot_entries = 0;
  std::uint64_t warm_entries = 0;
  std::uint64_t cold_entries = 0;
  std::uint64_t disk_free_bytes = 0;
  std::uint64_t disk_capacity_bytes = 0;
  double disk_free_ratio = 0.0;
  bool low_disk = false;
};

enum class AmfDisableReason : std::uint8_t {
  kNone = 0,
  kEnvOff,
  kNoPath,
  kIoError,
  kStateApiMissing,
};

enum class AmfLookupResult : std::uint8_t {
  kHit = 0,
  kMissFirstRun,
  kMissShortPrefix,
  kMissHashMismatch,
  kMissTokenMismatch,
  kMissLoadFailed,
};

const char * amf_disable_reason_name(AmfDisableReason r);
const char * amf_lookup_reason_name(AmfLookupResult r);

std::uint64_t amf_hash_file(const std::string & path);
std::uint64_t amf_hash_tenant_id(const std::string & tenant_id);
std::uint64_t amf_hash_tokens(const std::vector<llama_token> & tokens);
std::uint64_t amf_hash_tokens_prefix(const std::vector<llama_token> & tokens, std::size_t len);
std::uint32_t amf_float_bits(float v);
AmfKey amf_make_key(const AmfContext & ctx, std::uint64_t prefix_hash);

class AmfStore {
public:
  ~AmfStore();

  bool init_from_env();
  bool init(const std::string & path);

  bool enabled() const { return enabled_; }
  const char * disabled_reason() const { return amf_disable_reason_name(disable_reason_); }
  void log_disabled_once() const;

  std::size_t min_tokens() const { return min_tokens_; }
  double min_prompt_ms() const { return min_prompt_ms_; }
  double min_roi() const { return min_roi_; }
  double min_admit_roi() const { return min_admit_roi_; }
  double eviction_pressure() const { return eviction_pressure_; }
  std::uint64_t storage_budget_bytes() const { return storage_budget_bytes_; }
  double eviction_watermark() const { return eviction_watermark_; }
  std::uint64_t max_bytes() const { return max_bytes_; }
  std::uint64_t total_bytes() const { return total_bytes_; }
  std::uint64_t entry_count() const { return static_cast<std::uint64_t>(entries_.size()); }
  const AmfStats & stats() const { return stats_; }
  AmfStorageStats storage_stats(std::uint64_t now_ms) const;
  bool prewarm_complete() const { return prewarm_complete_.load(); }
  std::uint64_t prewarm_loaded_count() const { return prewarm_loaded_.load(); }
  std::uint64_t prewarm_failed_count() const { return prewarm_failed_.load(); }
  double warm_ratio() const;
  void shutdown();

  void set_min_prompt_ms(double v);
  void set_min_admit_roi(double v);
  void set_eviction_pressure(double v);

  std::uint64_t oldest_entry_age_ms(std::uint64_t now_ms) const;

  float reuse_score(std::size_t prompt_len, std::uint32_t prefix_len) const;
  float avg_prefix_length() const;
  float historical_accept_rate() const;
  float restore_cost_ms() const;

  struct AmfSignals {
    float reuse_score = 0.0f;
    float avg_prefix_length = 0.0f;
    float historical_accept_rate = 0.0f;
    float restore_cost_ms = 0.0f;
  };

  AmfSignals signals_for_prefix(std::size_t prompt_len, std::uint32_t prefix_len) const;

  AmfLookupResult find_longest_prefix(const AmfContext & ctx,
                                      const std::vector<llama_token> & tokens,
                                      AmfEntry * out_entry,
                                      std::vector<llama_token> * out_tokens);
  bool load_kv(const AmfEntry & entry, std::vector<std::uint8_t> * out) const;
  // Load KV blob directly into a caller-provided buffer (e.g. pinned memory),
  // avoiding the intermediate heap allocation in load_kv().  buf must be at
  // least entry.size_bytes bytes.  Sets *out_size to the actual bytes read.
  bool load_kv_into(const AmfEntry & entry,
                    void * buf, std::size_t buf_size,
                    std::size_t * out_size) const;
  bool store_entry(const AmfContext & ctx,
                   const std::vector<llama_token> & tokens,
                   const std::uint8_t * kv_data,
                   std::size_t kv_size,
                   std::uint64_t saved_ms);

  void note_hit(const AmfEntry & entry,
                std::uint64_t tokens_saved,
                std::uint64_t restore_ms,
                std::uint64_t saved_ms);
  void note_store(std::uint64_t prompt_ms);
  void note_failure(const AmfEntry & entry);
  void note_block();
  void note_block(const AmfEntry & entry);
  void note_miss();
  void evict_expired_entries(std::uint64_t now_ms);

private:
  static constexpr std::size_t kInvalidIndex = static_cast<std::size_t>(-1);

  struct KeyHash {
    std::size_t operator()(const AmfKey & k) const noexcept;
  };

  bool load_index();
  bool flush_index() const;
  bool write_tokens_file(const AmfEntry & entry, const std::vector<llama_token> & tokens) const;
  bool read_tokens_file(const AmfEntry & entry, std::vector<llama_token> * out) const;
  bool write_kv_file(const AmfEntry & entry, const std::uint8_t * data, std::size_t size) const;
  bool read_kv_file(const AmfEntry & entry, std::vector<std::uint8_t> * out) const;
  std::string entry_basename(const AmfEntry & entry) const;
  std::string entry_kv_path(const AmfEntry & entry) const;
  std::string entry_tok_path(const AmfEntry & entry) const;
  void evict_if_needed(std::size_t incoming_bytes);
  void evict_entry_at(std::size_t idx, const char * reason);
  void refresh_storage_pressure();
  bool disk_guard_allows(std::size_t incoming_bytes,
                         std::uint64_t * out_available,
                         std::uint64_t * out_capacity,
                         double * out_free_ratio) const;
  void log_store_fail(const char * stage, const char * reason, const char * path) const;
  AmfEntryTier classify_tier(const AmfEntry & entry, std::uint64_t now_ms) const;
  std::size_t pick_budget_eviction_candidate(AmfEntryTier tier, std::uint64_t now_ms) const;
  const char * tier_name(AmfEntryTier tier) const;
  const char * eviction_policy_name() const;
  std::uint64_t effective_ttl_ms(std::uint64_t access_count) const;
  const char * entry_state_name(AmfEntryState state) const;
  void start_prewarm_async();
  void run_prewarm(std::vector<AmfEntry> ranked);
  bool is_prewarmed_key(const AmfKey & key) const;
  void mark_prewarmed_key(const AmfKey & key);
  bool should_wait_warm_pending(const AmfEntry & entry) const;

  std::string dir_;
  std::string index_path_;
  bool enabled_ = false;
  mutable bool disabled_logged_ = false;
  AmfDisableReason disable_reason_ = AmfDisableReason::kNone;
  std::size_t min_tokens_ = 16;
  double min_prompt_ms_ = 0.0;
  double min_roi_ = 0.0;
  double min_admit_roi_ = 0.0;
  std::uint64_t max_bytes_ = 512ull * 1024ull * 1024ull;
  std::uint64_t storage_budget_bytes_ = 512ull * 1024ull * 1024ull;
  double eviction_watermark_ = 0.90;
  double critical_watermark_ = 0.98;
  std::uint64_t hot_ttl_ms_ = 5ull * 60ull * 1000ull;
  std::uint64_t warm_ttl_ms_ = 60ull * 60ull * 1000ull;
  AmfEvictionPolicy eviction_policy_ = AmfEvictionPolicy::kLru;
  std::uint64_t total_bytes_ = 0;
  double eviction_pressure_ = 1.0;
  std::uint64_t max_age_ms_ = 0;
  std::uint64_t default_ttl_ms_ = 24ull * 60ull * 60ull * 1000ull;
  std::uint64_t max_ttl_ms_ = 7ull * 24ull * 60ull * 60ull * 1000ull;
  bool model_version_check_ = true;
  std::uint32_t low_roi_hit_limit_ = 3;
  std::uint32_t roi_decline_streak_ = 0;
  bool admissions_disabled_ = false;
  bool prewarm_enabled_ = false;
  std::uint32_t prewarm_top_k_ = 10;
  std::uint64_t prewarm_timeout_ms_ = 30000;
  std::uint64_t prewarm_wait_ms_ = 100;
  bool disk_guard_enabled_ = true;
  std::uint64_t min_free_disk_bytes_ = 0;
  double min_free_disk_ratio_ = 0.0;
  mutable std::mutex prewarm_mu_;
  std::thread prewarm_thread_;
  std::unordered_map<AmfKey, bool, KeyHash> prewarmed_keys_;
  std::atomic<bool> prewarm_running_{false};
  std::atomic<bool> prewarm_complete_{true};
  std::atomic<bool> prewarm_cancel_{false};
  std::atomic<std::uint64_t> prewarm_loaded_{0};
  std::atomic<std::uint64_t> prewarm_failed_{0};
  AmfStats stats_{};

  std::vector<AmfEntry> entries_;
  std::unordered_map<AmfKey, std::size_t, KeyHash> index_;
};

}  // namespace korith::core
