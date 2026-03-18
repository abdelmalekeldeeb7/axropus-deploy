#include "../core/amf_store.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::uint64_t now_ms() {
  const auto now = std::chrono::system_clock::now();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
}

constexpr std::uint64_t kFNVOffset = 1469598103934665603ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

void hash_bytes_for_test(std::uint64_t & h, const std::uint8_t * data, std::size_t len) {
  for (std::size_t i = 0; i < len; ++i) {
    h ^= static_cast<std::uint64_t>(data[i]);
    h *= kFNVPrime;
  }
}

struct TestIndexHeader {
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint64_t count = 0;
  std::uint64_t checksum = 0;
};

struct TestIndexRecord {
  korith::core::AmfKey key{};
  std::uint64_t model_hash = 0;
  std::uint32_t prefix_len = 0;
  std::uint32_t reserved = 0;
  std::uint64_t seen_count = 0;
  std::uint64_t restore_count = 0;
  double ema_saved_ms = 0.0;
  double ema_restore_ms = 0.0;
  double ema_roi = 0.0;
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

bool rewrite_index_model_hash(const std::string & index_path, std::uint64_t model_hash) {
  std::ifstream in(index_path, std::ios::binary);
  if (!in) {
    return false;
  }
  TestIndexHeader hdr{};
  in.read(reinterpret_cast<char *>(&hdr), static_cast<std::streamsize>(sizeof(hdr)));
  if (!in || hdr.count == 0) {
    return false;
  }
  std::vector<TestIndexRecord> records(static_cast<std::size_t>(hdr.count));
  in.read(reinterpret_cast<char *>(records.data()),
          static_cast<std::streamsize>(records.size() * sizeof(TestIndexRecord)));
  if (!in) {
    return false;
  }
  in.close();

  records[0].model_hash = model_hash;
  std::uint64_t checksum = kFNVOffset;
  TestIndexHeader hdr_zero = hdr;
  hdr_zero.checksum = 0;
  hash_bytes_for_test(checksum, reinterpret_cast<const std::uint8_t *>(&hdr_zero), sizeof(hdr_zero));
  for (const TestIndexRecord & rec : records) {
    hash_bytes_for_test(checksum, reinterpret_cast<const std::uint8_t *>(&rec), sizeof(rec));
  }
  hdr.checksum = checksum;

  std::ofstream out(index_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  out.write(reinterpret_cast<const char *>(&hdr), static_cast<std::streamsize>(sizeof(hdr)));
  out.write(reinterpret_cast<const char *>(records.data()),
            static_cast<std::streamsize>(records.size() * sizeof(TestIndexRecord)));
  return static_cast<bool>(out);
}

void set_common_env(const std::string & dir) {
  (void) setenv("KORITH_ENABLE_AMF", "1", 1);
  (void) setenv("KORITH_AMF_PATH", dir.c_str(), 1);
  (void) setenv("KORITH_AMF_MIN_TOKENS", "2", 1);
}

korith::core::AmfContext make_ctx() {
  korith::core::AmfContext ctx{};
  ctx.model_hash = 0x1234;
  ctx.tenant_hash = 0;
  ctx.n_ctx = 4096;
  ctx.kv_version = 1;
  ctx.rope_base_bits = korith::core::amf_float_bits(0.0f);
  ctx.rope_scale_bits = korith::core::amf_float_bits(1.0f);
  ctx.sampling_hash = 0x1111;
  ctx.rng_hash = 0x2222;
  return ctx;
}

korith::core::AmfContext make_ctx_for_tenant(const std::string & tenant_id) {
  korith::core::AmfContext ctx = make_ctx();
  ctx.tenant_hash = korith::core::amf_hash_tenant_id(tenant_id);
  return ctx;
}

bool has_prefix(korith::core::AmfStore & store,
                const korith::core::AmfContext & ctx,
                const std::vector<llama_token> & tokens) {
  korith::core::AmfEntry entry{};
  std::vector<llama_token> out_tokens;
  const auto res = store.find_longest_prefix(ctx, tokens, &entry, &out_tokens);
  return (res == korith::core::AmfLookupResult::kHit) &&
      entry.prefix_len == tokens.size() &&
      out_tokens == tokens;
}

bool store_prefix(korith::core::AmfStore & store,
                  const korith::core::AmfContext & ctx,
                  const std::vector<llama_token> & tokens,
                  std::size_t kv_size,
                  std::uint8_t fill = 0xab) {
  std::vector<std::uint8_t> kv(kv_size, fill);
  return store.store_entry(ctx, tokens, kv.data(), kv.size(), 12);
}

bool note_hits(korith::core::AmfStore & store,
               const korith::core::AmfContext & ctx,
               const std::vector<llama_token> & tokens,
               int hits) {
  for (int i = 0; i < hits; ++i) {
    korith::core::AmfEntry entry{};
    std::vector<llama_token> out_tokens;
    if (store.find_longest_prefix(ctx, tokens, &entry, &out_tokens) != korith::core::AmfLookupResult::kHit) {
      return false;
    }
    store.note_hit(entry, tokens.size(), 1, 5);
  }
  return true;
}

bool wait_for_prewarm(korith::core::AmfStore & store, int timeout_ms = 3000) {
  const auto start = std::chrono::steady_clock::now();
  while (!store.prewarm_complete()) {
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    if (elapsed > timeout_ms) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return true;
}

int test_basic_lookup() {
  const std::string dir = "./build/amf_test_store_basic";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "1048576", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 1;
  }
  const korith::core::AmfContext ctx = make_ctx();

  std::vector<llama_token> tokens = {1, 2, 3, 4};
  if (!store_prefix(store, ctx, tokens, 16)) {
    return 2;
  }

  std::vector<llama_token> prompt = {1, 2, 3, 4};
  if (!has_prefix(store, ctx, prompt)) {
    return 3;
  }

  korith::core::AmfContext ctx_bad = ctx;
  ctx_bad.sampling_hash = 0x3333;
  if (has_prefix(store, ctx_bad, prompt)) {
    return 4;
  }
  ctx_bad = ctx;
  ctx_bad.rng_hash = 0x4444;
  if (has_prefix(store, ctx_bad, prompt)) {
    return 5;
  }
  return 0;
}

int test_tenant_hash_isolation() {
  const std::string dir = "./build/amf_test_store_tenant_isolation";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "1048576", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 6;
  }

  const auto ctx_a = make_ctx_for_tenant("tenant-a");
  const auto ctx_b = make_ctx_for_tenant("tenant-b");
  const std::vector<llama_token> tokens = {7, 8, 9, 10};
  if (!store_prefix(store, ctx_a, tokens, 16, 0x01)) {
    return 7;
  }
  if (!has_prefix(store, ctx_a, tokens)) {
    return 8;
  }
  if (has_prefix(store, ctx_b, tokens)) {
    return 9;
  }
  return 0;
}

int test_eviction_triggers_at_watermark() {
  const std::string dir = "./build/amf_test_store_watermark";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "100", 1);
  (void) setenv("KORITH_AMF_EVICTION_WATERMARK", "0.9", 1);
  (void) setenv("KORITH_AMF_EVICTION_POLICY", "lfu", 1);
  (void) setenv("KORITH_AMF_HOT_TTL_S", "60", 1);
  (void) setenv("KORITH_AMF_WARM_TTL_S", "120", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 11;
  }
  const korith::core::AmfContext ctx = make_ctx();

  if (!store_prefix(store, ctx, {11, 12}, 40, 0x01)) {
    return 12;
  }
  if (!store_prefix(store, ctx, {21, 22}, 40, 0x02)) {
    return 13;
  }
  if (!store_prefix(store, ctx, {31, 32}, 40, 0x03)) {
    return 14;
  }

  if (store.stats().evictions == 0) {
    return 15;
  }
  return 0;
}

int test_cold_before_warm() {
  const std::string dir = "./build/amf_test_store_tier_order";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "100", 1);
  (void) setenv("KORITH_AMF_EVICTION_WATERMARK", "0.9", 1);
  (void) setenv("KORITH_AMF_EVICTION_POLICY", "lfu", 1);
  (void) setenv("KORITH_AMF_HOT_TTL_S", "1", 1);
  (void) setenv("KORITH_AMF_WARM_TTL_S", "2", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 21;
  }
  const korith::core::AmfContext ctx = make_ctx();
  const std::vector<llama_token> cold = {41, 42};
  const std::vector<llama_token> warm = {51, 52};
  const std::vector<llama_token> hot = {61, 62};

  if (!store_prefix(store, ctx, cold, 40, 0x11)) {
    return 22;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(2200));
  if (!store_prefix(store, ctx, warm, 40, 0x22)) {
    return 23;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));
  if (!store_prefix(store, ctx, hot, 40, 0x33)) {
    return 24;
  }

  if (has_prefix(store, ctx, cold)) {
    return 25;
  }
  if (!has_prefix(store, ctx, warm)) {
    return 26;
  }
  if (!has_prefix(store, ctx, hot)) {
    return 27;
  }
  return 0;
}

int test_hot_preserved_until_critical() {
  const std::string dir = "./build/amf_test_store_hot_preserve";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "100", 1);
  (void) setenv("KORITH_AMF_EVICTION_WATERMARK", "0.9", 1);
  (void) setenv("KORITH_AMF_EVICTION_POLICY", "lfu", 1);
  (void) setenv("KORITH_AMF_HOT_TTL_S", "100", 1);
  (void) setenv("KORITH_AMF_WARM_TTL_S", "200", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 31;
  }
  const korith::core::AmfContext ctx = make_ctx();

  if (!store_prefix(store, ctx, {71, 72}, 45, 0x44)) {
    return 32;
  }
  if (!store_prefix(store, ctx, {81, 82}, 45, 0x55)) {
    return 33;
  }
  const std::uint64_t evictions_before = store.stats().evictions;
  if (!store_prefix(store, ctx, {91, 92}, 5, 0x66)) {
    return 34;
  }
  if (store.stats().evictions != evictions_before) {
    return 35;
  }
  if (!store_prefix(store, ctx, {101, 102}, 10, 0x77)) {
    return 36;
  }
  if (store.stats().evictions <= evictions_before) {
    return 37;
  }
  return 0;
}

int test_storage_stats() {
  const std::string dir = "./build/amf_test_store_stats";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "1000", 1);
  (void) setenv("KORITH_AMF_EVICTION_WATERMARK", "0.9", 1);
  (void) setenv("KORITH_AMF_HOT_TTL_S", "1", 1);
  (void) setenv("KORITH_AMF_WARM_TTL_S", "3", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 41;
  }
  const korith::core::AmfContext ctx = make_ctx();
  if (!store_prefix(store, ctx, {111, 112}, 32, 0x88)) {
    return 42;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(3200));
  if (!store_prefix(store, ctx, {121, 122}, 32, 0x99)) {
    return 43;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));
  if (!store_prefix(store, ctx, {131, 132}, 32, 0xaa)) {
    return 44;
  }

  const korith::core::AmfStorageStats storage = store.storage_stats(now_ms());
  if (storage.hot_entries != 1 || storage.warm_entries != 1 || storage.cold_entries != 1) {
    return 45;
  }
  if (storage.total_bytes != 96) {
    return 46;
  }
  if (storage.budget_bytes != 1000) {
    return 47;
  }
  if (!(storage.utilization > 0.09 && storage.utilization < 0.10)) {
    return 48;
  }
  return 0;
}

int test_expired_entries_miss() {
  const std::string dir = "./build/amf_test_store_ttl_expire";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "1000", 1);
  (void) setenv("KORITH_AMF_DEFAULT_TTL_S", "1", 1);
  (void) setenv("KORITH_AMF_MAX_TTL_S", "2", 1);
  (void) setenv("KORITH_AMF_MODEL_VERSION_CHECK", "1", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 51;
  }
  const korith::core::AmfContext ctx = make_ctx();
  const std::vector<llama_token> tokens = {141, 142};
  if (!store_prefix(store, ctx, tokens, 32, 0xbb)) {
    return 52;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));
  if (has_prefix(store, ctx, tokens)) {
    return 53;
  }
  if (store.entry_count() != 0) {
    return 54;
  }
  return 0;
}

int test_model_hash_mismatch_miss() {
  const std::string dir = "./build/amf_test_store_model_mismatch";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "1000", 1);
  (void) setenv("KORITH_AMF_DEFAULT_TTL_S", "30", 1);
  (void) setenv("KORITH_AMF_MAX_TTL_S", "60", 1);
  (void) setenv("KORITH_AMF_MODEL_VERSION_CHECK", "1", 1);

  {
    korith::core::AmfStore store;
    if (!store.init_from_env()) {
      return 61;
    }
    const korith::core::AmfContext ctx = make_ctx();
    if (!store_prefix(store, ctx, {151, 152}, 32, 0xcc)) {
      return 62;
    }
  }

  const std::string index_path = (std::filesystem::path(dir) / "amf_index.bin").string();
  if (!rewrite_index_model_hash(index_path, 0xfeedbeefull)) {
    return 63;
  }

  korith::core::AmfStore store2;
  if (!store2.init_from_env()) {
    return 64;
  }
  const korith::core::AmfContext ctx = make_ctx();
  if (has_prefix(store2, ctx, {151, 152})) {
    return 65;
  }
  if (store2.entry_count() != 0) {
    return 66;
  }
  return 0;
}

int test_adaptive_ttl_extension() {
  const std::string dir = "./build/amf_test_store_ttl_adaptive";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "1000", 1);
  (void) setenv("KORITH_AMF_DEFAULT_TTL_S", "1", 1);
  (void) setenv("KORITH_AMF_MAX_TTL_S", "4", 1);
  (void) setenv("KORITH_AMF_MODEL_VERSION_CHECK", "1", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 71;
  }
  const korith::core::AmfContext ctx = make_ctx();
  const std::vector<llama_token> tokens = {161, 162};
  if (!store_prefix(store, ctx, tokens, 32, 0xdd)) {
    return 72;
  }

  for (int i = 0; i < 8; ++i) {
    korith::core::AmfEntry entry{};
    std::vector<llama_token> out_tokens;
    if (store.find_longest_prefix(ctx, tokens, &entry, &out_tokens) != korith::core::AmfLookupResult::kHit) {
      return 73;
    }
    store.note_hit(entry, tokens.size(), 1, 5);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(2200));
  if (!has_prefix(store, ctx, tokens)) {
    return 74;
  }
  return 0;
}

int test_max_ttl_cap() {
  const std::string dir = "./build/amf_test_store_ttl_cap";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "1000", 1);
  (void) setenv("KORITH_AMF_DEFAULT_TTL_S", "1", 1);
  (void) setenv("KORITH_AMF_MAX_TTL_S", "2", 1);
  (void) setenv("KORITH_AMF_MODEL_VERSION_CHECK", "1", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 81;
  }
  const korith::core::AmfContext ctx = make_ctx();
  const std::vector<llama_token> tokens = {171, 172};
  if (!store_prefix(store, ctx, tokens, 32, 0xee)) {
    return 82;
  }

  for (int i = 0; i < 32; ++i) {
    korith::core::AmfEntry entry{};
    std::vector<llama_token> out_tokens;
    if (store.find_longest_prefix(ctx, tokens, &entry, &out_tokens) != korith::core::AmfLookupResult::kHit) {
      return 83;
    }
    store.note_hit(entry, tokens.size(), 1, 5);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(2300));
  if (has_prefix(store, ctx, tokens)) {
    return 84;
  }
  return 0;
}

int test_prewarm_top_k() {
  const std::string dir = "./build/amf_test_store_prewarm_topk";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "10000", 1);
  (void) setenv("KORITH_AMF_PREWARM", "0", 1);

  {
    korith::core::AmfStore store;
    if (!store.init_from_env()) {
      return 91;
    }
    const korith::core::AmfContext ctx = make_ctx();
    if (!store_prefix(store, ctx, {181, 182}, 64, 0x01)) {
      return 92;
    }
    if (!store_prefix(store, ctx, {191, 192}, 64, 0x02)) {
      return 93;
    }
    if (!store_prefix(store, ctx, {201, 202}, 64, 0x03)) {
      return 94;
    }
    if (!note_hits(store, ctx, {181, 182}, 6)) {
      return 95;
    }
    if (!note_hits(store, ctx, {191, 192}, 3)) {
      return 96;
    }
    store.shutdown();
  }

  (void) setenv("KORITH_AMF_PREWARM", "1", 1);
  (void) setenv("KORITH_AMF_PREWARM_TOP_K", "2", 1);
  (void) setenv("KORITH_AMF_PREWARM_TIMEOUT_S", "5", 1);
  korith::core::AmfStore store2;
  if (!store2.init_from_env()) {
    return 97;
  }
  if (!wait_for_prewarm(store2, 5000)) {
    return 98;
  }
  if (store2.prewarm_loaded_count() < 2) {
    return 99;
  }
  if (store2.warm_ratio() < 0.66) {
    return 100;
  }
  return 0;
}

int test_prewarm_timeout() {
  const std::string dir = "./build/amf_test_store_prewarm_timeout";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "10000", 1);
  (void) setenv("KORITH_AMF_PREWARM", "0", 1);

  {
    korith::core::AmfStore store;
    if (!store.init_from_env()) {
      return 101;
    }
    const korith::core::AmfContext ctx = make_ctx();
    if (!store_prefix(store, ctx, {211, 212}, 64, 0x11)) {
      return 102;
    }
    store.shutdown();
  }

  (void) setenv("KORITH_AMF_PREWARM", "1", 1);
  (void) setenv("KORITH_AMF_PREWARM_TOP_K", "10", 1);
  (void) setenv("KORITH_AMF_PREWARM_TIMEOUT_S", "0", 1);
  korith::core::AmfStore store2;
  if (!store2.init_from_env()) {
    return 103;
  }
  if (!wait_for_prewarm(store2, 2000)) {
    return 104;
  }
  if (store2.prewarm_loaded_count() != 0) {
    return 105;
  }
  return 0;
}

int test_shutdown_persists_index() {
  const std::string dir = "./build/amf_test_store_shutdown";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "10000", 1);
  (void) setenv("KORITH_AMF_PREWARM", "0", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 111;
  }
  const korith::core::AmfContext ctx = make_ctx();
  if (!store_prefix(store, ctx, {221, 222}, 64, 0x22)) {
    return 112;
  }
  store.shutdown();
  const std::filesystem::path index_path = std::filesystem::path(dir) / "amf_index.bin";
  if (!std::filesystem::exists(index_path)) {
    return 113;
  }
  if (std::filesystem::file_size(index_path) == 0) {
    return 114;
  }
  return 0;
}

int test_disk_guard_rejects_admission() {
  const std::string dir = "./build/amf_test_store_disk_guard";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);
  set_common_env(dir);
  (void) setenv("KORITH_AMF_MAX_BYTES", "10000", 1);
  (void) setenv("KORITH_AMF_DISK_GUARD", "1", 1);
  (void) setenv("KORITH_AMF_MIN_FREE_DISK_BYTES", "9000000000000000000", 1);
  (void) setenv("KORITH_AMF_MIN_FREE_DISK_PCT", "0.0", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 121;
  }
  const korith::core::AmfContext ctx = make_ctx();
  if (store_prefix(store, ctx, {231, 232}, 64, 0x33)) {
    return 122;
  }
  if (store.stats().admission_rejects == 0) {
    return 123;
  }
  if (store.stats().store_failures != 0) {
    return 124;
  }
  const korith::core::AmfStorageStats storage = store.storage_stats(now_ms());
  if (!storage.low_disk) {
    return 125;
  }
  return 0;
}

}  // namespace

int main() {
  const int rc_basic = test_basic_lookup();
  if (rc_basic != 0) {
    return rc_basic;
  }
  const int rc_tenant = test_tenant_hash_isolation();
  if (rc_tenant != 0) {
    return rc_tenant;
  }
  const int rc_watermark = test_eviction_triggers_at_watermark();
  if (rc_watermark != 0) {
    return rc_watermark;
  }
  const int rc_tier = test_cold_before_warm();
  if (rc_tier != 0) {
    return rc_tier;
  }
  const int rc_hot = test_hot_preserved_until_critical();
  if (rc_hot != 0) {
    return rc_hot;
  }
  const int rc_storage = test_storage_stats();
  if (rc_storage != 0) {
    return rc_storage;
  }
  const int rc_expired = test_expired_entries_miss();
  if (rc_expired != 0) {
    return rc_expired;
  }
  const int rc_model = test_model_hash_mismatch_miss();
  if (rc_model != 0) {
    return rc_model;
  }
  const int rc_adaptive = test_adaptive_ttl_extension();
  if (rc_adaptive != 0) {
    return rc_adaptive;
  }
  const int rc_cap = test_max_ttl_cap();
  if (rc_cap != 0) {
    return rc_cap;
  }
  const int rc_prewarm_topk = test_prewarm_top_k();
  if (rc_prewarm_topk != 0) {
    return rc_prewarm_topk;
  }
  const int rc_prewarm_timeout = test_prewarm_timeout();
  if (rc_prewarm_timeout != 0) {
    return rc_prewarm_timeout;
  }
  const int rc_shutdown = test_shutdown_persists_index();
  if (rc_shutdown != 0) {
    return rc_shutdown;
  }
  const int rc_disk_guard = test_disk_guard_rejects_admission();
  if (rc_disk_guard != 0) {
    return rc_disk_guard;
  }
  return 0;
}
