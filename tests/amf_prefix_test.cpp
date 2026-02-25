#include "../core/amf_store.h"

#include <cstdlib>
#include <filesystem>
#include <vector>

int main() {
  const char * dir = "./build/amf_test_store";
  (void) std::filesystem::remove_all(dir);
  (void) std::filesystem::create_directories(dir);

  (void) setenv("KORITH_ENABLE_AMF", "1", 1);
  (void) setenv("KORITH_AMF_PATH", dir, 1);
  (void) setenv("KORITH_AMF_MIN_TOKENS", "2", 1);

  korith::core::AmfStore store;
  if (!store.init_from_env()) {
    return 1;
  }

  korith::core::AmfContext ctx{};
  ctx.model_hash = 0x1234;
  ctx.n_ctx = 4096;
  ctx.kv_version = 1;
  ctx.rope_base_bits = korith::core::amf_float_bits(0.0f);
  ctx.rope_scale_bits = korith::core::amf_float_bits(1.0f);
  ctx.sampling_hash = 0x1111;
  ctx.rng_hash = 0x2222;

  std::vector<llama_token> tokens = {1, 2, 3, 4};
  std::vector<std::uint8_t> kv(16, 0xab);

  if (!store.store_entry(ctx, tokens, kv.data(), kv.size(), 12)) {
    return 2;
  }

  std::vector<llama_token> prompt = {1, 2, 3, 4, 5};
  korith::core::AmfEntry entry{};
  std::vector<llama_token> out_tokens;
  const auto res = store.find_longest_prefix(ctx, prompt, &entry, &out_tokens);
  if (res != korith::core::AmfLookupResult::kHit) {
    return 3;
  }
  if (entry.prefix_len != tokens.size()) {
    return 4;
  }

  korith::core::AmfContext ctx_bad = ctx;
  ctx_bad.sampling_hash = 0x3333;
  const auto res_bad = store.find_longest_prefix(ctx_bad, prompt, &entry, &out_tokens);
  if (res_bad != korith::core::AmfLookupResult::kMissHashMismatch) {
    return 5;
  }

  ctx_bad = ctx;
  ctx_bad.rng_hash = 0x4444;
  const auto res_rng = store.find_longest_prefix(ctx_bad, prompt, &entry, &out_tokens);
  if (res_rng != korith::core::AmfLookupResult::kMissHashMismatch) {
    return 6;
  }
  return 0;
}
