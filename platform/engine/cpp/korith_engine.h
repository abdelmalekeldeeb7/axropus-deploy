#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

struct korith_engine_params {
  int32_t n_ctx;
  int32_t n_batch;
  int32_t n_gpu;
  const char * model_path;
};

struct korith_engine_metrics {
  double prefill_ms;
  double decode_ms;
  double total_ms;
  double restore_ms;
  double verify_ms;
  double draft_ms;
  int32_t tokens_out;
  int32_t accepted_tokens;
  int32_t rejected_tokens;
};

bool korith_engine_init_model(const korith_engine_params * params);
bool korith_engine_prefill(const int32_t * tokens, int32_t n_tokens);
bool korith_engine_decode_step(int32_t * token_out);
bool korith_engine_get_logits(float * out_logits, int32_t n_vocab);
bool korith_engine_apply_kv_replay(const uint8_t * blob, std::size_t n_bytes);
bool korith_engine_export_snapshot(const char * path);
bool korith_engine_import_snapshot(const char * path);
bool korith_engine_verify_tokens(const int32_t * proposed_tokens, int32_t n_tokens, int32_t * accepted_prefix);
void korith_engine_get_metrics(korith_engine_metrics * out_metrics);
void korith_engine_shutdown();
bool korith_engine_cuda_probe();

}
