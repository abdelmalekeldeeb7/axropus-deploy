#include "korith_engine.h"

#include <algorithm>
#include <cstring>

namespace {

korith_engine_metrics g_metrics{};
bool g_ready = false;

}  // namespace

extern "C" {

bool korith_engine_init_model(const korith_engine_params * params) {
  if (!params || !params->model_path || params->model_path[0] == '\0') {
    return false;
  }
  g_metrics = {};
  g_ready = true;
  return true;
}

bool korith_engine_prefill(const int32_t * /*tokens*/, int32_t n_tokens) {
  if (!g_ready || n_tokens <= 0) {
    return false;
  }
  g_metrics.prefill_ms = std::max(0.0, static_cast<double>(n_tokens) * 0.01);
  return true;
}

bool korith_engine_decode_step(int32_t * token_out) {
  if (!g_ready || !token_out) {
    return false;
  }
  *token_out = 0;
  g_metrics.decode_ms += 0.01;
  g_metrics.total_ms += 0.01;
  g_metrics.tokens_out += 1;
  return true;
}

bool korith_engine_get_logits(float * out_logits, int32_t n_vocab) {
  if (!g_ready || !out_logits || n_vocab <= 0) {
    return false;
  }
  std::memset(out_logits, 0, static_cast<std::size_t>(n_vocab) * sizeof(float));
  return true;
}

bool korith_engine_apply_kv_replay(const uint8_t * blob, std::size_t n_bytes) {
  if (!g_ready || !blob || n_bytes == 0) {
    return false;
  }
  g_metrics.restore_ms = 0.05;
  return true;
}

bool korith_engine_export_snapshot(const char * path) {
  return g_ready && path && path[0] != '\0';
}

bool korith_engine_import_snapshot(const char * path) {
  if (!g_ready || !path || path[0] == '\0') {
    return false;
  }
  g_metrics.restore_ms = 0.05;
  return true;
}

bool korith_engine_verify_tokens(const int32_t * proposed_tokens, int32_t n_tokens, int32_t * accepted_prefix) {
  if (!g_ready || !accepted_prefix || !proposed_tokens || n_tokens < 0) {
    return false;
  }
  *accepted_prefix = n_tokens;
  g_metrics.verify_ms += 0.01 * static_cast<double>(n_tokens);
  g_metrics.accepted_tokens += n_tokens;
  return true;
}

void korith_engine_get_metrics(korith_engine_metrics * out_metrics) {
  if (!out_metrics) {
    return;
  }
  *out_metrics = g_metrics;
}

void korith_engine_shutdown() {
  g_ready = false;
  g_metrics = {};
}

 #ifndef KORITH_ENGINE_USE_CUDA
bool korith_engine_cuda_probe() {
  return false;
}
 #endif

}
