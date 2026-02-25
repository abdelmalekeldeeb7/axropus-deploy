#pragma once

#include <cstdint>

#if defined(KORITH_USE_CUDA_ACCEPT_SCAN)
#include <cuda_runtime.h>
#endif

namespace korith::core {

#if defined(KORITH_USE_CUDA_ACCEPT_SCAN)

struct GpuSpecVerifyResult {
  int32_t accepted_count = 0;
  int32_t first_mismatch = -1;
  int32_t mismatch_token = -1;
};

bool gpu_spec_verify_available();

// Greedy-argmax each logits row on GPU and compare against draft tokens with GPU accept-scan.
// Returns only compact comparison outputs to host.
bool gpu_spec_verify_greedy_from_host(
    const float * logits_rows_host,
    int32_t rows,
    int32_t n_vocab,
    const int32_t * draft_tokens_host,
    GpuSpecVerifyResult * out);

// Variant used by Spec V3 verify path to avoid building an intermediate
// host-side contiguous logits buffer.
// - first_logits_row_host: logits at the base verify position.
// - next_logits_rows_host: logits for subsequent rows (rows-1), contiguous.
bool gpu_spec_verify_greedy_segmented_from_host(
    const float * first_logits_row_host,
    const float * next_logits_rows_host,
    int32_t rows,
    int32_t n_vocab,
    const int32_t * draft_tokens_host,
    GpuSpecVerifyResult * out);

#else

struct GpuSpecVerifyResult {
  int32_t accepted_count = 0;
  int32_t first_mismatch = -1;
  int32_t mismatch_token = -1;
};

inline bool gpu_spec_verify_available() {
  return false;
}

inline bool gpu_spec_verify_greedy_from_host(
    const float *,
    int32_t,
    int32_t,
    const int32_t *,
    GpuSpecVerifyResult *) {
  return false;
}

inline bool gpu_spec_verify_greedy_segmented_from_host(
    const float *,
    const float *,
    int32_t,
    int32_t,
    const int32_t *,
    GpuSpecVerifyResult *) {
  return false;
}

#endif

}  // namespace korith::core
