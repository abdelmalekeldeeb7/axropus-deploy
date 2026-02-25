#pragma once

#include <cstdint>

#if defined(KORITH_USE_CUDA_ACCEPT_SCAN)
#include <cuda_runtime.h>
#endif

namespace korith::core {

#if defined(KORITH_USE_CUDA_ACCEPT_SCAN)

bool gpu_accept_scan_available();
const char * gpu_accept_scan_backend();

void gpu_accept_scan(const int32_t * draft_tokens_device,
                     const int32_t * target_tokens_device,
                     int block_size,
                     int32_t * accepted_count_device,
                     int32_t * first_mismatch_device,
                     cudaStream_t stream);

bool gpu_accept_scan_from_host(const int32_t * draft_tokens_host,
                               const int32_t * target_tokens_host,
                               int block_size,
                               int32_t * accepted_count_host,
                               int32_t * first_mismatch_host = nullptr);

#else

using cudaStream_t = void *;

inline bool gpu_accept_scan_available() {
  return false;
}

inline const char * gpu_accept_scan_backend() {
  return "none";
}

inline void gpu_accept_scan(const int32_t *,
                            const int32_t *,
                            int,
                            int32_t *,
                            int32_t *,
                            cudaStream_t) {}

inline bool gpu_accept_scan_from_host(const int32_t *,
                                      const int32_t *,
                                      int,
                                      int32_t *,
                                      int32_t * = nullptr) {
  return false;
}

#endif

}  // namespace korith::core
