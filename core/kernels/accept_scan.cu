#include "accept_scan.h"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace korith::core {
namespace {

constexpr int kMaxScanTokens = 64;
constexpr int kKernelThreads = 64;  // Two warps (covers 8/12/16/24 and future small blocks).

__global__ void set_scan_outputs_kernel(int32_t accepted,
                                        int32_t mismatch,
                                        int32_t * accepted_count_device,
                                        int32_t * first_mismatch_device) {
  if (threadIdx.x == 0) {
    if (accepted_count_device) {
      *accepted_count_device = accepted;
    }
    if (first_mismatch_device) {
      *first_mismatch_device = mismatch;
    }
  }
}

__global__ void accept_scan_kernel(const int32_t * draft_tokens,
                                   const int32_t * target_tokens,
                                   int block_size,
                                   int32_t * accepted_count_device,
                                   int32_t * first_mismatch_device) {
  __shared__ int warp_first_mismatch[2];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int safe_block_size = max(0, min(block_size, kMaxScanTokens));

  if (lane == 0 && warp < 2) {
    warp_first_mismatch[warp] = safe_block_size;
  }
  __syncthreads();

  const bool in_range = tid < safe_block_size;
  const bool mismatch = in_range && (draft_tokens[tid] != target_tokens[tid]);
  const unsigned int mismatch_mask = __ballot_sync(0xffffffffu, mismatch);

  if (lane == 0 && warp < 2) {
    int first = safe_block_size;
    if (mismatch_mask != 0u) {
      first = (warp * 32) + (__ffs(mismatch_mask) - 1);
    }
    warp_first_mismatch[warp] = first;
  }
  __syncthreads();

  if (tid == 0) {
    const int first = min(warp_first_mismatch[0], warp_first_mismatch[1]);
    const int32_t first_mismatch = (first < safe_block_size) ? first : -1;
    const int32_t accepted = (first_mismatch >= 0) ? first_mismatch : safe_block_size;

    if (accepted_count_device) {
      *accepted_count_device = accepted;
    }
    if (first_mismatch_device) {
      *first_mismatch_device = first_mismatch;
    }
  }
}

struct AcceptScanBuffers {
  int32_t * draft_tokens_device = nullptr;
  int32_t * target_tokens_device = nullptr;
  int32_t * accepted_count_device = nullptr;
  int32_t * first_mismatch_device = nullptr;
  int capacity = 0;
  cudaStream_t stream = nullptr;

  ~AcceptScanBuffers() {
    if (draft_tokens_device) {
      cudaFree(draft_tokens_device);
    }
    if (target_tokens_device) {
      cudaFree(target_tokens_device);
    }
    if (accepted_count_device) {
      cudaFree(accepted_count_device);
    }
    if (first_mismatch_device) {
      cudaFree(first_mismatch_device);
    }
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }

  bool ensure_capacity(int requested) {
    const int needed = std::clamp(requested, 1, kMaxScanTokens);
    if (stream == nullptr) {
      if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        stream = nullptr;
        return false;
      }
    }
    if (capacity >= needed && draft_tokens_device && target_tokens_device &&
        accepted_count_device && first_mismatch_device) {
      return true;
    }

    if (draft_tokens_device) {
      cudaFree(draft_tokens_device);
      draft_tokens_device = nullptr;
    }
    if (target_tokens_device) {
      cudaFree(target_tokens_device);
      target_tokens_device = nullptr;
    }
    if (accepted_count_device) {
      cudaFree(accepted_count_device);
      accepted_count_device = nullptr;
    }
    if (first_mismatch_device) {
      cudaFree(first_mismatch_device);
      first_mismatch_device = nullptr;
    }
    capacity = 0;

    const std::size_t bytes = static_cast<std::size_t>(needed) * sizeof(int32_t);
    if (cudaMalloc(reinterpret_cast<void **>(&draft_tokens_device), bytes) != cudaSuccess) {
      return false;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&target_tokens_device), bytes) != cudaSuccess) {
      return false;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&accepted_count_device), sizeof(int32_t)) != cudaSuccess) {
      return false;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&first_mismatch_device), sizeof(int32_t)) != cudaSuccess) {
      return false;
    }
    capacity = needed;
    return true;
  }
};

AcceptScanBuffers & host_bridge_buffers() {
  static thread_local AcceptScanBuffers buffers;
  return buffers;
}

}  // namespace

bool gpu_accept_scan_available() {
  static bool initialized = false;
  static bool available = false;
  if (!initialized) {
    initialized = true;
    int device_count = 0;
    const cudaError_t rc = cudaGetDeviceCount(&device_count);
    available = (rc == cudaSuccess) && (device_count > 0);
  }
  return available;
}

const char * gpu_accept_scan_backend() {
  return gpu_accept_scan_available() ? "cuda" : "none";
}

void gpu_accept_scan(const int32_t * draft_tokens_device,
                     const int32_t * target_tokens_device,
                     int block_size,
                     int32_t * accepted_count_device,
                     int32_t * first_mismatch_device,
                     cudaStream_t stream) {
  const int safe_block_size = std::clamp(block_size, 0, kMaxScanTokens);
  if (accepted_count_device == nullptr || first_mismatch_device == nullptr ||
      draft_tokens_device == nullptr || target_tokens_device == nullptr ||
      safe_block_size <= 0) {
    set_scan_outputs_kernel<<<1, 1, 0, stream>>>(0, -1, accepted_count_device, first_mismatch_device);
    return;
  }
  accept_scan_kernel<<<1, kKernelThreads, 0, stream>>>(
      draft_tokens_device,
      target_tokens_device,
      safe_block_size,
      accepted_count_device,
      first_mismatch_device);
}

bool gpu_accept_scan_from_host(const int32_t * draft_tokens_host,
                               const int32_t * target_tokens_host,
                               int block_size,
                               int32_t * accepted_count_host,
                               int32_t * first_mismatch_host) {
  if (!gpu_accept_scan_available()) {
    return false;
  }
  if (!draft_tokens_host || !target_tokens_host || !accepted_count_host) {
    return false;
  }

  if (block_size > kMaxScanTokens) {
    std::fprintf(stderr,
                 "[ACCEPT_SCAN] warning: block_size=%d exceeds kMaxScanTokens=%d; "
                 "clamping to %d. Spec depth should be capped upstream.\n",
                 block_size, kMaxScanTokens, kMaxScanTokens);
    (void) std::fflush(stderr);
  }

  const int safe_block_size = std::clamp(block_size, 0, kMaxScanTokens);
  if (safe_block_size <= 0) {
    *accepted_count_host = 0;
    if (first_mismatch_host) {
      *first_mismatch_host = -1;
    }
    return true;
  }

  AcceptScanBuffers & buffers = host_bridge_buffers();
  if (!buffers.ensure_capacity(safe_block_size)) {
    return false;
  }

  const std::size_t bytes = static_cast<std::size_t>(safe_block_size) * sizeof(int32_t);
  if (cudaMemcpyAsync(
          buffers.draft_tokens_device,
          draft_tokens_host,
          bytes,
          cudaMemcpyHostToDevice,
          buffers.stream) != cudaSuccess) {
    return false;
  }
  if (cudaMemcpyAsync(
          buffers.target_tokens_device,
          target_tokens_host,
          bytes,
          cudaMemcpyHostToDevice,
          buffers.stream) != cudaSuccess) {
    return false;
  }

  gpu_accept_scan(
      buffers.draft_tokens_device,
      buffers.target_tokens_device,
      safe_block_size,
      buffers.accepted_count_device,
      buffers.first_mismatch_device,
      buffers.stream);

  if (cudaMemcpyAsync(
          accepted_count_host,
          buffers.accepted_count_device,
          sizeof(int32_t),
          cudaMemcpyDeviceToHost,
          buffers.stream) != cudaSuccess) {
    return false;
  }

  if (first_mismatch_host) {
    if (cudaMemcpyAsync(
            first_mismatch_host,
            buffers.first_mismatch_device,
            sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            buffers.stream) != cudaSuccess) {
      return false;
    }
  }

  if (cudaStreamSynchronize(buffers.stream) != cudaSuccess) {
    return false;
  }

  return true;
}

}  // namespace korith::core

