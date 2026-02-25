#include "spec_verify.h"

#include "accept_scan.h"

#include <algorithm>
#include <cstddef>

namespace korith::core {
namespace {

constexpr int kMaxRows = 64;
constexpr int kThreads = 256;

struct DeviceVerifyResult {
  int32_t accepted_count;
  int32_t first_mismatch;
  int32_t mismatch_token;
};

__global__ void fused_verify_kernel(
    const float * logits,
    int32_t rows,
    int32_t n_vocab,
    const int32_t * draft_tokens,
    DeviceVerifyResult * out) {
  const int32_t tid = static_cast<int32_t>(threadIdx.x);

  extern __shared__ unsigned char smem_raw[];
  float * s_val = reinterpret_cast<float *>(smem_raw);
  int32_t * s_idx = reinterpret_cast<int32_t *>(s_val + blockDim.x);

  __shared__ int32_t s_done;
  __shared__ DeviceVerifyResult s_out;

  if (tid == 0) {
    s_done = 0;
    s_out.accepted_count = 0;
    s_out.first_mismatch = -1;
    s_out.mismatch_token = -1;
  }
  __syncthreads();

  if (!out || !logits || !draft_tokens || rows <= 0 || n_vocab <= 0) {
    if (tid == 0 && out) {
      *out = s_out;
    }
    return;
  }

  for (int32_t row = 0; row < rows; ++row) {
    if (s_done != 0) {
      break;
    }

    float best_val = -3.402823466e+38F;
    int32_t best_idx = 0;
    const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(n_vocab);
    for (int32_t i = tid; i < n_vocab; i += static_cast<int32_t>(blockDim.x)) {
      const float v = logits[base + static_cast<std::size_t>(i)];
      if (v > best_val || (v == best_val && i < best_idx)) {
        best_val = v;
        best_idx = i;
      }
    }

    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();

    for (int32_t offset = static_cast<int32_t>(blockDim.x) / 2; offset > 0; offset >>= 1) {
      if (tid < offset) {
        const float rhs_val = s_val[tid + offset];
        const int32_t rhs_idx = s_idx[tid + offset];
        const float lhs_val = s_val[tid];
        const int32_t lhs_idx = s_idx[tid];
        if (rhs_val > lhs_val || (rhs_val == lhs_val && rhs_idx < lhs_idx)) {
          s_val[tid] = rhs_val;
          s_idx[tid] = rhs_idx;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      const int32_t target_tok = s_idx[0];
      if (target_tok == draft_tokens[row]) {
        s_out.accepted_count += 1;
      } else {
        s_out.first_mismatch = row;
        s_out.mismatch_token = target_tok;
        s_done = 1;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[0] = s_out;
  }
}

struct SpecVerifyBuffers {
  float * logits_device = nullptr;
  int32_t * draft_tokens_device = nullptr;
  DeviceVerifyResult * result_device = nullptr;
  int32_t rows_capacity = 0;
  int32_t vocab_capacity = 0;
  cudaStream_t stream = nullptr;

  ~SpecVerifyBuffers() {
    if (logits_device) {
      cudaFree(logits_device);
    }
    if (draft_tokens_device) {
      cudaFree(draft_tokens_device);
    }
    if (result_device) {
      cudaFree(result_device);
    }
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }

  bool ensure_capacity(int32_t rows, int32_t n_vocab) {
    rows = std::clamp(rows, 1, kMaxRows);
    n_vocab = std::max<int32_t>(1, n_vocab);
    if (!stream) {
      if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        stream = nullptr;
        return false;
      }
    }
    if (rows <= rows_capacity && n_vocab <= vocab_capacity &&
        logits_device && draft_tokens_device && result_device) {
      return true;
    }

    if (logits_device) {
      cudaFree(logits_device);
      logits_device = nullptr;
    }
    if (draft_tokens_device) {
      cudaFree(draft_tokens_device);
      draft_tokens_device = nullptr;
    }
    if (result_device) {
      cudaFree(result_device);
      result_device = nullptr;
    }
    rows_capacity = 0;
    vocab_capacity = 0;

    const std::size_t logits_bytes =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(n_vocab) * sizeof(float);
    const std::size_t token_bytes = static_cast<std::size_t>(rows) * sizeof(int32_t);
    if (cudaMalloc(reinterpret_cast<void **>(&logits_device), logits_bytes) != cudaSuccess) {
      return false;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&draft_tokens_device), token_bytes) != cudaSuccess) {
      return false;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&result_device), sizeof(DeviceVerifyResult)) != cudaSuccess) {
      return false;
    }
    rows_capacity = rows;
    vocab_capacity = n_vocab;
    return true;
  }
};

SpecVerifyBuffers & verify_buffers() {
  static thread_local SpecVerifyBuffers buffers;
  return buffers;
}

}  // namespace

bool gpu_spec_verify_available() {
  return gpu_accept_scan_available();
}

bool gpu_spec_verify_greedy_segmented_from_host(
    const float * first_logits_row_host,
    const float * next_logits_rows_host,
    int32_t rows,
    int32_t n_vocab,
    const int32_t * draft_tokens_host,
    GpuSpecVerifyResult * out) {
  if (!out || !gpu_spec_verify_available() || !first_logits_row_host || !draft_tokens_host) {
    return false;
  }
  const int32_t safe_rows = std::clamp(rows, 0, kMaxRows);
  if (safe_rows <= 0 || n_vocab <= 0) {
    out->accepted_count = 0;
    out->first_mismatch = -1;
    out->mismatch_token = -1;
    return true;
  }
  if (safe_rows > 1 && !next_logits_rows_host) {
    return false;
  }

  SpecVerifyBuffers & buffers = verify_buffers();
  if (!buffers.ensure_capacity(safe_rows, n_vocab)) {
    return false;
  }

  const std::size_t row_bytes = static_cast<std::size_t>(n_vocab) * sizeof(float);
  if (cudaMemcpyAsync(
          buffers.logits_device,
          first_logits_row_host,
          row_bytes,
          cudaMemcpyHostToDevice,
          buffers.stream) != cudaSuccess) {
    return false;
  }
  if (safe_rows > 1) {
    const std::size_t tail_bytes =
        static_cast<std::size_t>(safe_rows - 1) * static_cast<std::size_t>(n_vocab) * sizeof(float);
    if (cudaMemcpyAsync(
            buffers.logits_device + static_cast<std::ptrdiff_t>(n_vocab),
            next_logits_rows_host,
            tail_bytes,
            cudaMemcpyHostToDevice,
            buffers.stream) != cudaSuccess) {
      return false;
    }
  }

  const std::size_t token_bytes = static_cast<std::size_t>(safe_rows) * sizeof(int32_t);
  if (cudaMemcpyAsync(
          buffers.draft_tokens_device,
          draft_tokens_host,
          token_bytes,
          cudaMemcpyHostToDevice,
          buffers.stream) != cudaSuccess) {
    return false;
  }

  const std::size_t shmem_bytes =
      static_cast<std::size_t>(kThreads) * (sizeof(float) + sizeof(int32_t));
  fused_verify_kernel<<<1, kThreads, shmem_bytes, buffers.stream>>>(
      buffers.logits_device,
      safe_rows,
      n_vocab,
      buffers.draft_tokens_device,
      buffers.result_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  DeviceVerifyResult result{};
  if (cudaMemcpyAsync(
          &result,
          buffers.result_device,
          sizeof(DeviceVerifyResult),
          cudaMemcpyDeviceToHost,
          buffers.stream) != cudaSuccess) {
    return false;
  }
  if (cudaStreamSynchronize(buffers.stream) != cudaSuccess) {
    return false;
  }

  out->accepted_count = std::clamp(result.accepted_count, 0, safe_rows);
  out->first_mismatch = result.first_mismatch;
  out->mismatch_token = result.mismatch_token;
  if (out->first_mismatch < 0 || out->first_mismatch >= safe_rows) {
    out->first_mismatch = -1;
    out->mismatch_token = -1;
  }
  return true;
}

bool gpu_spec_verify_greedy_from_host(
    const float * logits_rows_host,
    int32_t rows,
    int32_t n_vocab,
    const int32_t * draft_tokens_host,
    GpuSpecVerifyResult * out) {
  const int32_t safe_rows = std::clamp(rows, 0, kMaxRows);
  return gpu_spec_verify_greedy_segmented_from_host(
      logits_rows_host,
      safe_rows > 1 ? (logits_rows_host + static_cast<std::ptrdiff_t>(n_vocab)) : nullptr,
      rows,
      n_vocab,
      draft_tokens_host,
      out);
}

}  // namespace korith::core
