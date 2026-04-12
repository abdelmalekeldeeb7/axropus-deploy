/*
 * fp8_decode_attention.cu — Scalar FP8 decode attention kernel.
 *
 * Reads FP8 E4M3 KV from the AMF compressed pool, computes attention in
 * FP32, writes FP16 output. One CTA per (batch, head). Uses online softmax
 * with correct accumulator rescaling across tiles.
 *
 * NOT tensor-core-accelerated — this is a bandwidth-reduction kernel that
 * wins via 2x fewer HBM bytes read, not via MMA throughput.
 * TODO: WGMMA + TMA rewrite for full Hopper utilisation (requires H200).
 *
 * Kernel contract:
 *
 *   Inputs:
 *     Q               [B, H, 1, D]     half
 *     K_fp8           [B, H, T, D]     __nv_fp8_e4m3
 *     V_fp8           [B, H, T, D]     __nv_fp8_e4m3
 *     k_scale, v_scale                 float per-tensor scales
 *     mask            [B, 1, 1, T]     half (optional)
 *
 *   Output:
 *     O               [B, H, 1, D]     half
 *
 * Build note: This file is compiled by ``torch.utils.cpp_extension.load``
 * via dispatch.py::try_load_cuda_extension. When CUDA is unavailable the
 * kernel is never built and the pure-Python fallback is used instead.
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include <torch/extension.h>

namespace axropus {
namespace fp8 {

// ── Tile sizes ─────────────────────────────────────────────────────────────

constexpr int kQHeads      = 1;     // decode: one query token per step
constexpr int kKvTileLen   = 128;   // KV tokens per CTA tile
constexpr int kHeadDimMax  = 128;
constexpr int kNumWarps    = 4;     // 4 warps per CTA = 128 threads

// ── Helpers ────────────────────────────────────────────────────────────────

__device__ __forceinline__ float expf_fast(float x) {
    return __expf(x);
}

// Online softmax running stats.
struct SoftmaxState {
    float m;   // running max
    float s;   // running denominator (sum of exp)
};

__device__ __forceinline__ void online_softmax_update(
    SoftmaxState& state,
    float* scores,
    int len
) {
    float new_m = state.m;
    for (int i = 0; i < len; ++i) {
        new_m = fmaxf(new_m, scores[i]);
    }
    float scale = expf_fast(state.m - new_m);
    float new_s = state.s * scale;
    for (int i = 0; i < len; ++i) {
        float p = expf_fast(scores[i] - new_m);
        scores[i] = p;
        new_s += p;
    }
    state.m = new_m;
    state.s = new_s;
}

// ── Main kernel ────────────────────────────────────────────────────────────
//
// One CTA per (batch, head) pair. Thread block iterates through KV tiles
// and accumulates attention incrementally using the online softmax trick.

template<int HeadDim>
__global__ void fp8_decode_attention_kernel(
    const __half*       __restrict__ Q,          // [B, H, 1, D]
    const __nv_fp8_e4m3* __restrict__ K,         // [B, H, T, D]
    const __nv_fp8_e4m3* __restrict__ V,         // [B, H, T, D]
    __half*             __restrict__ O,          // [B, H, 1, D]
    int                 num_kv_tokens,           // T
    float               k_scale,
    float               v_scale,
    float               softmax_scale,
    const __half*       __restrict__ mask        // optional [B, 1, 1, T]
) {
    const int batch_id = blockIdx.y;
    const int head_id  = blockIdx.x;
    const int lane     = threadIdx.x;

    const int q_offset = (batch_id * gridDim.x + head_id) * HeadDim;
    const int kv_base  = (batch_id * gridDim.x + head_id) * num_kv_tokens * HeadDim;

    // Load Q into registers.
    __shared__ float q_shared[HeadDim];
    if (lane < HeadDim) {
        q_shared[lane] = __half2float(Q[q_offset + lane]);
    }
    __syncthreads();

    // Running softmax state.
    __shared__ SoftmaxState smem_state;
    if (lane == 0) {
        smem_state.m = -CUDART_INF_F;
        smem_state.s = 0.0f;
    }
    __syncthreads();

    // Running output accumulator in registers (FP32 for numerical stability).
    float out_acc[HeadDim];
    #pragma unroll
    for (int i = 0; i < HeadDim; ++i) out_acc[i] = 0.0f;

    // Iterate over KV tiles.
    const int n_tiles = (num_kv_tokens + kKvTileLen - 1) / kKvTileLen;
    __shared__ float scores_tile[kKvTileLen];
    __shared__ float old_max_smem;

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int kv_start = tile * kKvTileLen;
        const int kv_end   = min(kv_start + kKvTileLen, num_kv_tokens);
        const int tile_len = kv_end - kv_start;

        // Step 1: Compute Q · K^T for this tile.
        for (int t = lane; t < tile_len; t += blockDim.x) {
            const int token = kv_start + t;
            const int k_off = kv_base + token * HeadDim;
            float acc = 0.0f;
            #pragma unroll
            for (int d = 0; d < HeadDim; ++d) {
                const float k_val = static_cast<float>(K[k_off + d]) * k_scale;
                acc += q_shared[d] * k_val;
            }
            acc *= softmax_scale;
            if (mask != nullptr) {
                const int m_off = batch_id * num_kv_tokens + token;
                acc += __half2float(mask[m_off]);
            }
            scores_tile[t] = acc;
        }
        __syncthreads();

        // Step 2: Save old max, then compute new softmax stats (thread 0).
        if (lane == 0) {
            old_max_smem = smem_state.m;
            online_softmax_update(smem_state, scores_tile, tile_len);
        }
        __syncthreads();

        // Step 3: RESCALE previous accumulator by exp(old_max - new_max).
        // Without this, out_acc from earlier tiles carries weights computed
        // under the old max, producing incorrect attention output for any
        // sequence longer than one tile (128 tokens).
        float rescale = expf_fast(old_max_smem - smem_state.m);
        #pragma unroll
        for (int d = 0; d < HeadDim; ++d) {
            out_acc[d] *= rescale;
        }

        // Step 4: Accumulate attn(probs) · V for this tile.
        // Threads cooperate over tokens, reduce into shared memory.
        __shared__ float v_accum[kHeadDimMax];
        for (int d = lane; d < HeadDim; d += blockDim.x) {
            v_accum[d] = 0.0f;
        }
        __syncthreads();

        for (int t = lane; t < tile_len; t += blockDim.x) {
            const float p = scores_tile[t];
            const int v_off = kv_base + (kv_start + t) * HeadDim;
            for (int d = 0; d < HeadDim; ++d) {
                const float v_val = static_cast<float>(V[v_off + d]) * v_scale;
                atomicAdd(&v_accum[d], p * v_val);
            }
        }
        __syncthreads();

        for (int d = lane; d < HeadDim; d += blockDim.x) {
            out_acc[d] += v_accum[d];
        }
        __syncthreads();
    }

    // Final normalization.
    const float inv_s = 1.0f / fmaxf(smem_state.s, 1e-12f);

    // Write output.
    for (int d = lane; d < HeadDim; d += blockDim.x) {
        O[q_offset + d] = __float2half(out_acc[d] * inv_s);
    }
}

// ── Host launcher ──────────────────────────────────────────────────────────

torch::Tensor fp8_decode_attention(
    torch::Tensor q,            // [B, H, 1, D] half
    torch::Tensor k,            // [B, H, T, D] uint8 (FP8 bytes)
    torch::Tensor v,            // [B, H, T, D] uint8
    double k_scale,
    double v_scale,
    double softmax_scale,
    torch::optional<torch::Tensor> mask
) {
    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    TORCH_CHECK(k.is_cuda() && v.is_cuda(), "k, v must be on CUDA");
    TORCH_CHECK(q.dim() == 4 && q.size(2) == 1, "q must be [B, H, 1, D]");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k, v must be [B, H, T, D]");

    const int B = q.size(0);
    const int H = q.size(1);
    const int D = q.size(3);
    const int T = k.size(2);

    TORCH_CHECK(D == 128, "Only head_dim=128 is supported by this prototype");

    auto out = torch::empty({B, H, 1, D}, q.options());

    dim3 grid(H, B);
    dim3 block(128);

    const __half*      q_ptr    = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const __nv_fp8_e4m3* k_ptr  = reinterpret_cast<const __nv_fp8_e4m3*>(k.data_ptr<uint8_t>());
    const __nv_fp8_e4m3* v_ptr  = reinterpret_cast<const __nv_fp8_e4m3*>(v.data_ptr<uint8_t>());
    __half*            o_ptr    = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    const __half*      mask_ptr = mask.has_value()
        ? reinterpret_cast<const __half*>(mask.value().data_ptr<at::Half>())
        : nullptr;

    fp8_decode_attention_kernel<128><<<grid, block>>>(
        q_ptr, k_ptr, v_ptr, o_ptr,
        T,
        static_cast<float>(k_scale),
        static_cast<float>(v_scale),
        static_cast<float>(softmax_scale),
        mask_ptr
    );
    return out;
}

}  // namespace fp8
}  // namespace axropus

#endif  // __CUDACC__
