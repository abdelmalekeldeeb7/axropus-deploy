/*
 * int4_decode_attention.cu — Scalar INT4 decode attention kernel.
 *
 * Unpacks INT4 from uint8, dequantizes via per-block scale, computes
 * attention in FP32. 4x bandwidth reduction vs FP16.
 * TODO: Rewrite with __byte_perm + WGMMA for Hopper, mma.m16n8k16 for Ampere.
 *
 * Storage layout (matches INT4PerBlockCodec):
 *
 *   K_packed   [B, H, T/2, D]         uint8 (two int4 values per byte,
 *                                           low nibble first)
 *   K_scales   [B, H, T/BLOCK]        half
 *   zero_point 0 (symmetric)
 *
 * Dequant maps each nibble ``n`` (0..15) to ``(n - 8) * scale``.
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <torch/extension.h>

namespace axropus {
namespace int4 {

constexpr int kHeadDim   = 128;
constexpr int kBlockLen  = 128;    // per-block scale granularity
constexpr int kThreads   = 128;

// Unpack two 4-bit values from a byte and shift to signed range.
__device__ __forceinline__ void unpack_int4(uint8_t byte, int& lo, int& hi) {
    lo = static_cast<int>(byte & 0x0F) - 8;
    hi = static_cast<int>((byte >> 4) & 0x0F) - 8;
}

__global__ void int4_decode_attention_kernel(
    const __half*    __restrict__ Q,           // [B, H, 1, D] half
    const uint8_t*   __restrict__ K_packed,    // [B, H, T/2, D] uint8
    const uint8_t*   __restrict__ V_packed,    // [B, H, T/2, D] uint8
    const __half*    __restrict__ K_scales,    // [B, H, ceil(T/BLOCK)]
    const __half*    __restrict__ V_scales,    // [B, H, ceil(T/BLOCK)]
    __half*          __restrict__ O,           // [B, H, 1, D]
    int              num_kv_tokens,            // T
    float            softmax_scale,
    const __half*    __restrict__ mask         // optional
) {
    const int batch_id = blockIdx.y;
    const int head_id  = blockIdx.x;
    const int lane     = threadIdx.x;

    const int q_off  = (batch_id * gridDim.x + head_id) * kHeadDim;
    const int kv_off = (batch_id * gridDim.x + head_id) * num_kv_tokens * kHeadDim;
    const int scale_off = (batch_id * gridDim.x + head_id) * ((num_kv_tokens + kBlockLen - 1) / kBlockLen);

    // Load Q.
    __shared__ float q_shared[kHeadDim];
    if (lane < kHeadDim) {
        q_shared[lane] = __half2float(Q[q_off + lane]);
    }
    __syncthreads();

    // Running softmax.
    __shared__ float smem_max;
    __shared__ float smem_denom;
    if (lane == 0) {
        smem_max   = -CUDART_INF_F;
        smem_denom = 0.0f;
    }
    __syncthreads();

    float out_acc[kHeadDim];
    #pragma unroll
    for (int i = 0; i < kHeadDim; ++i) out_acc[i] = 0.0f;

    __shared__ float scores_tile[kBlockLen];
    __shared__ float old_max;

    const int n_blocks = (num_kv_tokens + kBlockLen - 1) / kBlockLen;
    for (int blk = 0; blk < n_blocks; ++blk) {
        const int token_start = blk * kBlockLen;
        const int token_end   = min(token_start + kBlockLen, num_kv_tokens);
        const int tile_len    = token_end - token_start;

        const float k_scale = __half2float(K_scales[scale_off + blk]);
        const float v_scale = __half2float(V_scales[scale_off + blk]);

        // Step 1: Score each KV token in the block.
        for (int t = lane; t < tile_len; t += kThreads) {
            const int token = token_start + t;
            float acc = 0.0f;
            const int k_byte_off = kv_off / 2 + token * (kHeadDim / 2);
            #pragma unroll
            for (int d2 = 0; d2 < kHeadDim / 2; ++d2) {
                const uint8_t byte = K_packed[k_byte_off + d2];
                int lo, hi;
                unpack_int4(byte, lo, hi);
                const float k0 = static_cast<float>(lo) * k_scale;
                const float k1 = static_cast<float>(hi) * k_scale;
                acc += q_shared[2 * d2] * k0;
                acc += q_shared[2 * d2 + 1] * k1;
            }
            acc *= softmax_scale;
            if (mask != nullptr) {
                const int m_off = batch_id * num_kv_tokens + token;
                acc += __half2float(mask[m_off]);
            }
            scores_tile[t] = acc;
        }
        __syncthreads();

        // Step 2: Save old max, then update running max/denom (thread 0).
        if (lane == 0) {
            old_max = smem_max;
            float local_max = smem_max;
            for (int t = 0; t < tile_len; ++t) {
                local_max = fmaxf(local_max, scores_tile[t]);
            }
            float rescale = __expf(smem_max - local_max);
            float local_denom = smem_denom * rescale;
            for (int t = 0; t < tile_len; ++t) {
                float p = __expf(scores_tile[t] - local_max);
                scores_tile[t] = p;
                local_denom += p;
            }
            smem_max   = local_max;
            smem_denom = local_denom;
        }
        __syncthreads();

        // Step 3: RESCALE previous accumulator by exp(old_max - new_max).
        // Without this, out_acc from earlier tiles carries weights computed
        // under the old max, producing incorrect attention output for any
        // sequence longer than one tile (128 tokens).
        float rescale_factor = __expf(old_max - smem_max);
        #pragma unroll
        for (int i = 0; i < kHeadDim; ++i) {
            out_acc[i] *= rescale_factor;
        }

        // Step 4: Accumulate probs · V. Threads cooperate over tokens
        // and reduce into shared memory.
        __shared__ float v_accum[kHeadDim];
        for (int d = lane; d < kHeadDim; d += kThreads) {
            v_accum[d] = 0.0f;
        }
        __syncthreads();

        for (int t = lane; t < tile_len; t += kThreads) {
            const float p = scores_tile[t];
            const int v_byte_off = kv_off / 2 + (token_start + t) * (kHeadDim / 2);
            for (int d2 = 0; d2 < kHeadDim / 2; ++d2) {
                const uint8_t byte = V_packed[v_byte_off + d2];
                int lo, hi;
                unpack_int4(byte, lo, hi);
                const float v0 = static_cast<float>(lo) * v_scale;
                const float v1 = static_cast<float>(hi) * v_scale;
                atomicAdd(&v_accum[2 * d2], p * v0);
                atomicAdd(&v_accum[2 * d2 + 1], p * v1);
            }
        }
        __syncthreads();

        for (int d = lane; d < kHeadDim; d += kThreads) {
            out_acc[d] += v_accum[d];
        }
        __syncthreads();
    }

    // Final normalization.
    const float inv_s = 1.0f / fmaxf(smem_denom, 1e-12f);
    for (int d = lane; d < kHeadDim; d += kThreads) {
        O[q_off + d] = __float2half(out_acc[d] * inv_s);
    }
}

torch::Tensor int4_decode_attention(
    torch::Tensor q,
    torch::Tensor k_packed,
    torch::Tensor v_packed,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    int64_t num_kv_tokens,
    double softmax_scale,
    torch::optional<torch::Tensor> mask
) {
    TORCH_CHECK(q.is_cuda() && k_packed.is_cuda(), "all inputs must be CUDA");

    const int B = q.size(0);
    const int H = q.size(1);
    const int D = q.size(3);
    TORCH_CHECK(D == kHeadDim, "head_dim must be ", kHeadDim);

    auto out = torch::empty({B, H, 1, D}, q.options());

    dim3 grid(H, B);
    dim3 block(kThreads);

    int4_decode_attention_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        k_packed.data_ptr<uint8_t>(),
        v_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(k_scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        static_cast<int>(num_kv_tokens),
        static_cast<float>(softmax_scale),
        mask.has_value() ? reinterpret_cast<const __half*>(mask.value().data_ptr<at::Half>()) : nullptr
    );
    return out;
}

}  // namespace int4
}  // namespace axropus

#endif  // __CUDACC__
