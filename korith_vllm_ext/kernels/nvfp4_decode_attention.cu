/*
 * nvfp4_decode_attention.cu — Blackwell NVFP4 decode attention kernel.
 *
 * This file is a stub pending access to a B200 for validation. The design
 * lives in §2.4 of the infrastructure design doc. The required UMMA
 * intrinsics are only available on SM100+, so the body is gated on
 * ``__CUDA_ARCH__ >= 1000``.
 *
 * The stub exists so that the JIT build in dispatch.py finds all three
 * kernel translation units and does not warn. When AMF is built for
 * Hopper the compiler simply drops the SM100 code path.
 *
 * TODO(B200): implement the TMA-loaded micro-block scaled NVFP4 path:
 *   - Use cp.async.bulk.tensor for K/V loads.
 *   - Invoke umma.m64n256k64.f32.e2m1.e2m1 for Q·K and probs·V.
 *   - Apply per-group FP8 E4M3 scales inline via umma ".scale_vec" suffix.
 *   - Store accumulators in TMEM (SM100 tensor memory) to free SMEM.
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace axropus {
namespace nvfp4 {

__global__ void nvfp4_decode_attention_kernel_placeholder(
    const __half* __restrict__ Q,
    const uint8_t* __restrict__ K_packed,
    const uint8_t* __restrict__ V_packed,
    const uint8_t* __restrict__ K_scales_fp8,
    const uint8_t* __restrict__ V_scales_fp8,
    __half*       __restrict__ O,
    int            num_kv_tokens,
    float          softmax_scale
) {
    // Placeholder: on SM100 the real implementation would go here. For
    // now we leave the output untouched so the Python fallback path runs.
    (void)Q; (void)K_packed; (void)V_packed;
    (void)K_scales_fp8; (void)V_scales_fp8; (void)O;
    (void)num_kv_tokens; (void)softmax_scale;
}

torch::Tensor nvfp4_decode_attention(
    torch::Tensor q,
    torch::Tensor k_packed,
    torch::Tensor v_packed,
    torch::Tensor k_scales,
    torch::Tensor v_scales,
    int64_t num_kv_tokens,
    double softmax_scale
) {
    TORCH_CHECK(false,
        "nvfp4_decode_attention: Blackwell kernel is a stub. Use the "
        "Python fallback path until B200 hardware is wired up.");
    return q;  // unreachable
}

}  // namespace nvfp4
}  // namespace axropus

#endif  // __CUDACC__
