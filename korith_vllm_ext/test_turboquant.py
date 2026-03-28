"""test_turboquant.py — Unit tests for TurboQuant KV cache compression.

Run:
    python -m pytest korith_vllm_ext/test_turboquant.py -v
    # or standalone:
    python korith_vllm_ext/test_turboquant.py
"""

from __future__ import annotations

import torch

from .turboquant_codec import TurboQuantCodec


def _roundtrip_test(
    dtype: torch.dtype,
    bits: int,
    head_dim: int = 128,
    n_blocks: int = 3,
    block_size: int = 16,
    num_kv_heads: int = 2,
) -> dict:
    """Compress → decompress and measure reconstruction quality."""
    label = f"{dtype} bits={bits}"
    shape = (n_blocks, block_size, num_kv_heads, head_dim)
    n_elem = 1
    for s in shape:
        n_elem *= s

    # Create random data in the target dtype, matching vLLM KV cache values.
    torch.manual_seed(0)
    original_f32 = torch.randn(n_elem)
    original = original_f32.to(dtype).contiguous()

    # Get raw bytes the same way save_kv_state does.
    raw_bytes = bytes(original.untyped_storage())
    assert len(raw_bytes) == n_elem * original.element_size(), (
        f"Raw bytes size mismatch: {len(raw_bytes)} != "
        f"{n_elem * original.element_size()}"
    )

    # Verify torch.frombuffer can read them back (the compress() input path).
    reconstructed_pre = torch.frombuffer(bytearray(raw_bytes), dtype=dtype)
    assert torch.equal(original, reconstructed_pre), (
        f"[{label}] frombuffer != original before compression"
    )

    # Compress.
    codec = TurboQuantCodec(bits=bits, seed=42)
    compressed = codec.compress(raw_bytes, head_dim, dtype)
    ratio = len(raw_bytes) / max(len(compressed), 1)

    # Decompress (CPU path).
    decompressed_bytes = codec.decompress(compressed, dtype)
    restored = torch.frombuffer(bytearray(decompressed_bytes), dtype=dtype)

    # Compare.
    orig_f32 = original.float()
    rest_f32 = restored.float()

    max_err = (orig_f32 - rest_f32).abs().max().item()
    mean_err = (orig_f32 - rest_f32).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        orig_f32.unsqueeze(0), rest_f32.unsqueeze(0)
    ).item()

    return {
        "label": label,
        "n_elem": n_elem,
        "raw_bytes": len(raw_bytes),
        "compressed_bytes": len(compressed),
        "ratio": ratio,
        "max_err": max_err,
        "mean_err": mean_err,
        "cos_sim": cos_sim,
    }


def _roundtrip_gpu_test(
    dtype: torch.dtype,
    bits: int,
    head_dim: int = 128,
    n_elem: int = 3 * 16 * 2 * 128,
) -> dict:
    """Test the decompress_to_gpu path."""
    label = f"GPU {dtype} bits={bits}"
    torch.manual_seed(0)
    original = torch.randn(n_elem).to(dtype).contiguous()
    raw_bytes = bytes(original.untyped_storage())

    codec = TurboQuantCodec(bits=bits, seed=42)
    compressed = codec.compress(raw_bytes, head_dim, dtype)

    # GPU decompress.
    gpu_tensor = codec.decompress_to_gpu(compressed, dtype, "cuda")
    restored = gpu_tensor.cpu()

    orig_f32 = original.float()
    rest_f32 = restored.float()

    max_err = (orig_f32 - rest_f32).abs().max().item()
    mean_err = (orig_f32 - rest_f32).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        orig_f32.unsqueeze(0), rest_f32.unsqueeze(0)
    ).item()

    return {
        "label": label,
        "max_err": max_err,
        "mean_err": mean_err,
        "cos_sim": cos_sim,
    }


def main() -> None:
    print("=" * 70)
    print("TurboQuant roundtrip test")
    print("=" * 70)

    results = []
    for dtype in [torch.float16, torch.bfloat16]:
        for bits in [3, 4]:
            r = _roundtrip_test(dtype, bits)
            results.append(r)
            print(
                f"[{r['label']}]  "
                f"ratio={r['ratio']:.2f}x  "
                f"max_err={r['max_err']:.6f}  "
                f"mean_err={r['mean_err']:.6f}  "
                f"cos_sim={r['cos_sim']:.6f}"
            )

    if torch.cuda.is_available():
        print()
        print("GPU decompress path:")
        for dtype in [torch.float16, torch.bfloat16]:
            for bits in [3, 4]:
                r = _roundtrip_gpu_test(dtype, bits)
                results.append(r)
                print(
                    f"[{r['label']}]  "
                    f"max_err={r['max_err']:.6f}  "
                    f"mean_err={r['mean_err']:.6f}  "
                    f"cos_sim={r['cos_sim']:.6f}"
                )

    print()
    # Check quality thresholds.
    all_ok = True
    for r in results:
        if r["cos_sim"] < 0.90:
            print(f"FAIL: {r['label']} cosine_similarity={r['cos_sim']:.6f} < 0.90")
            all_ok = False
    if all_ok:
        print("ALL TESTS PASSED — cosine similarity > 0.90 for all configs")
    else:
        print("SOME TESTS FAILED")


if __name__ == "__main__":
    main()
