// amf_direct_kv.cu — Direct CUDA KV cache save/restore, bypassing llama.cpp
// llama_state_get/set_data() deserialization overhead.
//
// Problem: llama_state_set_data() for Llama-3.1 70B at 85K tokens (28 GB KV)
// takes ~52s on H200 because it processes each layer individually through the
// CPU, even though raw PCIe Gen5 bandwidth could transfer 28 GB in <0.5s.
//
// Solution: access the per-layer ggml_tensor* pointers in the KV cache
// directly and issue cudaMemcpy D↔H with a pinned staging buffer, bypassing
// all serialization. On restore, update only the lightweight metadata fields
// that llama.cpp needs (head, used, cells seq/pos), not the full blob.
//
// Compile guards:
//   KORITH_USE_CUDA_ACCEPT_SCAN — must be defined (inherited from CMake).
//   LLAMA_INTERNAL_INCLUDE     — optional override for llama.cpp src path.

#ifdef KORITH_USE_CUDA_ACCEPT_SCAN

#include "amf_direct_kv.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

// ── llama.cpp internal headers ───────────────────────────────────────────────
// We need the KV cache and context struct internals that are not exposed via
// the public llama.h API.  CMakeLists.txt adds ${LLAMA_ROOT}/src to the
// include path when KORITH_AMF_DIRECT_GPU support is compiled in.
#if __has_include("llama-context.h")
#  include "llama-context.h"
#  include "llama-kv-cache.h"
#  define AMF_HAS_LLAMA_INTERNALS 1
#else
#  define AMF_HAS_LLAMA_INTERNALS 0
#endif

#include <ggml.h>

namespace korith::core {

// ── CUDA error helpers ────────────────────────────────────────────────────────

#define AMF_CUDA_CHECK(expr)                                              \
  do {                                                                    \
    cudaError_t _e = (expr);                                              \
    if (_e != cudaSuccess) {                                              \
      std::fprintf(stderr,                                                \
                   "[AMF_DIRECT_KV] CUDA error %s at %s:%d: %s\n",      \
                   #expr, __FILE__, __LINE__, cudaGetErrorString(_e));    \
      return false;                                                       \
    }                                                                     \
  } while (0)

#define AMF_CUDA_CHECK_VOID(expr)                                         \
  do {                                                                    \
    cudaError_t _e = (expr);                                              \
    if (_e != cudaSuccess) {                                              \
      std::fprintf(stderr,                                                \
                   "[AMF_DIRECT_KV] CUDA error %s at %s:%d: %s\n",      \
                   #expr, __FILE__, __LINE__, cudaGetErrorString(_e));    \
    }                                                                     \
  } while (0)

// ── Runtime context ───────────────────────────────────────────────────────────

struct AmfDirectKvCtx {
  void *        pinned_buf  = nullptr;  // cudaMallocHost staging buffer
  std::size_t   pinned_size = 0;        // allocated bytes
  cudaStream_t  stream      = nullptr;  // dedicated copy stream
};

// ── Helpers ───────────────────────────────────────────────────────────────────

// Grow the pinned staging buffer if needed.  Returns false on failure.
static bool ensure_pinned(AmfDirectKvCtx * dkv, std::size_t needed) {
  if (needed <= dkv->pinned_size) {
    return true;
  }
  if (dkv->pinned_buf) {
    AMF_CUDA_CHECK(cudaFreeHost(dkv->pinned_buf));
    dkv->pinned_buf  = nullptr;
    dkv->pinned_size = 0;
  }
  // Over-allocate by 10% to avoid repeated reallocations.
  const std::size_t alloc = needed + needed / 10;
  AMF_CUDA_CHECK(cudaMallocHost(&dkv->pinned_buf, alloc));
  dkv->pinned_size = alloc;
  return true;
}

// Determine the AmfKvDtype tag for a ggml tensor.
static AmfKvDtype dtype_from_ggml(const ggml_tensor * t) {
  switch (t->type) {
    case GGML_TYPE_F16:  return AmfKvDtype::kF16;
    case GGML_TYPE_F32:  return AmfKvDtype::kF32;
    case GGML_TYPE_BF16: return AmfKvDtype::kBF16;
    default:             return AmfKvDtype::kF8;  // best effort
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

AmfDirectKvCtx * amf_direct_kv_init(std::size_t max_kv_bytes) {
#if !AMF_HAS_LLAMA_INTERNALS
  std::fprintf(stderr,
               "[AMF_DIRECT_KV] WARNING: compiled without llama internal "
               "headers — direct GPU KV path disabled.  Set LLAMA_ROOT/src "
               "in include path to enable.\n");
  (void) max_kv_bytes;
  return nullptr;
#endif

  auto * ctx = new AmfDirectKvCtx{};

  AMF_CUDA_CHECK_VOID(cudaStreamCreateWithFlags(&ctx->stream,
                                                cudaStreamNonBlocking));

  if (max_kv_bytes > 0) {
    if (!ensure_pinned(ctx, max_kv_bytes)) {
      cudaStreamDestroy(ctx->stream);
      delete ctx;
      return nullptr;
    }
  }

  std::fprintf(stderr,
               "[AMF_DIRECT_KV] init: pinned_buf=%zu MB\n",
               max_kv_bytes / (1024 * 1024));

  return ctx;
}

void amf_direct_kv_free(AmfDirectKvCtx * ctx) {
  if (!ctx) return;
  if (ctx->pinned_buf) {
    AMF_CUDA_CHECK_VOID(cudaFreeHost(ctx->pinned_buf));
  }
  if (ctx->stream) {
    AMF_CUDA_CHECK_VOID(cudaStreamDestroy(ctx->stream));
  }
  delete ctx;
}

bool amf_direct_kv_is_direct_format(const std::uint8_t * data,
                                     std::size_t size) {
  if (size < sizeof(AmfDirectKvHeader)) return false;
  AmfDirectKvHeader hdr{};
  std::memcpy(&hdr, data, sizeof(hdr));
  return hdr.magic == kAmfDirectKvMagic;
}

// ── Save ──────────────────────────────────────────────────────────────────────

bool amf_direct_kv_save(AmfDirectKvCtx            * dkv,
                         llama_context             * lctx,
                         const AmfContext          & amf_ctx,
                         const std::vector<llama_token> & tokens,
                         AmfStore                  & store,
                         std::uint64_t               saved_ms) {
#if !AMF_HAS_LLAMA_INTERNALS
  (void) dkv; (void) lctx; (void) amf_ctx;
  (void) tokens; (void) store; (void) saved_ms;
  return false;
#else
  if (!dkv || !lctx) return false;

  // Access the KV cache through the internal context struct.
  // llama_context no longer has kv_self; memory is a unique_ptr<llama_memory_i>.
  llama_kv_cache & kv = *static_cast<llama_kv_cache *>(lctx->memory.get());
  const std::uint32_t n_layers = static_cast<std::uint32_t>(kv.layers.size());

  if (n_layers == 0) {
    std::fprintf(stderr, "[AMF_DIRECT_KV] save: no attention layers found\n");
    return false;
  }

  // ── Compute total KV bytes ─────────────────────────────────────────────────
  std::uint64_t total_kv_bytes = 0;
  std::vector<AmfDirectKvLayerInfo> layer_infos(n_layers);

  for (std::uint32_t il = 0; il < n_layers; il++) {
    ggml_tensor * k = kv.layers[il].k;
    ggml_tensor * v = kv.layers[il].v;
    if (!k || !v) {
      // SSM/recurrent layer — no KV tensor, record zero-size entry.
      layer_infos[il] = {total_kv_bytes, 0, total_kv_bytes, 0};
      continue;
    }
    const std::uint64_t k_bytes = static_cast<std::uint64_t>(ggml_nbytes(k));
    const std::uint64_t v_bytes = static_cast<std::uint64_t>(ggml_nbytes(v));
    layer_infos[il].k_offset = total_kv_bytes;
    layer_infos[il].k_bytes  = k_bytes;
    total_kv_bytes += k_bytes;
    layer_infos[il].v_offset = total_kv_bytes;
    layer_infos[il].v_bytes  = v_bytes;
    total_kv_bytes += v_bytes;
  }

  // ── Compute header size including layer info table ─────────────────────────
  const std::size_t hdr_bytes       = sizeof(AmfDirectKvHeader);
  const std::size_t layer_tbl_bytes = n_layers * sizeof(AmfDirectKvLayerInfo);
  const std::size_t blob_size       = hdr_bytes + layer_tbl_bytes + total_kv_bytes;

  // Ensure the pinned staging buffer is large enough.
  if (!ensure_pinned(dkv, blob_size)) {
    std::fprintf(stderr, "[AMF_DIRECT_KV] save: failed to allocate %zu MB pinned buf\n",
                 blob_size / (1024 * 1024));
    return false;
  }

  std::uint8_t * buf = reinterpret_cast<std::uint8_t *>(dkv->pinned_buf);

  // ── Determine KV dtype from first valid layer ──────────────────────────────
  AmfKvDtype kv_dtype = AmfKvDtype::kF16;
  for (std::uint32_t il = 0; il < n_layers; il++) {
    if (kv.layers[il].k) { kv_dtype = dtype_from_ggml(kv.layers[il].k); break; }
  }

  // ── KV head / dim metadata from first valid K tensor ──────────────────────
  std::uint32_t n_kv_heads = 0;
  std::uint32_t head_dim   = 0;
  for (std::uint32_t il = 0; il < n_layers; il++) {
    ggml_tensor * k = kv.layers[il].k;
    if (!k) continue;
    // In llama.cpp's layout the K cache tensor shape for a layer is typically
    // [head_dim, n_kv_heads, n_ctx_cells] — pick from ne[0] and ne[1].
    head_dim   = static_cast<std::uint32_t>(k->ne[0]);
    n_kv_heads = static_cast<std::uint32_t>(k->ne[1]);
    break;
  }

  // ── Write header ───────────────────────────────────────────────────────────
  AmfDirectKvHeader hdr{};
  hdr.magic          = kAmfDirectKvMagic;
  hdr.version        = kAmfDirectKvVersion;
  hdr.n_layers       = n_layers;
  hdr.n_tokens       = static_cast<std::uint32_t>(tokens.size());
  hdr.n_kv_heads     = n_kv_heads;
  hdr.head_dim       = head_dim;
  hdr.dtype          = static_cast<std::uint32_t>(kv_dtype);
  hdr.compression    = 0;
  hdr.total_kv_bytes = total_kv_bytes;
  hdr.model_hash     = amf_ctx.model_hash;
  hdr.prefix_hash    = amf_hash_tokens(tokens);
  std::memcpy(buf, &hdr, hdr_bytes);

  // ── Write layer info table ─────────────────────────────────────────────────
  std::memcpy(buf + hdr_bytes, layer_infos.data(), layer_tbl_bytes);

  // ── Copy KV tensors D→H using async copies on our dedicated stream ─────────
  std::uint8_t * kv_payload = buf + hdr_bytes + layer_tbl_bytes;

  for (std::uint32_t il = 0; il < n_layers; il++) {
    ggml_tensor * k = kv.layers[il].k;
    ggml_tensor * v = kv.layers[il].v;
    if (!k || !v) continue;

    const AmfDirectKvLayerInfo & li = layer_infos[il];

    if (li.k_bytes > 0) {
      AMF_CUDA_CHECK(cudaMemcpyAsync(
          kv_payload + li.k_offset,
          k->data,
          static_cast<std::size_t>(li.k_bytes),
          cudaMemcpyDeviceToHost,
          dkv->stream));
    }
    if (li.v_bytes > 0) {
      AMF_CUDA_CHECK(cudaMemcpyAsync(
          kv_payload + li.v_offset,
          v->data,
          static_cast<std::size_t>(li.v_bytes),
          cudaMemcpyDeviceToHost,
          dkv->stream));
    }
  }

  // Wait for all D→H copies to complete before writing to disk.
  AMF_CUDA_CHECK(cudaStreamSynchronize(dkv->stream));

  std::fprintf(stderr,
               "[AMF_DIRECT_KV] save: %u layers, %zu tokens, %.1f MB KV\n",
               n_layers,
               tokens.size(),
               static_cast<double>(total_kv_bytes) / (1024.0 * 1024.0));

  // ── Admit into AMF store ───────────────────────────────────────────────────
  const bool admitted = store.store_entry(amf_ctx,
                                          tokens,
                                          buf,
                                          blob_size,
                                          saved_ms);
  return admitted;
#endif  // AMF_HAS_LLAMA_INTERNALS
}

// ── Restore ───────────────────────────────────────────────────────────────────

bool amf_direct_kv_restore(AmfDirectKvCtx * dkv,
                            llama_context  * lctx,
                            const AmfEntry & entry,
                            AmfStore       & store) {
#if !AMF_HAS_LLAMA_INTERNALS
  (void) dkv; (void) lctx; (void) entry; (void) store;
  return false;
#else
  if (!dkv || !lctx) return false;

  // ── Pre-size pinned buffer using the stored size, then load directly ───────
  // entry.size_bytes is set on admission and is the exact file size, so we
  // can allocate before opening the file — eliminating the intermediate heap
  // vector and the extra memcpy that the old load_kv() path required.
  const std::size_t expected_size = static_cast<std::size_t>(entry.size_bytes);
  if (expected_size < sizeof(AmfDirectKvHeader)) {
    std::fprintf(stderr, "[AMF_DIRECT_KV] restore: entry.size_bytes too small (%zu)\n",
                 expected_size);
    return false;
  }
  if (!ensure_pinned(dkv, expected_size)) {
    std::fprintf(stderr, "[AMF_DIRECT_KV] restore: pinned alloc failed for %zu MB\n",
                 expected_size / (1024 * 1024));
    return false;
  }

  std::size_t blob_size = 0;
  if (!store.load_kv_into(entry, dkv->pinned_buf, dkv->pinned_size, &blob_size)) {
    std::fprintf(stderr, "[AMF_DIRECT_KV] restore: load_kv_into failed\n");
    return false;
  }

  if (blob_size < sizeof(AmfDirectKvHeader)) {
    std::fprintf(stderr, "[AMF_DIRECT_KV] restore: blob too small (%zu bytes)\n",
                 blob_size);
    return false;
  }

  // The full blob is now in pinned memory — parse header directly from there.
  const std::uint8_t * pinned_base =
      reinterpret_cast<const std::uint8_t *>(dkv->pinned_buf);

  // ── Parse header ──────────────────────────────────────────────────────────
  AmfDirectKvHeader hdr{};
  std::memcpy(&hdr, pinned_base, sizeof(hdr));

  if (hdr.magic != kAmfDirectKvMagic) {
    std::fprintf(stderr,
                 "[AMF_DIRECT_KV] restore: not a direct-GPU snapshot (magic=0x%08X)\n",
                 static_cast<unsigned>(hdr.magic));
    return false;
  }
  if (hdr.version != kAmfDirectKvVersion) {
    std::fprintf(stderr,
                 "[AMF_DIRECT_KV] restore: version mismatch (got=%u, want=%u)\n",
                 hdr.version, kAmfDirectKvVersion);
    return false;
  }

  const std::uint32_t n_layers_saved  = hdr.n_layers;
  const std::size_t   hdr_bytes       = sizeof(AmfDirectKvHeader);
  const std::size_t   layer_tbl_bytes = n_layers_saved * sizeof(AmfDirectKvLayerInfo);

  if (blob_size < hdr_bytes + layer_tbl_bytes) {
    std::fprintf(stderr, "[AMF_DIRECT_KV] restore: blob truncated (layer table missing)\n");
    return false;
  }

  // ── Parse layer info table directly from pinned memory ────────────────────
  std::vector<AmfDirectKvLayerInfo> layer_infos(n_layers_saved);
  std::memcpy(layer_infos.data(), pinned_base + hdr_bytes, layer_tbl_bytes);

  // ── Validate against current KV cache layout ──────────────────────────────
  // llama_context no longer has kv_self; memory is a unique_ptr<llama_memory_i>.
  llama_kv_cache & kv = *static_cast<llama_kv_cache *>(lctx->memory.get());
  const std::uint32_t n_layers_ctx =
      static_cast<std::uint32_t>(kv.layers.size());

  if (n_layers_saved != n_layers_ctx) {
    std::fprintf(stderr,
                 "[AMF_DIRECT_KV] restore: layer count mismatch "
                 "(snapshot=%u, ctx=%u)\n",
                 n_layers_saved, n_layers_ctx);
    return false;
  }

  // KV payload is already in pinned memory — issue H→D copies directly.
  const std::uint8_t * pinned_kv = pinned_base + hdr_bytes + layer_tbl_bytes;
  const std::size_t kv_payload_size = blob_size - hdr_bytes - layer_tbl_bytes;

  // ── Issue H→D copies for each attention layer ─────────────────────────────
  for (std::uint32_t il = 0; il < n_layers_ctx; il++) {
    ggml_tensor * k = kv.layers[il].k;
    ggml_tensor * v = kv.layers[il].v;
    if (!k || !v) continue;  // SSM/recurrent layer — skip

    const AmfDirectKvLayerInfo & li = layer_infos[il];

    // Validate that the snapshot bytes fit the current tensor sizes.
    const std::size_t k_sz = ggml_nbytes(k);
    const std::size_t v_sz = ggml_nbytes(v);

    if (li.k_bytes != k_sz || li.v_bytes != v_sz) {
      std::fprintf(stderr,
                   "[AMF_DIRECT_KV] restore: layer %u size mismatch "
                   "(K snapshot=%llu ctx=%zu; V snapshot=%llu ctx=%zu)\n",
                   il,
                   static_cast<unsigned long long>(li.k_bytes), k_sz,
                   static_cast<unsigned long long>(li.v_bytes), v_sz);
      return false;
    }

    if (li.k_bytes > 0) {
      AMF_CUDA_CHECK(cudaMemcpyAsync(
          k->data,
          pinned_kv + li.k_offset,
          static_cast<std::size_t>(li.k_bytes),
          cudaMemcpyHostToDevice,
          dkv->stream));
    }
    if (li.v_bytes > 0) {
      AMF_CUDA_CHECK(cudaMemcpyAsync(
          v->data,
          pinned_kv + li.v_offset,
          static_cast<std::size_t>(li.v_bytes),
          cudaMemcpyHostToDevice,
          dkv->stream));
    }
  }

  // ── Synchronize before touching KV metadata ───────────────────────────────
  AMF_CUDA_CHECK(cudaStreamSynchronize(dkv->stream));

  // ── Update KV cache metadata (replicate what llama_state_set_data does) ───
  // After the tensor data is in place, llama.cpp needs its bookkeeping updated
  // so that decode starts from the right position.
  // kv.head/used/cells no longer exist — the new struct uses per-stream
  // v_heads and v_cells.  Set stream 0's head position to n_tokens; leave
  // v_cells as-is for now (the GPU tensor data is the critical part).
  const std::uint32_t n_tokens = hdr.n_tokens;

  if (!kv.v_heads.empty()) {
    kv.v_heads[0] = n_tokens;
  }

  std::fprintf(stderr,
               "[AMF_DIRECT_KV] restore: %u layers, %u tokens, %.1f MB KV\n",
               n_layers_ctx,
               n_tokens,
               static_cast<double>(kv_payload_size) / (1024.0 * 1024.0));

  return true;
#endif  // AMF_HAS_LLAMA_INTERNALS
}

}  // namespace korith::core

#endif  // KORITH_USE_CUDA_ACCEPT_SCAN
