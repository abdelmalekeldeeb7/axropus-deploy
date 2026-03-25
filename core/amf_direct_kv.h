#pragma once

// amf_direct_kv.h — Direct CUDA KV cache save/restore, bypassing llama.cpp
// serialization overhead.
//
// Design:
//   - Format magic: 0x414D464B ("AMFK") distinguishes direct-GPU snapshots from
//     legacy llama_state_get/set_data blobs.
//   - Pinned host staging buffer allocated once on init and reused across
//     save/restore calls to maximise PCIe throughput.
//   - All GPU ↔ host transfers go through cudaMemcpyAsync on a dedicated stream
//     so they can be overlapped between layers.
//   - Restore path replicates only the KV metadata fields that llama.cpp's
//     llama_state_set_data() updates (kv_self.head/used/cells) — no full
//     deserialisation needed.
//
// Feature flag: set KORITH_AMF_DIRECT_GPU=1 to activate.  The header magic
// makes the format self-identifying, so save and restore auto-detect which
// path wrote the file.

#ifndef AMF_DIRECT_KV_H
#define AMF_DIRECT_KV_H

#include <cstddef>
#include <cstdint>

#include <llama.h>

#include "amf_store.h"

namespace korith::core {

// ── File-format header ────────────────────────────────────────────────────────

constexpr std::uint32_t kAmfDirectKvMagic   = 0x414D464Bu;  // "AMFK"
constexpr std::uint32_t kAmfDirectKvVersion = 2u;  // v2: reserved repurposed as compression codec

// Compression codec IDs stored in AmfDirectKvHeader::compression.
// Must stay in sync with turboquant_codec.py CODEC_* constants.
enum class AmfKvCompression : std::uint32_t {
  kNone       = 0,  // raw FP16/BF16 — legacy v1 and uncompressed v2
  kTurboQuant = 2,  // PolarQuant + QJL (Google Research, ICLR 2026)
};

// dtype tag stored in the header so restore can sanity-check against current
// model configuration.
enum class AmfKvDtype : std::uint32_t {
  kF16  = 0,
  kF32  = 1,
  kBF16 = 2,
  kF8   = 3,
};

// Packed header written at the start of every direct-GPU KV snapshot.
// Followed immediately by n_layers pairs of (K-blob, V-blob) in layer order.
struct AmfDirectKvHeader {
  std::uint32_t magic;           // kAmfDirectKvMagic
  std::uint32_t version;         // kAmfDirectKvVersion
  std::uint32_t n_layers;        // number of attention layers saved
  std::uint32_t n_tokens;        // tokens present in this snapshot
  std::uint32_t n_kv_heads;      // KV head count (post-GQA)
  std::uint32_t head_dim;        // dimension per head
  std::uint32_t dtype;           // AmfKvDtype cast to uint32
  std::uint32_t compression;     // AmfKvCompression: 0=none, 2=TurboQuant (was: reserved)
  std::uint64_t total_kv_bytes;  // UNCOMPRESSED bytes — used for VRAM allocation on restore
  std::uint64_t model_hash;      // from AmfKey (sanity-check on restore)
  std::uint64_t prefix_hash;     // from AmfKey (sanity-check on restore)
};
static_assert(sizeof(AmfDirectKvHeader) == 56,
              "AmfDirectKvHeader size changed — update format version");

// Per-layer metadata appended after the main header (one entry per attention
// layer). Enables variable-size K/V tensors across layers in future.
struct AmfDirectKvLayerInfo {
  std::uint64_t k_offset;  // byte offset of K data from start of KV payload
  std::uint64_t k_bytes;   // byte size of K tensor
  std::uint64_t v_offset;  // byte offset of V data from start of KV payload
  std::uint64_t v_bytes;   // byte size of V tensor
};

// ── Runtime context ───────────────────────────────────────────────────────────

// Opaque per-process context for direct-GPU KV I/O.  Allocate once on startup
// with amf_direct_kv_init() and free on shutdown with amf_direct_kv_free().
struct AmfDirectKvCtx;

// Initialise direct-GPU KV context.  Allocates a pinned host staging buffer
// large enough for max_kv_bytes.  Returns nullptr on failure.
AmfDirectKvCtx * amf_direct_kv_init(std::size_t max_kv_bytes);

// Free context and pinned staging buffer.
void amf_direct_kv_free(AmfDirectKvCtx * ctx);

// ── Primary API ───────────────────────────────────────────────────────────────

// Save the KV cache for llama_context ctx into the AMF store using the direct
// CUDA path.  The snapshot is keyed by amf_ctx + tokens.  On success the entry
// is admitted into the store and true is returned.
//
// Parameters:
//   dkv        - context returned by amf_direct_kv_init
//   lctx       - the llama_context whose KV cache to snapshot
//   amf_ctx    - AmfContext for key generation (must match restore side)
//   tokens     - full prefix token sequence
//   store      - the AMF store to write into
//   saved_ms   - caller-measured time saving (for ROI gate)
//
bool amf_direct_kv_save(AmfDirectKvCtx * dkv,
                        llama_context   * lctx,
                        const AmfContext & amf_ctx,
                        const std::vector<llama_token> & tokens,
                        AmfStore & store,
                        std::uint64_t saved_ms);

// Restore KV cache into llama_context ctx from AMF entry.  Returns true if
// the restore succeeded and the context is ready to continue decode.
//
// Parameters:
//   dkv        - context returned by amf_direct_kv_init
//   lctx       - the llama_context to restore into
//   entry      - the AMF entry that was found by find_longest_prefix
//   store      - the AMF store to read from
//
bool amf_direct_kv_restore(AmfDirectKvCtx * dkv,
                            llama_context   * lctx,
                            const AmfEntry  & entry,
                            AmfStore & store);

// Probe the first few bytes of a KV blob and return true when it was written
// by the direct-GPU path (magic == kAmfDirectKvMagic).
bool amf_direct_kv_is_direct_format(const std::uint8_t * data, std::size_t size);

}  // namespace korith::core

#endif  // AMF_DIRECT_KV_H
