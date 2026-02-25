// core/spec_v2.h
//
// Clean speculative decode path (v2).
// Bypasses holographic collapse / thermo controller / CP decision pipeline.
// Activated by KORITH_SPEC_V2=1.

#pragma once

#include "llama.h"
#include <cstdint>

struct SpecV2Result {
    uint64_t proposed  = 0;
    uint64_t accepted  = 0;
    double   draft_ms  = 0.0;
    double   verify_ms = 0.0;
    double   total_ms  = 0.0;
};

// Run one step of clean speculative decode.
// Generates up to max_this_call tokens using draft-verify loop.
// Returns number of tokens printed, or -1 on error.
int32_t spec_v2_step(
    llama_context * ctx_target,
    llama_batch & batch_target,
    const llama_vocab * vocab_target,
    int32_t n_vocab,
    llama_context * ctx_draft,
    llama_batch & batch_draft,
    llama_pos & pos_target,
    llama_pos & pos_draft,
    const float *& logits_target,
    const float *& logits_draft,
    bool & finished,
    int32_t max_this_call,
    uint64_t & printed_total,
    uint64_t max_tokens,
    SpecV2Result & result);

// Print final metrics summary to stderr.
void spec_v2_print_summary();

