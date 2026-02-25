// core/spec_v2.cpp
//
// Clean speculative decode path (v2).
// Bypasses holographic collapse / thermo controller / CP decision pipeline entirely.
// Activated by KORITH_SPEC_V2=1 environment variable.
//
// Algorithm per iteration:
//   1. Sample t0 from target logits (guaranteed correct) and print it.
//   2. Feed t0 to draft model.
//   3. Draft greedily proposes k tokens (k single-token forward passes on cheap 1B model).
//   4. Target verifies [t0, d0, ..., d_{k-1}] in ONE batch forward pass.
//   5. Accept matching tokens, reject at first mismatch, keep target's corrected token.
//   6. Rollback KV caches to last accepted position, decode correction into both models.
//
// No collapse controller. No thermo gate. No CP decision. No failsafe. No EMA.

#include "spec_v2.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// ── Greedy sampling ────────────────────────────────────────────────────────────

llama_token greedy_sample(const float * logits, int32_t n_vocab) {
    if (!logits || n_vocab <= 0) return LLAMA_TOKEN_NULL;
    int32_t best = 0;
    float best_val = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return static_cast<llama_token>(best);
}

// ── Token output to stdout ─────────────────────────────────────────────────────

int32_t emit_token(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int32_t n = llama_token_to_piece(
        vocab, token, buf, static_cast<int32_t>(sizeof(buf)), 0, false);
    if (n == 0) return 0;
    if (n > 0) {
        const auto want = static_cast<std::size_t>(n);
        return (std::fwrite(buf, 1, want, stdout) == want) ? n : -1;
    }
    const int32_t need = -n;
    std::vector<char> dyn(static_cast<std::size_t>(need));
    n = llama_token_to_piece(
        vocab, token, dyn.data(), static_cast<int32_t>(dyn.size()), 0, false);
    if (n <= 0) return -1;
    const auto want = static_cast<std::size_t>(n);
    return (std::fwrite(dyn.data(), 1, want, stdout) == want) ? n : -1;
}

// ── Single-token decode helper ─────────────────────────────────────────────────

bool decode_one(llama_context * ctx, llama_batch & batch,
                llama_pos pos, llama_token token, bool want_logits) {
    batch.n_tokens     = 1;
    batch.token[0]     = token;
    batch.pos[0]       = pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = want_logits ? 1 : 0;
    return llama_decode(ctx, batch) == 0;
}

// ── Dynamic k controller ───────────────────────────────────────────────────────
//
// Simple counter-based adjustment over a sliding window.
// No EMA. No gain calculation.

struct DynK {
    static constexpr int32_t kWindow  = 24;   // wider window → smoother adaptation
    static constexpr int32_t kMin     = 2;
    static constexpr int32_t kMax     = 12;
    static constexpr int32_t kInit    = 2;    // start conservative during warmup
    static constexpr int32_t kWarmup  = 32;   // tokens before enabling dynamic k

    int32_t k = kInit;
    int32_t proposed_buf[kWindow] = {};
    int32_t accepted_buf[kWindow] = {};
    int32_t idx   = 0;
    int32_t count = 0;
    uint64_t tokens_seen = 0;                 // total tokens for warmup tracking

    void record(int32_t proposed, int32_t accepted) {
        tokens_seen += static_cast<uint64_t>(accepted) + (proposed > accepted ? 1 : 0);

        proposed_buf[idx] = proposed;
        accepted_buf[idx] = accepted;
        idx = (idx + 1) % kWindow;
        if (count < kWindow) ++count;

        // Warmup: keep k=2 until enough tokens have been generated.
        if (tokens_seen < static_cast<uint64_t>(kWarmup)) return;

        if (count < kWindow) return;

        int32_t sp = 0, sa = 0;
        for (int i = 0; i < kWindow; ++i) {
            sp += proposed_buf[i];
            sa += accepted_buf[i];
        }
        const double rate = (sp > 0) ? static_cast<double>(sa) / sp : 0.0;
        if (rate > 0.85 && k < kMax) {
            ++k;
            std::fprintf(stderr, "[SPEC_V2_K] k=%d rate=%.3f (increase)\n", k, rate);
            (void) std::fflush(stderr);
        } else if (rate < 0.55 && k > kMin) {  // wider threshold → k stays higher
            --k;
            std::fprintf(stderr, "[SPEC_V2_K] k=%d rate=%.3f (decrease)\n", k, rate);
            (void) std::fflush(stderr);
        }
    }
};

// ── Cumulative metrics ─────────────────────────────────────────────────────────

struct CumulMetrics {
    uint64_t total_proposed  = 0;
    uint64_t total_accepted  = 0;
    double   draft_ms_total  = 0.0;
    double   verify_ms_total = 0.0;
    double   decode_ms_total = 0.0;
    uint64_t tokens_printed  = 0;
    uint64_t iterations      = 0;
};

static DynK         g_dyn_k;
static CumulMetrics  g_cum;
static bool          g_init_logged = false;

}  // anonymous namespace

// ── Main entry ─────────────────────────────────────────────────────────────────

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
    SpecV2Result & result)
{
    if (!g_init_logged) {
        g_init_logged = true;
        std::fprintf(stderr,
            "[SPEC_V2] init k=%d min=%d max=%d window=%d warmup=%d\n",
            DynK::kInit, DynK::kMin, DynK::kMax, DynK::kWindow, DynK::kWarmup);
        (void) std::fflush(stderr);
    }

    const auto call_t0 = std::chrono::steady_clock::now();
    int32_t  tokens_printed = 0;
    uint64_t step_proposed  = 0;
    uint64_t step_accepted  = 0;
    double   step_draft_ms  = 0.0;
    double   step_verify_ms = 0.0;

    while (tokens_printed < max_this_call &&
           (max_tokens == 0 || printed_total < max_tokens) &&
           !finished)
    {
        const llama_pos pos0 = pos_target;   // save start position

        // ── 1. Sample guaranteed-correct token from target logits ───────────

        const llama_token t0 = greedy_sample(logits_target, n_vocab);
        if (t0 == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab_target, t0)) {
            finished = true;
            break;
        }

        if (emit_token(vocab_target, t0) < 0) { finished = true; break; }
        tokens_printed++;
        printed_total++;
        if (max_tokens > 0 && printed_total >= max_tokens) { finished = true; break; }

        // ── 2. Feed t0 to draft model ──────────────────────────────────────

        const auto draft_t0 = std::chrono::steady_clock::now();

        if (!decode_one(ctx_draft, batch_draft, pos_draft, t0, true)) {
            // Draft failed — fall back to target-only for this token.
            if (!decode_one(ctx_target, batch_target, pos_target, t0, true)) {
                finished = true; break;
            }
            pos_target++;
            pos_draft++;
            logits_target = llama_get_logits(ctx_target);
            logits_draft  = nullptr;
            continue;
        }
        pos_draft++;
        logits_draft = llama_get_logits(ctx_draft);
        if (!logits_draft) {
            if (!decode_one(ctx_target, batch_target, pos_target, t0, true)) {
                finished = true; break;
            }
            pos_target++;
            logits_target = llama_get_logits(ctx_target);
            continue;
        }

        // ── 3. Draft proposes k tokens greedily ────────────────────────────

        const int32_t k = g_dyn_k.k;
        std::vector<llama_token> draft_toks;
        draft_toks.reserve(static_cast<std::size_t>(k));

        for (int32_t i = 0; i < k; ++i) {
            const llama_token d = greedy_sample(logits_draft, n_vocab);
            if (d == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab_target, d)) break;
            draft_toks.push_back(d);
            if (!decode_one(ctx_draft, batch_draft, pos_draft, d, true)) break;
            pos_draft++;
            logits_draft = llama_get_logits(ctx_draft);
            if (!logits_draft) break;
        }

        const auto draft_t1 = std::chrono::steady_clock::now();
        step_draft_ms += std::chrono::duration<double, std::milli>(draft_t1 - draft_t0).count();

        const int32_t n_draft = static_cast<int32_t>(draft_toks.size());

        if (n_draft == 0) {
            // Draft proposed nothing — just decode t0 into target normally.
            if (!decode_one(ctx_target, batch_target, pos0, t0, true)) {
                finished = true; break;
            }
            pos_target = pos0 + 1;
            logits_target = llama_get_logits(ctx_target);
            // Draft is already at pos0+1 from feeding t0.
            continue;
        }

        // ── 4. Target verifies [t0, d0, ..., d_{n-1}] in ONE batch ────────
        //
        // Batch layout:
        //   index 0:       t0       at position pos0
        //   index 1:       d0       at position pos0+1
        //   ...
        //   index n_draft: d_{n-1}  at position pos0+n_draft
        //
        // All logits requested so we can check each position.

        const auto verify_t0 = std::chrono::steady_clock::now();
        {
            const int32_t blen = n_draft + 1;
            batch_target.n_tokens = blen;

            batch_target.token[0]      = t0;
            batch_target.pos[0]        = pos0;
            batch_target.n_seq_id[0]   = 1;
            batch_target.seq_id[0][0]  = 0;
            batch_target.logits[0]     = 1;

            for (int32_t i = 0; i < n_draft; ++i) {
                const int32_t bi = i + 1;
                batch_target.token[bi]     = draft_toks[static_cast<std::size_t>(i)];
                batch_target.pos[bi]       = pos0 + 1 + i;
                batch_target.n_seq_id[bi]  = 1;
                batch_target.seq_id[bi][0] = 0;
                batch_target.logits[bi]    = 1;
            }

            if (llama_decode(ctx_target, batch_target) != 0) {
                finished = true;
                break;
            }
        }
        const auto verify_t1 = std::chrono::steady_clock::now();
        step_verify_ms += std::chrono::duration<double, std::milli>(verify_t1 - verify_t0).count();

        // ── 5. Compare draft vs target at each position ────────────────────
        //
        // llama_get_logits(ctx) returns the base of the logits output buffer.
        // After batch decode with all logits=1, logits for token i in the
        // batch are at: base + i * n_vocab.
        //
        // Logits at index i predict the NEXT token after position (pos0 + i):
        //   logits[0] → predicts token at pos0+1  → compare with d0
        //   logits[1] → predicts token at pos0+2  → compare with d1
        //   ...
        //   logits[n_draft] → predicts token at pos0+n_draft+1 → bonus

        const float * logits_base = llama_get_logits(ctx_target);
        if (!logits_base) { finished = true; break; }

        int32_t n_accepted = 0;
        llama_token correction = LLAMA_TOKEN_NULL;

        for (int32_t i = 0; i < n_draft; ++i) {
            const float * row = logits_base +
                static_cast<std::ptrdiff_t>(i) * n_vocab;
            const llama_token pred = greedy_sample(row, n_vocab);
            if (pred == draft_toks[static_cast<std::size_t>(i)]) {
                ++n_accepted;
            } else {
                correction = pred;
                break;
            }
        }

        step_proposed += static_cast<uint64_t>(n_draft);
        step_accepted += static_cast<uint64_t>(n_accepted);

        // Print accepted draft tokens.
        for (int32_t i = 0; i < n_accepted && !finished; ++i) {
            if (emit_token(vocab_target, draft_toks[static_cast<std::size_t>(i)]) < 0) {
                finished = true; break;
            }
            tokens_printed++;
            printed_total++;
            if (max_tokens > 0 && printed_total >= max_tokens) { finished = true; }
        }
        if (finished) break;

        if (n_accepted == n_draft) {
            // ── ALL ACCEPTED ───────────────────────────────────────────────
            //
            // Target KV is fully correct at [pos0, ..., pos0+n_draft].
            // Draft KV is also correct (same range, from the proposal loop).
            // Take logits from the last position in the batch for next iteration.

            logits_target = llama_get_logits_ith(ctx_target, n_draft);
            pos_target = pos0 + n_draft + 1;
            // pos_draft is already pos0 + n_draft + 1 from the proposal loop.
            // logits_draft is from the last draft decode (same position).

        } else {
            // ── MISMATCH at n_accepted ─────────────────────────────────────
            //
            // draft_toks[n_accepted] was wrong. Target says correction instead.

            const bool corr_eog = (correction == LLAMA_TOKEN_NULL) ||
                                   llama_vocab_is_eog(vocab_target, correction);

            if (!corr_eog) {
                if (emit_token(vocab_target, correction) < 0) { finished = true; break; }
                tokens_printed++;
                printed_total++;
                if (max_tokens > 0 && printed_total >= max_tokens) finished = true;
            } else {
                finished = true;
            }

            // Remove wrong KV entries from both models.
            // Both have entries at [pos0, ..., pos0+n_draft].
            // Keep [pos0, ..., pos0+n_accepted], remove the rest.
            const llama_pos rm_start = pos0 + n_accepted + 1;
            const llama_pos rm_end   = pos0 + n_draft + 1;    // exclusive

            if (rm_start < rm_end) {
                llama_memory_seq_rm(
                    llama_get_memory(ctx_target), 0, rm_start, rm_end);
                llama_memory_seq_rm(
                    llama_get_memory(ctx_draft), 0, rm_start, rm_end);
            }

            if (!corr_eog && !finished) {
                // Decode correction token into both models.
                const llama_pos corr_pos = pos0 + n_accepted + 1;

                if (!decode_one(ctx_target, batch_target, corr_pos, correction, true)) {
                    finished = true; break;
                }
                logits_target = llama_get_logits(ctx_target);
                pos_target = corr_pos + 1;

                if (!decode_one(ctx_draft, batch_draft, corr_pos, correction, true)) {
                    // Draft failed — keep going, it'll resync or fallback next iter.
                    pos_draft = pos_target;
                    logits_draft = nullptr;
                } else {
                    logits_draft = llama_get_logits(ctx_draft);
                    pos_draft = corr_pos + 1;
                }
            } else {
                pos_target = pos0 + n_accepted + 1;
                pos_draft  = pos_target;
            }
        }

        // Record for dynamic k adjustment.
        g_dyn_k.record(n_draft, n_accepted);
        g_cum.iterations++;
    }

    (void) std::fflush(stdout);

    const auto call_t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(call_t1 - call_t0).count();

    result.proposed  = step_proposed;
    result.accepted  = step_accepted;
    result.draft_ms  = step_draft_ms;
    result.verify_ms = step_verify_ms;
    result.total_ms  = total_ms;

    g_cum.total_proposed  += step_proposed;
    g_cum.total_accepted  += step_accepted;
    g_cum.draft_ms_total  += step_draft_ms;
    g_cum.verify_ms_total += step_verify_ms;
    g_cum.decode_ms_total += total_ms;
    g_cum.tokens_printed  += static_cast<uint64_t>(tokens_printed);

    if (step_proposed > 0) {
        const double rate = static_cast<double>(step_accepted) /
                            static_cast<double>(step_proposed);
        std::fprintf(stderr,
            "[SPEC_V2] tokens=%d proposed=%llu accepted=%llu rate=%.3f "
            "draft_ms=%.1f verify_ms=%.1f total_ms=%.1f k=%d\n",
            tokens_printed,
            static_cast<unsigned long long>(step_proposed),
            static_cast<unsigned long long>(step_accepted),
            rate, step_draft_ms, step_verify_ms, total_ms, g_dyn_k.k);
        (void) std::fflush(stderr);
    }

    return tokens_printed;
}

void spec_v2_print_summary() {
    const double rate = (g_cum.total_proposed > 0)
        ? static_cast<double>(g_cum.total_accepted) /
          static_cast<double>(g_cum.total_proposed)
        : 0.0;
    const double tps = (g_cum.decode_ms_total > 1e-3)
        ? (static_cast<double>(g_cum.tokens_printed) * 1000.0 / g_cum.decode_ms_total)
        : 0.0;

    std::fprintf(stderr,
        "\n[SPEC_V2_SUMMARY]\n"
        "  total_tokens:     %llu\n"
        "  proposed_tokens:  %llu\n"
        "  accepted_tokens:  %llu\n"
        "  acceptance_rate:  %.4f\n"
        "  draft_ms:         %.1f\n"
        "  verify_ms:        %.1f\n"
        "  total_decode_ms:  %.1f\n"
        "  effective_tps:    %.2f\n"
        "  iterations:       %llu\n"
        "  final_k:          %d\n",
        static_cast<unsigned long long>(g_cum.tokens_printed),
        static_cast<unsigned long long>(g_cum.total_proposed),
        static_cast<unsigned long long>(g_cum.total_accepted),
        rate,
        g_cum.draft_ms_total,
        g_cum.verify_ms_total,
        g_cum.decode_ms_total,
        tps,
        static_cast<unsigned long long>(g_cum.iterations),
        g_dyn_k.k);
    (void) std::fflush(stderr);
}

