#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initializes llama.cpp backend, loads the GGUF model, and creates a context.
// Returns true on success.
bool korith_init(const char * model_path, int n_ctx);

// Tokenizes + evaluates the prompt into the context (prefill).
// Returns the number of prompt tokens on success, or -1 on failure.
int korith_tokenize(const char * prompt);

// Advances decoding by exactly one token using greedy sampling.
// Returns the next token id, or -1 if finished or on error.
int korith_step(void);

// Frees all resources (model, context, sampler) and shuts down llama backend.
void korith_shutdown(void);

#ifdef __cplusplus
}  // extern "C"
#endif

