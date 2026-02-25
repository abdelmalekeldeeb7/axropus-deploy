#pragma once

#include <cstdint>

#if defined(KORITH_USE_CUDA_GRAPHS)
#include <cuda_runtime.h>
#endif

struct llama_context;
struct llama_batch;

namespace korith::core {

#if defined(KORITH_USE_CUDA_GRAPHS)

class CudaGraphDecodeSession {
public:
  CudaGraphDecodeSession() = default;
  ~CudaGraphDecodeSession();

  CudaGraphDecodeSession(const CudaGraphDecodeSession &) = delete;
  CudaGraphDecodeSession & operator=(const CudaGraphDecodeSession &) = delete;
  CudaGraphDecodeSession(CudaGraphDecodeSession && other) noexcept;
  CudaGraphDecodeSession & operator=(CudaGraphDecodeSession && other) noexcept;

  bool capture_begin(cudaStream_t stream);
  bool capture_end(cudaStream_t stream, int32_t batch_size, int32_t model_dim);
  bool replay(cudaStream_t stream) const;
  bool is_valid(int32_t batch_size, int32_t model_dim) const;
  void invalidate();

private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  int32_t batch_size_ = 0;
  int32_t model_dim_ = 0;
  bool valid_ = false;
};

bool cuda_graphs_enabled();
int32_t cuda_graph_cache_size();
const char * cuda_graph_backend();
void cuda_graph_log_config_once();
void cuda_graph_apply_backend_env_policy();

// Returns true when decode was executed via capture/replay path.
bool cuda_graph_decode_step(
    llama_context * ctx,
    llama_batch & batch,
    int32_t model_dim,
    int32_t * submit_us = nullptr);

void cuda_graph_decode_invalidate_context(const llama_context * ctx);
void cuda_graph_decode_invalidate_all();

#else

class CudaGraphDecodeSession {
public:
  bool capture_begin(void *) { return false; }
  bool capture_end(void *, int32_t, int32_t) { return false; }
  bool replay(void *) const { return false; }
  bool is_valid(int32_t, int32_t) const { return false; }
  void invalidate() {}
};

inline bool cuda_graphs_enabled() { return false; }
inline int32_t cuda_graph_cache_size() { return 0; }
inline const char * cuda_graph_backend() { return "none"; }
inline void cuda_graph_log_config_once() {}
inline void cuda_graph_apply_backend_env_policy() {}
inline bool cuda_graph_decode_step(llama_context *, llama_batch &, int32_t, int32_t *) { return false; }
inline void cuda_graph_decode_invalidate_context(const llama_context *) {}
inline void cuda_graph_decode_invalidate_all() {}

#endif

}  // namespace korith::core
