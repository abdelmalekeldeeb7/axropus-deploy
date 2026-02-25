#include "cuda_graph_decode.h"

#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <vector>

namespace korith::core {
namespace {

struct GraphCacheEntry {
  const llama_context * ctx = nullptr;
  int32_t batch_size = 0;
  int32_t model_dim = 0;
  CudaGraphDecodeSession session;
  std::uint64_t last_used = 0;
  bool capture_failed = false;
};

std::mutex g_graph_mu;
std::vector<GraphCacheEntry> g_graph_cache;
std::uint64_t g_graph_tick = 0;

int32_t parse_env_i32(const char * name, int32_t default_value, int32_t lo, int32_t hi) {
  const char * v = std::getenv(name);
  if (!v || v[0] == '\0') {
    return default_value;
  }
  char * end = nullptr;
  const long parsed = std::strtol(v, &end, 10);
  if (!end || end == v || *end != '\0') {
    return default_value;
  }
  return std::clamp(static_cast<int32_t>(parsed), lo, hi);
}

bool have_cuda_device() {
  int device_count = 0;
  const cudaError_t rc = cudaGetDeviceCount(&device_count);
  return (rc == cudaSuccess) && (device_count > 0);
}

GraphCacheEntry * find_entry_unlocked(
    const llama_context * ctx,
    int32_t batch_size,
    int32_t model_dim) {
  for (auto & entry : g_graph_cache) {
    if (entry.ctx == ctx && entry.batch_size == batch_size && entry.model_dim == model_dim) {
      return &entry;
    }
  }
  return nullptr;
}

void evict_if_needed_unlocked() {
  const int32_t cap = cuda_graph_cache_size();
  while (static_cast<int32_t>(g_graph_cache.size()) > cap && !g_graph_cache.empty()) {
    std::size_t lru_idx = 0;
    for (std::size_t i = 1; i < g_graph_cache.size(); ++i) {
      if (g_graph_cache[i].last_used < g_graph_cache[lru_idx].last_used) {
        lru_idx = i;
      }
    }
    g_graph_cache[lru_idx].session.invalidate();
    g_graph_cache.erase(g_graph_cache.begin() + static_cast<std::ptrdiff_t>(lru_idx));
  }
}

GraphCacheEntry & get_or_create_entry_unlocked(
    const llama_context * ctx,
    int32_t batch_size,
    int32_t model_dim) {
  if (GraphCacheEntry * existing = find_entry_unlocked(ctx, batch_size, model_dim)) {
    existing->last_used = ++g_graph_tick;
    return *existing;
  }

  GraphCacheEntry entry{};
  entry.ctx = ctx;
  entry.batch_size = batch_size;
  entry.model_dim = model_dim;
  entry.last_used = ++g_graph_tick;
  g_graph_cache.push_back(std::move(entry));
  evict_if_needed_unlocked();

  GraphCacheEntry * found = find_entry_unlocked(ctx, batch_size, model_dim);
  return *found;
}

}  // namespace

CudaGraphDecodeSession::~CudaGraphDecodeSession() {
  invalidate();
}

CudaGraphDecodeSession::CudaGraphDecodeSession(CudaGraphDecodeSession && other) noexcept {
  graph_ = other.graph_;
  graph_exec_ = other.graph_exec_;
  batch_size_ = other.batch_size_;
  model_dim_ = other.model_dim_;
  valid_ = other.valid_;
  other.graph_ = nullptr;
  other.graph_exec_ = nullptr;
  other.batch_size_ = 0;
  other.model_dim_ = 0;
  other.valid_ = false;
}

CudaGraphDecodeSession & CudaGraphDecodeSession::operator=(CudaGraphDecodeSession && other) noexcept {
  if (this == &other) {
    return *this;
  }
  invalidate();
  graph_ = other.graph_;
  graph_exec_ = other.graph_exec_;
  batch_size_ = other.batch_size_;
  model_dim_ = other.model_dim_;
  valid_ = other.valid_;
  other.graph_ = nullptr;
  other.graph_exec_ = nullptr;
  other.batch_size_ = 0;
  other.model_dim_ = 0;
  other.valid_ = false;
  return *this;
}

bool CudaGraphDecodeSession::capture_begin(cudaStream_t stream) {
  invalidate();
  return cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess;
}

bool CudaGraphDecodeSession::capture_end(cudaStream_t stream, int32_t batch_size, int32_t model_dim) {
  cudaGraph_t graph = nullptr;
  const cudaError_t end_rc = cudaStreamEndCapture(stream, &graph);
  if (end_rc != cudaSuccess || graph == nullptr) {
    if (graph) {
      cudaGraphDestroy(graph);
    }
    return false;
  }

  cudaGraphExec_t graph_exec = nullptr;
  const cudaError_t inst_rc = cudaGraphInstantiate(&graph_exec, graph, 0);
  if (inst_rc != cudaSuccess || graph_exec == nullptr) {
    cudaGraphDestroy(graph);
    return false;
  }

  graph_ = graph;
  graph_exec_ = graph_exec;
  batch_size_ = batch_size;
  model_dim_ = model_dim;
  valid_ = true;
  return true;
}

bool CudaGraphDecodeSession::replay(cudaStream_t stream) const {
  if (!valid_ || !graph_exec_) {
    return false;
  }
  return cudaGraphLaunch(graph_exec_, stream) == cudaSuccess;
}

bool CudaGraphDecodeSession::is_valid(int32_t batch_size, int32_t model_dim) const {
  return valid_ && batch_size_ == batch_size && model_dim_ == model_dim &&
      graph_ != nullptr && graph_exec_ != nullptr;
}

void CudaGraphDecodeSession::invalidate() {
  if (graph_exec_) {
    cudaGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
  valid_ = false;
  batch_size_ = 0;
  model_dim_ = 0;
}

bool cuda_graphs_enabled() {
  static bool initialized = false;
  static bool enabled = false;
  if (!initialized) {
    initialized = true;
    bool requested = have_cuda_device();
    if (const char * env = std::getenv("KORITH_CUDA_GRAPHS")) {
      requested = (env[0] != '\0') && (env[0] != '0');
    }
    enabled = requested && have_cuda_device();
  }
  return enabled;
}

int32_t cuda_graph_cache_size() {
  static bool initialized = false;
  static int32_t value = 4;
  if (!initialized) {
    initialized = true;
    value = parse_env_i32("KORITH_CUDA_GRAPH_CACHE_SIZE", 4, 1, 64);
  }
  return value;
}

const char * cuda_graph_backend() {
  return cuda_graphs_enabled() ? "cuda" : "none";
}

void cuda_graph_apply_backend_env_policy() {
  // Delegate graph execution to llama.cpp's internal CUDA-graph path when available.
  // We only control whether that backend path is permitted.
  if (cuda_graphs_enabled()) {
    (void) ::unsetenv("GGML_CUDA_DISABLE_GRAPHS");
  } else {
    (void) ::setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
  }
}

void cuda_graph_log_config_once() {
  static bool logged = false;
  if (logged) {
    return;
  }
  logged = true;
  std::fprintf(stderr,
               "[CUDA_GRAPH] enabled=%d cache_size=%d backend=%s\n",
               cuda_graphs_enabled() ? 1 : 0,
               cuda_graph_cache_size(),
               cuda_graph_backend());
  (void) std::fflush(stderr);
}

bool cuda_graph_decode_step(
    llama_context * ctx,
    llama_batch & batch,
    int32_t model_dim,
    int32_t * submit_us) {
  if (submit_us) {
    *submit_us = 0;
  }
  if (!ctx || !cuda_graphs_enabled()) {
    return false;
  }
  if (batch.n_tokens <= 0) {
    return false;
  }
  static bool capture_flag_checked = false;
  static bool capture_enabled = false;
  if (!capture_flag_checked) {
    capture_flag_checked = true;
    if (const char * env = std::getenv("KORITH_CUDA_GRAPH_CAPTURE")) {
      capture_enabled = (env[0] != '\0') && (env[0] != '0');
    } else if (const char * env_legacy = std::getenv("KORITH_CUDA_GRAPH_CAPTURE_EXPERIMENTAL")) {
      // Legacy compatibility for old benchmark scripts.
      capture_enabled = (env_legacy[0] != '\0') && (env_legacy[0] != '0');
    }
  }
  if (!capture_enabled) {
    return false;
  }

  cuda_graph_log_config_once();

  constexpr cudaStream_t kStream = nullptr;  // default stream; llama.cpp decode launches into this stream context.
  const int32_t batch_size = batch.n_tokens;

  bool capture_mode = false;
  {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphCacheEntry & entry = get_or_create_entry_unlocked(ctx, batch_size, model_dim);
    if (!entry.session.is_valid(batch_size, model_dim)) {
      if (entry.capture_failed) {
        return false;
      }
      capture_mode = true;
    } else {
      entry.last_used = ++g_graph_tick;
    }
  }

  const auto t0 = std::chrono::steady_clock::now();

  if (capture_mode) {
    bool capture_ok = false;
    {
      std::lock_guard<std::mutex> lock(g_graph_mu);
      GraphCacheEntry * entry = find_entry_unlocked(ctx, batch_size, model_dim);
      if (!entry) {
        return false;
      }
      entry->session.invalidate();
      if (entry->session.capture_begin(kStream)) {
        const int32_t rc = llama_decode(ctx, batch);
        const bool graph_ok = entry->session.capture_end(kStream, batch_size, model_dim);
        capture_ok = (rc == 0) && graph_ok;
      }
      entry->capture_failed = !capture_ok;
      entry->last_used = ++g_graph_tick;
      if (capture_ok) {
        std::fprintf(stderr,
                     "[CUDA_GRAPH] captured batch_size=%d model_dim=%d replaying=true\n",
                     batch_size,
                     model_dim);
        (void) std::fflush(stderr);
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (submit_us) {
      *submit_us = static_cast<int32_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    }
    return capture_ok;
  }

  bool replay_ok = false;
  {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    GraphCacheEntry * entry = find_entry_unlocked(ctx, batch_size, model_dim);
    if (!entry || !entry->session.is_valid(batch_size, model_dim)) {
      return false;
    }
    replay_ok = entry->session.replay(kStream);
    if (!replay_ok) {
      entry->session.invalidate();
      entry->capture_failed = true;
    } else {
      entry->last_used = ++g_graph_tick;
    }
  }
  if (!replay_ok) {
    return false;
  }
  if (cudaStreamSynchronize(kStream) != cudaSuccess) {
    return false;
  }

  const auto t1 = std::chrono::steady_clock::now();
  if (submit_us) {
    *submit_us = static_cast<int32_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
  }
  return true;
}

void cuda_graph_decode_invalidate_context(const llama_context * ctx) {
  if (!ctx) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_graph_mu);
  for (auto it = g_graph_cache.begin(); it != g_graph_cache.end();) {
    if (it->ctx == ctx) {
      it->session.invalidate();
      it = g_graph_cache.erase(it);
    } else {
      ++it;
    }
  }
}

void cuda_graph_decode_invalidate_all() {
  std::lock_guard<std::mutex> lock(g_graph_mu);
  for (auto & entry : g_graph_cache) {
    entry.session.invalidate();
  }
  g_graph_cache.clear();
}

}  // namespace korith::core
