#include <cuda_runtime.h>

extern "C" {

// v1 minimal placeholder kernels used by the Python kernel registry.
// Real fused kernels can replace these signatures without changing platform wiring.
__global__ void korith_kv_restore_kernel(const unsigned char * src, unsigned char * dst, size_t n) {
  const size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) {
    dst[i] = src[i];
  }
}

__global__ void korith_decode_attn_kernel(const float * q, const float * k, const float * v, float * out, int n) {
  const int i = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (i < n) {
    out[i] = q[i] + k[i] + v[i];
  }
}

__global__ void korith_rmsnorm_linear_kernel(const float * x, float * y, int n) {
  const int i = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (i < n) {
    y[i] = x[i];
  }
}

}

