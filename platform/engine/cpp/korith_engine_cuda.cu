#include <cuda_runtime.h>

namespace {

__global__ void korith_noop_kernel() {}

}  // namespace

extern "C" bool korith_engine_cuda_probe() {
  cudaError_t err = cudaSuccess;
  korith_noop_kernel<<<1, 1>>>();
  err = cudaDeviceSynchronize();
  return err == cudaSuccess;
}

