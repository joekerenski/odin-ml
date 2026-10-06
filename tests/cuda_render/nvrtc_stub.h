// What NVRTC provides implicitly, declared for a clang -fsyntax-only check.
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __device__ __attribute__((device))
struct __dim3 { unsigned int x, y, z; };
extern __device__ const __dim3 blockIdx, threadIdx, blockDim, gridDim;
__device__ float expf(float); __device__ float logf(float); __device__ float sqrtf(float);
__device__ float __uint_as_float(unsigned int); __device__ float __int_as_float(int);
__device__ void __syncthreads();
__device__ float __shfl_xor_sync(unsigned int mask, float v, int lane_mask);
