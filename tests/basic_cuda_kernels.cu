#include <cuda_runtime.h>
#include <math.h>

// Basic CUDA add kernel (Tests load/store/arith/masking)
extern "C" __global__ void cuda_add(float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = a[i] + b[i];
    }
}

// Maths intrinsics (Tests math dialect mapping)
extern "C" __global__ void maths_test(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = sinf(a[i]) + expf(a[i]);
    }
}

// Atomics (Tests vector reduction logic)
extern "C" __global__ void atomic_test(float *data, float *sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(sum, data[i]);
    }
}

// Shared memory (Tests Alloca conversion and barrier removal)
extern "C" __global__ void shared_mem_test(float *data) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    
    // Load into shared
    temp[tid] = data[tid];
    __syncthreads();
    
    // Reverse data using shared memory
    data[tid] = temp[255 - tid];
}