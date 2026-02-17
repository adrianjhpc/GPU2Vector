#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for performing reduction (sum) of an array
__global__ void reduceSum(int *g_input, int *g_output, int n) {
    extern __shared__ int s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // All threads must load something to shared memory to keep sync consistent
    if (i < n) {
        s_data[tid] = g_input[i];
    } else {
        s_data[tid] = 0;
    }

    // First sync: all threads reached this point
    __syncthreads();

    // Do reduction in shared memory for the Full block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        // All threads must reach this sync point every iteration
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) atomicAdd(g_output, s_data[0]);
}

int main() {
    int n = 2048;
    int size = n * sizeof(int);
    int *h_input, *h_output;
    int *d_input, *d_output;

    // Allocate host memory
    h_input = (int*)malloc(size);
    h_output = (int*)malloc(sizeof(int));

    // Initialize input array
    for(int i = 0; i < n; i++) {
        h_input[i] = 1;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, sizeof(int));

    // Copy from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 64;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    h_output = (int*)malloc(sizeof(int)*blocksPerGrid);

    // Initialize output array
    for(int i = 0; i < blocksPerGrid; i++) {
        h_output[i] = 1;
    }

    reduceSum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_input, d_output, n);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum is %d\n", *h_output);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

