#include <iostream>
#include <vector>

// This matches the signature of the CUDA kernel: 
// __global__ void add(float *a, float *b, int n)
// MLIR's "emit-c-interface" adds the _mlir_ciface_ prefix.
extern "C" void _mlir_ciface_add(float* a, float* b, int n);

int main() {
    int N = 1024;
    std::vector<float> dataA(N, 1.0f);
    std::vector<float> dataB(N, 2.0f);

    // Call the vectorised "GPU" kernel as a CPU function
    _mlir_ciface_add(dataA.data(), dataB.data(), N);

    // Verify result
    std::cout << "Result[0]: " << dataA[0] << " (Expected 3.0)" << std::endl;
    std::cout << "Result[N-1]: " << dataA[N-1] << " (Expected 3.0)" << std::endl;

    return 0;
}
