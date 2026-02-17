#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// MLIR C-Interface wrappers
extern "C" {
    void _mlir_ciface_cuda_add(float* a, float* b, int n);
    void _mlir_ciface_maths_test(float* a, int n);
    void _mlir_ciface_atomic_test(float* data, float* sum, int n);
    void _mlir_ciface_shared_mem_test(float* data);
}

void test_cuda_add() {
    int n = 1025; // Intentional odd number to test masking/tail loops
    std::vector<float> a(n, 1.0f), b(n, 2.0f);
    _mlir_ciface_cuda_add(a.data(), b.data(), n);
    assert(std::abs(a[0] - 3.0f) < 1e-5);
    assert(std::abs(a[1024] - 3.0f) < 1e-5);
    std::cout << "Test CUDA Add: PASSED\n";
}

void test_maths_intrinsics() {
    int n = 256;
    std::vector<float> data(n);
    std::vector<float> expected(n);
    
    for(int i = 0; i < n; ++i) {
        data[i] = (float)i * 0.1f;
        // The CUDA kernel does: a[i] = sinf(a[i]) + expf(a[i])
        expected[i] = std::sin(data[i]) + std::exp(data[i]);
    }

    _mlir_ciface_maths_test(data.data(), n);

    for(int i = 0; i < n; ++i) {
      assert(is_close(data[i], expected[i], 1e-5))
    }
    std::cout << "Test Maths Intrinsics: PASSED (sin + exp)\n";
}

void test_atomic_reduction() {
    int n = 1024;
    std::vector<float> data(n, 1.0f);
    float sum = 0.0f;
    _mlir_ciface_atomic_test(data.data(), &sum, n);
    assert(std::abs(sum - 1024.0f) < 1e-5);
    std::cout << "Test Atomic Reduction: PASSED\n";
}

void test_shared_mem() {
    int n = 256;
    std::vector<float> data(n);
    for(int i=0; i<n; ++i) data[i] = (float)i;
    _mlir_ciface_shared_mem_test(data.data());
    assert(data[0] == 255.0f);
    assert(data[255] == 0.0f);
    std::cout << "Test Shared Memory (Reverse): PASSED\n";
}

int main() {
    try {
        test_vec_add();
        test_atomic_reduction();
        test_shared_mem();
        std::cout << "\nALL TESTS PASSED\n";
    } catch (...) {
        std::cerr << "TEST FAILED!\n";
        return 1;
    }
    return 0;
}
