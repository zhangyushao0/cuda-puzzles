// Puzzle 01: Vector Addition — Test Suite
// Tests the vector_add kernel with various array sizes.

#include "test_utils.h"
#include "cuda_utils.h"

// Forward-declare the kernel (defined in puzzle.cu or solution.cu)
__global__ void vector_add(float* A, float* B, float* C, int N);

// ---------------------------------------------------------------
// Test 1: Small array with hardcoded values (N=8)
// ---------------------------------------------------------------
TEST_CASE(small_array) {
    const int N = 8;
    float A[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float B[N] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float C[N];
    float expected[N] = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    KERNEL_CHECK();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    if (!check_array_close(C, expected, N)) {
        throw std::runtime_error("Output does not match expected values");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ---------------------------------------------------------------
// Test 2: Medium array with seeded random values (N=1024)
// ---------------------------------------------------------------
TEST_CASE(medium_array) {
    const int N = 1024;
    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];
    float* expected = new float[N];

    // Fill with deterministic random values
    fill_random(A, N, 42, -10.0f, 10.0f);
    fill_random(B, N, 43, -10.0f, 10.0f);

    // Compute expected result on host
    for (int i = 0; i < N; i++) {
        expected[i] = A[i] + B[i];
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    KERNEL_CHECK();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    if (!check_array_close(C, expected, N)) {
        delete[] A; delete[] B; delete[] C; delete[] expected;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        throw std::runtime_error("Output does not match expected values");
    }

    // Cleanup
    delete[] A; delete[] B; delete[] C; delete[] expected;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// ---------------------------------------------------------------
// Test 3: Large array with seeded random values (N=1,000,000)
// ---------------------------------------------------------------
TEST_CASE(large_array) {
    const int N = 1000000;
    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];
    float* expected = new float[N];

    // Fill with deterministic random values
    fill_random(A, N, 100, -100.0f, 100.0f);
    fill_random(B, N, 200, -100.0f, 100.0f);

    // Compute expected result on host
    for (int i = 0; i < N; i++) {
        expected[i] = A[i] + B[i];
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    KERNEL_CHECK();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    if (!check_array_close(C, expected, N)) {
        delete[] A; delete[] B; delete[] C; delete[] expected;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        throw std::runtime_error("Output does not match expected values");
    }

    // Cleanup
    delete[] A; delete[] B; delete[] C; delete[] expected;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// ---------------------------------------------------------------
// Test 4: Edge case — N not divisible by block size (N=1000)
// ---------------------------------------------------------------
TEST_CASE(non_divisible_size) {
    const int N = 1000;  // Not divisible by 256
    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];
    float* expected = new float[N];

    // Fill with deterministic random values
    fill_random(A, N, 77, -5.0f, 5.0f);
    fill_random(B, N, 88, -5.0f, 5.0f);

    // Compute expected result on host
    for (int i = 0; i < N; i++) {
        expected[i] = A[i] + B[i];
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel — note: blocks * threads > N
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    KERNEL_CHECK();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    if (!check_array_close(C, expected, N)) {
        delete[] A; delete[] B; delete[] C; delete[] expected;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        throw std::runtime_error("Output does not match expected values");
    }

    // Cleanup
    delete[] A; delete[] B; delete[] C; delete[] expected;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    return RUN_ALL_TESTS();
}
