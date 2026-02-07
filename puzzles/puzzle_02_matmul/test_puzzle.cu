// Puzzle 02: Matrix Multiplication — Test Harness
//
// Tests:
//   1. 2x2 * 2x2 — hardcoded, hand-verifiable
//   2. 4x3 * 3x5 — hardcoded rectangular matrices
//   3. 64x128 * 128x64 — seeded random, CPU reference
//   4. 256x120 — LeNet FC1 dimensions (256 inputs -> 120 outputs)

#include "cuda_utils.h"
#include "test_utils.h"
#include <catch2/catch_test_macros.hpp>

// Include the kernel implementation
// Build system compiles this file twice:
//   puzzle_02_test          -> includes puzzle.cu
//   puzzle_02_test_solution -> includes solution.cu
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// CPU reference implementation for verification
void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Helper: run matmul kernel on GPU and copy result back to host
void run_matmul_gpu(const float* h_A, const float* h_B, float* h_C,
                    int M, int K, int N) {
    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// Test 1: 2x2 * 2x2 — hand-verifiable
// A = [1 2; 3 4], B = [5 6; 7 8]
// C = [1*5+2*7  1*6+2*8;  3*5+4*7  3*6+4*8] = [19 22; 43 50]
TEST_CASE("matmul_2x2", "[puzzle_02]") {
  const int M = 2, K = 2, N = 2;
  float h_A[] = {1, 2, 3, 4};
  float h_B[] = {5, 6, 7, 8};
  float h_C[4] = {0};
  float expected[] = {19, 22, 43, 50};

  run_matmul_gpu(h_A, h_B, h_C, M, K, N);

  REQUIRE(check_array_close(h_C, expected, M * N));
}

// Test 2: 4x3 * 3x5 — hardcoded rectangular matrices
// A (4x3):          B (3x5):
// [1 2 3]           [1  2  3  4  5]
// [4 5 6]           [6  7  8  9 10]
// [7 8 9]           [11 12 13 14 15]
// [10 11 12]
TEST_CASE("matmul_4x3_3x5", "[puzzle_02]") {
  const int M = 4, K = 3, N = 5;
  float h_A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float h_B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  // CPU reference
  float expected[M * N];
  matmul_cpu(h_A, h_B, expected, M, K, N);

  // Hand-verify first element: C[0,0] = 1*1 + 2*6 + 3*11 = 1+12+33 = 46
  REQUIRE(expected[0] == 46.0f);

  float h_C[M * N] = {0};
  run_matmul_gpu(h_A, h_B, h_C, M, K, N);

  REQUIRE(check_array_close(h_C, expected, M * N));
}

// Test 3: 64x128 * 128x64 — seeded random, compared against CPU reference
TEST_CASE("matmul_64x128_128x64", "[puzzle_02]") {
  const int M = 64, K = 128, N = 64;
  const int size_A = M * K;
  const int size_B = K * N;
  const int size_C = M * N;

  std::vector<float> h_A(size_A);
  std::vector<float> h_B(size_B);
  std::vector<float> h_C(size_C, 0.0f);
  std::vector<float> expected(size_C, 0.0f);

  fill_random(h_A.data(), size_A, 42, -1.0f, 1.0f);
  fill_random(h_B.data(), size_B, 43, -1.0f, 1.0f);

  matmul_cpu(h_A.data(), h_B.data(), expected.data(), M, K, N);
  run_matmul_gpu(h_A.data(), h_B.data(), h_C.data(), M, K, N);

  // Use slightly relaxed tolerance for larger matrices (accumulated FP error)
  REQUIRE(check_array_close(h_C.data(), expected.data(), size_C, 1e-3f, 1e-3f));
}

// Test 4: 256x120 — LeNet FC1 layer dimensions
// This is the exact shape of FC1 in LeNet-5: 256 inputs -> 120 outputs
// In a real FC layer: output = input * weights^T
// Here we test: A(8x256) * B(256x120) = C(8x120)  (batch of 8)
TEST_CASE("matmul_lenet_fc1_256x120", "[puzzle_02]") {
  const int M = 8;   // batch size
  const int K = 256; // input features (4*4*16 from conv layers)
  const int N = 120; // output features (FC1 output)
  const int size_A = M * K;
  const int size_B = K * N;
  const int size_C = M * N;

  std::vector<float> h_A(size_A);
  std::vector<float> h_B(size_B);
  std::vector<float> h_C(size_C, 0.0f);
  std::vector<float> expected(size_C, 0.0f);

  fill_random(h_A.data(), size_A, 100, -0.5f, 0.5f);
  fill_random(h_B.data(), size_B, 101, -0.5f, 0.5f);

  matmul_cpu(h_A.data(), h_B.data(), expected.data(), M, K, N);
  run_matmul_gpu(h_A.data(), h_B.data(), h_C.data(), M, K, N);

  REQUIRE(check_array_close(h_C.data(), expected.data(), size_C, 1e-3f, 1e-3f));
}
