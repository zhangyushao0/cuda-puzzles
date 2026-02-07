#include <catch2/catch_test_macros.hpp>
// Puzzle 04: ReLU Forward + Backward — Test Harness
//
// Tests:
//   1. Forward pass with negative, zero, and positive values
//   2. Backward pass — gradient masking verification
//   3. Round-trip: forward then backward end-to-end
//   4. Edge case: behavior at exactly x=0

#include "cuda_utils.h"
#include "test_utils.h"

// Include the kernel implementation
// Build system compiles this file twice:
//   puzzle_04_test          -> includes puzzle.cu
//   puzzle_04_test_solution -> includes solution.cu
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// Helper: run relu_forward on GPU
void run_relu_forward_gpu(const float* h_input, float* h_output, int n) {
    float *d_input, *d_output;
    size_t bytes = n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input,  bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_forward<<<blocks, threads>>>(d_input, d_output, n);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Helper: run relu_backward on GPU
void run_relu_backward_gpu(const float* h_grad_output, const float* h_input,
                           float* h_grad_input, int n) {
    float *d_grad_output, *d_input, *d_grad_input;
    size_t bytes = n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_grad_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_input,       bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_input,  bytes));
    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input,       h_input,       bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_backward<<<blocks, threads>>>(d_grad_output, d_input, d_grad_input, n);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_grad_input, d_grad_input, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_grad_input));
}

// Test 1: Forward pass with negative, zero, and positive values
//
// Input:    [-3.0, -1.0, -0.5,  0.0,  0.5,  1.0,  3.0,  0.001]
// Expected: [ 0.0,  0.0,  0.0,  0.0,  0.5,  1.0,  3.0,  0.001]
//
// Verifies:
//   - Negative values → clamped to 0
//   - Zero → stays 0
//   - Positive values → pass through unchanged
TEST_CASE("relu_forward_neg_zero_pos", "[puzzle_04_relu]") {
    const int n = 8;

    float h_input[]    = {-3.0f, -1.0f, -0.5f,  0.0f,  0.5f,  1.0f,  3.0f,  0.001f};
    float expected[]   = { 0.0f,  0.0f,  0.0f,  0.0f,  0.5f,  1.0f,  3.0f,  0.001f};
    float h_output[8]  = {0};

    run_relu_forward_gpu(h_input, h_output, n);

    REQUIRE(check_array_close(h_output, expected, n, 1e-6f, 1e-6f));
}

// Test 2: Backward pass — gradient routing verification
//
// Input:       [-2.0,  3.5, -0.1,  1.7,  0.0, -4.2,  0.8, -0.01]
// Grad output: [ 0.5, -1.2,  0.8, -0.3,  2.0,  0.1, -0.7,  1.5]
// Expected:    [ 0.0, -1.2,  0.0, -0.3,  0.0,  0.0, -0.7,  0.0]
//
// Verifies:
//   - Gradients pass through where input > 0
//   - Gradients blocked (zeroed) where input ≤ 0
//   - Gradient values preserved exactly (no scaling)
TEST_CASE("relu_backward_gradient_routing", "[puzzle_04_relu]") {
    const int n = 8;

    float h_input[]       = {-2.0f,  3.5f, -0.1f,  1.7f,  0.0f, -4.2f,  0.8f, -0.01f};
    float h_grad_output[] = { 0.5f, -1.2f,  0.8f, -0.3f,  2.0f,  0.1f, -0.7f,  1.5f};
    float expected[]      = { 0.0f, -1.2f,  0.0f, -0.3f,  0.0f,  0.0f, -0.7f,  0.0f};
    float h_grad_input[8] = {0};

    run_relu_backward_gpu(h_grad_output, h_input, h_grad_input, n);

    REQUIRE(check_array_close(h_grad_input, expected, n, 1e-6f, 1e-6f));
}

// Test 3: Round-trip — forward then backward, large random array
//
// Verifies end-to-end gradient flow through ReLU:
//   1. Forward: compute output = max(0, input)
//   2. Backward: compute grad_input from grad_output using original input
//   3. Check: grad_input matches CPU reference
//
// Uses 10000 elements to stress-test grid/block edge cases
TEST_CASE("relu_round_trip", "[puzzle_04_relu]") {
    const int n = 10000;

    std::vector<float> h_input(n);
    std::vector<float> h_grad_output(n);
    std::vector<float> h_output(n, 0.0f);
    std::vector<float> h_grad_input(n, 0.0f);
    std::vector<float> expected_output(n);
    std::vector<float> expected_grad(n);

    // Generate random data centered around 0 (mix of positive and negative)
    fill_random(h_input.data(),       n, 42,  -2.0f, 2.0f);
    fill_random(h_grad_output.data(), n, 43,  -1.0f, 1.0f);

    // CPU reference: forward
    for (int i = 0; i < n; i++) {
        expected_output[i] = (h_input[i] > 0.0f) ? h_input[i] : 0.0f;
    }

    // CPU reference: backward
    for (int i = 0; i < n; i++) {
        expected_grad[i] = (h_input[i] > 0.0f) ? h_grad_output[i] : 0.0f;
    }

    // GPU: forward
    run_relu_forward_gpu(h_input.data(), h_output.data(), n);

    REQUIRE(check_array_close(h_output.data(), expected_output.data(), n, 1e-6f, 1e-6f));

    // GPU: backward (using original input, not output)
    run_relu_backward_gpu(h_grad_output.data(), h_input.data(),
                          h_grad_input.data(), n);

    REQUIRE(check_array_close(h_grad_input.data(), expected_grad.data(), n, 1e-6f, 1e-6f));
}

// Test 4: Edge case — behavior at exactly x=0
//
// Mathematical convention: ReLU'(0) is undefined, but in practice we
// define it as 0 (the "subgradient" convention). This means:
//   - Forward: max(0, 0) = 0  (unambiguous)
//   - Backward: gradient is blocked at x=0 (convention: derivative = 0)
//
// This test uses values very close to zero on both sides to verify
// the boundary is handled correctly.
TEST_CASE("relu_edge_case_zero", "[puzzle_04_relu]") {
    const int n = 8;

    // Values at and very near zero
    float h_input[]    = { 0.0f,  -0.0f,  1e-7f, -1e-7f,  1e-30f, -1e-30f,  0.0f,  0.0f};
    float expected_fwd[] = { 0.0f,  0.0f,  1e-7f,  0.0f,  1e-30f,  0.0f,  0.0f,  0.0f};
    float h_output[8] = {0};

    // Forward: check boundary values
    run_relu_forward_gpu(h_input, h_output, n);

    REQUIRE(check_array_close(h_output, expected_fwd, n, 1e-38f, 1e-6f));

    // Backward: all grad_output = 1.0 to clearly see which gradients pass
    float h_grad_output[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float expected_bwd[]  = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    float h_grad_input[8] = {0};

    run_relu_backward_gpu(h_grad_output, h_input, h_grad_input, n);

    REQUIRE(check_array_close(h_grad_input, expected_bwd, n, 1e-6f, 1e-6f));
}

