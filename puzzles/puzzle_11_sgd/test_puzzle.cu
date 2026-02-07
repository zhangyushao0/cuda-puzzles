#include <catch2/catch_test_macros.hpp>
// Puzzle 11: SGD Optimizer — Test Harness
//
// Tests:
//   1. Hardcoded update — verify exact w -= lr*grad on small array
//   2. Multi-step loss decrease — SGD on quadratic loss, verify convergence
//   3. Gradient zeroing — verify zero_gradients sets all to 0.0f
//   4. LeNet param count — update ~44K parameters at scale

#include "cuda_utils.h"
#include "test_utils.h"

// Include the kernel implementation
// Build system compiles this file twice:
//   puzzle_11_test          -> includes puzzle.cu
//   puzzle_11_test_solution -> includes solution.cu
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// Helper: run sgd_update on GPU (modifies weights in-place)
void run_sgd_update_gpu(float* h_weights, const float* h_gradients,
                        float learning_rate, int n) {
    float *d_weights, *d_gradients;
    size_t bytes = n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_weights,   bytes));
    CUDA_CHECK(cudaMalloc(&d_gradients, bytes));
    CUDA_CHECK(cudaMemcpy(d_weights,   h_weights,   bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gradients, h_gradients, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sgd_update<<<blocks, threads>>>(d_weights, d_gradients, learning_rate, n);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_gradients));
}

// Helper: run zero_gradients on GPU
void run_zero_gradients_gpu(float* h_gradients, int n) {
    float *d_gradients;
    size_t bytes = n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_gradients, bytes));
    CUDA_CHECK(cudaMemcpy(d_gradients, h_gradients, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    zero_gradients<<<blocks, threads>>>(d_gradients, n);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_gradients, d_gradients, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_gradients));
}

// Helper: run sgd_update on GPU with persistent device memory (for multi-step)
// Returns device pointers; caller must free them
struct DevicePtrs {
    float* d_weights;
    float* d_gradients;
};

DevicePtrs upload_to_device(const float* h_weights, int n) {
    DevicePtrs ptrs;
    size_t bytes = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&ptrs.d_weights,   bytes));
    CUDA_CHECK(cudaMalloc(&ptrs.d_gradients, bytes));
    CUDA_CHECK(cudaMemcpy(ptrs.d_weights, h_weights, bytes, cudaMemcpyHostToDevice));
    return ptrs;
}

void download_weights(float* h_weights, const DevicePtrs& ptrs, int n) {
    CUDA_CHECK(cudaMemcpy(h_weights, ptrs.d_weights, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void free_device(DevicePtrs& ptrs) {
    CUDA_CHECK(cudaFree(ptrs.d_weights));
    CUDA_CHECK(cudaFree(ptrs.d_gradients));
}

// ============================================================
// Test 1: Hardcoded SGD update — verify exact w -= lr*grad
// ============================================================
//
// Weights:    [ 1.0,  2.0, -0.5,  0.0,  3.0, -1.0,  0.25,  4.0]
// Gradients:  [ 0.5, -1.0,  2.0,  0.0,  0.1, -0.3,  0.0,   1.5]
// lr = 0.1
//
// Expected:   w_new = w_old - 0.1 * grad
//   [1.0 - 0.05, 2.0 + 0.1, -0.5 - 0.2, 0.0 - 0.0, 3.0 - 0.01, -1.0 + 0.03, 0.25 - 0.0, 4.0 - 0.15]
// = [0.95,        2.1,       -0.7,        0.0,        2.99,        -0.97,        0.25,        3.85]

TEST_CASE("sgd_hardcoded_update", "[puzzle_11_sgd]") {
    const int n = 8;
    const float lr = 0.1f;

    float h_weights[]   = { 1.0f,  2.0f, -0.5f,  0.0f,  3.0f, -1.0f,  0.25f,  4.0f};
    float h_gradients[] = { 0.5f, -1.0f,  2.0f,  0.0f,  0.1f, -0.3f,  0.0f,   1.5f};
    float expected[]    = { 0.95f, 2.1f, -0.7f,  0.0f,  2.99f,-0.97f,  0.25f,  3.85f};

    run_sgd_update_gpu(h_weights, h_gradients, lr, n);

    REQUIRE(check_array_close(h_weights, expected, n, 1e-6f, 1e-6f));
}

// ============================================================
// Test 2: Multi-step SGD on quadratic loss — verify loss decreases
// ============================================================
//
// Quadratic loss: L(w) = 0.5 * Σ (w[i] - target[i])^2
// Gradient:       grad[i] = w[i] - target[i]
//
// With correct SGD steps, loss should strictly decrease each step.
// After enough steps, weights should converge toward the target.

TEST_CASE("sgd_multi_step_loss_decrease", "[puzzle_11_sgd]") {
    const int n = 16;
    const float lr = 0.1f;
    const int num_steps = 20;

    // Initial weights (far from target)
    float h_weights[16];
    float target[16];

    fill_random(h_weights, n, 100, -3.0f, 3.0f);
    fill_random(target,    n, 200,  0.0f, 1.0f);

    // Compute initial loss
    auto compute_loss = [&](const float* w) -> float {
        float loss = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = w[i] - target[i];
            loss += 0.5f * diff * diff;
        }
        return loss;
    };

    // Compute gradient of quadratic loss: grad[i] = w[i] - target[i]
    auto compute_grad = [&](const float* w, float* grad) {
        for (int i = 0; i < n; i++) {
            grad[i] = w[i] - target[i];
        }
    };

    float prev_loss = compute_loss(h_weights);
    float gradients[16];

    for (int step = 0; step < num_steps; step++) {
        // Compute gradient on CPU (simulating backward pass)
        compute_grad(h_weights, gradients);

        // Run SGD update on GPU
        run_sgd_update_gpu(h_weights, gradients, lr, n);

        // Check loss decreased
        float current_loss = compute_loss(h_weights);
        if (current_loss >= prev_loss) {
            fprintf(stderr, "  Step %d: loss increased from %.6f to %.6f\n",
                    step, prev_loss, current_loss);
            throw std::runtime_error("SGD multi-step: loss did not decrease");
        }
        prev_loss = current_loss;
    }

    // After 20 steps with lr=0.1 on quadratic, loss should be much smaller
    float final_loss = compute_loss(h_weights);
    float initial_loss_approx = 30.0f;  // rough upper bound for initial random weights
    if (final_loss > initial_loss_approx) {
        fprintf(stderr, "  Final loss %.6f is still very high\n", final_loss);
        throw std::runtime_error("SGD multi-step: did not converge");
    }
}

// ============================================================
// Test 3: Gradient zeroing — verify all elements set to 0.0f
// ============================================================
//
// Fill gradient array with non-zero values, run zero_gradients,
// verify every element is exactly 0.0f.

TEST_CASE("sgd_gradient_zeroing", "[puzzle_11_sgd]") {
    const int n = 1024;

    std::vector<float> h_gradients(n);
    fill_random(h_gradients.data(), n, 300, -10.0f, 10.0f);

    // Verify we have non-zero values before zeroing
    bool has_nonzero = false;
    for (int i = 0; i < n; i++) {
        if (h_gradients[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    if (!has_nonzero) {
        throw std::runtime_error("Test setup error: all gradients already zero");
    }

    // Run zero_gradients on GPU
    run_zero_gradients_gpu(h_gradients.data(), n);

    // Verify every element is exactly 0.0f
    std::vector<float> expected(n, 0.0f);
    REQUIRE(check_array_close(h_gradients.data(), expected.data(), n, 0.0f, 0.0f));
}

// ============================================================
// Test 4: LeNet param count — update ~44K parameters at scale
// ============================================================
//
// LeNet-5 total parameters:
//   Conv1: 6*(5*5*1+1)    =    156
//   Conv2: 16*(5*5*6+1)   =  2,416
//   FC1:   256*120+120     = 30,840
//   FC2:   120*84+84       = 10,164
//   FC3:   84*10+10        =    850
//   Total:                   44,426
//
// This test verifies SGD works correctly at LeNet scale by
// comparing GPU results against CPU reference.

TEST_CASE("sgd_lenet_param_count", "[puzzle_11_sgd]") {
    const int n = 44426;  // Total LeNet-5 parameters
    const float lr = 0.01f;

    std::vector<float> h_weights(n);
    std::vector<float> h_gradients(n);
    std::vector<float> expected(n);

    fill_random(h_weights.data(),   n, 400, -0.5f, 0.5f);
    fill_random(h_gradients.data(), n, 401, -0.1f, 0.1f);

    // CPU reference: w_new = w_old - lr * grad
    for (int i = 0; i < n; i++) {
        expected[i] = h_weights[i] - lr * h_gradients[i];
    }

    // Run on GPU
    run_sgd_update_gpu(h_weights.data(), h_gradients.data(), lr, n);

    REQUIRE(check_array_close(h_weights.data(), expected.data(), n, 1e-6f, 1e-6f));
}

