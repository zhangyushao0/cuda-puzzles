#include <catch2/catch_test_macros.hpp>
// Puzzle 06: Fully Connected Layer (Backward Pass) — Test Harness
//
// Tests:
//   1. Small example (batch=2, 3→2) — hardcoded, hand-verifiable
//   2. Numerical gradient check — perturb weight by ε, compare to analytical
//   3. LeNet FC1 dims (batch=4, 256→120) — backward pass dimensions
//   4. Shape verification — all gradient shapes match expected dimensions

#include "cuda_utils.h"
#include "test_utils.h"

// Include the kernel implementation
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// ============================================================
// CPU reference implementations for verification
// ============================================================

// FC forward pass on CPU (needed for numerical gradient check)
void fc_forward_cpu(const float* input, const float* weights, const float* bias,
                    float* output, int batch, int in_features, int out_features) {
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < out_features; j++) {
            float sum = 0.0f;
            for (int i = 0; i < in_features; i++) {
                sum += input[b * in_features + i] * weights[j * in_features + i];
            }
            output[b * out_features + j] = sum + bias[j];
        }
    }
}

// Weight gradient: dW[j][i] = Σ_b grad_output[b][j] × input[b][i]
void fc_backward_weights_cpu(const float* grad_output, const float* input,
                              float* grad_weights, int batch,
                              int in_features, int out_features) {
    for (int j = 0; j < out_features; j++) {
        for (int i = 0; i < in_features; i++) {
            float sum = 0.0f;
            for (int b = 0; b < batch; b++) {
                sum += grad_output[b * out_features + j] * input[b * in_features + i];
            }
            grad_weights[j * in_features + i] = sum;
        }
    }
}

// Bias gradient: db[j] = Σ_b grad_output[b][j]
void fc_backward_bias_cpu(const float* grad_output, float* grad_bias,
                           int batch, int out_features) {
    for (int j = 0; j < out_features; j++) {
        float sum = 0.0f;
        for (int b = 0; b < batch; b++) {
            sum += grad_output[b * out_features + j];
        }
        grad_bias[j] = sum;
    }
}

// Input gradient: dX[b][i] = Σ_j grad_output[b][j] × weights[j][i]
void fc_backward_input_cpu(const float* grad_output, const float* weights,
                            float* grad_input, int batch,
                            int in_features, int out_features) {
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < in_features; i++) {
            float sum = 0.0f;
            for (int j = 0; j < out_features; j++) {
                sum += grad_output[b * out_features + j] * weights[j * in_features + i];
            }
            grad_input[b * in_features + i] = sum;
        }
    }
}

// ============================================================
// GPU helper: run all three backward kernels
// ============================================================

void run_fc_backward_gpu(const float* h_grad_output, const float* h_input,
                          const float* h_weights,
                          float* h_grad_weights, float* h_grad_bias,
                          float* h_grad_input,
                          int batch, int in_features, int out_features) {
    float *d_grad_output, *d_input, *d_weights;
    float *d_grad_weights, *d_grad_bias, *d_grad_input;

    size_t grad_output_bytes = batch * out_features * sizeof(float);
    size_t input_bytes       = batch * in_features * sizeof(float);
    size_t weight_bytes      = out_features * in_features * sizeof(float);
    size_t bias_bytes        = out_features * sizeof(float);
    size_t grad_input_bytes  = batch * in_features * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_grad_output,  grad_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_input,        input_bytes));
    CUDA_CHECK(cudaMalloc(&d_weights,      weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_weights, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_bias,    bias_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_input,   grad_input_bytes));

    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, grad_output_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input,       h_input,       input_bytes,       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights,     h_weights,     weight_bytes,      cudaMemcpyHostToDevice));

    // Kernel 1: Weight gradient — dW = dY^T · X
    {
        dim3 block(16, 16);
        dim3 grid((in_features + block.x - 1) / block.x,
                  (out_features + block.y - 1) / block.y);
        fc_backward_weights<<<grid, block>>>(d_grad_output, d_input, d_grad_weights,
                                              batch, in_features, out_features);
        KERNEL_CHECK();
    }

    // Kernel 2: Bias gradient — db = Σ_batch dY
    {
        dim3 block(256);
        dim3 grid((out_features + block.x - 1) / block.x);
        fc_backward_bias<<<grid, block>>>(d_grad_output, d_grad_bias,
                                           batch, out_features);
        KERNEL_CHECK();
    }

    // Kernel 3: Input gradient — dX = dY · W
    {
        dim3 block(16, 16);
        dim3 grid((in_features + block.x - 1) / block.x,
                  (batch + block.y - 1) / block.y);
        fc_backward_input<<<grid, block>>>(d_grad_output, d_weights, d_grad_input,
                                            batch, in_features, out_features);
        KERNEL_CHECK();
    }

    CUDA_CHECK(cudaMemcpy(h_grad_weights, d_grad_weights, weight_bytes,      cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_bias,    d_grad_bias,    bias_bytes,        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_input,   d_grad_input,   grad_input_bytes,  cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_grad_weights));
    CUDA_CHECK(cudaFree(d_grad_bias));
    CUDA_CHECK(cudaFree(d_grad_input));
}

// ============================================================
// Test 1: Small example (batch=2, 3→2) — hardcoded, hand-verifiable
// ============================================================
//
// Input X (2×3):  [[1, 2, 3],
//                   [4, 5, 6]]
//
// Weights W (2×3): [[0.1, 0.2, 0.3],
//                    [0.4, 0.5, 0.6]]
//
// Bias b (2): [0.1, 0.2]
//
// Forward: Y = X · W^T + b
//   Y[0] = [1*0.1+2*0.2+3*0.3+0.1, 1*0.4+2*0.5+3*0.6+0.2] = [1.5, 3.4]
//   Y[1] = [4*0.1+5*0.2+6*0.3+0.1, 4*0.4+5*0.5+6*0.6+0.2] = [3.3, 7.3]
//
// Upstream gradient dY (2×2):  [[1.0, 0.5],
//                                [0.2, 0.8]]
//
// dW = dY^T · X:
//   dW[0][0] = 1.0*1 + 0.2*4 = 1.8
//   dW[0][1] = 1.0*2 + 0.2*5 = 3.0
//   dW[0][2] = 1.0*3 + 0.2*6 = 4.2
//   dW[1][0] = 0.5*1 + 0.8*4 = 3.7
//   dW[1][1] = 0.5*2 + 0.8*5 = 5.0
//   dW[1][2] = 0.5*3 + 0.8*6 = 6.3
//
// db = Σ_batch dY:
//   db[0] = 1.0 + 0.2 = 1.2
//   db[1] = 0.5 + 0.8 = 1.3
//
// dX = dY · W:
//   dX[0][0] = 1.0*0.1 + 0.5*0.4 = 0.30
//   dX[0][1] = 1.0*0.2 + 0.5*0.5 = 0.45
//   dX[0][2] = 1.0*0.3 + 0.5*0.6 = 0.60
//   dX[1][0] = 0.2*0.1 + 0.8*0.4 = 0.34
//   dX[1][1] = 0.2*0.2 + 0.8*0.5 = 0.44
//   dX[1][2] = 0.2*0.3 + 0.8*0.6 = 0.54

TEST_CASE("fc_backward_small_example", "[puzzle_06_fc_backward]") {
    const int batch = 2, in_features = 3, out_features = 2;

    float h_input[] = {1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f};

    float h_weights[] = {0.1f, 0.2f, 0.3f,
                         0.4f, 0.5f, 0.6f};

    float h_grad_output[] = {1.0f, 0.5f,
                              0.2f, 0.8f};

    float expected_dW[] = {1.8f, 3.0f, 4.2f,
                           3.7f, 5.0f, 6.3f};

    float expected_db[] = {1.2f, 1.3f};

    float expected_dX[] = {0.30f, 0.45f, 0.60f,
                           0.34f, 0.44f, 0.54f};

    float h_grad_weights[6] = {0};
    float h_grad_bias[2] = {0};
    float h_grad_input[6] = {0};

    run_fc_backward_gpu(h_grad_output, h_input, h_weights,
                         h_grad_weights, h_grad_bias, h_grad_input,
                         batch, in_features, out_features);

    bool pass = true;

    if (!check_array_close(h_grad_weights, expected_dW, out_features * in_features, 1e-4f, 1e-4f)) {
        fprintf(stderr, "  Weight gradient mismatch\n");
        pass = false;
    }

    if (!check_array_close(h_grad_bias, expected_db, out_features, 1e-4f, 1e-4f)) {
        fprintf(stderr, "  Bias gradient mismatch\n");
        pass = false;
    }

    if (!check_array_close(h_grad_input, expected_dX, batch * in_features, 1e-4f, 1e-4f)) {
        fprintf(stderr, "  Input gradient mismatch\n");
        pass = false;
    }

    if (!pass) {
        throw std::runtime_error("Small example (2×3→2) backward mismatch");
    }
}

// ============================================================
// Test 2: Numerical gradient check
// ============================================================
//
// For each weight w[j][i], compute:
//   numerical_grad = (L(w+ε) - L(w-ε)) / (2ε)
// where L is a simple sum-of-outputs loss.
//
// Compare to analytical dW[j][i] with relative error < 1e-3.

TEST_CASE("fc_backward_numerical_gradient_check", "[puzzle_06_fc_backward]") {
    const int batch = 2, in_features = 4, out_features = 3;
    const float epsilon = 1e-3f;
    const float rel_tol = 1e-3f;

    const int weight_size = out_features * in_features;
    const int input_size  = batch * in_features;
    const int output_size = batch * out_features;

    std::vector<float> h_input(input_size);
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_bias(out_features);

    fill_random(h_input.data(),   input_size,  600, -1.0f, 1.0f);
    fill_random(h_weights.data(), weight_size, 601, -1.0f, 1.0f);
    fill_random(h_bias.data(),    out_features, 602, -0.5f, 0.5f);

    // Forward pass to get output, then use dY = ones (sum loss: L = Σ Y)
    // This means ∂L/∂Y = 1 for all elements
    std::vector<float> h_grad_output(output_size, 1.0f);

    // Compute analytical gradients via GPU kernels
    std::vector<float> analytical_dW(weight_size, 0.0f);
    std::vector<float> analytical_db(out_features, 0.0f);
    std::vector<float> analytical_dX(input_size, 0.0f);

    run_fc_backward_gpu(h_grad_output.data(), h_input.data(), h_weights.data(),
                         analytical_dW.data(), analytical_db.data(), analytical_dX.data(),
                         batch, in_features, out_features);

    // Numerical gradient check for weights: perturb each weight by ε
    bool pass = true;
    int fail_count = 0;
    const int max_print = 5;

    for (int idx = 0; idx < weight_size; idx++) {
        // L(w + ε)
        std::vector<float> w_plus(h_weights);
        w_plus[idx] += epsilon;
        std::vector<float> out_plus(output_size);
        fc_forward_cpu(h_input.data(), w_plus.data(), h_bias.data(),
                       out_plus.data(), batch, in_features, out_features);
        float loss_plus = 0.0f;
        for (int k = 0; k < output_size; k++) loss_plus += out_plus[k];

        // L(w - ε)
        std::vector<float> w_minus(h_weights);
        w_minus[idx] -= epsilon;
        std::vector<float> out_minus(output_size);
        fc_forward_cpu(h_input.data(), w_minus.data(), h_bias.data(),
                       out_minus.data(), batch, in_features, out_features);
        float loss_minus = 0.0f;
        for (int k = 0; k < output_size; k++) loss_minus += out_minus[k];

        float numerical = (loss_plus - loss_minus) / (2.0f * epsilon);
        float analytical = analytical_dW[idx];

        // Relative error: |num - ana| / max(|num|, |ana|, 1e-8)
        float diff = fabsf(numerical - analytical);
        float denom = fmaxf(fabsf(numerical), fmaxf(fabsf(analytical), 1e-8f));
        float rel_err = diff / denom;

        if (rel_err > rel_tol) {
            if (fail_count < max_print) {
                fprintf(stderr, "  [FAIL] dW[%d]: numerical=%.6f, analytical=%.6f, rel_err=%.6f\n",
                        idx, numerical, analytical, rel_err);
            }
            fail_count++;
            pass = false;
        }
    }

    // Also check bias gradient numerically
    for (int j = 0; j < out_features; j++) {
        // L(b + ε)
        std::vector<float> b_plus(h_bias);
        b_plus[j] += epsilon;
        std::vector<float> out_plus(output_size);
        fc_forward_cpu(h_input.data(), h_weights.data(), b_plus.data(),
                       out_plus.data(), batch, in_features, out_features);
        float loss_plus = 0.0f;
        for (int k = 0; k < output_size; k++) loss_plus += out_plus[k];

        // L(b - ε)
        std::vector<float> b_minus(h_bias);
        b_minus[j] -= epsilon;
        std::vector<float> out_minus(output_size);
        fc_forward_cpu(h_input.data(), h_weights.data(), b_minus.data(),
                       out_minus.data(), batch, in_features, out_features);
        float loss_minus = 0.0f;
        for (int k = 0; k < output_size; k++) loss_minus += out_minus[k];

        float numerical = (loss_plus - loss_minus) / (2.0f * epsilon);
        float analytical = analytical_db[j];

        float diff = fabsf(numerical - analytical);
        float denom = fmaxf(fabsf(numerical), fmaxf(fabsf(analytical), 1e-8f));
        float rel_err = diff / denom;

        if (rel_err > rel_tol) {
            if (fail_count < max_print) {
                fprintf(stderr, "  [FAIL] db[%d]: numerical=%.6f, analytical=%.6f, rel_err=%.6f\n",
                        j, numerical, analytical, rel_err);
            }
            fail_count++;
            pass = false;
        }
    }

    // Also check input gradient numerically
    for (int idx = 0; idx < input_size; idx++) {
        // L(x + ε)
        std::vector<float> x_plus(h_input);
        x_plus[idx] += epsilon;
        std::vector<float> out_plus(output_size);
        fc_forward_cpu(x_plus.data(), h_weights.data(), h_bias.data(),
                       out_plus.data(), batch, in_features, out_features);
        float loss_plus = 0.0f;
        for (int k = 0; k < output_size; k++) loss_plus += out_plus[k];

        // L(x - ε)
        std::vector<float> x_minus(h_input);
        x_minus[idx] -= epsilon;
        std::vector<float> out_minus(output_size);
        fc_forward_cpu(x_minus.data(), h_weights.data(), h_bias.data(),
                       out_minus.data(), batch, in_features, out_features);
        float loss_minus = 0.0f;
        for (int k = 0; k < output_size; k++) loss_minus += out_minus[k];

        float numerical = (loss_plus - loss_minus) / (2.0f * epsilon);
        float analytical = analytical_dX[idx];

        float diff = fabsf(numerical - analytical);
        float denom = fmaxf(fabsf(numerical), fmaxf(fabsf(analytical), 1e-8f));
        float rel_err = diff / denom;

        if (rel_err > rel_tol) {
            if (fail_count < max_print) {
                fprintf(stderr, "  [FAIL] dX[%d]: numerical=%.6f, analytical=%.6f, rel_err=%.6f\n",
                        idx, numerical, analytical, rel_err);
            }
            fail_count++;
            pass = false;
        }
    }

    if (fail_count > max_print) {
        fprintf(stderr, "  ... and %d more gradient check failures\n", fail_count - max_print);
    }

    if (!pass) {
        throw std::runtime_error("Numerical gradient check failed");
    }
}

// ============================================================
// Test 3: LeNet FC1 dimensions (batch=4, 256→120)
// ============================================================

TEST_CASE("fc_backward_lenet_fc1_dims", "[puzzle_06_fc_backward]") {
    const int batch = 4, in_features = 256, out_features = 120;

    const int input_size   = batch * in_features;
    const int weight_size  = out_features * in_features;
    const int output_size  = batch * out_features;

    std::vector<float> h_input(input_size);
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_grad_output(output_size);

    fill_random(h_input.data(),       input_size,  700, -0.1f, 0.1f);
    fill_random(h_weights.data(),     weight_size, 701, -0.1f, 0.1f);
    fill_random(h_grad_output.data(), output_size, 702, -0.1f, 0.1f);

    // GPU results
    std::vector<float> gpu_dW(weight_size, 0.0f);
    std::vector<float> gpu_db(out_features, 0.0f);
    std::vector<float> gpu_dX(input_size, 0.0f);

    run_fc_backward_gpu(h_grad_output.data(), h_input.data(), h_weights.data(),
                         gpu_dW.data(), gpu_db.data(), gpu_dX.data(),
                         batch, in_features, out_features);

    // CPU reference
    std::vector<float> cpu_dW(weight_size, 0.0f);
    std::vector<float> cpu_db(out_features, 0.0f);
    std::vector<float> cpu_dX(input_size, 0.0f);

    fc_backward_weights_cpu(h_grad_output.data(), h_input.data(),
                             cpu_dW.data(), batch, in_features, out_features);
    fc_backward_bias_cpu(h_grad_output.data(),
                          cpu_db.data(), batch, out_features);
    fc_backward_input_cpu(h_grad_output.data(), h_weights.data(),
                           cpu_dX.data(), batch, in_features, out_features);

    bool pass = true;

    if (!check_array_close(gpu_dW.data(), cpu_dW.data(), weight_size, 1e-3f, 1e-3f)) {
        fprintf(stderr, "  LeNet FC1 weight gradient mismatch\n");
        pass = false;
    }

    if (!check_array_close(gpu_db.data(), cpu_db.data(), out_features, 1e-3f, 1e-3f)) {
        fprintf(stderr, "  LeNet FC1 bias gradient mismatch\n");
        pass = false;
    }

    if (!check_array_close(gpu_dX.data(), cpu_dX.data(), input_size, 1e-3f, 1e-3f)) {
        fprintf(stderr, "  LeNet FC1 input gradient mismatch\n");
        pass = false;
    }

    if (!pass) {
        throw std::runtime_error("LeNet FC1 backward (256→120) mismatch");
    }
}

// ============================================================
// Test 4: Shape verification
// ============================================================
//
// Verify that gradient shapes match expectations:
//   dW shape = (out_features × in_features) — same as W
//   db shape = (out_features)               — same as bias
//   dX shape = (batch × in_features)        — same as X
//
// We run the backward pass and verify the output arrays have
// the correct number of meaningful (non-garbage) elements by
// checking them against CPU reference at every index position.

TEST_CASE("fc_backward_shape_verification", "[puzzle_06_fc_backward]") {
    // Use non-square, non-power-of-2 dimensions to catch indexing bugs
    const int batch = 3, in_features = 5, out_features = 7;

    const int input_size   = batch * in_features;      // 3×5  = 15
    const int weight_size  = out_features * in_features; // 7×5  = 35
    const int output_size  = batch * out_features;      // 3×7  = 21

    std::vector<float> h_input(input_size);
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_grad_output(output_size);

    fill_random(h_input.data(),       input_size,  800, -1.0f, 1.0f);
    fill_random(h_weights.data(),     weight_size, 801, -1.0f, 1.0f);
    fill_random(h_grad_output.data(), output_size, 802, -1.0f, 1.0f);

    // GPU results — initialize to sentinel values to detect unwritten elements
    const float sentinel = -999.0f;
    std::vector<float> gpu_dW(weight_size, sentinel);
    std::vector<float> gpu_db(out_features, sentinel);
    std::vector<float> gpu_dX(input_size, sentinel);

    run_fc_backward_gpu(h_grad_output.data(), h_input.data(), h_weights.data(),
                         gpu_dW.data(), gpu_db.data(), gpu_dX.data(),
                         batch, in_features, out_features);

    // CPU reference
    std::vector<float> cpu_dW(weight_size, 0.0f);
    std::vector<float> cpu_db(out_features, 0.0f);
    std::vector<float> cpu_dX(input_size, 0.0f);

    fc_backward_weights_cpu(h_grad_output.data(), h_input.data(),
                             cpu_dW.data(), batch, in_features, out_features);
    fc_backward_bias_cpu(h_grad_output.data(),
                          cpu_db.data(), batch, out_features);
    fc_backward_input_cpu(h_grad_output.data(), h_weights.data(),
                           cpu_dX.data(), batch, in_features, out_features);

    bool pass = true;

    // Verify dW: exactly out_features × in_features elements written correctly
    printf("  Checking dW shape: (%d × %d) = %d elements\n",
           out_features, in_features, weight_size);
    if (!check_array_close(gpu_dW.data(), cpu_dW.data(), weight_size, 1e-4f, 1e-4f)) {
        fprintf(stderr, "  dW shape/values mismatch\n");
        pass = false;
    }

    // Verify db: exactly out_features elements written correctly
    printf("  Checking db shape: (%d) = %d elements\n",
           out_features, out_features);
    if (!check_array_close(gpu_db.data(), cpu_db.data(), out_features, 1e-4f, 1e-4f)) {
        fprintf(stderr, "  db shape/values mismatch\n");
        pass = false;
    }

    // Verify dX: exactly batch × in_features elements written correctly
    printf("  Checking dX shape: (%d × %d) = %d elements\n",
           batch, in_features, input_size);
    if (!check_array_close(gpu_dX.data(), cpu_dX.data(), input_size, 1e-4f, 1e-4f)) {
        fprintf(stderr, "  dX shape/values mismatch\n");
        pass = false;
    }

    if (!pass) {
        throw std::runtime_error("Shape verification failed");
    }
}

