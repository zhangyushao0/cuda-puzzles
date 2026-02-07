#include <catch2/catch_test_macros.hpp>
// Puzzle 09: Conv2D Backward — Weight & Bias Gradients — Test Harness
//
// Tests:
//   1. Tiny 3×3 filter on 5×5 input — hardcoded, hand-verifiable
//   2. Numerical gradient check — perturb each weight/bias by ε, compare analytical
//   3. LeNet Conv1 dims: batch=1, 1×28×28 input, 6 filters of 5×5
//   4. LeNet Conv2 dims: batch=1, 6×12×12 input, 16 filters of 5×5

#include "cuda_utils.h"
#include "test_utils.h"

// Include the kernel implementation
// Build system compiles this file twice:
//   puzzle_09_test          -> includes puzzle.cu
//   puzzle_09_test_solution -> includes solution.cu
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// ============================================================
// CPU reference implementations
// ============================================================

// Conv2D forward pass on CPU (needed for numerical gradient check)
void conv2d_forward_cpu(const float* input, const float* filters, const float* bias,
                        float* output, int batch, int C_in, int H, int W,
                        int C_out, int F) {
    int H_out = H - F + 1;
    int W_out = W - F + 1;

    for (int b = 0; b < batch; b++) {
        for (int k = 0; k < C_out; k++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float sum = bias[k];
                    for (int c = 0; c < C_in; c++) {
                        for (int fh = 0; fh < F; fh++) {
                            for (int fw = 0; fw < F; fw++) {
                                int in_idx = b * (C_in * H * W) + c * (H * W)
                                           + (oh + fh) * W + (ow + fw);
                                int f_idx  = k * (C_in * F * F) + c * (F * F)
                                           + fh * F + fw;
                                sum += input[in_idx] * filters[f_idx];
                            }
                        }
                    }
                    output[b * (C_out * H_out * W_out) + k * (H_out * W_out)
                         + oh * W_out + ow] = sum;
                }
            }
        }
    }
}

// Weight gradient: ∂L/∂W[k][c][fh][fw] = Σ_b Σ_h Σ_w grad_out[b][k][h][w] × input[b][c][h+fh][w+fw]
void conv2d_backward_weights_cpu(const float* grad_output, const float* input,
                                  float* grad_weights,
                                  int batch, int C_in, int H, int W,
                                  int C_out, int F) {
    int H_out = H - F + 1;
    int W_out = W - F + 1;

    for (int k = 0; k < C_out; k++) {
        for (int c = 0; c < C_in; c++) {
            for (int fh = 0; fh < F; fh++) {
                for (int fw = 0; fw < F; fw++) {
                    float sum = 0.0f;
                    for (int b = 0; b < batch; b++) {
                        for (int h = 0; h < H_out; h++) {
                            for (int w = 0; w < W_out; w++) {
                                int go_idx = b * (C_out * H_out * W_out) + k * (H_out * W_out)
                                           + h * W_out + w;
                                int in_idx = b * (C_in * H * W) + c * (H * W)
                                           + (h + fh) * W + (w + fw);
                                sum += grad_output[go_idx] * input[in_idx];
                            }
                        }
                    }
                    grad_weights[k * (C_in * F * F) + c * (F * F) + fh * F + fw] = sum;
                }
            }
        }
    }
}

// Bias gradient: ∂L/∂bias[k] = Σ_b Σ_h Σ_w grad_out[b][k][h][w]
void conv2d_backward_bias_cpu(const float* grad_output, float* grad_bias,
                               int batch, int C_out, int H_out, int W_out) {
    for (int k = 0; k < C_out; k++) {
        float sum = 0.0f;
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    sum += grad_output[b * (C_out * H_out * W_out) + k * (H_out * W_out)
                                     + h * W_out + w];
                }
            }
        }
        grad_bias[k] = sum;
    }
}

// ============================================================
// GPU helper: run backward weight/bias kernels
// ============================================================

void run_conv2d_backward_gpu(const float* h_grad_output, const float* h_input,
                              float* h_grad_weights, float* h_grad_bias,
                              int batch, int C_in, int H, int W,
                              int C_out, int F) {
    int H_out = H - F + 1;
    int W_out = W - F + 1;

    size_t go_bytes = (size_t)batch * C_out * H_out * W_out * sizeof(float);
    size_t in_bytes = (size_t)batch * C_in * H * W * sizeof(float);
    size_t gw_bytes = (size_t)C_out * C_in * F * F * sizeof(float);
    size_t gb_bytes = (size_t)C_out * sizeof(float);

    float *d_grad_output, *d_input, *d_grad_weights, *d_grad_bias;

    CUDA_CHECK(cudaMalloc(&d_grad_output,  go_bytes));
    CUDA_CHECK(cudaMalloc(&d_input,        in_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_weights, gw_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_bias,    gb_bytes));

    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, go_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input,       h_input,       in_bytes, cudaMemcpyHostToDevice));

    // Kernel 1: Weight gradient
    {
        int total_weights = C_out * C_in * F * F;
        int threads = 256;
        int blocks = (total_weights + threads - 1) / threads;
        conv2d_backward_weights<<<blocks, threads>>>(d_grad_output, d_input, d_grad_weights,
                                                      batch, C_in, H, W, C_out, F);
        KERNEL_CHECK();
    }

    // Kernel 2: Bias gradient
    {
        int threads = 256;
        int blocks = (C_out + threads - 1) / threads;
        conv2d_backward_bias<<<blocks, threads>>>(d_grad_output, d_grad_bias,
                                                   batch, C_out, H_out, W_out);
        KERNEL_CHECK();
    }

    CUDA_CHECK(cudaMemcpy(h_grad_weights, d_grad_weights, gw_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_bias,    d_grad_bias,    gb_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_grad_weights));
    CUDA_CHECK(cudaFree(d_grad_bias));
}

// ============================================================
// Test 1: Tiny 3×3 filter on 5×5 input — hardcoded, hand-verifiable
// ============================================================
//
// Setup:
//   Input (1×1×5×5):  1..25 filled row-major
//   Filter (1×1×3×3): all 0.1
//   Bias: [0.0]
//   grad_output (1×1×3×3): all 1.0
//
// Forward output (3×3): each element = 0.1 × sum_of_3x3_patch
//   (not needed directly — we're testing backward)
//
// Weight gradient ∂L/∂W[0][0][fh][fw]:
//   = Σ_{h,w} grad_out[0][0][h][w] × input[0][0][h+fh][w+fw]
//   Since grad_out = all 1.0:
//   ∂L/∂W[0][0][fh][fw] = Σ_{h=0}^{2} Σ_{w=0}^{2} input[h+fh][w+fw]
//
// For fh=0, fw=0: sum of input[0:3, 0:3] = 1+2+3+6+7+8+11+12+13 = 63
// For fh=0, fw=1: sum of input[0:3, 1:4] = 2+3+4+7+8+9+12+13+14 = 72
// For fh=0, fw=2: sum of input[0:3, 2:5] = 3+4+5+8+9+10+13+14+15 = 81
// For fh=1, fw=0: sum of input[1:4, 0:3] = 6+7+8+11+12+13+16+17+18 = 108
// For fh=1, fw=1: sum of input[1:4, 1:4] = 7+8+9+12+13+14+17+18+19 = 117
// For fh=1, fw=2: sum of input[1:4, 2:5] = 8+9+10+13+14+15+18+19+20 = 126
// For fh=2, fw=0: sum of input[2:5, 0:3] = 11+12+13+16+17+18+21+22+23 = 153
// For fh=2, fw=1: sum of input[2:5, 1:4] = 12+13+14+17+18+19+22+23+24 = 162
// For fh=2, fw=2: sum of input[2:5, 2:5] = 13+14+15+18+19+20+23+24+25 = 171
//
// Bias gradient: ∂L/∂bias[0] = sum of grad_out = 9 × 1.0 = 9.0

TEST_CASE("conv_bw_tiny_3x3", "[puzzle_09_conv_backward_weights]") {
    const int batch = 1, C_in = 1, H = 5, W = 5, C_out = 1, F = 3;
    const int H_out = H - F + 1;  // 3
    const int W_out = W - F + 1;  // 3

    // Input: 1..25
    float h_input[25];
    for (int i = 0; i < 25; i++) h_input[i] = (float)(i + 1);

    // grad_output: all 1.0 (3×3)
    float h_grad_output[9];
    for (int i = 0; i < 9; i++) h_grad_output[i] = 1.0f;

    // Expected weight gradient (hand-computed above)
    float expected_dW[] = { 63.0f,  72.0f,  81.0f,
                           108.0f, 117.0f, 126.0f,
                           153.0f, 162.0f, 171.0f};

    // Expected bias gradient
    float expected_db[] = {9.0f};

    float h_grad_weights[9] = {0};
    float h_grad_bias[1] = {0};

    run_conv2d_backward_gpu(h_grad_output, h_input,
                             h_grad_weights, h_grad_bias,
                             batch, C_in, H, W, C_out, F);

    bool pass = true;

    if (!check_array_close(h_grad_weights, expected_dW, C_out * C_in * F * F, 1e-4f, 1e-4f)) {
        fprintf(stderr, "  Weight gradient mismatch\n");
        pass = false;
    }

    if (!check_array_close(h_grad_bias, expected_db, C_out, 1e-4f, 1e-4f)) {
        fprintf(stderr, "  Bias gradient mismatch\n");
        pass = false;
    }

    if (!pass) {
        throw std::runtime_error("Tiny 3x3 conv backward mismatch");
    }
}

// ============================================================
// Test 2: Numerical gradient check
// ============================================================
//
// For each weight w[k][c][fh][fw], compute:
//   numerical_grad = (L(w+ε) - L(w-ε)) / (2ε)
// where L is a simple sum-of-outputs loss (L = Σ output).
//
// Since ∂L/∂output = 1 everywhere (sum loss), grad_output = all 1s.
// Compare numerical to analytical with relative error < 1e-3.

TEST_CASE("conv_bw_numerical_gradient_check", "[puzzle_09_conv_backward_weights]") {
    // Use small dimensions to keep numerical error manageable
    const int batch = 1, C_in = 1, H = 5, W = 5, C_out = 1, F = 3;
    const int H_out = H - F + 1;  // 3
    const int W_out = W - F + 1;  // 3
    const float epsilon = 1e-3f;
    const float rel_tol = 1e-3f;

    const int input_size  = batch * C_in * H * W;
    const int weight_size = C_out * C_in * F * F;
    const int output_size = batch * C_out * H_out * W_out;

    std::vector<float> h_input(input_size);
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_bias(C_out);

    fill_random(h_input.data(),   input_size,  900, -1.0f, 1.0f);
    fill_random(h_weights.data(), weight_size, 901, -1.0f, 1.0f);
    fill_random(h_bias.data(),    C_out,       902, -0.5f, 0.5f);

    // grad_output = all 1s (sum loss: L = Σ output)
    std::vector<float> h_grad_output(output_size, 1.0f);

    // Compute analytical gradients via GPU kernels
    std::vector<float> analytical_dW(weight_size, 0.0f);
    std::vector<float> analytical_db(C_out, 0.0f);

    run_conv2d_backward_gpu(h_grad_output.data(), h_input.data(),
                             analytical_dW.data(), analytical_db.data(),
                             batch, C_in, H, W, C_out, F);

    // Numerical gradient check for weights
    // Use double-precision accumulation for the loss to avoid cancellation errors
    bool pass = true;
    int fail_count = 0;
    const int max_print = 5;

    for (int idx = 0; idx < weight_size; idx++) {
        // L(w + ε)
        std::vector<float> w_plus(h_weights);
        w_plus[idx] += epsilon;
        std::vector<float> out_plus(output_size);
        conv2d_forward_cpu(h_input.data(), w_plus.data(), h_bias.data(),
                           out_plus.data(), batch, C_in, H, W, C_out, F);
        double loss_plus = 0.0;
        for (int i = 0; i < output_size; i++) loss_plus += (double)out_plus[i];

        // L(w - ε)
        std::vector<float> w_minus(h_weights);
        w_minus[idx] -= epsilon;
        std::vector<float> out_minus(output_size);
        conv2d_forward_cpu(h_input.data(), w_minus.data(), h_bias.data(),
                           out_minus.data(), batch, C_in, H, W, C_out, F);
        double loss_minus = 0.0;
        for (int i = 0; i < output_size; i++) loss_minus += (double)out_minus[i];

        float numerical = (float)((loss_plus - loss_minus) / (2.0 * epsilon));
        float analytical = analytical_dW[idx];

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

    // Numerical gradient check for biases
    for (int k = 0; k < C_out; k++) {
        // L(b + ε)
        std::vector<float> b_plus(h_bias);
        b_plus[k] += epsilon;
        std::vector<float> out_plus(output_size);
        conv2d_forward_cpu(h_input.data(), h_weights.data(), b_plus.data(),
                           out_plus.data(), batch, C_in, H, W, C_out, F);
        double loss_plus = 0.0;
        for (int i = 0; i < output_size; i++) loss_plus += (double)out_plus[i];

        // L(b - ε)
        std::vector<float> b_minus(h_bias);
        b_minus[k] -= epsilon;
        std::vector<float> out_minus(output_size);
        conv2d_forward_cpu(h_input.data(), h_weights.data(), b_minus.data(),
                           out_minus.data(), batch, C_in, H, W, C_out, F);
        double loss_minus = 0.0;
        for (int i = 0; i < output_size; i++) loss_minus += (double)out_minus[i];

        float numerical = (float)((loss_plus - loss_minus) / (2.0 * epsilon));
        float analytical = analytical_db[k];

        float diff = fabsf(numerical - analytical);
        float denom = fmaxf(fabsf(numerical), fmaxf(fabsf(analytical), 1e-8f));
        float rel_err = diff / denom;

        if (rel_err > rel_tol) {
            if (fail_count < max_print) {
                fprintf(stderr, "  [FAIL] db[%d]: numerical=%.6f, analytical=%.6f, rel_err=%.6f\n",
                        k, numerical, analytical, rel_err);
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
// Test 3: LeNet Conv1 dims — batch=1, 1×28×28 → 6×5×5 weight gradient
// ============================================================

TEST_CASE("conv_bw_lenet_conv1", "[puzzle_09_conv_backward_weights]") {
    const int batch = 1, C_in = 1, H = 28, W = 28, C_out = 6, F = 5;
    const int H_out = H - F + 1;  // 24
    const int W_out = W - F + 1;  // 24

    const int input_size  = batch * C_in * H * W;
    const int go_size     = batch * C_out * H_out * W_out;
    const int weight_size = C_out * C_in * F * F;

    std::vector<float> h_input(input_size);
    std::vector<float> h_grad_output(go_size);

    fill_random(h_input.data(),       input_size, 910, -0.5f, 0.5f);
    fill_random(h_grad_output.data(), go_size,    911, -0.5f, 0.5f);

    // GPU results
    std::vector<float> gpu_dW(weight_size, 0.0f);
    std::vector<float> gpu_db(C_out, 0.0f);

    run_conv2d_backward_gpu(h_grad_output.data(), h_input.data(),
                             gpu_dW.data(), gpu_db.data(),
                             batch, C_in, H, W, C_out, F);

    // CPU reference
    std::vector<float> cpu_dW(weight_size, 0.0f);
    std::vector<float> cpu_db(C_out, 0.0f);

    conv2d_backward_weights_cpu(h_grad_output.data(), h_input.data(),
                                 cpu_dW.data(), batch, C_in, H, W, C_out, F);
    conv2d_backward_bias_cpu(h_grad_output.data(),
                              cpu_db.data(), batch, C_out, H_out, W_out);

    bool pass = true;

    printf("  Checking dW shape: (%d × %d × %d × %d) = %d elements\n",
           C_out, C_in, F, F, weight_size);
    if (!check_array_close(gpu_dW.data(), cpu_dW.data(), weight_size, 1e-3f, 1e-3f)) {
        fprintf(stderr, "  LeNet Conv1 weight gradient mismatch\n");
        pass = false;
    }

    printf("  Checking db shape: (%d) = %d elements\n", C_out, C_out);
    if (!check_array_close(gpu_db.data(), cpu_db.data(), C_out, 1e-3f, 1e-3f)) {
        fprintf(stderr, "  LeNet Conv1 bias gradient mismatch\n");
        pass = false;
    }

    if (!pass) {
        throw std::runtime_error("LeNet Conv1 backward (1×28×28 → 6×5×5) mismatch");
    }
}

// ============================================================
// Test 4: LeNet Conv2 dims — batch=1, 6×12×12 → 16×5×5 weight gradient
// ============================================================

TEST_CASE("conv_bw_lenet_conv2", "[puzzle_09_conv_backward_weights]") {
    const int batch = 1, C_in = 6, H = 12, W = 12, C_out = 16, F = 5;
    const int H_out = H - F + 1;  // 8
    const int W_out = W - F + 1;  // 8

    const int input_size  = batch * C_in * H * W;
    const int go_size     = batch * C_out * H_out * W_out;
    const int weight_size = C_out * C_in * F * F;

    std::vector<float> h_input(input_size);
    std::vector<float> h_grad_output(go_size);

    fill_random(h_input.data(),       input_size, 920, -0.5f, 0.5f);
    fill_random(h_grad_output.data(), go_size,    921, -0.5f, 0.5f);

    // GPU results
    std::vector<float> gpu_dW(weight_size, 0.0f);
    std::vector<float> gpu_db(C_out, 0.0f);

    run_conv2d_backward_gpu(h_grad_output.data(), h_input.data(),
                             gpu_dW.data(), gpu_db.data(),
                             batch, C_in, H, W, C_out, F);

    // CPU reference
    std::vector<float> cpu_dW(weight_size, 0.0f);
    std::vector<float> cpu_db(C_out, 0.0f);

    conv2d_backward_weights_cpu(h_grad_output.data(), h_input.data(),
                                 cpu_dW.data(), batch, C_in, H, W, C_out, F);
    conv2d_backward_bias_cpu(h_grad_output.data(),
                              cpu_db.data(), batch, C_out, H_out, W_out);

    bool pass = true;

    printf("  Checking dW shape: (%d × %d × %d × %d) = %d elements\n",
           C_out, C_in, F, F, weight_size);
    if (!check_array_close(gpu_dW.data(), cpu_dW.data(), weight_size, 1e-3f, 1e-3f)) {
        fprintf(stderr, "  LeNet Conv2 weight gradient mismatch\n");
        pass = false;
    }

    printf("  Checking db shape: (%d) = %d elements\n", C_out, C_out);
    if (!check_array_close(gpu_db.data(), cpu_db.data(), C_out, 1e-3f, 1e-3f)) {
        fprintf(stderr, "  LeNet Conv2 bias gradient mismatch\n");
        pass = false;
    }

    if (!pass) {
        throw std::runtime_error("LeNet Conv2 backward (6×12×12 → 16×5×5) mismatch");
    }
}

