// Puzzle 10: Conv2D Backward Pass — Input Gradient — Test Harness
//
// Tests:
//   1. Tiny example: 5×5 input, 3×3 filter — same setup as Puzzle 07, hand-verifiable
//   2. Numerical gradient check — perturb each input element by ε, compare to analytical
//   3. LeNet Conv1: grad_input shape 28×28×1 (from 24×24×6 grad_output)
//   4. LeNet Conv2: grad_input shape 12×12×6 (from 8×8×16 grad_output)

#include "cuda_utils.h"
#include "test_utils.h"

// Include the kernel implementation
// Build system compiles this file twice:
//   puzzle_10_test          -> includes puzzle.cu
//   puzzle_10_test_solution -> includes solution.cu
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// ============================================================
// CPU reference: conv2d forward (needed for numerical gradient check)
// ============================================================

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

// ============================================================
// CPU reference: conv2d backward input
// ============================================================

// ∂L/∂input[b][c][h][w] = Σ_k Σ_fh Σ_fw grad_output[b][k][h-fh][w-fw] × W[k][c][fh][fw]
void conv2d_backward_input_cpu(const float* grad_output, const float* filters,
                                float* grad_input,
                                int batch, int C_in, int H, int W,
                                int C_out, int F) {
    int H_out = H - F + 1;
    int W_out = W - F + 1;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < C_in; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float sum = 0.0f;
                    for (int k = 0; k < C_out; k++) {
                        for (int fh = 0; fh < F; fh++) {
                            for (int fw = 0; fw < F; fw++) {
                                int oh = h - fh;
                                int ow = w - fw;
                                if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                                    int go_idx = b * (C_out * H_out * W_out)
                                               + k * (H_out * W_out)
                                               + oh * W_out + ow;
                                    int f_idx  = k * (C_in * F * F)
                                               + c * (F * F)
                                               + fh * F + fw;
                                    sum += grad_output[go_idx] * filters[f_idx];
                                }
                            }
                        }
                    }
                    grad_input[b * (C_in * H * W) + c * (H * W) + h * W + w] = sum;
                }
            }
        }
    }
}

// ============================================================
// GPU helper: run conv2d_backward_input kernel
// ============================================================

void run_conv2d_backward_input_gpu(const float* h_grad_output,
                                    const float* h_filters,
                                    float* h_grad_input,
                                    int batch, int C_in, int H, int W,
                                    int C_out, int F) {
    int H_out = H - F + 1;
    int W_out = W - F + 1;

    size_t go_bytes = (size_t)batch * C_out * H_out * W_out * sizeof(float);
    size_t f_bytes  = (size_t)C_out * C_in * F * F * sizeof(float);
    size_t gi_bytes = (size_t)batch * C_in * H * W * sizeof(float);

    float *d_grad_output, *d_filters, *d_grad_input;

    CUDA_CHECK(cudaMalloc(&d_grad_output, go_bytes));
    CUDA_CHECK(cudaMalloc(&d_filters,     f_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_input,  gi_bytes));

    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, go_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filters,     h_filters,     f_bytes,  cudaMemcpyHostToDevice));

    int total = batch * C_in * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_backward_input<<<blocks, threads>>>(d_grad_output, d_filters, d_grad_input,
                                                batch, C_in, H, W, C_out, F);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_grad_input, d_grad_input, gi_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_filters));
    CUDA_CHECK(cudaFree(d_grad_input));
}

// ============================================================
// Test 1: Tiny example — 5×5 input, 1→1 channel, 3×3 filter
// ============================================================
//
// Same setup as Puzzle 07's tiny test:
//   Input: 1×1×5×5, values 1..25
//   Filter: 1×1×3×3, all ones
//   Bias: 0.0
//
// Forward output (1×1×3×3):
//   [63, 72, 81; 108, 117, 126; 153, 162, 171]
//
// For the backward pass, we use grad_output = all ones (1×1×3×3).
// This means: grad_input[h][w] = Σ_{fh,fw} W[fh][fw]  (for valid oh,ow)
//   = count of output positions that used input[h][w]
//
// Since W is all ones, grad_input[h][w] = number of output positions
// whose receptive field includes (h,w).
//
// Expected grad_input (5×5):
//   [1, 2, 3, 2, 1]     ← corners: 1 output uses them
//   [2, 4, 6, 4, 2]     ← edges: 2 or 4 outputs
//   [3, 6, 9, 6, 3]     ← center row: 3, 6, or 9 outputs
//   [2, 4, 6, 4, 2]
//   [1, 2, 3, 2, 1]

TEST_CASE(conv_backward_input_tiny) {
    const int batch = 1, C_in = 1, H = 5, W = 5, C_out = 1, F = 3;

    // Filter: all ones
    float h_filters[9];
    for (int i = 0; i < 9; i++) h_filters[i] = 1.0f;

    // grad_output: all ones (3×3)
    float h_grad_output[9];
    for (int i = 0; i < 9; i++) h_grad_output[i] = 1.0f;

    float expected[] = {
        1.0f, 2.0f, 3.0f, 2.0f, 1.0f,
        2.0f, 4.0f, 6.0f, 4.0f, 2.0f,
        3.0f, 6.0f, 9.0f, 6.0f, 3.0f,
        2.0f, 4.0f, 6.0f, 4.0f, 2.0f,
        1.0f, 2.0f, 3.0f, 2.0f, 1.0f
    };

    float h_grad_input[25] = {0};
    run_conv2d_backward_input_gpu(h_grad_output, h_filters, h_grad_input,
                                   batch, C_in, H, W, C_out, F);

    if (!check_array_close(h_grad_input, expected, 25, 1e-4f, 1e-4f)) {
        throw std::runtime_error("Tiny conv backward input (5x5, 3x3 filter) mismatch");
    }
}

// ============================================================
// Test 2: Numerical gradient check
// ============================================================
//
// For each input element input[idx], compute:
//   numerical_grad = (L(input+ε) - L(input-ε)) / (2ε)
// where L = Σ output (sum loss), so grad_output = all ones.
//
// Compare to analytical grad_input[idx] with relative error < 1e-2.
// (float32 finite-difference has limited precision due to accumulation noise)

TEST_CASE(conv_backward_input_numerical_gradient_check) {
    const int batch = 1, C_in = 1, H = 5, W = 5, C_out = 1, F = 3;
    const float epsilon = 1e-3f;
    const float rel_tol = 1e-2f;

    const int H_out = H - F + 1;  // 4
    const int W_out = W - F + 1;  // 4

    const int input_size  = batch * C_in * H * W;       // 72
    const int filter_size = C_out * C_in * F * F;       // 36
    const int output_size = batch * C_out * H_out * W_out; // 32

    std::vector<float> h_input(input_size);
    std::vector<float> h_filters(filter_size);
    std::vector<float> h_bias(C_out, 0.0f);

    fill_random(h_input.data(),   input_size,  1000, -1.0f, 1.0f);
    fill_random(h_filters.data(), filter_size, 1001, -1.0f, 1.0f);

    // Use sum loss: L = Σ output, so grad_output = all ones
    std::vector<float> h_grad_output(output_size, 1.0f);

    // Compute analytical gradient via GPU kernel
    std::vector<float> analytical_gi(input_size, 0.0f);
    run_conv2d_backward_input_gpu(h_grad_output.data(), h_filters.data(),
                                   analytical_gi.data(),
                                   batch, C_in, H, W, C_out, F);

    // Numerical gradient check: perturb each input element
    bool pass = true;
    int fail_count = 0;
    const int max_print = 5;

    for (int idx = 0; idx < input_size; idx++) {
        // L(input + ε)
        std::vector<float> in_plus(h_input);
        in_plus[idx] += epsilon;
        std::vector<float> out_plus(output_size);
        conv2d_forward_cpu(in_plus.data(), h_filters.data(), h_bias.data(),
                           out_plus.data(), batch, C_in, H, W, C_out, F);
        float loss_plus = 0.0f;
        for (int k = 0; k < output_size; k++) loss_plus += out_plus[k];

        // L(input - ε)
        std::vector<float> in_minus(h_input);
        in_minus[idx] -= epsilon;
        std::vector<float> out_minus(output_size);
        conv2d_forward_cpu(in_minus.data(), h_filters.data(), h_bias.data(),
                           out_minus.data(), batch, C_in, H, W, C_out, F);
        float loss_minus = 0.0f;
        for (int k = 0; k < output_size; k++) loss_minus += out_minus[k];

        float numerical = (loss_plus - loss_minus) / (2.0f * epsilon);
        float analytical = analytical_gi[idx];

        // Relative error: |num - ana| / max(|num|, |ana|, 1e-8)
        float diff = fabsf(numerical - analytical);
        float denom = fmaxf(fabsf(numerical), fmaxf(fabsf(analytical), 1e-8f));
        float rel_err = diff / denom;

        if (rel_err > rel_tol) {
            if (fail_count < max_print) {
                fprintf(stderr, "  [FAIL] grad_input[%d]: numerical=%.6f, analytical=%.6f, rel_err=%.6f\n",
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
// Test 3: LeNet Conv1 — grad_input shape 28×28×1
// ============================================================
//
// Conv1 forward: input 1×28×28, filter (6,1,5,5) → output 6×24×24
// Conv1 backward: grad_output 6×24×24 → grad_input 1×28×28

TEST_CASE(conv_backward_input_lenet_conv1) {
    const int batch = 1, C_in = 1, H = 28, W = 28, C_out = 6, F = 5;
    const int H_out = H - F + 1;  // 24
    const int W_out = W - F + 1;  // 24

    const int go_size = batch * C_out * H_out * W_out;  // 3456
    const int f_size  = C_out * C_in * F * F;           // 150
    const int gi_size = batch * C_in * H * W;           // 784

    std::vector<float> h_grad_output(go_size);
    std::vector<float> h_filters(f_size);
    std::vector<float> gpu_gi(gi_size, 0.0f);
    std::vector<float> cpu_gi(gi_size, 0.0f);

    fill_random(h_grad_output.data(), go_size, 1010, -0.5f, 0.5f);
    fill_random(h_filters.data(),     f_size,  1011, -0.2f, 0.2f);

    conv2d_backward_input_cpu(h_grad_output.data(), h_filters.data(),
                               cpu_gi.data(), batch, C_in, H, W, C_out, F);

    run_conv2d_backward_input_gpu(h_grad_output.data(), h_filters.data(),
                                   gpu_gi.data(), batch, C_in, H, W, C_out, F);

    if (!check_array_close(gpu_gi.data(), cpu_gi.data(), gi_size, 1e-3f, 1e-3f)) {
        throw std::runtime_error("LeNet Conv1 backward input (28x28x1) mismatch");
    }
}

// ============================================================
// Test 4: LeNet Conv2 — grad_input shape 12×12×6
// ============================================================
//
// Conv2 forward: input 6×12×12, filter (16,6,5,5) → output 16×8×8
// Conv2 backward: grad_output 16×8×8 → grad_input 6×12×12

TEST_CASE(conv_backward_input_lenet_conv2) {
    const int batch = 1, C_in = 6, H = 12, W = 12, C_out = 16, F = 5;
    const int H_out = H - F + 1;  // 8
    const int W_out = W - F + 1;  // 8

    const int go_size = batch * C_out * H_out * W_out;  // 1024
    const int f_size  = C_out * C_in * F * F;           // 2400
    const int gi_size = batch * C_in * H * W;           // 864

    std::vector<float> h_grad_output(go_size);
    std::vector<float> h_filters(f_size);
    std::vector<float> gpu_gi(gi_size, 0.0f);
    std::vector<float> cpu_gi(gi_size, 0.0f);

    fill_random(h_grad_output.data(), go_size, 1020, -0.5f, 0.5f);
    fill_random(h_filters.data(),     f_size,  1021, -0.1f, 0.1f);

    conv2d_backward_input_cpu(h_grad_output.data(), h_filters.data(),
                               cpu_gi.data(), batch, C_in, H, W, C_out, F);

    run_conv2d_backward_input_gpu(h_grad_output.data(), h_filters.data(),
                                   gpu_gi.data(), batch, C_in, H, W, C_out, F);

    if (!check_array_close(gpu_gi.data(), cpu_gi.data(), gi_size, 1e-3f, 1e-3f)) {
        throw std::runtime_error("LeNet Conv2 backward input (12x12x6) mismatch");
    }
}

int main() {
    return RUN_ALL_TESTS();
}
