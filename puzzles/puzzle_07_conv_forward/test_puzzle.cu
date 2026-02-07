#include <catch2/catch_test_macros.hpp>
// Puzzle 07: 2D Convolution (Forward Pass) — Test Harness
//
// Tests:
//   1. Single 3×3 filter on 5×5 input — hardcoded, hand-verifiable
//   2. LeNet Conv1: 28×28×1 → 24×24×6 (filter 5×5)
//   3. LeNet Conv2: 12×12×6 → 8×8×16 (filter 5×5)
//   4. Batch processing: batch=8 through Conv1
//   5. Dimension check: verifies output H/W match formula

#include "cuda_utils.h"
#include "test_utils.h"

// Include the kernel implementation
// Build system compiles this file twice:
//   puzzle_07_test          -> includes puzzle.cu
//   puzzle_07_test_solution -> includes solution.cu
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// CPU reference implementation for verification
// out[b][k][oh][ow] = bias[k] + Σ_c Σ_fh Σ_fw input[b][c][oh+fh][ow+fw] * filter[k][c][fh][fw]
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

// Helper: run conv2d_forward kernel on GPU and copy result back to host
void run_conv2d_forward_gpu(const float* h_input, const float* h_filters,
                            const float* h_bias, float* h_output,
                            int batch, int C_in, int H, int W,
                            int C_out, int F) {
    int H_out = H - F + 1;
    int W_out = W - F + 1;

    size_t input_bytes  = (size_t)batch * C_in * H * W * sizeof(float);
    size_t filter_bytes = (size_t)C_out * C_in * F * F * sizeof(float);
    size_t bias_bytes   = (size_t)C_out * sizeof(float);
    size_t output_bytes = (size_t)batch * C_out * H_out * W_out * sizeof(float);

    float *d_input, *d_filters, *d_bias, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input,   input_bytes));
    CUDA_CHECK(cudaMalloc(&d_filters, filter_bytes));
    CUDA_CHECK(cudaMalloc(&d_bias,    bias_bytes));
    CUDA_CHECK(cudaMalloc(&d_output,  output_bytes));

    CUDA_CHECK(cudaMemcpy(d_input,   h_input,   input_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filters, h_filters, filter_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias,    h_bias,    bias_bytes,   cudaMemcpyHostToDevice));

    int total = batch * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_forward<<<blocks, threads>>>(d_input, d_filters, d_bias, d_output,
                                         batch, C_in, H, W, C_out, F);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filters));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
}

// Test 1: Single 3×3 filter on 5×5 input — hardcoded, hand-verifiable
//
// Input (1×1×5×5):  1..25 filled row-major
// Filter (1×1×3×3): all ones
// Bias: 0.0
//
// Each output element = sum of a 3×3 patch in the input.
// Output (1×1×3×3):
//   out[0][0] = 1+2+3+6+7+8+11+12+13 = 63
//   out[0][1] = 2+3+4+7+8+9+12+13+14 = 72
//   out[0][2] = 3+4+5+8+9+10+13+14+15 = 81
//   out[1][0] = 6+7+8+11+12+13+16+17+18 = 108
//   out[1][1] = 7+8+9+12+13+14+17+18+19 = 117
//   out[1][2] = 8+9+10+13+14+15+18+19+20 = 126
//   out[2][0] = 11+12+13+16+17+18+21+22+23 = 153
//   out[2][1] = 12+13+14+17+18+19+22+23+24 = 162
//   out[2][2] = 13+14+15+18+19+20+23+24+25 = 171
TEST_CASE("conv_single_3x3", "[puzzle_07_conv_forward]") {
    const int batch = 1, C_in = 1, H = 5, W = 5, C_out = 1, F = 3;
    const int H_out = H - F + 1;  // 3
    const int W_out = W - F + 1;  // 3

    // Input: 1..25
    float h_input[25];
    for (int i = 0; i < 25; i++) h_input[i] = (float)(i + 1);

    // Filter: all ones
    float h_filters[9];
    for (int i = 0; i < 9; i++) h_filters[i] = 1.0f;

    float h_bias[] = {0.0f};

    float expected[] = {63.0f, 72.0f, 81.0f,
                        108.0f, 117.0f, 126.0f,
                        153.0f, 162.0f, 171.0f};

    float h_output[9] = {0};
    run_conv2d_forward_gpu(h_input, h_filters, h_bias, h_output,
                           batch, C_in, H, W, C_out, F);

    REQUIRE(check_array_close(h_output, expected, H_out * W_out, 1e-4f, 1e-4f));
}

// Test 2: LeNet Conv1 — 28×28×1 → 24×24×6, filter 5×5
// Conv1 takes the single-channel 28×28 MNIST image and produces 6 feature maps
TEST_CASE("conv_lenet_conv1", "[puzzle_07_conv_forward]") {
    const int batch = 1, C_in = 1, H = 28, W = 28, C_out = 6, F = 5;
    const int H_out = H - F + 1;  // 24
    const int W_out = W - F + 1;  // 24

    const int input_size  = batch * C_in * H * W;
    const int filter_size = C_out * C_in * F * F;
    const int output_size = batch * C_out * H_out * W_out;

    std::vector<float> h_input(input_size);
    std::vector<float> h_filters(filter_size);
    std::vector<float> h_bias(C_out);
    std::vector<float> h_output(output_size, 0.0f);
    std::vector<float> expected(output_size, 0.0f);

    fill_random(h_input.data(),   input_size,  700, -0.5f, 0.5f);
    fill_random(h_filters.data(), filter_size, 701, -0.2f, 0.2f);
    fill_random(h_bias.data(),    C_out,       702, -0.1f, 0.1f);

    conv2d_forward_cpu(h_input.data(), h_filters.data(), h_bias.data(),
                       expected.data(), batch, C_in, H, W, C_out, F);

    run_conv2d_forward_gpu(h_input.data(), h_filters.data(), h_bias.data(),
                           h_output.data(), batch, C_in, H, W, C_out, F);

    if (!check_array_close(h_output.data(), expected.data(), output_size, 1e-3f, 1e-3f)) {
        throw std::runtime_error("LeNet Conv1 (28x28x1 -> 24x24x6) mismatch");
    }
}

// Test 3: LeNet Conv2 — 12×12×6 → 8×8×16, filter 5×5
// Conv2 takes 6 feature maps (after pooling) and produces 16 feature maps
TEST_CASE("conv_lenet_conv2", "[puzzle_07_conv_forward]") {
    const int batch = 1, C_in = 6, H = 12, W = 12, C_out = 16, F = 5;
    const int H_out = H - F + 1;  // 8
    const int W_out = W - F + 1;  // 8

    const int input_size  = batch * C_in * H * W;
    const int filter_size = C_out * C_in * F * F;
    const int output_size = batch * C_out * H_out * W_out;

    std::vector<float> h_input(input_size);
    std::vector<float> h_filters(filter_size);
    std::vector<float> h_bias(C_out);
    std::vector<float> h_output(output_size, 0.0f);
    std::vector<float> expected(output_size, 0.0f);

    fill_random(h_input.data(),   input_size,  710, -0.5f, 0.5f);
    fill_random(h_filters.data(), filter_size, 711, -0.1f, 0.1f);
    fill_random(h_bias.data(),    C_out,       712, -0.1f, 0.1f);

    conv2d_forward_cpu(h_input.data(), h_filters.data(), h_bias.data(),
                       expected.data(), batch, C_in, H, W, C_out, F);

    run_conv2d_forward_gpu(h_input.data(), h_filters.data(), h_bias.data(),
                           h_output.data(), batch, C_in, H, W, C_out, F);

    if (!check_array_close(h_output.data(), expected.data(), output_size, 1e-3f, 1e-3f)) {
        throw std::runtime_error("LeNet Conv2 (12x12x6 -> 8x8x16) mismatch");
    }
}

// Test 4: Batch processing — batch=8 through Conv1 (28×28×1 → 24×24×6)
// Verifies the kernel handles multiple images in a batch correctly
TEST_CASE("conv_batch_processing", "[puzzle_07_conv_forward]") {
    const int batch = 8, C_in = 1, H = 28, W = 28, C_out = 6, F = 5;
    const int H_out = H - F + 1;  // 24
    const int W_out = W - F + 1;  // 24

    const int input_size  = batch * C_in * H * W;
    const int filter_size = C_out * C_in * F * F;
    const int output_size = batch * C_out * H_out * W_out;

    std::vector<float> h_input(input_size);
    std::vector<float> h_filters(filter_size);
    std::vector<float> h_bias(C_out);
    std::vector<float> h_output(output_size, 0.0f);
    std::vector<float> expected(output_size, 0.0f);

    fill_random(h_input.data(),   input_size,  720, -0.5f, 0.5f);
    fill_random(h_filters.data(), filter_size, 721, -0.2f, 0.2f);
    fill_random(h_bias.data(),    C_out,       722, -0.1f, 0.1f);

    conv2d_forward_cpu(h_input.data(), h_filters.data(), h_bias.data(),
                       expected.data(), batch, C_in, H, W, C_out, F);

    run_conv2d_forward_gpu(h_input.data(), h_filters.data(), h_bias.data(),
                           h_output.data(), batch, C_in, H, W, C_out, F);

    if (!check_array_close(h_output.data(), expected.data(), output_size, 1e-3f, 1e-3f)) {
        throw std::runtime_error("Batch Conv1 (8×28×28×1 -> 8×24×24×6) mismatch");
    }
}

// Test 5: Dimension check — verifies output dimensions match the formula
// H_out = H - F + 1, W_out = W - F + 1
// Conv1: (28-5)+1 = 24, Conv2: (12-5)+1 = 8
TEST_CASE("conv_output_dimensions", "[puzzle_07_conv_forward]") {
    // Test Conv1 dimensions: 28×28 with 5×5 filter → 24×24
    {
        const int batch = 1, C_in = 1, H = 28, W = 28, C_out = 1, F = 5;
        const int H_out = H - F + 1;
        const int W_out = W - F + 1;

        if (H_out != 24 || W_out != 24) {
            throw std::runtime_error("Conv1 dimension formula wrong: expected 24x24");
        }

        const int input_size  = batch * C_in * H * W;
        const int filter_size = C_out * C_in * F * F;
        const int output_size = batch * C_out * H_out * W_out;

        // Run a real convolution to verify the kernel produces the right number of outputs
        std::vector<float> h_input(input_size, 1.0f);
        std::vector<float> h_filters(filter_size, 0.0f);
        std::vector<float> h_bias(C_out, 1.0f);  // bias=1, filters=0 → output should be all 1s
        std::vector<float> h_output(output_size, 0.0f);

        run_conv2d_forward_gpu(h_input.data(), h_filters.data(), h_bias.data(),
                               h_output.data(), batch, C_in, H, W, C_out, F);

        // Every output element should be bias[0] = 1.0 (since filters are zero)
        std::vector<float> expected(output_size, 1.0f);
        if (!check_array_close(h_output.data(), expected.data(), output_size, 1e-5f, 1e-5f)) {
            throw std::runtime_error("Conv1 dimension check: output values wrong (expected all 1.0)");
        }
    }

    // Test Conv2 dimensions: 12×12 with 5×5 filter → 8×8
    {
        const int batch = 1, C_in = 6, H = 12, W = 12, C_out = 1, F = 5;
        const int H_out = H - F + 1;
        const int W_out = W - F + 1;

        if (H_out != 8 || W_out != 8) {
            throw std::runtime_error("Conv2 dimension formula wrong: expected 8x8");
        }

        const int input_size  = batch * C_in * H * W;
        const int filter_size = C_out * C_in * F * F;
        const int output_size = batch * C_out * H_out * W_out;

        std::vector<float> h_input(input_size, 1.0f);
        std::vector<float> h_filters(filter_size, 0.0f);
        std::vector<float> h_bias(C_out, 2.0f);  // bias=2, filters=0 → output should be all 2s
        std::vector<float> h_output(output_size, 0.0f);

        run_conv2d_forward_gpu(h_input.data(), h_filters.data(), h_bias.data(),
                               h_output.data(), batch, C_in, H, W, C_out, F);

        std::vector<float> expected(output_size, 2.0f);
        if (!check_array_close(h_output.data(), expected.data(), output_size, 1e-5f, 1e-5f)) {
            throw std::runtime_error("Conv2 dimension check: output values wrong (expected all 2.0)");
        }
    }
}

