#include <catch2/catch_test_macros.hpp>
// Puzzle 03: Fully Connected Layer (Forward Pass) — Test Harness
//
// Tests:
//   1. Single sample, 4→3 — hardcoded, hand-verifiable
//   2. Batch=8, 256→120 — LeNet FC1 dimensions
//   3. Batch=8, 120→84  — LeNet FC2 dimensions
//   4. Batch=8, 84→10   — LeNet FC3 (output layer)

#include "cuda_utils.h"
#include "test_utils.h"

// Include the kernel implementation
// Build system compiles this file twice:
//   puzzle_03_test          -> includes puzzle.cu
//   puzzle_03_test_solution -> includes solution.cu
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// CPU reference implementation for verification
// output[b][j] = Σ_i input[b][i] * weights[j][i] + bias[j]
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

// Helper: run fc_forward kernel on GPU and copy result back to host
void run_fc_forward_gpu(const float* h_input, const float* h_weights,
                        const float* h_bias, float* h_output,
                        int batch, int in_features, int out_features) {
    float *d_input, *d_weights, *d_bias, *d_output;

    size_t input_bytes  = batch * in_features * sizeof(float);
    size_t weight_bytes = out_features * in_features * sizeof(float);
    size_t bias_bytes   = out_features * sizeof(float);
    size_t output_bytes = batch * out_features * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input,   input_bytes));
    CUDA_CHECK(cudaMalloc(&d_weights, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_bias,    bias_bytes));
    CUDA_CHECK(cudaMalloc(&d_output,  output_bytes));

    CUDA_CHECK(cudaMemcpy(d_input,   h_input,   input_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, weight_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias,    h_bias,    bias_bytes,   cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((out_features + blockDim.x - 1) / blockDim.x,
                 (batch + blockDim.y - 1) / blockDim.y);

    fc_forward<<<gridDim, blockDim>>>(d_input, d_weights, d_bias, d_output,
                                      batch, in_features, out_features);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
}

// Test 1: Single sample, 4→3 — hardcoded, hand-verifiable
//
// Input (1×4):  [1.0, 2.0, 3.0, 4.0]
//
// Weights (3×4):
//   neuron 0: [0.1, 0.2, 0.3, 0.4]
//   neuron 1: [0.5, 0.6, 0.7, 0.8]
//   neuron 2: [1.0, -1.0, 0.5, -0.5]
//
// Bias (3): [0.1, 0.2, 0.3]
//
// output[0] = 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 + 0.1 = 0.1+0.4+0.9+1.6+0.1 = 3.1
// output[1] = 1*0.5 + 2*0.6 + 3*0.7 + 4*0.8 + 0.2 = 0.5+1.2+2.1+3.2+0.2 = 7.2
// output[2] = 1*1.0 + 2*(-1.0) + 3*0.5 + 4*(-0.5) + 0.3 = 1-2+1.5-2+0.3 = -1.2
TEST_CASE("fc_single_sample_4to3", "[puzzle_03_fc_forward]") {
    const int batch = 1, in_features = 4, out_features = 3;

    float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f};

    float h_weights[] = {
        0.1f, 0.2f, 0.3f, 0.4f,    // neuron 0
        0.5f, 0.6f, 0.7f, 0.8f,    // neuron 1
        1.0f, -1.0f, 0.5f, -0.5f   // neuron 2
    };

    float h_bias[] = {0.1f, 0.2f, 0.3f};

    float expected[] = {3.1f, 7.2f, -1.2f};

    float h_output[3] = {0};
    run_fc_forward_gpu(h_input, h_weights, h_bias, h_output,
                       batch, in_features, out_features);

    REQUIRE(check_array_close(h_output, expected, batch * out_features, 1e-4f, 1e-4f));
}

// Test 2: Batch=8, 256→120 — LeNet FC1 dimensions
// FC1 takes the 4×4×16=256 flattened feature maps and outputs 120 features
TEST_CASE("fc_lenet_fc1_256to120", "[puzzle_03_fc_forward]") {
    const int batch = 8, in_features = 256, out_features = 120;
    const int input_size  = batch * in_features;
    const int weight_size = out_features * in_features;
    const int output_size = batch * out_features;

    std::vector<float> h_input(input_size);
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_bias(out_features);
    std::vector<float> h_output(output_size, 0.0f);
    std::vector<float> expected(output_size, 0.0f);

    // Use small values to keep accumulated sums reasonable
    fill_random(h_input.data(),   input_size,  200, -0.1f, 0.1f);
    fill_random(h_weights.data(), weight_size, 201, -0.1f, 0.1f);
    fill_random(h_bias.data(),    out_features, 202, -0.5f, 0.5f);

    fc_forward_cpu(h_input.data(), h_weights.data(), h_bias.data(),
                   expected.data(), batch, in_features, out_features);

    run_fc_forward_gpu(h_input.data(), h_weights.data(), h_bias.data(),
                       h_output.data(), batch, in_features, out_features);

    if (!check_array_close(h_output.data(), expected.data(), output_size, 1e-3f, 1e-3f)) {
        throw std::runtime_error("LeNet FC1 (256→120) mismatch");
    }
}

// Test 3: Batch=8, 120→84 — LeNet FC2 dimensions
TEST_CASE("fc_lenet_fc2_120to84", "[puzzle_03_fc_forward]") {
    const int batch = 8, in_features = 120, out_features = 84;
    const int input_size  = batch * in_features;
    const int weight_size = out_features * in_features;
    const int output_size = batch * out_features;

    std::vector<float> h_input(input_size);
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_bias(out_features);
    std::vector<float> h_output(output_size, 0.0f);
    std::vector<float> expected(output_size, 0.0f);

    fill_random(h_input.data(),   input_size,  300, -0.1f, 0.1f);
    fill_random(h_weights.data(), weight_size, 301, -0.1f, 0.1f);
    fill_random(h_bias.data(),    out_features, 302, -0.5f, 0.5f);

    fc_forward_cpu(h_input.data(), h_weights.data(), h_bias.data(),
                   expected.data(), batch, in_features, out_features);

    run_fc_forward_gpu(h_input.data(), h_weights.data(), h_bias.data(),
                       h_output.data(), batch, in_features, out_features);

    if (!check_array_close(h_output.data(), expected.data(), output_size, 1e-3f, 1e-3f)) {
        throw std::runtime_error("LeNet FC2 (120→84) mismatch");
    }
}

// Test 4: Batch=8, 84→10 — LeNet FC3 (output layer)
// This is the final layer that produces class scores for 10 digits
TEST_CASE("fc_lenet_fc3_84to10", "[puzzle_03_fc_forward]") {
    const int batch = 8, in_features = 84, out_features = 10;
    const int input_size  = batch * in_features;
    const int weight_size = out_features * in_features;
    const int output_size = batch * out_features;

    std::vector<float> h_input(input_size);
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_bias(out_features);
    std::vector<float> h_output(output_size, 0.0f);
    std::vector<float> expected(output_size, 0.0f);

    fill_random(h_input.data(),   input_size,  400, -0.1f, 0.1f);
    fill_random(h_weights.data(), weight_size, 401, -0.1f, 0.1f);
    fill_random(h_bias.data(),    out_features, 402, -0.5f, 0.5f);

    fc_forward_cpu(h_input.data(), h_weights.data(), h_bias.data(),
                   expected.data(), batch, in_features, out_features);

    run_fc_forward_gpu(h_input.data(), h_weights.data(), h_bias.data(),
                       h_output.data(), batch, in_features, out_features);

    if (!check_array_close(h_output.data(), expected.data(), output_size, 1e-3f, 1e-3f)) {
        throw std::runtime_error("LeNet FC3 (84→10) mismatch");
    }
}

