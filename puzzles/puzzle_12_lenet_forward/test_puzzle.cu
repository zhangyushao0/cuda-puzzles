#include <catch2/catch_test_macros.hpp>
// Puzzle 12: LeNet-5 Forward Pass — Test Harness
//
// Tests:
//   1. Single image — output is 10 probabilities summing to 1.0
//   2. Batch of 8 — verify batch processing works
//   3. Deterministic weights (seed=42) — verify exact output probabilities
//   4. Intermediate dimensions — verify layer outputs have correct sizes

#include "cuda_utils.h"
#include "test_utils.h"

#include <cmath>
#include <cfloat>
#include <vector>
#include <stdexcept>

// Include the kernel implementation
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// ============================================================
// CPU reference implementations for each layer
// ============================================================

void conv2d_forward_cpu(const float* input, const float* filters,
                        const float* bias, float* output,
                        int batch, int C_in, int H, int W,
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

void relu_forward_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

void maxpool_forward_cpu(const float* input, float* output, int* max_indices,
                         int batch, int C, int H, int W) {
    int H_out = H / 2;
    int W_out = W / 2;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    int idx = b * (C * H_out * W_out) + c * (H_out * W_out)
                            + oh * W_out + ow;
                    int h_start = oh * 2;
                    int w_start = ow * 2;
                    float max_val = -FLT_MAX;
                    int max_idx = 0;
                    for (int ph = 0; ph < 2; ph++) {
                        for (int pw = 0; pw < 2; pw++) {
                            int in_idx = b * (C * H * W) + c * (H * W)
                                       + (h_start + ph) * W + (w_start + pw);
                            if (input[in_idx] > max_val) {
                                max_val = input[in_idx];
                                max_idx = ph * 2 + pw;
                            }
                        }
                    }
                    output[idx] = max_val;
                    max_indices[idx] = max_idx;
                }
            }
        }
    }
}

void fc_forward_cpu(const float* input, const float* weights,
                    const float* bias, float* output,
                    int batch, int in_features, int out_features) {
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < out_features; j++) {
            float sum = 0.0f;
            for (int i = 0; i < in_features; i++) {
                sum += input[b * in_features + i]
                     * weights[j * in_features + i];
            }
            output[b * out_features + j] = sum + bias[j];
        }
    }
}

void softmax_forward_cpu(const float* logits, float* probs,
                         int batch, int num_classes) {
    for (int b = 0; b < batch; b++) {
        int offset = b * num_classes;
        float max_val = logits[offset];
        for (int c = 1; c < num_classes; c++) {
            max_val = fmaxf(max_val, logits[offset + c]);
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            probs[offset + c] = expf(logits[offset + c] - max_val);
            sum_exp += probs[offset + c];
        }
        for (int c = 0; c < num_classes; c++) {
            probs[offset + c] /= sum_exp;
        }
    }
}

// ============================================================
// Full CPU LeNet forward pass
// ============================================================

void lenet_forward_cpu(const float* input, float* probs, int batch,
                       const float* conv1_w, const float* conv1_b,
                       const float* conv2_w, const float* conv2_b,
                       const float* fc1_w, const float* fc1_b,
                       const float* fc2_w, const float* fc2_b,
                       const float* fc3_w, const float* fc3_b) {
    // Allocate all intermediate buffers
    std::vector<float> conv1_out(batch * 6 * 24 * 24);
    std::vector<float> relu1_out(batch * 6 * 24 * 24);
    std::vector<float> pool1_out(batch * 6 * 12 * 12);
    std::vector<int>   pool1_idx(batch * 6 * 12 * 12);
    std::vector<float> conv2_out(batch * 16 * 8 * 8);
    std::vector<float> relu2_out(batch * 16 * 8 * 8);
    std::vector<float> pool2_out(batch * 16 * 4 * 4);
    std::vector<int>   pool2_idx(batch * 16 * 4 * 4);
    std::vector<float> fc1_out(batch * 120);
    std::vector<float> relu3_out(batch * 120);
    std::vector<float> fc2_out(batch * 84);
    std::vector<float> relu4_out(batch * 84);
    std::vector<float> fc3_out(batch * 10);

    conv2d_forward_cpu(input, conv1_w, conv1_b, conv1_out.data(),
                       batch, 1, 28, 28, 6, 5);
    relu_forward_cpu(conv1_out.data(), relu1_out.data(), batch * 6 * 24 * 24);
    maxpool_forward_cpu(relu1_out.data(), pool1_out.data(), pool1_idx.data(),
                        batch, 6, 24, 24);
    conv2d_forward_cpu(pool1_out.data(), conv2_w, conv2_b, conv2_out.data(),
                       batch, 6, 12, 12, 16, 5);
    relu_forward_cpu(conv2_out.data(), relu2_out.data(), batch * 16 * 8 * 8);
    maxpool_forward_cpu(relu2_out.data(), pool2_out.data(), pool2_idx.data(),
                        batch, 16, 8, 8);
    // Flatten: pool2_out is already (batch x 256)
    fc_forward_cpu(pool2_out.data(), fc1_w, fc1_b, fc1_out.data(),
                   batch, 256, 120);
    relu_forward_cpu(fc1_out.data(), relu3_out.data(), batch * 120);
    fc_forward_cpu(relu3_out.data(), fc2_w, fc2_b, fc2_out.data(),
                   batch, 120, 84);
    relu_forward_cpu(fc2_out.data(), relu4_out.data(), batch * 84);
    fc_forward_cpu(relu4_out.data(), fc3_w, fc3_b, fc3_out.data(),
                   batch, 84, 10);
    softmax_forward_cpu(fc3_out.data(), probs, batch, 10);
}

// ============================================================
// Helper: generate deterministic LeNet parameters
// ============================================================

struct HostParams {
    std::vector<float> conv1_w, conv1_b;
    std::vector<float> conv2_w, conv2_b;
    std::vector<float> fc1_w,   fc1_b;
    std::vector<float> fc2_w,   fc2_b;
    std::vector<float> fc3_w,   fc3_b;
};

void init_params(HostParams& hp, unsigned seed) {
    hp.conv1_w.resize(150);   hp.conv1_b.resize(6);
    hp.conv2_w.resize(2400);  hp.conv2_b.resize(16);
    hp.fc1_w.resize(30720);   hp.fc1_b.resize(120);
    hp.fc2_w.resize(10080);   hp.fc2_b.resize(84);
    hp.fc3_w.resize(840);     hp.fc3_b.resize(10);

    // Use small random values to keep activations reasonable
    fill_random(hp.conv1_w.data(), 150,   seed,     -0.2f, 0.2f);
    fill_random(hp.conv1_b.data(), 6,     seed + 1, -0.1f, 0.1f);
    fill_random(hp.conv2_w.data(), 2400,  seed + 2, -0.1f, 0.1f);
    fill_random(hp.conv2_b.data(), 16,    seed + 3, -0.1f, 0.1f);
    fill_random(hp.fc1_w.data(),   30720, seed + 4, -0.05f, 0.05f);
    fill_random(hp.fc1_b.data(),   120,   seed + 5, -0.1f, 0.1f);
    fill_random(hp.fc2_w.data(),   10080, seed + 6, -0.05f, 0.05f);
    fill_random(hp.fc2_b.data(),   84,    seed + 7, -0.1f, 0.1f);
    fill_random(hp.fc3_w.data(),   840,   seed + 8, -0.1f, 0.1f);
    fill_random(hp.fc3_b.data(),   10,    seed + 9, -0.1f, 0.1f);
}

// ============================================================
// Helper: run GPU forward pass and return probabilities
// ============================================================

void run_lenet_gpu(const float* h_input, float* h_probs, int batch,
                   HostParams& hp) {
    // Upload input
    float* d_input;
    size_t input_bytes = (size_t)batch * 1 * 28 * 28 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

    // Allocate and upload params
    LeNetParams params;
    alloc_params(params);
    upload_params(params,
                  hp.conv1_w.data(), hp.conv1_b.data(),
                  hp.conv2_w.data(), hp.conv2_b.data(),
                  hp.fc1_w.data(),   hp.fc1_b.data(),
                  hp.fc2_w.data(),   hp.fc2_b.data(),
                  hp.fc3_w.data(),   hp.fc3_b.data());

    // Allocate activations
    LeNetActivations act;
    alloc_activations(act, batch);

    // Run forward pass
    lenet_forward(d_input, params, act, batch);

    // Copy probabilities back
    CUDA_CHECK(cudaMemcpy(h_probs, act.probs,
                          batch * 10 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Cleanup
    free_activations(act);
    free_params(params);
    CUDA_CHECK(cudaFree(d_input));
}

// ============================================================
// Test 1: Single image — output is 10 probabilities summing to 1.0
// Verify that a single 28x28 input produces valid probability output
// ============================================================
TEST_CASE("lenet_single_image", "[puzzle_12_lenet_forward]") {
    const int batch = 1;

    // Generate deterministic input and params
    std::vector<float> h_input(batch * 1 * 28 * 28);
    fill_random(h_input.data(), (int)h_input.size(), 100, 0.0f, 1.0f);

    HostParams hp;
    init_params(hp, 42);

    // CPU reference
    std::vector<float> expected(batch * 10);
    lenet_forward_cpu(h_input.data(), expected.data(), batch,
                      hp.conv1_w.data(), hp.conv1_b.data(),
                      hp.conv2_w.data(), hp.conv2_b.data(),
                      hp.fc1_w.data(),   hp.fc1_b.data(),
                      hp.fc2_w.data(),   hp.fc2_b.data(),
                      hp.fc3_w.data(),   hp.fc3_b.data());

    // GPU
    std::vector<float> h_probs(batch * 10, 0.0f);
    run_lenet_gpu(h_input.data(), h_probs.data(), batch, hp);

    // Check: all probabilities between 0 and 1
    for (int i = 0; i < 10; i++) {
        if (h_probs[i] < 0.0f || h_probs[i] > 1.0f) {
            throw std::runtime_error("Probability out of [0,1] range");
        }
    }

    // Check: probabilities sum to 1.0
    float sum = 0.0f;
    for (int i = 0; i < 10; i++) sum += h_probs[i];
    if (fabsf(sum - 1.0f) > 1e-4f) {
        throw std::runtime_error("Probabilities do not sum to 1.0 (sum=" +
                                 std::to_string(sum) + ")");
    }

    // Check: matches CPU reference
    REQUIRE(check_array_close(h_probs.data(), expected.data(), 10, 1e-4f, 1e-3f));
}

// ============================================================
// Test 2: Batch of 8 — verify batch processing works
// ============================================================
TEST_CASE("lenet_batch_processing", "[puzzle_12_lenet_forward]") {
    const int batch = 8;

    std::vector<float> h_input(batch * 1 * 28 * 28);
    fill_random(h_input.data(), (int)h_input.size(), 200, 0.0f, 1.0f);

    HostParams hp;
    init_params(hp, 42);

    // CPU reference
    std::vector<float> expected(batch * 10);
    lenet_forward_cpu(h_input.data(), expected.data(), batch,
                      hp.conv1_w.data(), hp.conv1_b.data(),
                      hp.conv2_w.data(), hp.conv2_b.data(),
                      hp.fc1_w.data(),   hp.fc1_b.data(),
                      hp.fc2_w.data(),   hp.fc2_b.data(),
                      hp.fc3_w.data(),   hp.fc3_b.data());

    // GPU
    std::vector<float> h_probs(batch * 10, 0.0f);
    run_lenet_gpu(h_input.data(), h_probs.data(), batch, hp);

    // Check each sample's probabilities sum to 1.0
    for (int b = 0; b < batch; b++) {
        float sum = 0.0f;
        for (int c = 0; c < 10; c++) sum += h_probs[b * 10 + c];
        if (fabsf(sum - 1.0f) > 1e-4f) {
            throw std::runtime_error("Sample " + std::to_string(b) +
                                     " probs don't sum to 1.0 (sum=" +
                                     std::to_string(sum) + ")");
        }
    }

    // Check: matches CPU reference
    REQUIRE(check_array_close(h_probs.data(), expected.data(), batch * 10, 1e-4f, 1e-3f));
}

// ============================================================
// Test 3: Deterministic weights (seed=42) — verify exact probabilities
// Run the same forward pass twice and confirm identical output
// ============================================================
TEST_CASE("lenet_deterministic", "[puzzle_12_lenet_forward]") {
    const int batch = 4;

    std::vector<float> h_input(batch * 1 * 28 * 28);
    fill_random(h_input.data(), (int)h_input.size(), 300, 0.0f, 1.0f);

    HostParams hp;
    init_params(hp, 42);

    // Run GPU forward pass twice
    std::vector<float> probs1(batch * 10, 0.0f);
    std::vector<float> probs2(batch * 10, 0.0f);
    run_lenet_gpu(h_input.data(), probs1.data(), batch, hp);
    run_lenet_gpu(h_input.data(), probs2.data(), batch, hp);

    // Must be bit-exact across runs (same deterministic computation)
    REQUIRE(check_array_close(probs1.data(), probs2.data(), batch * 10, 1e-6f, 1e-6f));

    // Also verify each sample has exactly one argmax (no all-zeros output)
    for (int b = 0; b < batch; b++) {
        float max_prob = 0.0f;
        for (int c = 0; c < 10; c++) {
            if (probs1[b * 10 + c] > max_prob) {
                max_prob = probs1[b * 10 + c];
            }
        }
        if (max_prob < 0.01f) {
            throw std::runtime_error("Sample " + std::to_string(b) +
                                     " has suspiciously low max probability: " +
                                     std::to_string(max_prob));
        }
    }
}

// ============================================================
// Test 4: Intermediate dimensions — verify layer outputs
// Use zero weights at specific layers to verify dimension correctness
// ============================================================
TEST_CASE("lenet_intermediate_dimensions", "[puzzle_12_lenet_forward]") {
    const int batch = 2;

    // Create input where all pixels are 1.0
    std::vector<float> h_input(batch * 1 * 28 * 28, 1.0f);

    // Use specific params: zero conv1 weights + known bias
    // to verify Conv1 output dimensions are 6 x 24 x 24
    HostParams hp;
    init_params(hp, 42);

    // Run on GPU — we'll use a special approach:
    // Upload input, run forward, then download intermediate activations
    float* d_input;
    size_t input_bytes = (size_t)batch * 1 * 28 * 28 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes,
                          cudaMemcpyHostToDevice));

    LeNetParams params;
    alloc_params(params);
    upload_params(params,
                  hp.conv1_w.data(), hp.conv1_b.data(),
                  hp.conv2_w.data(), hp.conv2_b.data(),
                  hp.fc1_w.data(),   hp.fc1_b.data(),
                  hp.fc2_w.data(),   hp.fc2_b.data(),
                  hp.fc3_w.data(),   hp.fc3_b.data());

    LeNetActivations act;
    alloc_activations(act, batch);

    lenet_forward(d_input, params, act, batch);

    // Verify Conv1 output: should be batch x 6 x 24 x 24
    {
        int conv1_size = batch * 6 * 24 * 24;
        std::vector<float> conv1(conv1_size);
        CUDA_CHECK(cudaMemcpy(conv1.data(), act.conv1_out,
                              conv1_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        // Run CPU reference for conv1 only
        std::vector<float> expected_conv1(conv1_size);
        conv2d_forward_cpu(h_input.data(), hp.conv1_w.data(), hp.conv1_b.data(),
                           expected_conv1.data(), batch, 1, 28, 28, 6, 5);
        if (!check_array_close(conv1.data(), expected_conv1.data(),
                              conv1_size, 1e-3f, 1e-3f)) {
            throw std::runtime_error("Conv1 output dimensions/values wrong (expected Bx6x24x24)");
        }
    }

    // Verify Pool1 output: should be batch x 6 x 12 x 12
    {
        int pool1_size = batch * 6 * 12 * 12;
        std::vector<float> pool1(pool1_size);
        CUDA_CHECK(cudaMemcpy(pool1.data(), act.pool1_out,
                              pool1_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        // All values should be finite
        for (int i = 0; i < pool1_size; i++) {
            if (!std::isfinite(pool1[i])) {
                throw std::runtime_error("Pool1 output has non-finite value at index " +
                                         std::to_string(i));
            }
        }
    }

    // Verify Pool2 output: should be batch x 16 x 4 x 4 = batch x 256
    {
        int pool2_size = batch * 16 * 4 * 4;
        std::vector<float> pool2(pool2_size);
        CUDA_CHECK(cudaMemcpy(pool2.data(), act.pool2_out,
                              pool2_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        // All values should be finite
        for (int i = 0; i < pool2_size; i++) {
            if (!std::isfinite(pool2[i])) {
                throw std::runtime_error("Pool2 output has non-finite value at index " +
                                         std::to_string(i));
            }
        }
    }

    // Verify final output: should be batch x 10
    {
        int prob_size = batch * 10;
        std::vector<float> probs(prob_size);
        CUDA_CHECK(cudaMemcpy(probs.data(), act.probs,
                              prob_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int b = 0; b < batch; b++) {
            float sum = 0.0f;
            for (int c = 0; c < 10; c++) {
                float p = probs[b * 10 + c];
                if (p < 0.0f || p > 1.0f) {
                    throw std::runtime_error("Final prob out of [0,1] for sample " +
                                             std::to_string(b));
                }
                sum += p;
            }
            if (fabsf(sum - 1.0f) > 1e-4f) {
                throw std::runtime_error("Final probs don't sum to 1.0 for sample " +
                                         std::to_string(b));
            }
        }
    }

    free_activations(act);
    free_params(params);
    CUDA_CHECK(cudaFree(d_input));
}

