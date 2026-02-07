// Puzzle 13: LeNet-5 Backward Pass — Test Harness
//
// Tests:
//   1. Forward+backward with gradient shapes verified
//   2. Sampled numerical gradient check (5 random params per layer)
//   3. Loss decrease after SGD step
//   4. Gradient magnitude sanity check

#include "cuda_utils.h"
#include "test_utils.h"

#include <cmath>
#include <cfloat>
#include <vector>
#include <stdexcept>
#include <random>
#include <algorithm>

// Include the kernel implementation
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// ============================================================
// CPU reference implementations for forward pass layers
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
// CPU forward pass that returns all intermediates
// ============================================================

struct CpuActivations {
    std::vector<float> conv1_out, relu1_out, pool1_out;
    std::vector<int>   pool1_idx;
    std::vector<float> conv2_out, relu2_out, pool2_out;
    std::vector<int>   pool2_idx;
    std::vector<float> fc1_out, relu3_out, fc2_out, relu4_out;
    std::vector<float> fc3_out, probs;
};

void lenet_forward_cpu(const float* input, CpuActivations& a, int batch,
                       const float* conv1_w, const float* conv1_b,
                       const float* conv2_w, const float* conv2_b,
                       const float* fc1_w, const float* fc1_b,
                       const float* fc2_w, const float* fc2_b,
                       const float* fc3_w, const float* fc3_b) {
    a.conv1_out.resize(batch * 6 * 24 * 24);
    a.relu1_out.resize(batch * 6 * 24 * 24);
    a.pool1_out.resize(batch * 6 * 12 * 12);
    a.pool1_idx.resize(batch * 6 * 12 * 12);
    a.conv2_out.resize(batch * 16 * 8 * 8);
    a.relu2_out.resize(batch * 16 * 8 * 8);
    a.pool2_out.resize(batch * 16 * 4 * 4);
    a.pool2_idx.resize(batch * 16 * 4 * 4);
    a.fc1_out.resize(batch * 120);
    a.relu3_out.resize(batch * 120);
    a.fc2_out.resize(batch * 84);
    a.relu4_out.resize(batch * 84);
    a.fc3_out.resize(batch * 10);
    a.probs.resize(batch * 10);

    conv2d_forward_cpu(input, conv1_w, conv1_b, a.conv1_out.data(), batch, 1, 28, 28, 6, 5);
    relu_forward_cpu(a.conv1_out.data(), a.relu1_out.data(), batch * 6 * 24 * 24);
    maxpool_forward_cpu(a.relu1_out.data(), a.pool1_out.data(), a.pool1_idx.data(), batch, 6, 24, 24);
    conv2d_forward_cpu(a.pool1_out.data(), conv2_w, conv2_b, a.conv2_out.data(), batch, 6, 12, 12, 16, 5);
    relu_forward_cpu(a.conv2_out.data(), a.relu2_out.data(), batch * 16 * 8 * 8);
    maxpool_forward_cpu(a.relu2_out.data(), a.pool2_out.data(), a.pool2_idx.data(), batch, 16, 8, 8);
    fc_forward_cpu(a.pool2_out.data(), fc1_w, fc1_b, a.fc1_out.data(), batch, 256, 120);
    relu_forward_cpu(a.fc1_out.data(), a.relu3_out.data(), batch * 120);
    fc_forward_cpu(a.relu3_out.data(), fc2_w, fc2_b, a.fc2_out.data(), batch, 120, 84);
    relu_forward_cpu(a.fc2_out.data(), a.relu4_out.data(), batch * 84);
    fc_forward_cpu(a.relu4_out.data(), fc3_w, fc3_b, a.fc3_out.data(), batch, 84, 10);
    softmax_forward_cpu(a.fc3_out.data(), a.probs.data(), batch, 10);
}

// ============================================================
// CPU cross-entropy loss (average over batch)
// ============================================================

float cross_entropy_loss_cpu(const float* probs, const float* labels,
                             int batch, int num_classes) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < num_classes; c++) {
            total_loss -= labels[b * num_classes + c]
                        * logf(probs[b * num_classes + c] + 1e-10f);
        }
    }
    return total_loss / batch;
}

// ============================================================
// Host parameter structure and helpers
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

// Get a flat pointer to all params in order, and sizes
struct ParamInfo {
    float* host_ptr;
    int    size;
    const char* name;
};

std::vector<ParamInfo> get_param_infos(HostParams& hp) {
    return {
        {hp.conv1_w.data(), 150,   "conv1_w"},
        {hp.conv1_b.data(), 6,     "conv1_b"},
        {hp.conv2_w.data(), 2400,  "conv2_w"},
        {hp.conv2_b.data(), 16,    "conv2_b"},
        {hp.fc1_w.data(),   30720, "fc1_w"},
        {hp.fc1_b.data(),   120,   "fc1_b"},
        {hp.fc2_w.data(),   10080, "fc2_w"},
        {hp.fc2_b.data(),   84,    "fc2_b"},
        {hp.fc3_w.data(),   840,   "fc3_w"},
        {hp.fc3_b.data(),   10,    "fc3_b"}
    };
}

// ============================================================
// Helper: run forward + backward on GPU, return gradients
// ============================================================

struct GpuResult {
    std::vector<float> probs;
    std::vector<float> grad_conv1_w, grad_conv1_b;
    std::vector<float> grad_conv2_w, grad_conv2_b;
    std::vector<float> grad_fc1_w,   grad_fc1_b;
    std::vector<float> grad_fc2_w,   grad_fc2_b;
    std::vector<float> grad_fc3_w,   grad_fc3_b;
};

void run_forward_backward_gpu(const float* h_input, const float* h_labels,
                              int batch, HostParams& hp, GpuResult& result) {
    // Upload input
    float* d_input;
    size_t input_bytes = (size_t)batch * 1 * 28 * 28 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

    // Upload labels
    float* d_labels;
    size_t label_bytes = (size_t)batch * 10 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_labels, label_bytes));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels, label_bytes, cudaMemcpyHostToDevice));

    // Allocate and upload params
    LeNetParams params;
    alloc_params(params);
    upload_params(params,
                  hp.conv1_w.data(), hp.conv1_b.data(),
                  hp.conv2_w.data(), hp.conv2_b.data(),
                  hp.fc1_w.data(),   hp.fc1_b.data(),
                  hp.fc2_w.data(),   hp.fc2_b.data(),
                  hp.fc3_w.data(),   hp.fc3_b.data());

    // Allocate activations and run forward pass
    LeNetActivations act;
    alloc_activations(act, batch);
    lenet_forward(d_input, params, act, batch);

    // Allocate gradient buffers
    LeNetGrads grads;
    alloc_grads(grads);
    LeNetGradActivations ga;
    alloc_grad_activations(ga, batch);

    // Run backward pass
    lenet_backward(d_input, d_labels, params, act, grads, ga, batch);

    // Copy results back
    result.probs.resize(batch * 10);
    CUDA_CHECK(cudaMemcpy(result.probs.data(), act.probs,
                          batch * 10 * sizeof(float), cudaMemcpyDeviceToHost));

    result.grad_conv1_w.resize(150);
    result.grad_conv1_b.resize(6);
    result.grad_conv2_w.resize(2400);
    result.grad_conv2_b.resize(16);
    result.grad_fc1_w.resize(30720);
    result.grad_fc1_b.resize(120);
    result.grad_fc2_w.resize(10080);
    result.grad_fc2_b.resize(84);
    result.grad_fc3_w.resize(840);
    result.grad_fc3_b.resize(10);

    CUDA_CHECK(cudaMemcpy(result.grad_conv1_w.data(), grads.conv1_w, 150   * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grad_conv1_b.data(), grads.conv1_b, 6     * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grad_conv2_w.data(), grads.conv2_w, 2400  * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grad_conv2_b.data(), grads.conv2_b, 16    * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grad_fc1_w.data(),   grads.fc1_w,   30720 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grad_fc1_b.data(),   grads.fc1_b,   120   * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grad_fc2_w.data(),   grads.fc2_w,   10080 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grad_fc2_b.data(),   grads.fc2_b,   84    * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grad_fc3_w.data(),   grads.fc3_w,   840   * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.grad_fc3_b.data(),   grads.fc3_b,   10    * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    free_grad_activations(ga);
    free_grads(grads);
    free_activations(act);
    free_params(params);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_labels));
}

// ============================================================
// Test 1: Forward+backward with shapes verified
// Verify that gradient shapes match parameter shapes and all gradients are finite
// ============================================================
TEST_CASE(backward_shapes_verified) {
    const int batch = 4;

    std::vector<float> h_input(batch * 1 * 28 * 28);
    fill_random(h_input.data(), (int)h_input.size(), 100, 0.0f, 1.0f);

    // Create one-hot labels
    std::vector<float> h_labels(batch * 10, 0.0f);
    h_labels[0 * 10 + 3] = 1.0f;  // sample 0 -> class 3
    h_labels[1 * 10 + 7] = 1.0f;  // sample 1 -> class 7
    h_labels[2 * 10 + 1] = 1.0f;  // sample 2 -> class 1
    h_labels[3 * 10 + 9] = 1.0f;  // sample 3 -> class 9

    HostParams hp;
    init_params(hp, 42);

    GpuResult result;
    run_forward_backward_gpu(h_input.data(), h_labels.data(), batch, hp, result);

    // Verify gradient shapes (by checking sizes of downloaded arrays)
    struct ShapeCheck {
        const float* data;
        int size;
        const char* name;
    };
    std::vector<ShapeCheck> checks = {
        {result.grad_conv1_w.data(), 150,   "grad_conv1_w"},
        {result.grad_conv1_b.data(), 6,     "grad_conv1_b"},
        {result.grad_conv2_w.data(), 2400,  "grad_conv2_w"},
        {result.grad_conv2_b.data(), 16,    "grad_conv2_b"},
        {result.grad_fc1_w.data(),   30720, "grad_fc1_w"},
        {result.grad_fc1_b.data(),   120,   "grad_fc1_b"},
        {result.grad_fc2_w.data(),   10080, "grad_fc2_w"},
        {result.grad_fc2_b.data(),   84,    "grad_fc2_b"},
        {result.grad_fc3_w.data(),   840,   "grad_fc3_w"},
        {result.grad_fc3_b.data(),   10,    "grad_fc3_b"},
    };

    for (auto& chk : checks) {
        // Check all values are finite
        for (int i = 0; i < chk.size; i++) {
            if (!std::isfinite(chk.data[i])) {
                throw std::runtime_error(std::string(chk.name) +
                    " has non-finite value at index " + std::to_string(i));
            }
        }
        // Check not all zeros (gradients should be non-trivial)
        float max_abs = 0.0f;
        for (int i = 0; i < chk.size; i++) {
            max_abs = fmaxf(max_abs, fabsf(chk.data[i]));
        }
        if (max_abs < 1e-12f) {
            throw std::runtime_error(std::string(chk.name) +
                " is all zeros — backward pass may not be connected");
        }
    }

    // Verify forward pass produced valid probabilities
    for (int b = 0; b < batch; b++) {
        float sum = 0.0f;
        for (int c = 0; c < 10; c++) {
            sum += result.probs[b * 10 + c];
        }
        if (fabsf(sum - 1.0f) > 1e-4f) {
            throw std::runtime_error("Forward pass probs don't sum to 1.0");
        }
    }
}

// ============================================================
// Test 2: Sampled numerical gradient check
// For each parameter group, perturb 5 random params and compare
// ∂L/∂p ≈ (L(p+ε) - L(p-ε)) / (2ε)
// ============================================================
TEST_CASE(numerical_gradient_check) {
    const int batch = 2;
    const float eps = 1e-3f;
    const float rel_tol = 1e-2f;  // relaxed for deep chain
    const int samples_per_layer = 5;

    std::vector<float> h_input(batch * 1 * 28 * 28);
    fill_random(h_input.data(), (int)h_input.size(), 200, 0.0f, 1.0f);

    std::vector<float> h_labels(batch * 10, 0.0f);
    h_labels[0 * 10 + 2] = 1.0f;
    h_labels[1 * 10 + 5] = 1.0f;

    HostParams hp;
    init_params(hp, 42);

    // Get analytical gradients
    GpuResult result;
    run_forward_backward_gpu(h_input.data(), h_labels.data(), batch, hp, result);

    // Map gradient results to param infos
    struct GradInfo {
        float* grad_ptr;
        float* param_ptr;
        int    size;
        const char* name;
    };
    std::vector<GradInfo> grad_infos = {
        {result.grad_conv1_w.data(), hp.conv1_w.data(), 150,   "conv1_w"},
        {result.grad_conv1_b.data(), hp.conv1_b.data(), 6,     "conv1_b"},
        {result.grad_conv2_w.data(), hp.conv2_w.data(), 2400,  "conv2_w"},
        {result.grad_conv2_b.data(), hp.conv2_b.data(), 16,    "conv2_b"},
        {result.grad_fc1_w.data(),   hp.fc1_w.data(),   30720, "fc1_w"},
        {result.grad_fc1_b.data(),   hp.fc1_b.data(),   120,   "fc1_b"},
        {result.grad_fc2_w.data(),   hp.fc2_w.data(),   10080, "fc2_w"},
        {result.grad_fc2_b.data(),   hp.fc2_b.data(),   84,    "fc2_b"},
        {result.grad_fc3_w.data(),   hp.fc3_w.data(),   840,   "fc3_w"},
        {result.grad_fc3_b.data(),   hp.fc3_b.data(),   10,    "fc3_b"},
    };

    std::mt19937 rng(12345);
    int total_checked = 0;
    int total_passed = 0;

    for (auto& gi : grad_infos) {
        // Pick random indices to check
        std::vector<int> indices(gi.size);
        for (int i = 0; i < gi.size; i++) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), rng);
        int n_check = std::min(samples_per_layer, gi.size);

        for (int s = 0; s < n_check; s++) {
            int idx = indices[s];
            float orig = gi.param_ptr[idx];

            // L(p + eps)
            gi.param_ptr[idx] = orig + eps;
            CpuActivations a_plus;
            lenet_forward_cpu(h_input.data(), a_plus, batch,
                              hp.conv1_w.data(), hp.conv1_b.data(),
                              hp.conv2_w.data(), hp.conv2_b.data(),
                              hp.fc1_w.data(),   hp.fc1_b.data(),
                              hp.fc2_w.data(),   hp.fc2_b.data(),
                              hp.fc3_w.data(),   hp.fc3_b.data());
            float loss_plus = cross_entropy_loss_cpu(a_plus.probs.data(),
                                                     h_labels.data(), batch, 10);

            // L(p - eps)
            gi.param_ptr[idx] = orig - eps;
            CpuActivations a_minus;
            lenet_forward_cpu(h_input.data(), a_minus, batch,
                              hp.conv1_w.data(), hp.conv1_b.data(),
                              hp.conv2_w.data(), hp.conv2_b.data(),
                              hp.fc1_w.data(),   hp.fc1_b.data(),
                              hp.fc2_w.data(),   hp.fc2_b.data(),
                              hp.fc3_w.data(),   hp.fc3_b.data());
            float loss_minus = cross_entropy_loss_cpu(a_minus.probs.data(),
                                                      h_labels.data(), batch, 10);

            // Restore
            gi.param_ptr[idx] = orig;

            // Numerical gradient (averaged over batch, so divide analytical by batch too)
            float numerical_grad = (loss_plus - loss_minus) / (2.0f * eps);
            float analytical_grad = gi.grad_ptr[idx] / batch;  // our backward produces sum, not mean

            float abs_diff = fabsf(numerical_grad - analytical_grad);
            float denom = fmaxf(fabsf(numerical_grad) + fabsf(analytical_grad), 1e-8f);
            float rel_error = abs_diff / denom;

            total_checked++;
            if (rel_error < rel_tol || abs_diff < 1e-4f) {
                total_passed++;
            } else {
                fprintf(stderr, "  [WARN] %s[%d]: numerical=%.6f analytical=%.6f rel_err=%.4f\n",
                        gi.name, idx, numerical_grad, analytical_grad, rel_error);
            }
        }
    }

    float pass_rate = (float)total_passed / total_checked;
    if (pass_rate < 0.8f) {
        throw std::runtime_error("Numerical gradient check: only " +
            std::to_string(total_passed) + "/" + std::to_string(total_checked) +
            " passed (need 80%+)");
    }
}

// ============================================================
// Test 3: Loss decrease after SGD step
// Run forward, backward, SGD update, forward again -> loss should decrease
// ============================================================
TEST_CASE(loss_decreases_after_sgd) {
    const int batch = 4;
    const float lr = 0.01f;

    std::vector<float> h_input(batch * 1 * 28 * 28);
    fill_random(h_input.data(), (int)h_input.size(), 300, 0.0f, 1.0f);

    std::vector<float> h_labels(batch * 10, 0.0f);
    h_labels[0 * 10 + 0] = 1.0f;
    h_labels[1 * 10 + 3] = 1.0f;
    h_labels[2 * 10 + 7] = 1.0f;
    h_labels[3 * 10 + 9] = 1.0f;

    HostParams hp;
    init_params(hp, 42);

    // Upload input and labels
    float* d_input;
    float* d_labels;
    CUDA_CHECK(cudaMalloc(&d_input,  (size_t)batch * 784 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, (size_t)batch * 10  * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input,  h_input.data(),  batch * 784 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), batch * 10  * sizeof(float), cudaMemcpyHostToDevice));

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
    LeNetGrads grads;
    alloc_grads(grads);
    LeNetGradActivations ga;
    alloc_grad_activations(ga, batch);

    // Forward pass 1 — compute initial loss
    lenet_forward(d_input, params, act, batch);

    float* d_losses;
    CUDA_CHECK(cudaMalloc(&d_losses, batch * sizeof(float)));
    {
        int blocks = (batch + 255) / 256;
        cross_entropy_loss<<<blocks, 256>>>(act.probs, d_labels, d_losses, batch, 10);
        KERNEL_CHECK();
    }
    std::vector<float> h_losses(batch);
    CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, batch * sizeof(float), cudaMemcpyDeviceToHost));
    float loss1 = 0.0f;
    for (int b = 0; b < batch; b++) loss1 += h_losses[b];
    loss1 /= batch;

    // Backward pass
    lenet_backward(d_input, d_labels, params, act, grads, ga, batch);

    // SGD update on all parameters
    const int threads = 256;
    auto sgd = [&](float* w, float* g, int n) {
        int blocks = (n + threads - 1) / threads;
        sgd_update<<<blocks, threads>>>(w, g, lr / batch, n);
        KERNEL_CHECK();
    };
    sgd(params.conv1_w, grads.conv1_w, 150);
    sgd(params.conv1_b, grads.conv1_b, 6);
    sgd(params.conv2_w, grads.conv2_w, 2400);
    sgd(params.conv2_b, grads.conv2_b, 16);
    sgd(params.fc1_w,   grads.fc1_w,   30720);
    sgd(params.fc1_b,   grads.fc1_b,   120);
    sgd(params.fc2_w,   grads.fc2_w,   10080);
    sgd(params.fc2_b,   grads.fc2_b,   84);
    sgd(params.fc3_w,   grads.fc3_w,   840);
    sgd(params.fc3_b,   grads.fc3_b,   10);

    // Forward pass 2 — compute new loss
    lenet_forward(d_input, params, act, batch);
    {
        int blocks = (batch + 255) / 256;
        cross_entropy_loss<<<blocks, 256>>>(act.probs, d_labels, d_losses, batch, 10);
        KERNEL_CHECK();
    }
    CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, batch * sizeof(float), cudaMemcpyDeviceToHost));
    float loss2 = 0.0f;
    for (int b = 0; b < batch; b++) loss2 += h_losses[b];
    loss2 /= batch;

    // Loss should decrease
    if (loss2 >= loss1) {
        throw std::runtime_error("Loss did not decrease after SGD step: " +
            std::to_string(loss1) + " -> " + std::to_string(loss2));
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_losses));
    free_grad_activations(ga);
    free_grads(grads);
    free_activations(act);
    free_params(params);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_labels));
}

// ============================================================
// Test 4: Gradient magnitude sanity check
// Verify gradients are in reasonable ranges (not exploding/vanishing)
// ============================================================
TEST_CASE(gradient_magnitude_sanity) {
    const int batch = 4;

    std::vector<float> h_input(batch * 1 * 28 * 28);
    fill_random(h_input.data(), (int)h_input.size(), 400, 0.0f, 1.0f);

    std::vector<float> h_labels(batch * 10, 0.0f);
    h_labels[0 * 10 + 4] = 1.0f;
    h_labels[1 * 10 + 2] = 1.0f;
    h_labels[2 * 10 + 8] = 1.0f;
    h_labels[3 * 10 + 0] = 1.0f;

    HostParams hp;
    init_params(hp, 42);

    GpuResult result;
    run_forward_backward_gpu(h_input.data(), h_labels.data(), batch, hp, result);

    struct GradCheck {
        const float* data;
        int size;
        const char* name;
        float max_expected;  // reasonable upper bound for max|grad|
    };

    // These bounds are generous — just checking for explosions
    std::vector<GradCheck> checks = {
        {result.grad_conv1_w.data(), 150,   "grad_conv1_w", 100.0f},
        {result.grad_conv1_b.data(), 6,     "grad_conv1_b", 100.0f},
        {result.grad_conv2_w.data(), 2400,  "grad_conv2_w", 100.0f},
        {result.grad_conv2_b.data(), 16,    "grad_conv2_b", 100.0f},
        {result.grad_fc1_w.data(),   30720, "grad_fc1_w",   100.0f},
        {result.grad_fc1_b.data(),   120,   "grad_fc1_b",   100.0f},
        {result.grad_fc2_w.data(),   10080, "grad_fc2_w",   100.0f},
        {result.grad_fc2_b.data(),   84,    "grad_fc2_b",   100.0f},
        {result.grad_fc3_w.data(),   840,   "grad_fc3_w",   100.0f},
        {result.grad_fc3_b.data(),   10,    "grad_fc3_b",   100.0f},
    };

    for (auto& chk : checks) {
        float max_abs = 0.0f;
        float min_abs = FLT_MAX;
        float sum_sq = 0.0f;

        for (int i = 0; i < chk.size; i++) {
            float val = fabsf(chk.data[i]);
            max_abs = fmaxf(max_abs, val);
            if (val > 0.0f) min_abs = fminf(min_abs, val);
            sum_sq += chk.data[i] * chk.data[i];
        }
        float rms = sqrtf(sum_sq / chk.size);

        // Check: not exploding
        if (max_abs > chk.max_expected) {
            throw std::runtime_error(std::string(chk.name) +
                " has exploding gradient: max|grad|=" + std::to_string(max_abs));
        }

        // Check: not all vanished (RMS should be above some tiny threshold)
        if (rms < 1e-10f) {
            throw std::runtime_error(std::string(chk.name) +
                " has vanishing gradient: RMS=" + std::to_string(rms));
        }

        // Check: all finite
        for (int i = 0; i < chk.size; i++) {
            if (!std::isfinite(chk.data[i])) {
                throw std::runtime_error(std::string(chk.name) +
                    " has non-finite gradient at index " + std::to_string(i));
            }
        }
    }
}

int main() {
    return RUN_ALL_TESTS();
}
