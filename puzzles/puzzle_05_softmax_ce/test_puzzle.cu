// Puzzle 05: Softmax + Cross-Entropy Loss — Test Harness
//
// Tests:
//   1. Softmax probabilities sum to 1.0
//   2. Numerical stability with large logits [1000, 1001, 1002]
//   3. Cross-entropy loss correctness
//   4. Backward gradient correctness (probs - labels)
//   5. Round-trip forward→backward pipeline consistency

#include "cuda_utils.h"
#include "test_utils.h"

#include <cmath>
#include <stdexcept>

// Include the kernel implementation
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// ============================================================
// CPU reference implementations
// ============================================================

void softmax_forward_cpu(const float* logits, float* probs,
                         int batch, int num_classes) {
    for (int b = 0; b < batch; b++) {
        int offset = b * num_classes;

        // Find max (for numerical stability)
        float max_val = logits[offset];
        for (int c = 1; c < num_classes; c++) {
            max_val = fmaxf(max_val, logits[offset + c]);
        }

        // Compute exp(z - max) and sum
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            probs[offset + c] = expf(logits[offset + c] - max_val);
            sum_exp += probs[offset + c];
        }

        // Normalize
        for (int c = 0; c < num_classes; c++) {
            probs[offset + c] /= sum_exp;
        }
    }
}

void cross_entropy_loss_cpu(const float* probs, const float* labels,
                            float* losses, int batch, int num_classes) {
    for (int b = 0; b < batch; b++) {
        int offset = b * num_classes;
        float loss = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            loss -= labels[offset + c] * logf(probs[offset + c] + 1e-10f);
        }
        losses[b] = loss;
    }
}

void softmax_ce_backward_cpu(const float* probs, const float* labels,
                             float* grad_logits, int batch, int num_classes) {
    for (int b = 0; b < batch; b++) {
        int offset = b * num_classes;
        for (int c = 0; c < num_classes; c++) {
            grad_logits[offset + c] = probs[offset + c] - labels[offset + c];
        }
    }
}

// ============================================================
// GPU helper: run softmax_forward on GPU and copy results back
// ============================================================
void run_softmax_forward_gpu(const float* h_logits, float* h_probs,
                             int batch, int num_classes) {
    float *d_logits, *d_probs;
    size_t data_bytes = batch * num_classes * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_logits, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_probs, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits, data_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    softmax_forward<<<blocks, threads>>>(d_logits, d_probs, batch, num_classes);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_probs, d_probs, data_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_probs));
}

// GPU helper: run cross_entropy_loss on GPU and copy results back
void run_cross_entropy_loss_gpu(const float* h_probs, const float* h_labels,
                                float* h_losses, int batch, int num_classes) {
    float *d_probs, *d_labels, *d_losses;
    size_t data_bytes = batch * num_classes * sizeof(float);
    size_t loss_bytes = batch * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_probs, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_labels, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_losses, loss_bytes));
    CUDA_CHECK(cudaMemcpy(d_probs, h_probs, data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels, data_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    cross_entropy_loss<<<blocks, threads>>>(d_probs, d_labels, d_losses,
                                            batch, num_classes);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_losses, d_losses, loss_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_losses));
}

// GPU helper: run softmax_ce_backward on GPU and copy results back
void run_softmax_ce_backward_gpu(const float* h_probs, const float* h_labels,
                                 float* h_grad, int batch, int num_classes) {
    float *d_probs, *d_labels, *d_grad;
    size_t data_bytes = batch * num_classes * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_probs, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_labels, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_probs, h_probs, data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels, data_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    softmax_ce_backward<<<blocks, threads>>>(d_probs, d_labels, d_grad,
                                             batch, num_classes);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_grad, d_grad, data_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_grad));
}

// ============================================================
// Test 1: Softmax probabilities sum to 1.0
// ============================================================
// Verifies the fundamental property: softmax output is a valid
// probability distribution (all values in [0,1], sum = 1).
TEST_CASE(softmax_probs_sum_to_one) {
    const int batch = 4, num_classes = 10;
    std::vector<float> h_logits(batch * num_classes);
    std::vector<float> h_probs(batch * num_classes, 0.0f);

    // Use varied logits to exercise the kernel
    fill_random(h_logits.data(), batch * num_classes, 500, -5.0f, 5.0f);

    run_softmax_forward_gpu(h_logits.data(), h_probs.data(), batch, num_classes);

    // Check each sample's probabilities sum to 1.0
    for (int b = 0; b < batch; b++) {
        float sum = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            float p = h_probs[b * num_classes + c];
            // Each probability must be non-negative
            if (p < 0.0f || p > 1.0f) {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "Sample %d, class %d: prob %.6f out of [0,1]", b, c, p);
                throw std::runtime_error(msg);
            }
            sum += p;
        }
        if (fabsf(sum - 1.0f) > 1e-5f) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "Sample %d: probs sum to %.8f, expected 1.0", b, sum);
            throw std::runtime_error(msg);
        }
    }
}

// ============================================================
// Test 2: Numerical stability with large logits
// ============================================================
// This is the CRITICAL test: naive softmax (without max-subtraction)
// will produce NaN/Inf for these logits. Only a stable implementation
// using the max-subtraction trick will pass.
TEST_CASE(softmax_numerical_stability) {
    const int batch = 2, num_classes = 3;

    // Large logits that would cause exp() overflow without max-subtraction
    float h_logits[] = {
        1000.0f, 1001.0f, 1002.0f,   // Sample 0: very large
        -1000.0f, -999.0f, -998.0f    // Sample 1: very negative (tests underflow)
    };

    std::vector<float> h_probs(batch * num_classes, 0.0f);

    run_softmax_forward_gpu(h_logits, h_probs.data(), batch, num_classes);

    // Expected (same as softmax([0, 1, 2]) due to max-subtraction):
    // exp(0) = 1.000, exp(1) = 2.718, exp(2) = 7.389
    // sum = 11.107
    // probs = [0.0900, 0.2447, 0.6652]
    float expected_probs[] = {
        0.0900306f, 0.244728f, 0.665241f,   // softmax([0,1,2])
        0.0900306f, 0.244728f, 0.665241f    // softmax([0,1,2]) — same after shift
    };

    // First: verify no NaN or Inf
    for (int i = 0; i < batch * num_classes; i++) {
        if (std::isnan(h_probs[i]) || std::isinf(h_probs[i])) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "NaN/Inf detected at index %d (value=%.6f). "
                     "Max-subtraction trick not implemented?", i, h_probs[i]);
            throw std::runtime_error(msg);
        }
    }

    // Verify probabilities sum to 1.0 for each sample
    for (int b = 0; b < batch; b++) {
        float sum = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum += h_probs[b * num_classes + c];
        }
        if (fabsf(sum - 1.0f) > 1e-5f) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "Large logits sample %d: probs sum to %.8f, expected 1.0", b, sum);
            throw std::runtime_error(msg);
        }
    }

    // Verify values match expected
    if (!check_array_close(h_probs.data(), expected_probs, batch * num_classes,
                           1e-4f, 1e-4f)) {
        throw std::runtime_error("Large logits: softmax values don't match expected");
    }
}

// ============================================================
// Test 3: Cross-entropy loss correctness
// ============================================================
// Verifies CE loss against CPU reference with random probabilities and
// one-hot labels.
TEST_CASE(cross_entropy_loss_correctness) {
    const int batch = 8, num_classes = 10;

    // Generate random logits, compute softmax on CPU to get valid probs
    std::vector<float> h_logits(batch * num_classes);
    std::vector<float> h_probs(batch * num_classes);
    std::vector<float> h_labels(batch * num_classes, 0.0f);

    fill_random(h_logits.data(), batch * num_classes, 501, -3.0f, 3.0f);
    softmax_forward_cpu(h_logits.data(), h_probs.data(), batch, num_classes);

    // Create one-hot labels: assign each sample a random true class
    std::mt19937 gen(502);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
    for (int b = 0; b < batch; b++) {
        int true_class = class_dist(gen);
        h_labels[b * num_classes + true_class] = 1.0f;
    }

    // CPU reference
    std::vector<float> expected_losses(batch, 0.0f);
    cross_entropy_loss_cpu(h_probs.data(), h_labels.data(),
                           expected_losses.data(), batch, num_classes);

    // GPU result
    std::vector<float> h_losses(batch, 0.0f);
    run_cross_entropy_loss_gpu(h_probs.data(), h_labels.data(),
                               h_losses.data(), batch, num_classes);

    if (!check_array_close(h_losses.data(), expected_losses.data(), batch,
                           1e-4f, 1e-4f)) {
        throw std::runtime_error("Cross-entropy loss mismatch vs CPU reference");
    }
}

// ============================================================
// Test 4: Backward gradient correctness
// ============================================================
// Verifies that grad_logits[b][c] = probs[b][c] - labels[b][c]
TEST_CASE(backward_gradient_correctness) {
    const int batch = 8, num_classes = 10;

    // Generate valid probabilities via softmax on CPU
    std::vector<float> h_logits(batch * num_classes);
    std::vector<float> h_probs(batch * num_classes);
    std::vector<float> h_labels(batch * num_classes, 0.0f);

    fill_random(h_logits.data(), batch * num_classes, 503, -2.0f, 2.0f);
    softmax_forward_cpu(h_logits.data(), h_probs.data(), batch, num_classes);

    // One-hot labels
    std::mt19937 gen(504);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
    for (int b = 0; b < batch; b++) {
        int true_class = class_dist(gen);
        h_labels[b * num_classes + true_class] = 1.0f;
    }

    // CPU reference
    std::vector<float> expected_grad(batch * num_classes, 0.0f);
    softmax_ce_backward_cpu(h_probs.data(), h_labels.data(),
                            expected_grad.data(), batch, num_classes);

    // GPU result
    std::vector<float> h_grad(batch * num_classes, 0.0f);
    run_softmax_ce_backward_gpu(h_probs.data(), h_labels.data(),
                                h_grad.data(), batch, num_classes);

    if (!check_array_close(h_grad.data(), expected_grad.data(),
                           batch * num_classes, 1e-5f, 1e-5f)) {
        throw std::runtime_error("Backward gradient mismatch vs CPU reference");
    }

    // Extra check: for the true class, gradient should be negative (prob < 1)
    // For wrong classes, gradient should be positive (prob > 0)
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < num_classes; c++) {
            float g = h_grad[b * num_classes + c];
            float label = h_labels[b * num_classes + c];
            if (label > 0.5f) {
                // True class: grad = prob - 1 < 0 (unless prob ≈ 1)
                if (g > 0.0f) {
                    char msg[256];
                    snprintf(msg, sizeof(msg),
                             "Sample %d, true class %d: grad %.6f should be <= 0",
                             b, c, g);
                    throw std::runtime_error(msg);
                }
            }
        }
    }
}

// ============================================================
// Test 5: Round-trip forward→backward consistency
// ============================================================
// Runs the full pipeline: logits → softmax → CE loss → backward
// Verifies all three kernels work together correctly.
TEST_CASE(round_trip_forward_backward) {
    const int batch = 16, num_classes = 10;
    const int total = batch * num_classes;

    std::vector<float> h_logits(total);
    std::vector<float> h_probs_gpu(total, 0.0f);
    std::vector<float> h_labels(total, 0.0f);
    std::vector<float> h_losses_gpu(batch, 0.0f);
    std::vector<float> h_grad_gpu(total, 0.0f);

    // CPU references
    std::vector<float> h_probs_cpu(total, 0.0f);
    std::vector<float> h_losses_cpu(batch, 0.0f);
    std::vector<float> h_grad_cpu(total, 0.0f);

    fill_random(h_logits.data(), total, 505, -4.0f, 4.0f);

    // One-hot labels
    std::mt19937 gen(506);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
    for (int b = 0; b < batch; b++) {
        int true_class = class_dist(gen);
        h_labels[b * num_classes + true_class] = 1.0f;
    }

    // CPU pipeline
    softmax_forward_cpu(h_logits.data(), h_probs_cpu.data(), batch, num_classes);
    cross_entropy_loss_cpu(h_probs_cpu.data(), h_labels.data(),
                           h_losses_cpu.data(), batch, num_classes);
    softmax_ce_backward_cpu(h_probs_cpu.data(), h_labels.data(),
                            h_grad_cpu.data(), batch, num_classes);

    // GPU pipeline
    run_softmax_forward_gpu(h_logits.data(), h_probs_gpu.data(),
                            batch, num_classes);
    run_cross_entropy_loss_gpu(h_probs_gpu.data(), h_labels.data(),
                               h_losses_gpu.data(), batch, num_classes);
    run_softmax_ce_backward_gpu(h_probs_gpu.data(), h_labels.data(),
                                h_grad_gpu.data(), batch, num_classes);

    // Verify softmax output
    if (!check_array_close(h_probs_gpu.data(), h_probs_cpu.data(), total,
                           1e-5f, 1e-5f)) {
        throw std::runtime_error("Round-trip: softmax output mismatch");
    }

    // Verify CE loss
    if (!check_array_close(h_losses_gpu.data(), h_losses_cpu.data(), batch,
                           1e-4f, 1e-4f)) {
        throw std::runtime_error("Round-trip: cross-entropy loss mismatch");
    }

    // Verify backward gradient
    if (!check_array_close(h_grad_gpu.data(), h_grad_cpu.data(), total,
                           1e-5f, 1e-5f)) {
        throw std::runtime_error("Round-trip: backward gradient mismatch");
    }

    // Verify gradient property: sum over classes ≈ 0 for each sample
    // (since Σ probs = 1 and Σ labels = 1, Σ (probs - labels) = 0)
    for (int b = 0; b < batch; b++) {
        float grad_sum = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            grad_sum += h_grad_gpu[b * num_classes + c];
        }
        if (fabsf(grad_sum) > 1e-5f) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "Round-trip: sample %d gradient sum = %.8f, expected ~0",
                     b, grad_sum);
            throw std::runtime_error(msg);
        }
    }
}

int main() {
    return RUN_ALL_TESTS();
}
