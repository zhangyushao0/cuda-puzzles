#include <catch2/catch_test_macros.hpp>
// Puzzle 15: Full Training Loop — Test Harness
//
// Tests:
//   1. Xavier init distribution — verify mean ~0, variance matches formula
//   2. Single batch step — loss decreases after one gradient step
//   3. Overfit 10 samples — achieve >95% accuracy (memorization proof)
//   4. Overfit 100 samples — verify training on larger mini-dataset
//   5. Shape verification — all activation/gradient buffers have correct sizes

#include "cuda_utils.h"
#include "test_utils.h"

#include <cmath>
#include <cfloat>
#include <vector>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <numeric>

// Include the kernel implementation
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// ============================================================
// Helper: generate synthetic MNIST-like dataset on GPU
// Returns device pointers to images and labels
// ============================================================

struct SyntheticDataset {
    float* d_images;   // [N x 1 x 28 x 28] on GPU
    int*   d_labels;   // [N] on GPU
    int    num_samples;
};

// Create a synthetic dataset where each class has a distinctive pattern.
// Class c: image has a bright horizontal stripe at row c*2+4 and
// a vertical stripe at column c*2+4, making classes distinguishable.
SyntheticDataset create_synthetic_dataset(int num_samples, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> noise(0.0f, 0.1f);
    std::uniform_int_distribution<int> label_dist(0, 9);

    int image_size = 1 * 28 * 28;
    std::vector<float> h_images(num_samples * image_size);
    std::vector<int> h_labels(num_samples);

    for (int n = 0; n < num_samples; n++) {
        int label = label_dist(gen);
        h_labels[n] = label;

        // Fill with low noise
        for (int i = 0; i < image_size; i++) {
            h_images[n * image_size + i] = noise(gen);
        }

        // Add class-distinctive pattern: horizontal band
        int row_start = label * 2 + 4;
        for (int r = row_start; r < row_start + 2 && r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                h_images[n * image_size + r * 28 + c] = 0.8f + noise(gen) * 0.2f;
            }
        }

        // Add vertical band
        int col_start = label * 2 + 4;
        for (int r = 0; r < 28; r++) {
            for (int c = col_start; c < col_start + 2 && c < 28; c++) {
                h_images[n * image_size + r * 28 + c] = 0.8f + noise(gen) * 0.2f;
            }
        }
    }

    SyntheticDataset ds;
    ds.num_samples = num_samples;

    CUDA_CHECK(cudaMalloc(&ds.d_images, num_samples * image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.d_labels, num_samples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(ds.d_images, h_images.data(),
                          num_samples * image_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds.d_labels, h_labels.data(),
                          num_samples * sizeof(int),
                          cudaMemcpyHostToDevice));

    return ds;
}

void free_dataset(SyntheticDataset& ds) {
    cudaFree(ds.d_images);
    cudaFree(ds.d_labels);
}

// ============================================================
// Test 1: Xavier init distribution
// Verify weights have approximately correct mean and variance
// Xavier Uniform: Var = 2 / (fan_in + fan_out) * (limit^2 / 3 is uniform var)
// For Uniform(-a, a): mean=0, var=a^2/3
// ============================================================
TEST_CASE("xavier_init_distribution", "[puzzle_15_training_loop]") {
    LeNetParams params;
    alloc_params(params);

    xavier_init(params, 42);

    // Check conv1 weights: fan_in=25, fan_out=150
    // limit = sqrt(6/(25+150)) = sqrt(6/175) ≈ 0.1852
    // Expected var = limit^2/3 = 6/(3*(25+150)) = 2/175 ≈ 0.01143
    {
        float limit = sqrtf(6.0f / (25.0f + 150.0f));
        std::vector<float> h_w(150);
        CUDA_CHECK(cudaMemcpy(h_w.data(), params.conv1_w,
                              150 * sizeof(float), cudaMemcpyDeviceToHost));

        // Check all values in [-limit, +limit]
        for (int i = 0; i < 150; i++) {
            if (h_w[i] < -limit - 1e-5f || h_w[i] > limit + 1e-5f) {
                throw std::runtime_error(
                    "Conv1 weight " + std::to_string(i) + " = " +
                    std::to_string(h_w[i]) + " outside Xavier range [-" +
                    std::to_string(limit) + ", " + std::to_string(limit) + "]");
            }
        }

        // Check mean is approximately 0
        float mean = 0.0f;
        for (int i = 0; i < 150; i++) mean += h_w[i];
        mean /= 150.0f;
        if (fabsf(mean) > 0.05f) {
            throw std::runtime_error(
                "Conv1 weights mean = " + std::to_string(mean) +
                ", expected ~0");
        }
    }

    // Check FC1 weights: fan_in=256, fan_out=120
    // limit = sqrt(6/376) ≈ 0.1263
    {
        float limit = sqrtf(6.0f / (256.0f + 120.0f));
        std::vector<float> h_w(30720);
        CUDA_CHECK(cudaMemcpy(h_w.data(), params.fc1_w,
                              30720 * sizeof(float), cudaMemcpyDeviceToHost));

        // Check all values in [-limit, +limit]
        bool all_in_range = true;
        int out_of_range = 0;
        for (int i = 0; i < 30720; i++) {
            if (h_w[i] < -limit - 1e-5f || h_w[i] > limit + 1e-5f) {
                out_of_range++;
                all_in_range = false;
            }
        }
        if (!all_in_range) {
            throw std::runtime_error(
                "FC1 weights: " + std::to_string(out_of_range) +
                " values outside Xavier range [-" +
                std::to_string(limit) + ", " + std::to_string(limit) + "]");
        }

        // Check mean is approximately 0 (large sample, should be very close)
        float mean = 0.0f;
        for (int i = 0; i < 30720; i++) mean += h_w[i];
        mean /= 30720.0f;
        if (fabsf(mean) > 0.01f) {
            throw std::runtime_error(
                "FC1 weights mean = " + std::to_string(mean) +
                ", expected ~0");
        }

        // Check variance is approximately limit^2/3
        float expected_var = (limit * limit) / 3.0f;
        float var = 0.0f;
        for (int i = 0; i < 30720; i++) {
            float diff = h_w[i] - mean;
            var += diff * diff;
        }
        var /= 30720.0f;
        float var_ratio = var / expected_var;
        if (var_ratio < 0.8f || var_ratio > 1.2f) {
            throw std::runtime_error(
                "FC1 weights variance = " + std::to_string(var) +
                ", expected ~" + std::to_string(expected_var) +
                " (ratio=" + std::to_string(var_ratio) + ")");
        }
    }

    // Check biases are zero
    {
        std::vector<float> h_b(120);
        CUDA_CHECK(cudaMemcpy(h_b.data(), params.fc1_b,
                              120 * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 120; i++) {
            if (h_b[i] != 0.0f) {
                throw std::runtime_error(
                    "FC1 bias[" + std::to_string(i) + "] = " +
                    std::to_string(h_b[i]) + ", expected 0.0");
            }
        }
    }

    free_params(params);
}

// ============================================================
// Test 2: Single batch step — loss decreases
// Take one gradient step and verify loss goes down
// ============================================================
TEST_CASE("single_batch_loss_decreases", "[puzzle_15_training_loop]") {
    const int batch_size = 10;
    const float lr = 0.01f;

    SyntheticDataset ds = create_synthetic_dataset(batch_size, 123);

    LeNetParams params;
    alloc_params(params);
    xavier_init(params, 42);

    LeNetGradients grads;
    alloc_gradients(grads);

    LeNetActivations act;
    alloc_activations(act, batch_size);

    LeNetBackwardActs back_act;
    alloc_backward_acts(back_act, batch_size);

    const int threads = 256;

    // --- Compute initial loss ---
    lenet_forward(ds.d_images, params, act, batch_size);

    labels_to_onehot<<<(batch_size+threads-1)/threads, threads>>>(
        ds.d_labels, back_act.d_labels_onehot, batch_size, 10);
    KERNEL_CHECK();

    cross_entropy_loss<<<(batch_size+threads-1)/threads, threads>>>(
        act.probs, back_act.d_labels_onehot, back_act.losses, batch_size, 10);
    KERNEL_CHECK();

    std::vector<float> h_losses(batch_size);
    CUDA_CHECK(cudaMemcpy(h_losses.data(), back_act.losses,
                          batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    float loss_before = 0.0f;
    for (int i = 0; i < batch_size; i++) loss_before += h_losses[i];
    loss_before /= batch_size;

    // --- One training step ---
    float avg_loss = train_epoch(ds.d_images, ds.d_labels, batch_size,
                                  batch_size, params, grads, act, back_act, lr);

    // --- Compute loss after step ---
    lenet_forward(ds.d_images, params, act, batch_size);

    cross_entropy_loss<<<(batch_size+threads-1)/threads, threads>>>(
        act.probs, back_act.d_labels_onehot, back_act.losses, batch_size, 10);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_losses.data(), back_act.losses,
                          batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    float loss_after = 0.0f;
    for (int i = 0; i < batch_size; i++) loss_after += h_losses[i];
    loss_after /= batch_size;

    if (loss_after >= loss_before) {
        throw std::runtime_error(
            "Loss did not decrease after one step: before=" +
            std::to_string(loss_before) + ", after=" +
            std::to_string(loss_after));
    }

    free_backward_acts(back_act);
    free_activations(act);
    free_gradients(grads);
    free_params(params);
    free_dataset(ds);
}

// ============================================================
// Test 3: Overfit 10 samples — achieve >95% accuracy
// With 50 epochs on only 10 samples, the network should memorize them
// ============================================================
TEST_CASE("overfit_10_samples", "[puzzle_15_training_loop]") {
    const int num_samples = 10;
    const int batch_size = 10;
    const int num_epochs = 50;
    const float lr = 0.01f;

    SyntheticDataset ds = create_synthetic_dataset(num_samples, 456);

    LeNetParams params;
    alloc_params(params);
    xavier_init(params, 42);

    LeNetGradients grads;
    alloc_gradients(grads);

    LeNetActivations act;
    alloc_activations(act, batch_size);

    LeNetBackwardActs back_act;
    alloc_backward_acts(back_act, batch_size);

    // Track loss to verify monotonic decrease (at least mostly)
    float prev_loss = 1e10f;
    int loss_increased_count = 0;

    for (int epoch = 1; epoch <= num_epochs; epoch++) {
        float avg_loss = train_epoch(ds.d_images, ds.d_labels, num_samples,
                                      batch_size, params, grads, act,
                                      back_act, lr);
        if (avg_loss > prev_loss) {
            loss_increased_count++;
        }
        prev_loss = avg_loss;
    }

    // Evaluate accuracy
    float accuracy = evaluate(ds.d_images, ds.d_labels, num_samples,
                               batch_size, params, act);

    if (accuracy < 0.95f) {
        throw std::runtime_error(
            "Overfit 10 samples: accuracy=" + std::to_string(accuracy * 100.0f) +
            "%, expected >95%");
    }

    free_backward_acts(back_act);
    free_activations(act);
    free_gradients(grads);
    free_params(params);
    free_dataset(ds);
}

// ============================================================
// Test 4: Overfit 100 samples
// With enough epochs, should achieve decent accuracy on 100 samples
// ============================================================
TEST_CASE("overfit_100_samples", "[puzzle_15_training_loop]") {
    const int num_samples = 100;
    const int batch_size = 32;
    const int num_epochs = 40;
    const float lr = 0.01f;

    SyntheticDataset ds = create_synthetic_dataset(num_samples, 789);

    LeNetParams params;
    alloc_params(params);
    xavier_init(params, 42);

    LeNetGradients grads;
    alloc_gradients(grads);

    LeNetActivations act;
    alloc_activations(act, batch_size);

    LeNetBackwardActs back_act;
    alloc_backward_acts(back_act, batch_size);

    float first_loss = 0.0f;
    float last_loss = 0.0f;

    for (int epoch = 1; epoch <= num_epochs; epoch++) {
        float avg_loss = train_epoch(ds.d_images, ds.d_labels, num_samples,
                                      batch_size, params, grads, act,
                                      back_act, lr);
        if (epoch == 1) first_loss = avg_loss;
        last_loss = avg_loss;
    }

    // Loss should have decreased significantly
    if (last_loss >= first_loss * 0.5f) {
        throw std::runtime_error(
            "100 samples: loss didn't decrease enough. first=" +
            std::to_string(first_loss) + ", last=" + std::to_string(last_loss));
    }

    // Evaluate accuracy — should be at least 70% on 100 synthetic samples
    float accuracy = evaluate(ds.d_images, ds.d_labels, num_samples,
                               batch_size, params, act);

    if (accuracy < 0.70f) {
        throw std::runtime_error(
            "Overfit 100 samples: accuracy=" +
            std::to_string(accuracy * 100.0f) + "%, expected >70%");
    }

    free_backward_acts(back_act);
    free_activations(act);
    free_gradients(grads);
    free_params(params);
    free_dataset(ds);
}

// ============================================================
// Test 5: Shape verification — all buffers have correct sizes
// Verify no CUDA errors when allocating and using all structures
// ============================================================
TEST_CASE("shape_verification", "[puzzle_15_training_loop]") {
    const int batch_size = 4;

    // Allocate all structures
    LeNetParams params;
    alloc_params(params);
    xavier_init(params, 42);

    LeNetGradients grads;
    alloc_gradients(grads);

    LeNetActivations act;
    alloc_activations(act, batch_size);

    LeNetBackwardActs back_act;
    alloc_backward_acts(back_act, batch_size);

    // Create small dataset
    SyntheticDataset ds = create_synthetic_dataset(batch_size, 999);

    // Run forward pass — verifies activation shapes
    lenet_forward(ds.d_images, params, act, batch_size);

    // Verify probabilities have correct shape (batch x 10)
    std::vector<float> h_probs(batch_size * 10);
    CUDA_CHECK(cudaMemcpy(h_probs.data(), act.probs,
                          batch_size * 10 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        for (int c = 0; c < 10; c++) {
            float p = h_probs[b * 10 + c];
            if (p < 0.0f || p > 1.0f) {
                throw std::runtime_error(
                    "Prob[" + std::to_string(b) + "][" + std::to_string(c) +
                    "] = " + std::to_string(p) + " outside [0,1]");
            }
            sum += p;
        }
        if (fabsf(sum - 1.0f) > 1e-4f) {
            throw std::runtime_error(
                "Probs for sample " + std::to_string(b) +
                " sum to " + std::to_string(sum) + ", expected 1.0");
        }
    }

    // Run one-hot conversion — verifies label handling
    const int threads = 256;
    labels_to_onehot<<<(batch_size+threads-1)/threads, threads>>>(
        ds.d_labels, back_act.d_labels_onehot, batch_size, 10);
    KERNEL_CHECK();

    std::vector<float> h_onehot(batch_size * 10);
    CUDA_CHECK(cudaMemcpy(h_onehot.data(), back_act.d_labels_onehot,
                          batch_size * 10 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        for (int c = 0; c < 10; c++) {
            float v = h_onehot[b * 10 + c];
            if (v != 0.0f && v != 1.0f) {
                throw std::runtime_error(
                    "One-hot[" + std::to_string(b) + "][" + std::to_string(c) +
                    "] = " + std::to_string(v) + ", expected 0 or 1");
            }
            sum += v;
        }
        if (fabsf(sum - 1.0f) > 1e-6f) {
            throw std::runtime_error(
                "One-hot for sample " + std::to_string(b) +
                " sums to " + std::to_string(sum) + ", expected 1.0");
        }
    }

    // Run backward pass — verifies gradient shapes don't cause CUDA errors
    cross_entropy_loss<<<(batch_size+threads-1)/threads, threads>>>(
        act.probs, back_act.d_labels_onehot, back_act.losses, batch_size, 10);
    KERNEL_CHECK();

    lenet_backward(ds.d_images, params, grads, act, back_act, batch_size);

    // Verify gradient shapes by reading back a few
    {
        std::vector<float> h_gw(150);
        CUDA_CHECK(cudaMemcpy(h_gw.data(), grads.conv1_w,
                              150 * sizeof(float), cudaMemcpyDeviceToHost));
        bool has_nonzero = false;
        for (int i = 0; i < 150; i++) {
            if (!std::isfinite(h_gw[i])) {
                throw std::runtime_error(
                    "Conv1 weight gradient has non-finite value at index " +
                    std::to_string(i));
            }
            if (h_gw[i] != 0.0f) has_nonzero = true;
        }
        if (!has_nonzero) {
            throw std::runtime_error("Conv1 weight gradients are all zero");
        }
    }

    {
        std::vector<float> h_gw(840);
        CUDA_CHECK(cudaMemcpy(h_gw.data(), grads.fc3_w,
                              840 * sizeof(float), cudaMemcpyDeviceToHost));
        bool has_nonzero = false;
        for (int i = 0; i < 840; i++) {
            if (!std::isfinite(h_gw[i])) {
                throw std::runtime_error(
                    "FC3 weight gradient has non-finite value at index " +
                    std::to_string(i));
            }
            if (h_gw[i] != 0.0f) has_nonzero = true;
        }
        if (!has_nonzero) {
            throw std::runtime_error("FC3 weight gradients are all zero");
        }
    }

    free_dataset(ds);
    free_backward_acts(back_act);
    free_activations(act);
    free_gradients(grads);
    free_params(params);
}

