// Puzzle 08: Max Pooling (Forward + Backward) — Test Harness
//
// Tests:
//   1. Forward 4×4→2×2 hardcoded — hand-verifiable max selection and indices
//   2. Backward gradient routing — verifies gradient routes ONLY to max positions
//   3. LeNet Pool1 (24×24×6→12×12×6) and Pool2 (8×8×16→4×4×16) dimensions
//   4. All-equal edge case — all elements in a window are equal

#include "cuda_utils.h"
#include "test_utils.h"
#include <cfloat>

// Include the kernel implementation
// Build system compiles this file twice:
//   puzzle_08_test          -> includes puzzle.cu
//   puzzle_08_test_solution -> includes solution.cu
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// ============================================================
// CPU reference implementations for verification
// ============================================================

// CPU max pooling forward: 2×2, stride 2, NCHW layout
void maxpool_forward_cpu(const float* input, float* output, int* max_indices,
                         int batch, int C, int H, int W) {
    int H_out = H / 2;
    int W_out = W / 2;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    int h_start = oh * 2;
                    int w_start = ow * 2;

                    float max_val = -FLT_MAX;
                    int max_idx = 0;

                    for (int ph = 0; ph < 2; ph++) {
                        for (int pw = 0; pw < 2; pw++) {
                            int in_idx = b * (C * H * W) + c * (H * W)
                                       + (h_start + ph) * W + (w_start + pw);
                            float val = input[in_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = ph * 2 + pw;
                            }
                        }
                    }

                    int out_idx = b * (C * H_out * W_out) + c * (H_out * W_out)
                                + oh * W_out + ow;
                    output[out_idx] = max_val;
                    max_indices[out_idx] = max_idx;
                }
            }
        }
    }
}

// CPU max pooling backward: route gradients to max positions only
void maxpool_backward_cpu(const float* grad_output, const int* max_indices,
                          float* grad_input,
                          int batch, int C, int H, int W) {
    int H_out = H / 2;
    int W_out = W / 2;

    // Zero-initialize grad_input
    for (int i = 0; i < batch * C * H * W; i++) {
        grad_input[i] = 0.0f;
    }

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    int out_idx = b * (C * H_out * W_out) + c * (H_out * W_out)
                                + oh * W_out + ow;

                    int local_idx = max_indices[out_idx];
                    int ph = local_idx / 2;
                    int pw = local_idx % 2;

                    int h_start = oh * 2;
                    int w_start = ow * 2;

                    int in_idx = b * (C * H * W) + c * (H * W)
                               + (h_start + ph) * W + (w_start + pw);

                    grad_input[in_idx] = grad_output[out_idx];
                }
            }
        }
    }
}

// ============================================================
// GPU helpers
// ============================================================

// Run maxpool_forward on GPU and copy results back to host
void run_maxpool_forward_gpu(const float* h_input, float* h_output,
                             int* h_max_indices,
                             int batch, int C, int H, int W) {
    int H_out = H / 2;
    int W_out = W / 2;

    size_t input_bytes   = (size_t)batch * C * H * W * sizeof(float);
    size_t output_bytes  = (size_t)batch * C * H_out * W_out * sizeof(float);
    size_t indices_bytes = (size_t)batch * C * H_out * W_out * sizeof(int);

    float *d_input, *d_output;
    int *d_max_indices;

    CUDA_CHECK(cudaMalloc(&d_input,       input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output,      output_bytes));
    CUDA_CHECK(cudaMalloc(&d_max_indices, indices_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

    int total = batch * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    maxpool_forward<<<blocks, threads>>>(d_input, d_output, d_max_indices,
                                          batch, C, H, W);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_output,      d_output,      output_bytes,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_indices, d_max_indices, indices_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_max_indices));
}

// Run maxpool_backward on GPU and copy results back to host
void run_maxpool_backward_gpu(const float* h_grad_output, const int* h_max_indices,
                              float* h_grad_input,
                              int batch, int C, int H, int W) {
    int H_out = H / 2;
    int W_out = W / 2;

    size_t grad_output_bytes = (size_t)batch * C * H_out * W_out * sizeof(float);
    size_t indices_bytes     = (size_t)batch * C * H_out * W_out * sizeof(int);
    size_t grad_input_bytes  = (size_t)batch * C * H * W * sizeof(float);

    float *d_grad_output, *d_grad_input;
    int *d_max_indices;

    CUDA_CHECK(cudaMalloc(&d_grad_output, grad_output_bytes));
    CUDA_CHECK(cudaMalloc(&d_max_indices, indices_bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_input,  grad_input_bytes));

    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, grad_output_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max_indices, h_max_indices, indices_bytes,     cudaMemcpyHostToDevice));

    // Zero-initialize grad_input on device
    CUDA_CHECK(cudaMemset(d_grad_input, 0, grad_input_bytes));

    int total = batch * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    maxpool_backward<<<blocks, threads>>>(d_grad_output, d_max_indices, d_grad_input,
                                           batch, C, H, W);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_grad_input, d_grad_input, grad_input_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_max_indices));
    CUDA_CHECK(cudaFree(d_grad_input));
}

// ============================================================
// Test 1: Forward 4×4→2×2 hardcoded — hand-verifiable
// ============================================================
//
// Input (1×1×4×4):
//   ┌────┬────┬────┬────┐
//   │  1 │  3 │  5 │  2 │
//   ├────┼────┼────┼────┤
//   │  4 │  2 │  6 │  1 │
//   ├────┼────┼────┼────┤
//   │  7 │  8 │  3 │  9 │
//   ├────┼────┼────┼────┤
//   │  5 │  6 │  4 │  7 │
//   └────┴────┴────┴────┘
//
// Expected output (1×1×2×2):
//   max(1,3,4,2)=4   max(5,2,6,1)=6
//   max(7,8,5,6)=8   max(3,9,4,7)=9
//
// Expected max_indices:
//   4 is at (1,0) → index 2    6 is at (1,0) → index 2
//   8 is at (0,1) → index 1    9 is at (0,1) → index 1

TEST_CASE(maxpool_forward_4x4) {
    const int batch = 1, C = 1, H = 4, W = 4;
    const int H_out = 2, W_out = 2;

    float h_input[] = {
        1.0f, 3.0f, 5.0f, 2.0f,
        4.0f, 2.0f, 6.0f, 1.0f,
        7.0f, 8.0f, 3.0f, 9.0f,
        5.0f, 6.0f, 4.0f, 7.0f
    };

    float expected_output[] = {4.0f, 6.0f,
                               8.0f, 9.0f};

    int expected_indices[] = {2, 2,
                              1, 1};

    float h_output[4] = {0};
    int h_max_indices[4] = {-1, -1, -1, -1};

    run_maxpool_forward_gpu(h_input, h_output, h_max_indices,
                            batch, C, H, W);

    // Check output values
    if (!check_array_close(h_output, expected_output, H_out * W_out, 1e-5f, 1e-5f)) {
        throw std::runtime_error("4x4 forward: output values mismatch");
    }

    // Check max indices
    bool indices_ok = true;
    for (int i = 0; i < H_out * W_out; i++) {
        if (h_max_indices[i] != expected_indices[i]) {
            fprintf(stderr, "  [FAIL] max_indices[%d]: expected %d, got %d\n",
                    i, expected_indices[i], h_max_indices[i]);
            indices_ok = false;
        }
    }
    if (!indices_ok) {
        throw std::runtime_error("4x4 forward: max indices mismatch");
    }
}

// ============================================================
// Test 2: Backward gradient routing — verifies ONLY max positions get gradient
// ============================================================
//
// Uses the same 4×4 input as Test 1.
// grad_output = [[1.0, 2.0], [3.0, 4.0]]
//
// Expected grad_input (4×4):
//   ┌────┬────┬────┬────┐
//   │  0 │  0 │  0 │  0 │
//   ├────┼────┼────┼────┤
//   │ 1.0│  0 │ 2.0│  0 │    ← max positions get gradient
//   ├────┼────┼────┼────┤
//   │  0 │ 3.0│  0 │ 4.0│    ← max positions get gradient
//   ├────┼────┼────┼────┤
//   │  0 │  0 │  0 │  0 │
//   └────┴────┴────┴────┘

TEST_CASE(maxpool_backward_grad_routing) {
    const int batch = 1, C = 1, H = 4, W = 4;

    // First run forward to get max_indices
    float h_input[] = {
        1.0f, 3.0f, 5.0f, 2.0f,
        4.0f, 2.0f, 6.0f, 1.0f,
        7.0f, 8.0f, 3.0f, 9.0f,
        5.0f, 6.0f, 4.0f, 7.0f
    };

    float h_output[4] = {0};
    int h_max_indices[4] = {0};

    run_maxpool_forward_gpu(h_input, h_output, h_max_indices,
                            batch, C, H, W);

    // Now run backward
    float h_grad_output[] = {1.0f, 2.0f,
                             3.0f, 4.0f};

    float expected_grad_input[] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 2.0f, 0.0f,
        0.0f, 3.0f, 0.0f, 4.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };

    float h_grad_input[16] = {0};

    run_maxpool_backward_gpu(h_grad_output, h_max_indices, h_grad_input,
                             batch, C, H, W);

    // Check gradient values
    if (!check_array_close(h_grad_input, expected_grad_input, H * W, 1e-5f, 1e-5f)) {
        throw std::runtime_error("Backward: gradient routing mismatch");
    }

    // Verify sparsity: exactly 4 non-zero elements out of 16
    int nonzero_count = 0;
    for (int i = 0; i < H * W; i++) {
        if (h_grad_input[i] != 0.0f) nonzero_count++;
    }
    if (nonzero_count != 4) {
        fprintf(stderr, "  Expected 4 non-zero gradients, got %d\n", nonzero_count);
        throw std::runtime_error("Backward: wrong number of non-zero gradients");
    }
}

// ============================================================
// Test 3: LeNet Pool1 (24×24×6→12×12×6) and Pool2 (8×8×16→4×4×16)
// ============================================================

TEST_CASE(maxpool_lenet_dims) {
    bool pass = true;

    // Pool1: 24×24×6 → 12×12×6
    {
        const int batch = 1, C = 6, H = 24, W = 24;
        const int H_out = 12, W_out = 12;

        const int input_size  = batch * C * H * W;
        const int output_size = batch * C * H_out * W_out;

        std::vector<float> h_input(input_size);
        std::vector<float> h_output(output_size, 0.0f);
        std::vector<int>   h_max_indices(output_size, -1);
        std::vector<float> expected_output(output_size, 0.0f);
        std::vector<int>   expected_indices(output_size, -1);

        fill_random(h_input.data(), input_size, 800, -1.0f, 1.0f);

        // CPU reference
        maxpool_forward_cpu(h_input.data(), expected_output.data(),
                            expected_indices.data(), batch, C, H, W);

        // GPU
        run_maxpool_forward_gpu(h_input.data(), h_output.data(),
                                h_max_indices.data(), batch, C, H, W);

        if (!check_array_close(h_output.data(), expected_output.data(), output_size, 1e-5f, 1e-5f)) {
            fprintf(stderr, "  LeNet Pool1 forward (24x24x6 -> 12x12x6) output mismatch\n");
            pass = false;
        }

        // Verify backward at LeNet Pool1 dimensions
        std::vector<float> h_grad_output(output_size);
        fill_random(h_grad_output.data(), output_size, 801, -1.0f, 1.0f);

        std::vector<float> h_grad_input(input_size, 0.0f);
        std::vector<float> expected_grad_input(input_size, 0.0f);

        maxpool_backward_cpu(h_grad_output.data(), expected_indices.data(),
                             expected_grad_input.data(), batch, C, H, W);

        run_maxpool_backward_gpu(h_grad_output.data(), h_max_indices.data(),
                                 h_grad_input.data(), batch, C, H, W);

        if (!check_array_close(h_grad_input.data(), expected_grad_input.data(), input_size, 1e-5f, 1e-5f)) {
            fprintf(stderr, "  LeNet Pool1 backward (24x24x6) gradient mismatch\n");
            pass = false;
        }
    }

    // Pool2: 8×8×16 → 4×4×16
    {
        const int batch = 1, C = 16, H = 8, W = 8;
        const int H_out = 4, W_out = 4;

        const int input_size  = batch * C * H * W;
        const int output_size = batch * C * H_out * W_out;

        std::vector<float> h_input(input_size);
        std::vector<float> h_output(output_size, 0.0f);
        std::vector<int>   h_max_indices(output_size, -1);
        std::vector<float> expected_output(output_size, 0.0f);
        std::vector<int>   expected_indices(output_size, -1);

        fill_random(h_input.data(), input_size, 810, -1.0f, 1.0f);

        // CPU reference
        maxpool_forward_cpu(h_input.data(), expected_output.data(),
                            expected_indices.data(), batch, C, H, W);

        // GPU
        run_maxpool_forward_gpu(h_input.data(), h_output.data(),
                                h_max_indices.data(), batch, C, H, W);

        if (!check_array_close(h_output.data(), expected_output.data(), output_size, 1e-5f, 1e-5f)) {
            fprintf(stderr, "  LeNet Pool2 forward (8x8x16 -> 4x4x16) output mismatch\n");
            pass = false;
        }

        // Verify backward at LeNet Pool2 dimensions
        std::vector<float> h_grad_output(output_size);
        fill_random(h_grad_output.data(), output_size, 811, -1.0f, 1.0f);

        std::vector<float> h_grad_input(input_size, 0.0f);
        std::vector<float> expected_grad_input(input_size, 0.0f);

        maxpool_backward_cpu(h_grad_output.data(), expected_indices.data(),
                             expected_grad_input.data(), batch, C, H, W);

        run_maxpool_backward_gpu(h_grad_output.data(), h_max_indices.data(),
                                 h_grad_input.data(), batch, C, H, W);

        if (!check_array_close(h_grad_input.data(), expected_grad_input.data(), input_size, 1e-5f, 1e-5f)) {
            fprintf(stderr, "  LeNet Pool2 backward (8x8x16) gradient mismatch\n");
            pass = false;
        }
    }

    if (!pass) {
        throw std::runtime_error("LeNet pooling dimensions mismatch");
    }
}

// ============================================================
// Test 4: All-equal edge case — all elements in each window are equal
// ============================================================
//
// When all values in a 2×2 window are equal, the max is that value
// and the index should be deterministic (first encountered = index 0
// with the top-left-first scan order).
// The gradient should go to exactly one position per window.

TEST_CASE(maxpool_all_equal) {
    const int batch = 1, C = 1, H = 4, W = 4;
    const int H_out = 2, W_out = 2;

    // All values in each 2×2 window are the same
    float h_input[] = {
        5.0f, 5.0f, 3.0f, 3.0f,
        5.0f, 5.0f, 3.0f, 3.0f,
        7.0f, 7.0f, 1.0f, 1.0f,
        7.0f, 7.0f, 1.0f, 1.0f
    };

    // Expected: max of each window = the repeated value
    float expected_output[] = {5.0f, 3.0f,
                               7.0f, 1.0f};

    // When all equal, first element wins → index 0 (top-left)
    int expected_indices[] = {0, 0,
                              0, 0};

    float h_output[4] = {0};
    int h_max_indices[4] = {-1, -1, -1, -1};

    run_maxpool_forward_gpu(h_input, h_output, h_max_indices,
                            batch, C, H, W);

    // Check output values
    if (!check_array_close(h_output, expected_output, H_out * W_out, 1e-5f, 1e-5f)) {
        throw std::runtime_error("All-equal: output values mismatch");
    }

    // Check indices — all should be 0 (top-left)
    bool indices_ok = true;
    for (int i = 0; i < H_out * W_out; i++) {
        if (h_max_indices[i] != expected_indices[i]) {
            fprintf(stderr, "  [FAIL] all-equal max_indices[%d]: expected %d, got %d\n",
                    i, expected_indices[i], h_max_indices[i]);
            indices_ok = false;
        }
    }
    if (!indices_ok) {
        throw std::runtime_error("All-equal: max indices mismatch");
    }

    // Verify backward with all-equal: gradient goes to index 0 (top-left) of each window
    float h_grad_output[] = {1.0f, 2.0f,
                             3.0f, 4.0f};

    float expected_grad_input[] = {
        1.0f, 0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 4.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };

    float h_grad_input[16] = {0};

    run_maxpool_backward_gpu(h_grad_output, h_max_indices, h_grad_input,
                             batch, C, H, W);

    if (!check_array_close(h_grad_input, expected_grad_input, H * W, 1e-5f, 1e-5f)) {
        throw std::runtime_error("All-equal: backward gradient routing mismatch");
    }
}

int main() {
    return RUN_ALL_TESTS();
}
