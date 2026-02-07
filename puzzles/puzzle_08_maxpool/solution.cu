// Puzzle 08: Max Pooling (Forward + Backward) — Reference Solution
//
// Forward: Select max from each 2×2 window, save index of max element.
// Backward: Route gradient only to the max position in each window.
//
// Pool size 2×2, stride 2 (non-overlapping). NCHW layout.
// Each thread computes one output element using a 1D grid.

#include <cuda_runtime.h>
#include <cfloat>

// Forward pass: max pooling with index saving
__global__ void maxpool_forward(const float* input, float* output,
                                int* max_indices,
                                int batch, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H / 2;
    int W_out = W / 2;
    int total = batch * C * H_out * W_out;

    if (idx >= total) return;

    // Decode 4D indices from flat index (NCHW order)
    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int c  = (idx / (W_out * H_out)) % C;
    int b  = idx / (W_out * H_out * C);

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

    output[idx] = max_val;
    max_indices[idx] = max_idx;
}

// Backward pass: route gradient to max position only
// IMPORTANT: grad_input must be zero-initialized before this kernel
__global__ void maxpool_backward(const float* grad_output, const int* max_indices,
                                 float* grad_input,
                                 int batch, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H / 2;
    int W_out = W / 2;
    int total = batch * C * H_out * W_out;

    if (idx >= total) return;

    // Decode 4D indices from flat index (NCHW order)
    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int c  = (idx / (W_out * H_out)) % C;
    int b  = idx / (W_out * H_out * C);

    int local_idx = max_indices[idx];
    int ph = local_idx / 2;
    int pw = local_idx % 2;

    int h_start = oh * 2;
    int w_start = ow * 2;

    int in_idx = b * (C * H * W) + c * (H * W)
               + (h_start + ph) * W + (w_start + pw);

    grad_input[in_idx] = grad_output[idx];
}
