// Puzzle 08: Max Pooling (Forward + Backward)
//
// Implement TWO CUDA kernels for 2×2 max pooling with stride 2:
//
//   1. maxpool_forward:  Selects max from each 2×2 window, saves max indices
//   2. maxpool_backward: Routes gradient ONLY to max position in each window
//
// See README.md for detailed explanation and hints.

#include <cuda_runtime.h>

// TODO: Implement the max pooling forward pass kernel
//
// Parameters:
//   input       - input tensor,  shape (batch, C, H, W),           NCHW layout
//   output      - output tensor, shape (batch, C, H/2, W/2),       NCHW layout
//   max_indices - index of max element in each 2×2 window (0-3),   same shape as output
//   batch       - number of samples in the batch
//   C           - number of channels
//   H, W        - input spatial dimensions (height, width)
//
// Pool size: 2×2, stride: 2 (non-overlapping windows)
// Output dimensions: H_out = H/2, W_out = W/2
//
// Steps:
//   1. Compute flat thread index: idx = blockIdx.x * blockDim.x + threadIdx.x
//   2. Compute H_out, W_out and total output count; bounds-check idx
//   3. Decode 4D indices (b, c, oh, ow) from idx
//   4. Compute window start: h_start = oh * 2, w_start = ow * 2
//   5. Find max value and its local index (0-3) in the 2×2 window
//   6. Write max value to output and local index to max_indices
//
// Local index encoding within 2×2 window:
//   ┌───┬───┐
//   │ 0 │ 1 │    local_idx = ph * 2 + pw
//   ├───┼───┤    where ph ∈ {0,1}, pw ∈ {0,1}
//   │ 2 │ 3 │
//   └───┴───┘
__global__ void maxpool_forward(const float* input, float* output,
                                int* max_indices,
                                int batch, int C, int H, int W) {
    // TODO: Your implementation here
}

// TODO: Implement the max pooling backward pass kernel
//
// Parameters:
//   grad_output  - upstream gradient, shape (batch, C, H/2, W/2),   NCHW layout
//   max_indices  - saved max positions from forward pass,            same shape as grad_output
//   grad_input   - output: gradient w.r.t. input, shape (batch, C, H, W), NCHW layout
//   batch        - number of samples in the batch
//   C            - number of channels
//   H, W         - INPUT spatial dimensions (not output!)
//
// The backward pass routes each gradient value to the position
// that was the maximum during the forward pass. All other positions
// in the 2×2 window receive zero gradient.
//
// Steps:
//   1. Compute flat thread index and bounds-check
//   2. Decode 4D indices (b, c, oh, ow) from idx
//   3. Read max_indices[idx] to get local position (0-3)
//   4. Decode local position: ph = local_idx / 2, pw = local_idx % 2
//   5. Compute input position: h = oh*2 + ph, w = ow*2 + pw
//   6. Write: grad_input[b][c][h][w] = grad_output[idx]
//
// IMPORTANT: grad_input must be zero-initialized before calling this kernel.
// Only the max positions receive non-zero gradients.
__global__ void maxpool_backward(const float* grad_output, const int* max_indices,
                                 float* grad_input,
                                 int batch, int C, int H, int W) {
    // TODO: Your implementation here
}
