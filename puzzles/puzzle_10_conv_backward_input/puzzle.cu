// Puzzle 10: Conv2D Backward Pass — Input Gradient
//
// Implement a CUDA kernel to compute the input gradient for a Conv2D layer:
//   ∂L/∂input[b][c][h][w] = Σ_k Σ_fh Σ_fw grad_output[b][k][h-fh][w-fw] × W[k][c][fh][fw]
//
// This is the gradient that flows backward through a convolutional layer
// to the preceding layer during backpropagation.
//
// Conceptually, this is a "full" convolution of grad_output with a
// 180°-rotated filter. In practice, we use bounds checking instead
// of literal padding.
//
// See README.md for detailed explanation and hints.

#include <cuda_runtime.h>

// TODO: Implement the Conv2D backward input gradient kernel
//
// Parameters:
//   grad_output - upstream gradient, shape (batch, C_out, H_out, W_out), NCHW layout
//   filters     - filter weights,    shape (C_out, C_in, F, F),         NCHW layout
//   grad_input  - output: input gradient, shape (batch, C_in, H, W),   NCHW layout
//   batch       - number of samples in the batch
//   C_in        - number of input channels
//   H, W        - input spatial dimensions (height, width)
//   C_out       - number of output channels (number of filters)
//   F           - filter spatial size (square: F×F)
//
// Output dimensions (from forward pass):
//   H_out = H - F + 1
//   W_out = W - F + 1
//
// Steps:
//   1. Compute flat thread index: idx = blockIdx.x * blockDim.x + threadIdx.x
//   2. Bounds check: if idx >= batch * C_in * H * W, return
//   3. Decode 4D indices (b, c, h, w) from idx
//   4. Compute H_out, W_out
//   5. Initialize sum = 0
//   6. Triple loop over k, fh, fw:
//      - Compute oh = h - fh, ow = w - fw
//      - Bounds check: skip if oh < 0 || oh >= H_out || ow < 0 || ow >= W_out
//      - Accumulate grad_output[b][k][oh][ow] × filters[k][c][fh][fw]
//   7. Write result to grad_input[b][c][h][w]
//
// Memory indexing (NCHW):
//   grad_output[b][k][oh][ow] = grad_output[b*(C_out*H_out*W_out) + k*(H_out*W_out) + oh*W_out + ow]
//   filters[k][c][fh][fw]     = filters[k*(C_in*F*F) + c*(F*F) + fh*F + fw]
//   grad_input[b][c][h][w]    = grad_input[b*(C_in*H*W) + c*(H*W) + h*W + w]
__global__ void conv2d_backward_input(const float* grad_output,
                                       const float* filters,
                                       float* grad_input,
                                       int batch, int C_in, int H, int W,
                                       int C_out, int F) {
    // TODO: Your implementation here
}
