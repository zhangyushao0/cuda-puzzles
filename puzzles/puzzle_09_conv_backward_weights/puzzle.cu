// Puzzle 09: Conv2D Backward — Weight & Bias Gradients
//
// Implement two CUDA kernels:
//
// Kernel 1: conv2d_backward_weights
//   ∂L/∂W[k][c][fh][fw] = Σ_b Σ_h Σ_w grad_out[b][k][h][w] × input[b][c][h+fh][w+fw]
//
// Kernel 2: conv2d_backward_bias
//   ∂L/∂bias[k] = Σ_b Σ_h Σ_w grad_out[b][k][h][w]
//
// The weight gradient is a correlation between the input and upstream gradient.
// The bias gradient sums the upstream gradient over batch and spatial dimensions.
//
// See README.md for detailed explanation and hints.

#include <cuda_runtime.h>

// TODO: Implement the Conv2D weight gradient kernel
//
// Parameters:
//   grad_output  - upstream gradient, shape (batch, C_out, H_out, W_out), NCHW
//   input        - input tensor,      shape (batch, C_in, H, W),          NCHW
//   grad_weights - weight gradient,   shape (C_out, C_in, F, F),          NCHW
//   batch        - number of samples in the batch
//   C_in         - number of input channels
//   H, W         - input spatial dimensions (height, width)
//   C_out        - number of output channels (number of filters)
//   F            - filter spatial size (square: F×F)
//
// Steps:
//   1. Compute flat thread index: idx = blockIdx.x * blockDim.x + threadIdx.x
//   2. Compute H_out = H - F + 1, W_out = W - F + 1; bounds-check idx < C_out*C_in*F*F
//   3. Decode 4D indices (k, c, fh, fw) from idx
//   4. Initialize sum = 0
//   5. Triple loop over b, h, w accumulating grad_output × input
//   6. Write result to grad_weights[k][c][fh][fw]
//
// Memory indexing (NCHW):
//   grad_output[b][k][h][w]      = grad_output[b*(C_out*H_out*W_out) + k*(H_out*W_out) + h*W_out + w]
//   input[b][c][h][w]            = input[b*(C_in*H*W) + c*(H*W) + h*W + w]
//   grad_weights[k][c][fh][fw]   = grad_weights[k*(C_in*F*F) + c*(F*F) + fh*F + fw]
__global__ void conv2d_backward_weights(const float* grad_output, const float* input,
                                         float* grad_weights,
                                         int batch, int C_in, int H, int W,
                                         int C_out, int F) {
    // TODO: Your implementation here
}

// TODO: Implement the Conv2D bias gradient kernel
//
// Parameters:
//   grad_output - upstream gradient, shape (batch, C_out, H_out, W_out), NCHW
//   grad_bias   - bias gradient,     size C_out
//   batch       - number of samples in the batch
//   C_out       - number of output channels
//   H_out, W_out - output spatial dimensions
//
// Steps:
//   1. Compute channel index: k = blockIdx.x * blockDim.x + threadIdx.x
//   2. Bounds check: if k >= C_out, return
//   3. Initialize sum = 0
//   4. Triple loop over b, h, w accumulating grad_output[b][k][h][w]
//   5. Write result to grad_bias[k]
__global__ void conv2d_backward_bias(const float* grad_output, float* grad_bias,
                                      int batch, int C_out, int H_out, int W_out) {
    // TODO: Your implementation here
}
