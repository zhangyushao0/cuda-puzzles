// Puzzle 07: 2D Convolution (Forward Pass)
//
// Implement a CUDA kernel for a Conv2D forward pass:
//   out[b][k][oh][ow] = bias[k]
//                      + Σ_c Σ_fh Σ_fw input[b][c][oh+fh][ow+fw] × filter[k][c][fh][fw]
//
// This is the fundamental operation of every CNN — a sliding window
// dot product across spatial dimensions and input channels.
//
// See README.md for detailed explanation and hints.

#include <cuda_runtime.h>

// TODO: Implement the Conv2D forward pass kernel
//
// Parameters:
//   input   - input tensor,  shape (batch, C_in, H, W),       NCHW layout
//   filters - filter weights, shape (C_out, C_in, F, F),      NCHW layout
//   bias    - bias vector,   size C_out
//   output  - output tensor, shape (batch, C_out, H_out, W_out), NCHW layout
//   batch   - number of samples in the batch
//   C_in    - number of input channels
//   H, W    - input spatial dimensions (height, width)
//   C_out   - number of output channels (number of filters)
//   F       - filter spatial size (square: F×F)
//
// Output dimensions (no padding, stride=1):
//   H_out = H - F + 1
//   W_out = W - F + 1
//
// Steps:
//   1. Compute flat thread index: idx = blockIdx.x * blockDim.x + threadIdx.x
//   2. Compute H_out, W_out and total output count; bounds-check idx
//   3. Decode 4D indices (b, k, oh, ow) from idx
//   4. Initialize sum = bias[k]
//   5. Triple loop over c, fh, fw accumulating input × filter
//   6. Write result to output[b][k][oh][ow]
//
// Memory indexing (NCHW):
//   input[b][c][h][w]       = input[b*(C_in*H*W) + c*(H*W) + h*W + w]
//   filters[k][c][fh][fw]   = filters[k*(C_in*F*F) + c*(F*F) + fh*F + fw]
//   output[b][k][oh][ow]    = output[b*(C_out*H_out*W_out) + k*(H_out*W_out) + oh*W_out + ow]
__global__ void conv2d_forward(const float* input, const float* filters,
                               const float* bias, float* output,
                               int batch, int C_in, int H, int W,
                               int C_out, int F) {
    // TODO: Your implementation here
}
