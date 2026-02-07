// Puzzle 04: ReLU Forward + Backward
//
// Implement TWO CUDA kernels:
//
// 1. relu_forward:  y[i] = max(0, x[i])
// 2. relu_backward: grad_input[i] = (input[i] > 0) ? grad_output[i] : 0
//
// This is the first puzzle with a backward pass!
// The backward kernel implements the chain rule for ReLU.
//
// See README.md for detailed explanation and hints.

#include <cuda_runtime.h>

// TODO: Implement the ReLU forward pass kernel
//
// Parameters:
//   input  - input activations, size n
//   output - output activations, size n
//   n      - total number of elements
//
// Steps:
//   1. Calculate thread index: i = blockIdx.x * blockDim.x + threadIdx.x
//   2. Bounds check: if i >= n, return
//   3. Apply ReLU: output[i] = max(0, input[i])
__global__ void relu_forward(const float* input, float* output, int n) {
    // TODO: Your implementation here
}

// TODO: Implement the ReLU backward pass kernel
//
// The backward pass computes how the loss gradient flows through ReLU.
// ReLU's derivative is 1 where input > 0, and 0 elsewhere.
// By the chain rule: grad_input = grad_output * (dy/dx)
//
// Parameters:
//   grad_output - gradient from the layer above (dL/dy), size n
//   input       - original input to the forward pass (needed for mask), size n
//   grad_input  - gradient to pass to the layer below (dL/dx), size n
//   n           - total number of elements
//
// Steps:
//   1. Calculate thread index: i = blockIdx.x * blockDim.x + threadIdx.x
//   2. Bounds check: if i >= n, return
//   3. Apply gradient mask: grad_input[i] = (input[i] > 0) ? grad_output[i] : 0
//
// Key insight: You need the ORIGINAL INPUT (not the output) to decide
// whether to pass the gradient through.
__global__ void relu_backward(const float* grad_output, const float* input,
                              float* grad_input, int n) {
    // TODO: Your implementation here
}
