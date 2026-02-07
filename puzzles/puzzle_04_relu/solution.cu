// Puzzle 04: ReLU Forward + Backward — Reference Solution
//
// Forward:  y[i] = max(0, x[i])
// Backward: grad_input[i] = (input[i] > 0) ? grad_output[i] : 0
//
// Both kernels are element-wise operations using 1D grids.
// The backward pass demonstrates gradient masking via the chain rule.

#include <cuda_runtime.h>

__global__ void relu_forward(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

__global__ void relu_backward(const float* grad_output, const float* input,
                              float* grad_input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
    }
}
