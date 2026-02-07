// Puzzle 06: Fully Connected Layer (Backward Pass) — Reference Solution
//
// Given upstream gradient dY = ∂L/∂Y (shape: batch × out_features):
//   dW = dY^T · X       (weight gradient)
//   db = Σ_batch dY     (bias gradient)
//   dX = dY · W         (input gradient)

#include <cuda_runtime.h>

// Kernel 1: Weight gradient
// dW[j][i] = Σ_b grad_output[b][j] × input[b][i]
// Matrix form: dW = dY^T · X    (O×B)·(B×I) = (O×I)
__global__ void fc_backward_weights(float* grad_output, float* input,
                                     float* grad_weights, int batch,
                                     int in_features, int out_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // in_features dimension
    int j = blockIdx.y * blockDim.y + threadIdx.y;   // out_features dimension

    if (j < out_features && i < in_features) {
        float sum = 0.0f;
        for (int b = 0; b < batch; b++) {
            sum += grad_output[b * out_features + j] * input[b * in_features + i];
        }
        grad_weights[j * in_features + i] = sum;
    }
}

// Kernel 2: Bias gradient
// db[j] = Σ_b grad_output[b][j]
__global__ void fc_backward_bias(float* grad_output, float* grad_bias,
                                  int batch, int out_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < out_features) {
        float sum = 0.0f;
        for (int b = 0; b < batch; b++) {
            sum += grad_output[b * out_features + j];
        }
        grad_bias[j] = sum;
    }
}

// Kernel 3: Input gradient
// dX[b][i] = Σ_j grad_output[b][j] × weights[j][i]
// Matrix form: dX = dY · W    (B×O)·(O×I) = (B×I)
__global__ void fc_backward_input(float* grad_output, float* weights,
                                   float* grad_input, int batch,
                                   int in_features, int out_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // in_features dimension
    int b = blockIdx.y * blockDim.y + threadIdx.y;   // batch dimension

    if (b < batch && i < in_features) {
        float sum = 0.0f;
        for (int jj = 0; jj < out_features; jj++) {
            sum += grad_output[b * out_features + jj] * weights[jj * in_features + i];
        }
        grad_input[b * in_features + i] = sum;
    }
}
