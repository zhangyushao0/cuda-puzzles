// Puzzle 03: Fully Connected Layer (Forward Pass) — Reference Solution
//
// output[b][j] = Σ_i input[b][i] × weights[j][i] + bias[j]
//
// Each thread computes one output element using a 2D grid.
// This is matmul (Puzzle 02) with bias addition: Y = X · W^T + b

#include <cuda_runtime.h>

__global__ void fc_forward(float* input, float* weights, float* bias,
                           float* output, int batch, int in_features,
                           int out_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // output feature
    int b = blockIdx.y * blockDim.y + threadIdx.y;   // batch index

    if (b < batch && j < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[b * in_features + i] * weights[j * in_features + i];
        }
        output[b * out_features + j] = sum + bias[j];
    }
}
