// Puzzle 03: Fully Connected Layer (Forward Pass)
//
// Implement a CUDA kernel for an FC layer forward pass:
//   output[b][j] = Σ_i input[b][i] × weights[j][i] + bias[j]
//
// This is a matrix multiplication (Puzzle 02) plus bias addition.
// Each thread computes one output element.
//
// See README.md for detailed explanation and hints.

#include <cuda_runtime.h>

// TODO: Implement the FC forward pass kernel
//
// Parameters:
//   input       - input activations, size (batch × in_features), row-major
//   weights     - weight matrix, size (out_features × in_features), row-major
//   bias        - bias vector, size (out_features)
//   output      - output activations, size (batch × out_features), row-major
//   batch       - number of samples in the batch
//   in_features - number of input features per sample
//   out_features - number of output features per sample
//
// Steps:
//   1. Calculate b (batch index) and j (output feature) from 2D thread indices
//   2. Check bounds: b < batch and j < out_features
//   3. Loop over i = 0..in_features-1, accumulating input[b][i] * weights[j][i]
//   4. Add bias[j] to the accumulated sum
//   5. Write result to output[b][j]
//
// Memory indexing (row-major):
//   input[b][i]   = input[b * in_features + i]
//   weights[j][i] = weights[j * in_features + i]
//   output[b][j]  = output[b * out_features + j]
__global__ void fc_forward(float* input, float* weights, float* bias,
                           float* output, int batch, int in_features,
                           int out_features) {
    // TODO: Your implementation here
}
