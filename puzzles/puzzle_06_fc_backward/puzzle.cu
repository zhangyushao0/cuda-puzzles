// Puzzle 06: Fully Connected Layer (Backward Pass)
//
// Implement THREE CUDA kernels for FC layer backward pass:
//
// Given upstream gradient dY = ∂L/∂Y (shape: batch × out_features):
//
//   1. fc_backward_weights:  dW[j][i] = Σ_b dY[b][j] × X[b][i]
//   2. fc_backward_bias:     db[j]    = Σ_b dY[b][j]
//   3. fc_backward_input:    dX[b][i] = Σ_j dY[b][j] × W[j][i]
//
// See README.md for detailed derivation and hints.

#include <cuda_runtime.h>

// TODO: Implement the weight gradient kernel
//
// Parameters:
//   grad_output  - upstream gradient ∂L/∂Y, size (batch × out_features), row-major
//   input        - input activations X, size (batch × in_features), row-major
//   grad_weights - output: ∂L/∂W, size (out_features × in_features), row-major
//   batch        - number of samples in the batch
//   in_features  - number of input features per sample
//   out_features - number of output features per sample
//
// Formula: dW[j][i] = Σ_b grad_output[b][j] × input[b][i]
// Matrix form: dW = dY^T · X    (O×B) · (B×I) = (O×I)
//
// Thread mapping: one thread per (j, i) element of dW
//   j = blockIdx.y * blockDim.y + threadIdx.y  (out_features dimension)
//   i = blockIdx.x * blockDim.x + threadIdx.x  (in_features dimension)
__global__ void fc_backward_weights(float* grad_output, float* input,
                                     float* grad_weights, int batch,
                                     int in_features, int out_features) {
    // TODO: Your implementation here
}

// TODO: Implement the bias gradient kernel
//
// Parameters:
//   grad_output - upstream gradient ∂L/∂Y, size (batch × out_features), row-major
//   grad_bias   - output: ∂L/∂b, size (out_features)
//   batch       - number of samples in the batch
//   out_features - number of output features per sample
//
// Formula: db[j] = Σ_b grad_output[b][j]
// This is a sum-reduction over the batch dimension.
//
// Thread mapping: one thread per output feature j
//   j = blockIdx.x * blockDim.x + threadIdx.x
__global__ void fc_backward_bias(float* grad_output, float* grad_bias,
                                  int batch, int out_features) {
    // TODO: Your implementation here
}

// TODO: Implement the input gradient kernel
//
// Parameters:
//   grad_output - upstream gradient ∂L/∂Y, size (batch × out_features), row-major
//   weights     - weight matrix W, size (out_features × in_features), row-major
//   grad_input  - output: ∂L/∂X, size (batch × in_features), row-major
//   batch       - number of samples in the batch
//   in_features - number of input features per sample
//   out_features - number of output features per sample
//
// Formula: dX[b][i] = Σ_j grad_output[b][j] × weights[j][i]
// Matrix form: dX = dY · W    (B×O) · (O×I) = (B×I)
//
// Thread mapping: one thread per (b, i) element of dX
//   i = blockIdx.x * blockDim.x + threadIdx.x  (in_features dimension)
//   b = blockIdx.y * blockDim.y + threadIdx.y   (batch dimension)
__global__ void fc_backward_input(float* grad_output, float* weights,
                                   float* grad_input, int batch,
                                   int in_features, int out_features) {
    // TODO: Your implementation here
}
