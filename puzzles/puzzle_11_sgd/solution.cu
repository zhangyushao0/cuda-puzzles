// Puzzle 11: SGD Optimizer — Reference Solution
//
// sgd_update:     weights[i] -= learning_rate * gradients[i]
// zero_gradients: gradients[i] = 0.0f
//
// Both kernels are element-wise operations using 1D grids.
// SGD is the simplest optimizer: just step in the negative gradient direction.

#include <cuda_runtime.h>

__global__ void sgd_update(float* weights, const float* gradients,
                           float learning_rate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        weights[i] -= learning_rate * gradients[i];
    }
}

__global__ void zero_gradients(float* gradients, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        gradients[i] = 0.0f;
    }
}
