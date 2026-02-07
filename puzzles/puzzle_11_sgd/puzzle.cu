// Puzzle 11: SGD Optimizer
//
// Implement TWO CUDA kernels:
//
// 1. sgd_update:     weights[i] -= learning_rate * gradients[i]
// 2. zero_gradients: gradients[i] = 0.0f
//
// SGD (Stochastic Gradient Descent) is the optimizer that actually
// updates network parameters using the gradients computed by the
// backward pass (Puzzles 4 and 6).
//
// See README.md for detailed explanation and hints.

#include <cuda_runtime.h>

// TODO: Implement the SGD weight update kernel
//
// Parameters:
//   weights       - parameter array to update IN-PLACE, size n
//   gradients     - gradient array (from backward pass), size n
//   learning_rate - step size (scalar, same for all parameters)
//   n             - total number of parameters
//
// Steps:
//   1. Calculate thread index: i = blockIdx.x * blockDim.x + threadIdx.x
//   2. Bounds check: if i >= n, return
//   3. Update: weights[i] = weights[i] - learning_rate * gradients[i]
__global__ void sgd_update(float* weights, const float* gradients,
                           float learning_rate, int n) {
    // TODO: Your implementation here
}

// TODO: Implement the gradient zeroing kernel
//
// After each SGD step, gradients must be reset to zero before
// the next forward/backward pass. Otherwise gradients would
// accumulate across mini-batches, giving incorrect updates.
//
// Parameters:
//   gradients - gradient array to zero out, size n
//   n         - total number of elements
//
// Steps:
//   1. Calculate thread index: i = blockIdx.x * blockDim.x + threadIdx.x
//   2. Bounds check: if i >= n, return
//   3. Zero: gradients[i] = 0.0f
__global__ void zero_gradients(float* gradients, int n) {
    // TODO: Your implementation here
}
