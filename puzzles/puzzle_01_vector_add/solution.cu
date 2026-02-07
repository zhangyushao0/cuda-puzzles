// Puzzle 01: Vector Addition — Reference Solution
// Computes C[i] = A[i] + B[i] using a grid stride loop.

#include "common.h"

__global__ void vector_add(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
