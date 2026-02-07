// Puzzle 02: Matrix Multiplication (Naive) — Reference Solution
//
// C(M x N) = A(M x K) * B(K x N)
//
// Each thread computes one element of the output matrix using a 2D grid.
// Bounds checking ensures threads outside the matrix do nothing.

#include <cuda_runtime.h>

__global__ void matmul(float* A, float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
