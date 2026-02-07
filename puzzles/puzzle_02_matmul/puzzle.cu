// Puzzle 02: Matrix Multiplication (Naive)
//
// Implement a CUDA kernel that multiplies two matrices:
//   C(M x N) = A(M x K) * B(K x N)
//
// Each thread computes one element of the output matrix.
// Use a 2D grid: row from blockIdx.y/threadIdx.y, col from blockIdx.x/threadIdx.x.
//
// See README.md for detailed explanation and hints.

#include <cuda_runtime.h>

// TODO: Implement the matrix multiplication kernel
//
// Parameters:
//   A - input matrix of size M x K (row-major)
//   B - input matrix of size K x N (row-major)
//   C - output matrix of size M x N (row-major)
//   M - number of rows in A and C
//   K - shared dimension (cols of A, rows of B)
//   N - number of columns in B and C
//
// Steps:
//   1. Calculate row and col from 2D thread indices
//   2. Check bounds: row < M and col < N
//   3. Loop over k = 0..K-1, accumulating A[row][k] * B[k][col]
//   4. Write result to C[row][col]
//
// Remember: element [i][j] of an MxN matrix is at index i*N+j in row-major order
__global__ void matmul(float* A, float* B, float* C, int M, int K, int N) {
    // TODO: Your implementation here
}
