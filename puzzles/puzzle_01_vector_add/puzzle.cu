// Puzzle 01: Vector Addition
// Implement a CUDA kernel that computes C[i] = A[i] + B[i]
//
// See README.md for concepts and hints.

#include "common.h"

// =============================================================
// TODO: Implement the vector_add kernel
//
// Each thread should compute one or more elements of C using
// the grid stride loop pattern:
//   1. Compute global thread index (tid)
//   2. Compute stride (total number of threads in the grid)
//   3. Loop: for each index i from tid to N (step by stride)
//      set C[i] = A[i] + B[i]
// =============================================================
__global__ void vector_add(float* A, float* B, float* C, int N) {
    // TODO: Your code here

}
