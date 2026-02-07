# Puzzle 01: Vector Addition

## Overview

Vector addition is the "Hello, World!" of GPU programming.
Given two arrays A and B, compute C where each element is:

```
C[i] = A[i] + B[i]    for all i in [0, N)
```

This puzzle introduces CUDA's core execution model: launching
a **kernel** function that runs across thousands of threads
in parallel, each computing one (or more) output elements.

## Concepts You'll Learn

- `__global__` keyword: marks a function as a GPU kernel
- Thread indexing: `blockIdx.x`, `blockDim.x`, `threadIdx.x`
- Grid/block launch configuration: `kernel<<<blocks, threads>>>()`
- Grid stride loop pattern for handling arbitrary array sizes
- CUDA memory management: `cudaMalloc`, `cudaMemcpy`, `cudaFree`

## Thread-to-Element Mapping

Each thread computes one element of the output array C.
The global thread index maps directly to the array index:

```
tid = blockIdx.x * blockDim.x + threadIdx.x

Grid (2 blocks, 4 threads each = 8 threads total)
┌─────────────────────────┬─────────────────────────┐
│       Block 0           │       Block 1           │
│  t0   t1   t2   t3     │  t4   t5   t6   t7     │
└──┬────┬────┬────┬───────┴──┬────┬────┬────┬──────┘
   │    │    │    │          │    │    │    │
   ▼    ▼    ▼    ▼          ▼    ▼    ▼    ▼
A: [1.0][2.0][3.0][4.0]    [5.0][6.0][7.0][8.0]
 +  +    +    +    +         +    +    +    +
B: [8.0][7.0][6.0][5.0]    [4.0][3.0][2.0][1.0]
 =  =    =    =    =         =    =    =    =
C: [9.0][9.0][9.0][9.0]    [9.0][9.0][9.0][9.0]
   [0]  [1]  [2]  [3]      [4]  [5]  [6]  [7]
```

## Grid Stride Loop

What if N > total number of threads? Use a **grid stride loop**:

```
int tid    = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;  // total threads

for (int i = tid; i < N; i += stride) {
    C[i] = A[i] + B[i];
}
```

This pattern lets a fixed number of threads process an
arbitrarily large array:

```
N = 12, Grid = 2 blocks x 4 threads = 8 threads

Pass 1 (i = tid):
  t0→[0]  t1→[1]  t2→[2]  t3→[3]
  t4→[4]  t5→[5]  t6→[6]  t7→[7]

Pass 2 (i = tid + stride):
  t0→[8]  t1→[9]  t2→[10] t3→[11]
  t4→done t5→done t6→done t7→done
```

## Your Task

Open `puzzle.cu` and implement the `vector_add` kernel.
The host code (memory allocation, kernel launch, cleanup)
is provided — you only need to write the kernel body.

## Step-by-Step Guide

1. Compute the global thread index from block/thread IDs
2. Compute the total stride (total number of threads)
3. Loop from your thread index to N, stepping by stride
4. In each iteration, write `C[i] = A[i] + B[i]`
5. Guard against out-of-bounds access (i < N)

## Hints

### Hint 1 (Gentle)
The global thread index combines the block index and the
thread index within that block. Think about how blocks are
arranged in a 1D grid and how threads are arranged within
each block.

### Hint 2 (Medium)
```
tid = blockIdx.x * blockDim.x + threadIdx.x
```
`blockDim.x` is the number of threads per block.
`blockIdx.x` tells you which block this thread is in.
`threadIdx.x` tells you the thread's position within its
block.

### Hint 3 (Strong)
```cuda
__global__ void vector_add(float* A, float* B,
                           float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = ???; i < ???; i += ???) {
        C[i] = ???;
    }
}
```
Replace each `???` with the correct expression. The loop
starts at `tid`, ends before `N`, steps by `stride`, and
the body computes the element-wise sum.

## Testing

Build and run the tests to verify your solution:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target puzzle_01_test
./build/puzzle_01_test
```

The test suite (see `test_puzzle.cu`) uses the test
framework from `common/test_utils.h`. It runs 4 tests:

1. **small_array**: N=8, hardcoded values
2. **medium_array**: N=1024, seeded random values
3. **large_array**: N=1,000,000, seeded random values
4. **non_divisible_size**: N not divisible by block size

All 4 tests must pass for the puzzle to be complete.
