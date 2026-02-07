# Puzzle 02: Matrix Multiplication (Naive)

## Overview

Multiply two matrices on the GPU. Each thread computes one element of
the output matrix. This is the **foundational operation** behind fully
connected (dense) layers in neural networks.

**Why this matters for LeNet:**
Every fully connected layer is essentially a matrix multiplication
plus a bias vector. When we later implement FC layers, the core
computation is exactly this kernel:

```
FC Layer:   Y = X * W^T + b
                 ^^^^
                matmul!
```

LeNet's FC layers use these exact dimensions:
- FC1: input(batch x 256) x weights(256 x 120)  -> output(batch x 120)
- FC2: input(batch x 120) x weights(120 x 84)   -> output(batch x 84)
- FC3: input(batch x 84)  x weights(84 x 10)    -> output(batch x 10)

---

## Matrix Multiplication: The Math

Given matrices A and B, compute C = A x B:

```
A is M x K          B is K x N          C is M x N
┌───────────┐      ┌───────────┐      ┌───────────┐
│           │      │           │      │           │
│   M rows  │  x   │   K rows  │  =   │   M rows  │
│   K cols  │      │   N cols  │      │   N cols  │
│           │      │           │      │           │
└───────────┘      └───────────┘      └───────────┘
```

Each element C[i][j] is computed as:

```
         K-1
C[i,j] = Sum  A[i,k] * B[k,j]
         k=0
```

### Concrete 2x2 Example

```
A (2x2)       B (2x2)         C (2x2)
┌─────────┐   ┌─────────┐     ┌───────────────────────────┐
│ 1    2  │   │ 5    6  │     │ 1*5+2*7=19   1*6+2*8=22  │
│ 3    4  │ x │ 7    8  │  =  │ 3*5+4*7=43   3*6+4*8=50  │
└─────────┘   └─────────┘     └───────────────────────────┘
```

### How Element C[i,j] Is Computed

To compute a single output element, we walk across row i of A
and down column j of B, multiplying corresponding elements:

```
        col j
         |
         v
A        B               C
Row i -> [a0 a1 a2]     [b0]            [    ...    ]
                         [b1]  <-- col j  [ ... c_ij ]
                         [b2]            [    ...    ]

c_ij = a0*b0 + a1*b1 + a2*b2
       ^^^^    ^^^^    ^^^^
       k=0     k=1     k=2
```

---

## Row-Major Memory Layout

C/C++ stores 2D arrays in **row-major** order: rows are contiguous
in memory. This means element `[i][j]` of an MxN matrix lives at
index `i * N + j`:

```
Logical 2D view (3x4 matrix):

     col 0  col 1  col 2  col 3
     ─────  ─────  ─────  ─────
row 0: a00    a01    a02    a03
row 1: a10    a11    a12    a13
row 2: a20    a21    a22    a23

Physical 1D memory layout:

Index:   0     1     2     3     4     5     6     7     8     9    10    11
       ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
       │ a00 │ a01 │ a02 │ a03 │ a10 │ a11 │ a12 │ a13 │ a20 │ a21 │ a22 │ a23 │
       └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
       ├──── row 0 ─────────────┤──── row 1 ─────────────┤──── row 2 ─────────────┤

Address of element [i][j] = base + i * N + j
                                       ^
                                   number of columns (row width)
```

For matmul, this means:
- `A[i][k]` lives at `A[i * K + k]`
- `B[k][j]` lives at `B[k * N + j]`
- `C[i][j]` lives at `C[i * N + j]`

---

## Thread Mapping: One Thread Per Output Element

We use a **2D grid** of threads where each thread computes exactly
one element of the output matrix C:

```
Thread (row, col) computes C[row][col]

row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x

Grid of thread blocks (each block is 16x16 threads):

         Block(0,0)    Block(1,0)    Block(2,0)
        ┌───────────┬───────────┬───────────┐
        │ 16x16     │ 16x16     │ 16x16     │
        │ threads   │ threads   │ threads   │  <- row block 0
        ├───────────┼───────────┼───────────┤
        │ 16x16     │ 16x16     │ 16x16     │
        │ threads   │ threads   │ threads   │  <- row block 1
        ├───────────┼───────────┼───────────┤
        │ 16x16     │ 16x16     │ 16x16     │
        │ threads   │ threads   │ threads   │  <- row block 2
        └───────────┴───────────┴───────────┘
         col blk 0   col blk 1   col blk 2

IMPORTANT: The grid may be larger than the matrix.
Threads outside the matrix must do nothing (bounds check!).
```

---

## Kernel Signature

```cuda
__global__ void matmul(float* A, float* B, float* C,
                       int M, int K, int N);
```

**Parameters:**
- `A`: Input matrix, size M x K (row-major)
- `B`: Input matrix, size K x N (row-major)
- `C`: Output matrix, size M x N (row-major)
- `M`: Number of rows in A (and C)
- `K`: Number of columns in A = number of rows in B
- `N`: Number of columns in B (and C)

**Launch configuration:**
```cuda
dim3 blockDim(16, 16);  // 16x16 = 256 threads per block
dim3 gridDim((N + 15) / 16, (M + 15) / 16);
matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
```

Note: gridDim.x covers columns (N), gridDim.y covers rows (M).

---

## Step-by-Step Guide

1. **Calculate your position**: Compute `row` and `col` from block
   and thread indices (use 2D indexing: y for row, x for col)

2. **Bounds check**: If `row >= M` or `col >= N`, return immediately.
   The grid may be larger than the matrix.

3. **Accumulate the dot product**: Loop over `k` from 0 to K-1,
   summing `A[row * K + k] * B[k * N + col]`

4. **Write the result**: Store the sum in `C[row * N + col]`

---

## Hints

<details>
<summary>Hint 1 (Mild): Thread index calculation</summary>

Use 2D block/thread indices:
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```
</details>

<details>
<summary>Hint 2 (Medium): The accumulation loop</summary>

```cuda
float sum = 0.0f;
for (int k = 0; k < K; k++) {
    sum += A[???] * B[???];  // Fill in the indexing!
}
```
</details>

<details>
<summary>Hint 3 (Strong): Almost complete solution</summary>

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```
</details>
