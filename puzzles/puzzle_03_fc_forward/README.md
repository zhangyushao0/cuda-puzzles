# Puzzle 03: Fully Connected Layer (Forward Pass)

## Overview

Implement a **fully connected (FC) layer** forward pass on the GPU.
This is the first puzzle that implements an actual neural network layer!

A fully connected layer takes an input vector and produces an output vector
where **every input is connected to every output** through a learnable weight,
plus a bias term:

```
Y = X · W^T + b
```

**Why this matters for LeNet:**
LeNet-5 has three fully connected layers that transform the flattened
feature maps into class predictions:

```
Conv layers → [flatten] → FC1(256→120) → FC2(120→84) → FC3(84→10) → predictions
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          This is what we're building!
```

**Connection to Puzzle 02:**
An FC layer is just a **matrix multiplication** (Puzzle 02) plus a
**bias broadcast**. If you solved matmul, you're 90% of the way there!

```
FC layer  =  matmul  +  bias addition
              ↑              ↑
          Puzzle 02     new concept
```

---

## What Does a Neuron Do?

A single neuron takes multiple inputs, multiplies each by a learned
weight, sums them up, and adds a bias:

```
          SINGLE NEURON
          ─────────────
  Inputs        Weights
  ──────        ───────
  x[0] ──── w[0] ──┐
                    │
  x[1] ──── w[1] ──┤
                    ├──→ Σ ──→ (+bias) ──→ output
  x[2] ──── w[2] ──┤
                    │
  x[3] ──── w[3] ──┘

  output = x[0]*w[0] + x[1]*w[1] + x[2]*w[2] + x[3]*w[3] + bias
         = Σ_i x[i] * w[i] + bias
```

A **fully connected layer** is just many neurons stacked together,
each with its own set of weights and bias:

```
     FULLY CONNECTED LAYER (in_features=4, out_features=3)
     ─────────────────────────────────────────────────────

     Inputs                    Outputs
     ──────                    ───────
                ┌─ w[0][0..3] ─→ Σ+b[0] ─→ y[0]    (neuron 0)
     x[0] ─────┤
     x[1] ─────┼─ w[1][0..3] ─→ Σ+b[1] ─→ y[1]    (neuron 1)
     x[2] ─────┤
     x[3] ─────┘─ w[2][0..3] ─→ Σ+b[2] ─→ y[2]    (neuron 2)

     Each output neuron j:
       y[j] = Σ_i x[i] * w[j][i] + bias[j]
```

---

## The Math

### Per-element formula

For a single sample in a batch:

```
y[b][j] = Σ_i x[b][i] × w[j][i] + bias[j]

where:
  b = sample index in the batch   (0 to batch-1)
  j = output feature index         (0 to out_features-1)
  i = input feature index          (0 to in_features-1)
```

### Matrix form

```
Y = X · W^T + b

X:      (batch × in_features)       — input activations
W:      (out_features × in_features) — weight matrix
W^T:    (in_features × out_features) — transposed for matmul
b:      (out_features)               — bias vector (broadcast across batch)
Y:      (batch × out_features)       — output activations
```

### Why W^T (transposed)?

The weight matrix W is stored as `(out_features × in_features)` because
each row of W holds the weights for one output neuron. But for the
matrix multiplication `X · W^T` to work, we need the shapes to align:

```
X    ×    W^T    =    Y
(B×I)  (I×O)      (B×O)

Where: B=batch, I=in_features, O=out_features
```

In our kernel, we don't explicitly transpose W. Instead, we just
index into W differently — accessing `W[j][i]` = `W[j * in_features + i]`,
which naturally gives us row j of W (the weights for output neuron j).

### Dimension example: LeNet FC layers

```
Input (32×256) × Weights (120×256)^T + Bias (120) → Output (32×120)
        ↑                   ↑                ↑              ↑
     batch=32          out=120×in=256    one per output   batch=32
     in=256            stored row-major   broadcast       out=120

FC1: Input(batch×256) × W(120×256)^T + b(120) → Output(batch×120)
FC2: Input(batch×120) × W(84×120)^T  + b(84)  → Output(batch×84)
FC3: Input(batch×84)  × W(10×84)^T   + b(10)  → Output(batch×10)
```

---

## Memory Layout

All matrices are **row-major** (same as Puzzle 02):

```
Input  (batch × in_features):   input[b * in_features + i]
Weight (out_features × in_features): weights[j * in_features + i]
Bias   (out_features):          bias[j]
Output (batch × out_features):  output[b * out_features + j]
```

---

## Thread Mapping: One Thread Per Output Element

Each thread computes one output element `output[b][j]`:

```
Thread (b, j) computes output[b][j]

b = blockIdx.y * blockDim.y + threadIdx.y    (batch dimension)
j = blockIdx.x * blockDim.x + threadIdx.x    (output feature)

Grid layout:
         out_features (j) →
    ┌──────────────────────┐
    │  Thread(0,0)  (0,1)  │
b   │  Thread(1,0)  (1,1)  │  Each thread loops over
↓   │  Thread(2,0)  (2,1)  │  in_features to compute
    │     ...       ...     │  one dot product + bias
    └──────────────────────┘
```

---

## Kernel Signature

```cuda
__global__ void fc_forward(float* input, float* weights, float* bias,
                           float* output, int batch, int in_features,
                           int out_features);
```

**Parameters:**
- `input`: Input activations, size batch × in_features (row-major)
- `weights`: Weight matrix, size out_features × in_features (row-major)
- `bias`: Bias vector, size out_features
- `output`: Output activations, size batch × out_features (row-major)
- `batch`: Number of samples in the batch
- `in_features`: Number of input features per sample
- `out_features`: Number of output features per sample

**Launch configuration:**
```cuda
dim3 blockDim(16, 16);  // 256 threads per block
dim3 gridDim((out_features + 15) / 16, (batch + 15) / 16);
fc_forward<<<gridDim, blockDim>>>(d_input, d_weights, d_bias, d_output,
                                  batch, in_features, out_features);
```

---

## Step-by-Step Guide

1. **Calculate your position**: Compute `b` (batch index) and `j`
   (output feature index) from block/thread indices (2D grid).

2. **Bounds check**: If `b >= batch` or `j >= out_features`, return
   immediately. The grid may be larger than the output matrix.

3. **Accumulate the dot product**: Loop over `i` from 0 to
   `in_features - 1`, summing `input[b][i] × weights[j][i]`.

4. **Add bias**: After the loop, add `bias[j]` to the sum.

5. **Write the result**: Store in `output[b * out_features + j]`.

---

## Hints

<details>
<summary>Hint 1 (Mild): Thread index calculation</summary>

Use 2D block/thread indices, just like Puzzle 02:
```cuda
int b = blockIdx.y * blockDim.y + threadIdx.y;  // batch index
int j = blockIdx.x * blockDim.x + threadIdx.x;  // output feature
```
</details>

<details>
<summary>Hint 2 (Medium): The accumulation loop</summary>

This is almost identical to Puzzle 02's matmul inner loop,
but you index into `weights` by row j (not column j):
```cuda
float sum = 0.0f;
for (int i = 0; i < in_features; i++) {
    sum += input[???] * weights[???];  // Fill in the indexing!
}
sum += bias[j];
```
</details>

<details>
<summary>Hint 3 (Strong): Almost complete solution</summary>

```cuda
int b = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;

if (b < batch && j < out_features) {
    float sum = 0.0f;
    for (int i = 0; i < in_features; i++) {
        sum += input[b * in_features + i] * weights[j * in_features + i];
    }
    output[b * out_features + j] = sum + bias[j];
}
```
</details>

---

## Testing

Build and run the tests to verify your solution:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target puzzle_03_test
./build/puzzle_03_test
```

The test suite runs 4 tests:

1. **fc_single_sample_4to3**: Single sample, 4→3 — hardcoded, hand-verifiable
2. **fc_lenet_fc1_256to120**: Batch=8, 256→120 — LeNet FC1 dimensions
3. **fc_lenet_fc2_120to84**: Batch=8, 120→84 — LeNet FC2 dimensions
4. **fc_lenet_fc3_84to10**: Batch=8, 84→10 — LeNet FC3 (output layer)

All 4 tests must pass for the puzzle to be complete.
