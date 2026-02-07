# Puzzle 06: Fully Connected Layer (Backward Pass)

## Overview

Implement the **backward pass** for a fully connected (FC) layer on the GPU.
Training a neural network requires computing gradients of the loss with respect
to every learnable parameter. This puzzle computes **three separate gradients**:

1. **Weight gradient** `dW` — how the loss changes as each weight changes
2. **Bias gradient** `db` — how the loss changes as each bias changes
3. **Input gradient** `dX` — how the loss changes as each input changes (for backprop to earlier layers)

**Why this matters for LeNet:**
During training, after the forward pass computes predictions, the backward
pass flows gradients from the loss back through each layer:

```
Loss → ∂L/∂Y → FC3 backward → FC2 backward → FC1 backward → Conv layers...
                    ↓               ↓               ↓
                 dW3, db3        dW2, db2        dW1, db1
                (update)        (update)        (update)
```

**Connection to Puzzle 03:**
The forward pass computes `Y = X · W^T + b`. The backward pass reverses
this: given `∂L/∂Y` (upstream gradient), compute `∂L/∂W`, `∂L/∂b`, and `∂L/∂X`.

---

## The Math: Deriving the Backward Pass

### Starting point: the forward pass

Recall from Puzzle 03:

```
Y = X · W^T + b

  X:  (B × I)   — input activations
  W:  (O × I)   — weight matrix
  b:  (O)       — bias vector (broadcast across batch)
  Y:  (B × O)   — output activations

Where: B=batch, I=in_features, O=out_features
```

Per-element:

```
y[b][j] = Σ_i x[b][i] * w[j][i] + bias[j]
```

### Given: upstream gradient

The upstream gradient `dY = ∂L/∂Y` has the same shape as `Y`:

```
dY: (B × O)    — ∂L/∂Y, gradient of the loss w.r.t. each output element
```

### Gradient 1: ∂L/∂W (weight gradient)

By the chain rule:

```
∂L/∂w[j][i] = Σ_b  ∂L/∂y[b][j] × ∂y[b][j]/∂w[j][i]
```

Since `y[b][j] = Σ_k x[b][k] * w[j][k] + bias[j]`, the partial derivative
`∂y[b][j]/∂w[j][i] = x[b][i]`. Therefore:

```
∂L/∂w[j][i] = Σ_b  dY[b][j] × x[b][i]
```

In matrix form:

```
dW = dY^T · X

  dY^T: (O × B)
  X:    (B × I)
  dW:   (O × I)  ✓ same shape as W
```

### Dimensional verification for dW

```
dW shape = (O × B) · (B × I) = (O × I)  ✓ matches W shape (out_features × in_features)
```

### Gradient 2: ∂L/∂b (bias gradient)

```
∂L/∂bias[j] = Σ_b  ∂L/∂y[b][j] × ∂y[b][j]/∂bias[j]
```

Since `∂y[b][j]/∂bias[j] = 1`:

```
∂L/∂bias[j] = Σ_b  dY[b][j]
```

In matrix form — sum over the batch dimension:

```
db[j] = Σ_b dY[b][j]

  dY: (B × O)
  db: (O)      ✓ same shape as bias
```

### Dimensional verification for db

```
db shape = sum over B of (B × O) = (O)  ✓ matches bias shape (out_features)
```

### Gradient 3: ∂L/∂X (input gradient)

```
∂L/∂x[b][i] = Σ_j  ∂L/∂y[b][j] × ∂y[b][j]/∂x[b][i]
```

Since `∂y[b][j]/∂x[b][i] = w[j][i]`:

```
∂L/∂x[b][i] = Σ_j  dY[b][j] × w[j][i]
```

In matrix form:

```
dX = dY · W

  dY: (B × O)
  W:  (O × I)
  dX: (B × I)  ✓ same shape as X
```

### Dimensional verification for dX

```
dX shape = (B × O) · (O × I) = (B × I)  ✓ matches X shape (batch × in_features)
```

---

## Summary of All Three Gradients

```
┌─────────────────────────────────────────────────────────────────┐
│  Given: dY = ∂L/∂Y  (B × O)                                    │
│                                                                 │
│  ∂L/∂W = dY^T · X         (O×B) · (B×I) = (O×I)  ← weight grad│
│  ∂L/∂b = Σ_batch dY       sum over B of (B×O) = (O) ← bias grad│
│  ∂L/∂X = dY · W           (B×O) · (O×I) = (B×I)  ← input grad │
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Layout

All matrices are **row-major**:

```
Input     X:   (B × I)   →  x[b * in_features + i]
Weights   W:   (O × I)   →  w[j * in_features + i]
Bias      b:   (O)       →  bias[j]
Output    Y:   (B × O)   →  y[b * out_features + j]
Upstream  dY:  (B × O)   →  grad_output[b * out_features + j]
Grad W    dW:  (O × I)   →  grad_weights[j * in_features + i]
Grad b    db:  (O)       →  grad_bias[j]
Grad X    dX:  (B × I)   →  grad_input[b * in_features + i]
```

---

## Kernel 1: fc_backward_weights

Each thread computes one element of `dW[j][i]`:

```
dW[j][i] = Σ_b  dY[b][j] × X[b][i]

Thread (j, i) loops over b = 0..batch-1
```

```
Grid layout:
         in_features (i) →
    ┌──────────────────────┐
    │  Thread(0,0)  (0,1)  │
 j  │  Thread(1,0)  (1,1)  │  Each thread loops over
 ↓  │  Thread(2,0)  (2,1)  │  batch to accumulate
    │     ...       ...     │  one gradient element
    └──────────────────────┘
```

## Kernel 2: fc_backward_bias

Each thread computes one element of `db[j]`:

```
db[j] = Σ_b  dY[b][j]

Thread j loops over b = 0..batch-1
```

This is a simple **reduction over the batch dimension** — a 1D kernel
where each thread sums one column of dY.

## Kernel 3: fc_backward_input

Each thread computes one element of `dX[b][i]`:

```
dX[b][i] = Σ_j  dY[b][j] × W[j][i]

Thread (b, i) loops over j = 0..out_features-1
```

```
Grid layout:
         in_features (i) →
    ┌──────────────────────┐
    │  Thread(0,0)  (0,1)  │
 b  │  Thread(1,0)  (1,1)  │  Each thread loops over
 ↓  │  Thread(2,0)  (2,1)  │  out_features to accumulate
    │     ...       ...     │  one gradient element
    └──────────────────────┘
```

---

## Kernel Signatures

```cuda
__global__ void fc_backward_weights(float* grad_output, float* input,
                                     float* grad_weights, int batch,
                                     int in_features, int out_features);

__global__ void fc_backward_bias(float* grad_output, float* grad_bias,
                                  int batch, int out_features);

__global__ void fc_backward_input(float* grad_output, float* weights,
                                   float* grad_input, int batch,
                                   int in_features, int out_features);
```

---

## Hints

<details>
<summary>Hint 1 (Mild): Weight gradient accumulation</summary>

Each element `dW[j][i]` is a dot product between column `j` of `dY^T`
and column `i` of `X^T` — equivalently, loop over `b`:
```cuda
float sum = 0.0f;
for (int b = 0; b < batch; b++) {
    sum += grad_output[b * out_features + j] * input[b * in_features + i];
}
grad_weights[j * in_features + i] = sum;
```
</details>

<details>
<summary>Hint 2 (Mild): Bias gradient is just column sums</summary>

```cuda
float sum = 0.0f;
for (int b = 0; b < batch; b++) {
    sum += grad_output[b * out_features + j];
}
grad_bias[j] = sum;
```
</details>

<details>
<summary>Hint 3 (Medium): Input gradient is a matmul with W (not W^T)</summary>

Unlike forward pass which uses `W^T`, the input gradient uses `W` directly:
```cuda
float sum = 0.0f;
for (int jj = 0; jj < out_features; jj++) {
    sum += grad_output[b * out_features + jj] * weights[jj * in_features + i];
}
grad_input[b * in_features + i] = sum;
```
</details>

---

## Testing

Build and run the tests to verify your solution:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target puzzle_06_test
./build/puzzle_06_test
```

The test suite runs 4 tests:

1. **fc_backward_small_example**: Small 2×3→2 — hardcoded, hand-verifiable
2. **fc_backward_numerical_gradient_check**: Perturb weights by ε, verify analytical vs numerical gradients
3. **fc_backward_lenet_fc1_dims**: Batch=4, 256→120 — LeNet FC1 backward dimensions
4. **fc_backward_shape_verification**: Verify all gradient shapes match expected dimensions

All 4 tests must pass for the puzzle to be complete.
