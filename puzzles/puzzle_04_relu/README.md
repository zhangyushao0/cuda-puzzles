# Puzzle 04: ReLU Forward + Backward

## Overview

Implement **two** CUDA kernels: ReLU **forward** pass and ReLU **backward** pass.
This is the first puzzle with a backward pass — welcome to **backpropagation**!

ReLU (Rectified Linear Unit) is the most common activation function in deep
learning. It's dead simple in the forward direction, and its backward pass
introduces the fundamental concept of **gradient masking**.

**Why this matters for LeNet:**
LeNet-5 applies ReLU after every convolutional and fully connected layer:

```
Input → Conv1 → ReLU → Pool → Conv2 → ReLU → Pool → FC1 → ReLU → FC2 → ReLU → FC3
                ^^^^                    ^^^^          ^^^^          ^^^^
                This is what we're building (forward AND backward)!
```

**Connection to Puzzle 03:**
In Puzzle 03 we computed `Y = X · W^T + b` (linear transform). But neural
networks need **non-linearity** to learn complex patterns. ReLU adds that:

```
FC layer output → ReLU → next layer input
    (linear)     (non-linear)
```

Without ReLU, stacking FC layers would just give another linear transform.
ReLU is what makes deep networks **deep**.

---

## The Forward Pass

### Formula

```
y = max(0, x)
```

That's it. If the input is positive, pass it through. If negative, output zero.

### The ReLU Graph

```
  output (y)
      │
    3 ┤                          ╱
      │                        ╱
    2 ┤                      ╱
      │                    ╱
    1 ┤                  ╱
      │                ╱
    0 ┤══════════════╱─────────── x = 0
      │            ╱
   -1 ┤          (input passes through if x > 0,
      │           clamped to 0 if x ≤ 0)
      ├──┬──┬──┬──┬──┬──┬──┬──┤
       -4 -3 -2 -1  0  1  2  3   input (x)
```

### Element-wise operation

ReLU operates independently on each element — no interaction between elements:

```
Input:  [-2.0,  3.5, -0.1,  0.0,  1.7, -4.2]
         ↓      ↓     ↓     ↓     ↓     ↓
ReLU:   max(0,·) for each element
         ↓      ↓     ↓     ↓     ↓     ↓
Output: [ 0.0,  3.5,  0.0,  0.0,  1.7,  0.0]
```

---

## The Backward Pass: Your First Gradient!

### Why do we need a backward pass?

Training a neural network means adjusting weights to reduce error. To know
**which direction** to adjust, we need **gradients** — the derivative of the
loss with respect to each parameter. Backpropagation computes these gradients
by applying the **chain rule** backwards through the network.

### The Chain Rule — Core of Backpropagation

The chain rule says: if `y = f(x)` and we know `dL/dy` (how the loss changes
with y), then we can compute `dL/dx` (how the loss changes with x):

```
dL     dL     dy
── = ── × ──
dx     dy     dx

"gradient       "gradient       "local
 flowing          coming          derivative
 backward"        from above"     of this op"
```

In code, we call these:
- `grad_output` = `dL/dy` — gradient arriving from the layer above
- `grad_input`  = `dL/dx` — gradient we pass to the layer below
- `dy/dx`       — local derivative of this operation

### Computational Graph: Forward + Backward

```
                    FORWARD PASS (left to right)
                    ════════════════════════════

              x ─────────→ [ ReLU: y = max(0,x) ] ─────────→ y ──→ ... ──→ Loss
                                                                            │
                    BACKWARD PASS (right to left)                           │
                    ═════════════════════════════                            │
                                                                            ↓
     dL/dx ←──── [ multiply by local gradient ] ←──── dL/dy ←── ... ←── dL/dL = 1
                        dy/dx = {1 if x>0
                                {0 if x≤0
```

### ReLU's Local Derivative

```
dy     d                ⎧ 1   if x > 0
── = ── max(0, x) =     ⎨
dx    dx                ⎩ 0   if x ≤ 0
```

### The Backward Formula

Applying the chain rule:

```
                    ⎧ grad_output[i]    if input[i] > 0
grad_input[i] =    ⎨
                    ⎩ 0.0               if input[i] ≤ 0
```

Or equivalently:

```
grad_input[i] = (input[i] > 0) ? grad_output[i] : 0.0f
```

### Gradient Masking: The Key Insight

ReLU backward is a **gradient mask**: it uses the same mask as the forward
pass (positive → pass, negative → block) but applies it to gradients instead
of activations:

```
          Forward:                    Backward:
          ────────                    ─────────
input:   [-2.0,  3.5, -0.1,  1.7]   grad_output: [0.5, -1.2,  0.8, -0.3]
          ↓      ↓     ↓     ↓                     ↓      ↓     ↓     ↓
mask:    [ 0,    1,    0,    1  ]    same mask:   [ 0,    1,    0,    1  ]
          ↓      ↓     ↓     ↓                     ↓      ↓     ↓     ↓
output:  [ 0.0,  3.5,  0.0,  1.7]   grad_input:  [0.0, -1.2,  0.0, -0.3]

Negative inputs BLOCK both activations AND gradients.
The gradient "dies" where the neuron is inactive.
```

This is why we need the **original input** during the backward pass — to
reconstruct the mask. The output alone isn't sufficient because we need to
distinguish "was zero because input was zero" from "was zero because input
was negative."

---

## Kernel Signatures

### Forward Kernel

```cuda
__global__ void relu_forward(const float* input, float* output, int n);
```

**Parameters:**
- `input`: Input activations, size n
- `output`: Output activations, size n
- `n`: Total number of elements

### Backward Kernel

```cuda
__global__ void relu_backward(const float* grad_output, const float* input,
                              float* grad_input, int n);
```

**Parameters:**
- `grad_output`: Gradient from the layer above (dL/dy), size n
- `input`: **Original input** to the forward pass (needed for mask), size n
- `grad_input`: Gradient to pass to the layer below (dL/dx), size n
- `n`: Total number of elements

**Launch configuration (both kernels):**
```cuda
int threads = 256;
int blocks = (n + threads - 1) / threads;
relu_forward<<<blocks, threads>>>(d_input, d_output, n);
relu_backward<<<blocks, threads>>>(d_grad_output, d_input, d_grad_input, n);
```

---

## Step-by-Step Guide

### Forward kernel:

1. **Calculate your index**: `int i = blockIdx.x * blockDim.x + threadIdx.x`
2. **Bounds check**: If `i >= n`, return immediately
3. **Apply ReLU**: `output[i] = (input[i] > 0.0f) ? input[i] : 0.0f`

### Backward kernel:

1. **Calculate your index**: `int i = blockIdx.x * blockDim.x + threadIdx.x`
2. **Bounds check**: If `i >= n`, return immediately
3. **Apply gradient mask**: `grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f`

Note: Both kernels use the **same** mask condition (`input[i] > 0`), but
the forward applies it to the input value while the backward applies it
to the incoming gradient.

---

## Hints

<details>
<summary>Hint 1 (Mild): Thread index and bounds</summary>

Use the standard 1D grid pattern from Puzzle 01:
```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= n) return;
```
</details>

<details>
<summary>Hint 2 (Medium): Forward pass</summary>

ReLU forward is a simple conditional:
```cuda
output[i] = fmaxf(0.0f, input[i]);
// or equivalently:
output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
```
</details>

<details>
<summary>Hint 3 (Strong): Backward pass — almost complete</summary>

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
}
```
The key insight: you must use the **original input** (not the output)
to decide whether to pass the gradient through.
</details>

---

## Testing

Build and run the tests to verify your solution:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target puzzle_04_test
./build/puzzle_04_test
```

The test suite runs 4 tests:

1. **relu_forward_neg_zero_pos**: Forward pass with negative, zero, and positive values
2. **relu_backward_gradient_routing**: Backward pass — verifies gradient masking
3. **relu_round_trip**: Forward then backward — end-to-end gradient flow
4. **relu_edge_case_zero**: Boundary behavior at exactly x=0

All 4 tests must pass for the puzzle to be complete.
