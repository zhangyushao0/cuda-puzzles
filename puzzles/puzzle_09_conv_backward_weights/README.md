# Puzzle 09: Conv2D Backward — Weight & Bias Gradients

## Overview

Implement CUDA kernels for the **weight and bias gradients** of a 2D
convolution — the core of training any CNN. During backpropagation,
given the upstream gradient `∂L/∂out`, we need to compute how to update
each filter weight and each bias to reduce the loss.

**Why this matters for LeNet:**
LeNet-5 has two convolutional layers. To train them, we need gradients
for every filter weight and bias:

```
Forward:  Input(1×28×28) → Conv1(6×5×5) → 6×24×24 → Pool → Conv2(16×5×5) → 16×8×8
                               ↑                                ↑
Backward: ∂L/∂W₁ ← ∂L/∂out₁               ∂L/∂W₂ ← ∂L/∂out₂
          ∂L/∂b₁ ← ∂L/∂out₁               ∂L/∂b₂ ← ∂L/∂out₂
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    This is what we're building!
```

**Connection to Puzzle 07 (Forward) and Puzzle 06 (FC Backward):**
Just as the FC backward pass (Puzzle 06) computed `dW = dY^T · X`, the
conv weight gradient computes a **correlation** between the input and the
upstream gradient — but with spatial sliding instead of matrix multiply.

---

## What Is the Weight Gradient?

The weight gradient tells us: *"How much does each filter weight
contribute to the loss?"* It's computed as a **correlation** between
the input activations and the upstream gradient.

### Key Insight: Which Inputs Affect Each Weight?

Consider a single 3×3 filter sliding over a 5×5 input producing a 3×3 output.
**Weight `W[fh][fw]`** is multiplied with a different input element at
every output position — the gradient sums all those contributions:

```
  For weight W[0][0] — which inputs contribute?
  ─────────────────────────────────────────────

  Output position (0,0): W[0][0] × in[0][0]  →  grad_out[0][0] × in[0][0]
  Output position (0,1): W[0][0] × in[0][1]  →  grad_out[0][1] × in[0][1]
  Output position (0,2): W[0][0] × in[0][2]  →  grad_out[0][2] × in[0][2]
  Output position (1,0): W[0][0] × in[1][0]  →  grad_out[1][0] × in[1][0]
  Output position (1,1): W[0][0] × in[1][1]  →  grad_out[1][1] × in[1][1]
  ...
  Output position (2,2): W[0][0] × in[2][2]  →  grad_out[2][2] × in[2][2]

  ∂L/∂W[0][0] = Σ_{h,w} grad_out[h][w] × in[h+0][w+0]
               = Σ_{h,w} grad_out[h][w] × in[h][w]
```

### ASCII Diagram: Input Regions Contributing to Each Weight

```
  Input (5×5)                          grad_output (3×3)
  ┌───┬───┬───┬───┬───┐               ┌───┬───┬───┐
  │ a │ b │ c │ d │ e │               │ G │ H │ I │
  ├───┼───┼───┼───┼───┤               ├───┼───┼───┤
  │ f │ g │ h │ i │ j │               │ J │ K │ L │
  ├───┼───┼───┼───┼───┤               ├───┼───┼───┤
  │ k │ l │ m │ n │ o │               │ M │ N │ O │
  ├───┼───┼───┼───┼───┤               └───┴───┴───┘
  │ p │ q │ r │ s │ t │
  ├───┼───┼───┼───┼───┤
  │ u │ v │ w │ x │ y │
  └───┴───┴───┴───┴───┘

  Weight W[fh][fw] uses input patch starting at (fh, fw):

  ∂L/∂W[0][0] = G·a + H·b + I·c       ← input[0:3, 0:3]
              + J·f + K·g + L·h
              + M·k + N·l + O·m

  ∂L/∂W[0][1] = G·b + H·c + I·d       ← input[0:3, 1:4]
              + J·g + K·h + L·i
              + M·l + N·m + O·n

  ∂L/∂W[1][0] = G·f + H·g + I·h       ← input[1:4, 0:3]
              + J·k + K·l + L·m
              + M·p + N·q + O·r

  Pattern: ∂L/∂W[fh][fw] = Σ_{h,w} grad_out[h][w] × in[h+fh][w+fw]
           This is a CORRELATION (not convolution — no filter flip!)
```

---

## The Math

### Weight Gradient Formula

```
∂L/∂W[k][c][fh][fw] = Σ_{b=0}^{B-1}
                         Σ_{h=0}^{H_out-1}
                           Σ_{w=0}^{W_out-1}
                             grad_out[b][k][h][w] × input[b][c][h+fh][w+fw]

where:
  k   = output channel (filter index)   (0 to K-1)
  c   = input channel                   (0 to C-1)
  fh  = filter height offset            (0 to F-1)
  fw  = filter width offset             (0 to F-1)
  b   = batch index                     (0 to B-1)
  h,w = output spatial position         (0 to H_out-1, 0 to W_out-1)
  H_out = H - F + 1
  W_out = W - F + 1
```

### Bias Gradient Formula

The bias gradient is simpler — each bias affects all spatial positions
in its output channel, so we just sum over batch and spatial dims:

```
∂L/∂bias[k] = Σ_{b=0}^{B-1}
                Σ_{h=0}^{H_out-1}
                  Σ_{w=0}^{W_out-1}
                    grad_out[b][k][h][w]

This is just the sum of all elements in grad_output for channel k,
across all batches and spatial positions.
```

### Shape Summary

```
Input:       (B, C, H, W)
Filters:     (K, C, F, F)
grad_output: (B, K, H_out, W_out)     ← from upstream

∂L/∂W:      (K, C, F, F)              ← same shape as Filters
∂L/∂bias:   (K)                       ← same shape as bias
```

---

## NCHW Memory Layout

Same layout as Puzzle 07:

```
input[b][c][h][w]         = input[b*(C*H*W) + c*(H*W) + h*W + w]
grad_out[b][k][h][w]      = grad_out[b*(K*H_out*W_out) + k*(H_out*W_out) + h*W_out + w]
grad_weights[k][c][fh][fw]= grad_weights[k*(C*F*F) + c*(F*F) + fh*F + fw]
```

---

## Thread Mapping

### Kernel 1: `conv2d_backward_weights`

One thread per weight element `grad_weights[k][c][fh][fw]`:

```
total_weights = K × C × F × F
idx = blockIdx.x * blockDim.x + threadIdx.x

Decode 4D indices from flat idx:
  fw = idx % F
  fh = (idx / F) % F
  c  = (idx / (F * F)) % C
  k  = idx / (F * F * C)

Each thread loops over batch × H_out × W_out to accumulate:
  sum += grad_out[b][k][h][w] × input[b][c][h+fh][w+fw]
```

### Kernel 2: `conv2d_backward_bias`

One thread per output channel:

```
k = blockIdx.x * blockDim.x + threadIdx.x

Each thread loops over batch × H_out × W_out to accumulate:
  sum += grad_out[b][k][h][w]
```

---

## Kernel Signatures

```cuda
__global__ void conv2d_backward_weights(
    const float* grad_output, const float* input,
    float* grad_weights,
    int batch, int C_in, int H, int W,
    int C_out, int F);

__global__ void conv2d_backward_bias(
    const float* grad_output, float* grad_bias,
    int batch, int C_out, int H_out, int W_out);
```

**Launch configurations:**
```cuda
// Weights kernel
int total_weights = C_out * C_in * F * F;
int threads = 256;
int blocks = (total_weights + threads - 1) / threads;
conv2d_backward_weights<<<blocks, threads>>>(...);

// Bias kernel
int bias_blocks = (C_out + 255) / 256;
conv2d_backward_bias<<<bias_blocks, 256>>>(...);
```

---

## Step-by-Step Guide

### Kernel 1: Weight Gradient

1. **Compute flat index**: `idx = blockIdx.x * blockDim.x + threadIdx.x`
2. **Bounds check**: If `idx >= K * C * F * F`, return
3. **Decode 4D position**: Extract `k`, `c`, `fh`, `fw` from `idx`
4. **Initialize sum** to 0
5. **Triple nested loop**: Over `b`, `h`, `w` — accumulate
   `grad_out[b][k][h][w] × input[b][c][h+fh][w+fw]`
6. **Write** `sum` to `grad_weights[k][c][fh][fw]`

### Kernel 2: Bias Gradient

1. **Compute channel index**: `k = blockIdx.x * blockDim.x + threadIdx.x`
2. **Bounds check**: If `k >= K`, return
3. **Initialize sum** to 0
4. **Triple nested loop**: Over `b`, `h`, `w` — accumulate
   `grad_out[b][k][h][w]`
5. **Write** `sum` to `grad_bias[k]`

---

## Hints

<details>
<summary>Hint 1 (Mild): Decoding weight indices</summary>

The weight gradient shape is `(K, C, F, F)`. To decode from flat index:
```cuda
int fw = idx % F;
int fh = (idx / F) % F;
int c  = (idx / (F * F)) % C_in;
int k  = idx / (F * F * C_in);
```
</details>

<details>
<summary>Hint 2 (Medium): The accumulation loop</summary>

Each thread (one weight element) must sum over all positions where that
weight was used in the forward pass:
```cuda
float sum = 0.0f;
for (int b = 0; b < batch; b++) {
    for (int h = 0; h < H_out; h++) {
        for (int w = 0; w < W_out; w++) {
            int go_idx = b*(C_out*H_out*W_out) + k*(H_out*W_out) + h*W_out + w;
            int in_idx = b*(C_in*H_in*W_in) + c*(H_in*W_in) + (h+fh)*W_in + (w+fw);
            sum += grad_output[go_idx] * input[in_idx];
        }
    }
}
```
</details>

<details>
<summary>Hint 3 (Strong): Almost complete weight gradient kernel</summary>

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int H_out = H - F + 1;
int W_out = W - F + 1;
int total = C_out * C_in * F * F;
if (idx >= total) return;

int fw = idx % F;
int fh = (idx / F) % F;
int c  = (idx / (F * F)) % C_in;
int k  = idx / (F * F * C_in);

float sum = 0.0f;
for (int b = 0; b < batch; b++) {
    for (int h = 0; h < H_out; h++) {
        for (int w = 0; w < W_out; w++) {
            sum += grad_output[b*(C_out*H_out*W_out) + k*(H_out*W_out) + h*W_out + w]
                 * input[b*(C_in*H*W) + c*(H*W) + (h+fh)*W + (w+fw)];
        }
    }
}
grad_weights[k*(C_in*F*F) + c*(F*F) + fh*F + fw] = sum;
```
</details>

---

## Testing

Build and run the tests to verify your solution:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target puzzle_09_test
./build/puzzle_09_test
```

The test suite runs 4 tests:

1. **conv_bw_tiny_3x3**: Single-channel 5×5 input with 3×3 filter — hand-verifiable
2. **conv_bw_numerical_gradient_check**: Perturb each weight by ε, verify relative error < 1e-3
3. **conv_bw_lenet_conv1**: LeNet Conv1 dims — batch=1, 1×28×28 → 6×5×5 weight gradient
4. **conv_bw_lenet_conv2**: LeNet Conv2 dims — batch=1, 6×12×12 → 16×5×5 weight gradient

All 4 tests must pass for the puzzle to be complete.
