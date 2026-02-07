# Puzzle 08: Max Pooling (Forward + Backward)

## Overview

Implement **2×2 max pooling** with stride 2 on the GPU — both the forward
pass and the backward pass. Max pooling downsamples spatial dimensions by
selecting the maximum value in each non-overlapping 2×2 window, reducing
computation in subsequent layers while preserving the strongest activations.

**Why this matters for LeNet:**
LeNet-5 uses max pooling after each convolutional layer to halve spatial
dimensions:

```
Conv1 output (6×24×24) → Pool1 → 6×12×12
                                    ↓
                          Conv2 → 16×8×8 → Pool2 → 16×4×4
                                                      ↓
                                                   Flatten → FC layers
```

**Connection to Puzzle 07:**
Conv2D produces feature maps; max pooling reduces their spatial size.
Together they form the "conv → pool" pattern repeated throughout CNNs.

---

## What Does Max Pooling Do?

### Forward Pass: Select the Maximum

A 2×2 max pool with stride 2 slides a 2×2 window across the input,
taking the **maximum** from each window:

```
  Input (4×4)                     Output (2×2)
  ┌────┬────┬────┬────┐          ┌────┬────┐
  │  1 │  3 │  5 │  2 │          │    │    │
  ├────┼────┼────┼────┤    →     │  4 │  6 │
  │  4 │  2 │  6 │  1 │          ├────┼────┤
  ├────┼────┼────┼────┤          │  8 │  9 │
  │  7 │  8 │  3 │  9 │          └────┴────┘
  ├────┼────┼────┼────┤
  │  5 │  6 │  4 │  7 │
  └────┴────┴────┴────┘

  Window (0,0): max(1, 3, 4, 2) = 4
  Window (0,1): max(5, 2, 6, 1) = 6
  Window (1,0): max(7, 8, 5, 6) = 8
  Window (1,1): max(3, 9, 4, 7) = 9
```

### Saving Max Indices (Critical for Backward Pass!)

During the forward pass, we must record **which element was the maximum**
in each 2×2 window. This information is needed during backpropagation:

```
  Input (4×4)                 max_indices (2×2)
  ┌────┬────┬────┬────┐      ┌────┬────┐
  │  1 │  3 │  5 │  2 │      │    │    │
  ├────┼────┼────┼────┤  →   │  4 │  6 │    Stores flat offset within
  │ [4]│  2 │ [6]│  1 │      ├────┼────┤    the 2×2 window (0-3):
  ├────┼────┼────┼────┤      │  1 │  3 │      0=top-left, 1=top-right
  │  7 │ [8]│  3 │ [9]│      └────┴────┘      2=bottom-left, 3=bottom-right
  ├────┼────┼────┼────┤
  │  5 │  6 │  4 │  7 │      Window (0,0): max at row=1,col=0 → index 2
  └────┴────┴────┴────┘      Window (0,1): max at row=0,col=0 → index 0
                              Window (1,0): max at row=0,col=1 → index 1
                              Window (1,1): max at row=0,col=1 → index 1
```

**Why save indices?** Without them, the backward pass would have to
re-examine each 2×2 window to find where the max was. Saving indices
makes the backward pass O(1) per output element instead of O(pool_size²).

### Backward Pass: Route Gradients to Max Positions Only

The gradient flows **only** to the position that was the maximum. All
other positions in the 2×2 window receive zero gradient:

```
  grad_output (2×2)            grad_input (4×4)
  ┌────┬────┐                  ┌────┬────┬────┬────┐
  │ g0 │ g1 │                  │  0 │  0 │  0 │  0 │
  ├────┼────┤       →          ├────┼────┼────┼────┤
  │ g2 │ g3 │                  │ g0 │  0 │ g1 │  0 │  ← only max positions
  └────┴────┘                  ├────┼────┼────┼────┤    receive gradient
                               │  0 │ g2 │  0 │ g3 │
                               ├────┼────┼────┼────┤
                               │  0 │  0 │  0 │  0 │
                               └────┴────┴────┴────┘

  grad_input[max_idx] = grad_output[pool_idx]
  grad_input[other]   = 0
```

**Intuition:** Only the maximum element contributed to the output, so only
it should receive the gradient. This is why max pooling creates sparse
gradients — most input positions get zero gradient.

---

## Output Dimension Formula

With pool size P and stride S:

```
H_out = H_in / S
W_out = W_in / S

For LeNet (P=2, S=2):
  Pool1: 24×24 → 12×12
  Pool2:  8×8  →  4×4
```

### LeNet Pooling Dimensions

```
Pool1: input 24×24×6  → output 12×12×6   (after Conv1)
Pool2: input  8×8×16  → output  4×4×16   (after Conv2)
```

---

## NCHW Memory Layout

All tensors use **NCHW** layout, consistent with Conv2D (Puzzle 07):

```
NCHW Layout:  data[b][c][h][w] = data[ b*(C*H*W) + c*(H*W) + h*W + w ]

Tensor Shapes:
  Input:       (batch, C, H_in,  W_in)
  Output:      (batch, C, H_out, W_out)
  Max Indices: (batch, C, H_out, W_out)    ← int array, same shape as output
```

### Indexing Formulas

```cuda
// Input:       input[b*(C*H*W) + c*(H*W) + h*W + w]
// Output:      output[b*(C*H_out*W_out) + c*(H_out*W_out) + oh*W_out + ow]
// Max Indices: max_indices[b*(C*H_out*W_out) + c*(H_out*W_out) + oh*W_out + ow]
```

---

## The Math

### Forward Pass

For each output position `(b, c, oh, ow)`:

```
h_start = oh * stride     (= oh * 2)
w_start = ow * stride     (= ow * 2)

output[b][c][oh][ow] = max over (ph, pw) in {0,1}×{0,1}:
                          input[b][c][h_start + ph][w_start + pw]

max_indices[b][c][oh][ow] = ph * 2 + pw   (offset of max within 2×2 window)
```

### Backward Pass

For each output position `(b, c, oh, ow)`:

```
h_start = oh * stride
w_start = ow * stride
local_idx = max_indices[b][c][oh][ow]     (0, 1, 2, or 3)
ph = local_idx / 2
pw = local_idx % 2

grad_input[b][c][h_start + ph][w_start + pw] += grad_output[b][c][oh][ow]
```

All other positions in the 2×2 window remain zero.

---

## Thread Mapping: One Thread Per Output Element

Each thread processes one output position `(b, c, oh, ow)`:

```
total_outputs = batch × C × H_out × W_out
idx = blockIdx.x * blockDim.x + threadIdx.x

Decode 4D indices from flat idx:
  ow = idx % W_out
  oh = (idx / W_out) % H_out
  c  = (idx / (W_out * H_out)) % C
  b  = idx / (W_out * H_out * C)
```

---

## Kernel Signatures

### Forward Pass

```cuda
__global__ void maxpool_forward(const float* input, float* output,
                                int* max_indices,
                                int batch, int C, int H, int W);
```

**Parameters:**
- `input`:       Input tensor, shape (batch, C, H, W) in NCHW layout
- `output`:      Output tensor, shape (batch, C, H/2, W/2)
- `max_indices`: Index of max element in each 2×2 window (0-3), same shape as output
- `batch`, `C`:  Batch size and number of channels
- `H`, `W`:      Input spatial dimensions

### Backward Pass

```cuda
__global__ void maxpool_backward(const float* grad_output, const int* max_indices,
                                 float* grad_input,
                                 int batch, int C, int H, int W);
```

**Parameters:**
- `grad_output`:  Upstream gradient, shape (batch, C, H/2, W/2)
- `max_indices`:  Saved max positions from forward pass
- `grad_input`:   Output: gradient w.r.t. input, shape (batch, C, H, W)
- `batch`, `C`:   Batch size and number of channels
- `H`, `W`:       **Input** spatial dimensions (grad_input shape)

**Launch configuration:**
```cuda
int H_out = H / 2, W_out = W / 2;
int total = batch * C * H_out * W_out;
int threads = 256;
int blocks = (total + threads - 1) / threads;
maxpool_forward<<<blocks, threads>>>(d_input, d_output, d_max_indices,
                                      batch, C, H, W);
maxpool_backward<<<blocks, threads>>>(d_grad_output, d_max_indices, d_grad_input,
                                       batch, C, H, W);
```

---

## Step-by-Step Guide

### Forward Pass

1. **Compute flat index**: `idx = blockIdx.x * blockDim.x + threadIdx.x`
2. **Bounds check**: If `idx >= batch * C * H_out * W_out`, return
3. **Decode 4D position**: Extract `b`, `c`, `oh`, `ow`
4. **Compute window start**: `h_start = oh * 2`, `w_start = ow * 2`
5. **Find max in 2×2 window**: Loop over `ph` ∈ {0,1}, `pw` ∈ {0,1}
6. **Write output value** and **save max index** (0-3)

### Backward Pass

1. **Compute flat index** and bounds check (same as forward)
2. **Decode 4D position**: Same as forward
3. **Read max index**: `local_idx = max_indices[idx]`
4. **Compute source position**: `ph = local_idx / 2`, `pw = local_idx % 2`
5. **Route gradient**: `grad_input[b][c][h_start+ph][w_start+pw] = grad_output[idx]`

---

## Hints

<details>
<summary>Hint 1 (Mild): Finding max in a 2×2 window</summary>

```cuda
int h_start = oh * 2;
int w_start = ow * 2;
float max_val = -INFINITY;
int max_idx = 0;

for (int ph = 0; ph < 2; ph++) {
    for (int pw = 0; pw < 2; pw++) {
        float val = input[b*(C*H*W) + c*(H*W) + (h_start+ph)*W + (w_start+pw)];
        if (val > max_val) {
            max_val = val;
            max_idx = ph * 2 + pw;
        }
    }
}
```
</details>

<details>
<summary>Hint 2 (Medium): Backward gradient routing</summary>

```cuda
int local_idx = max_indices[out_idx];
int ph = local_idx / 2;
int pw = local_idx % 2;

int h_start = oh * 2;
int w_start = ow * 2;
int in_idx = b*(C*H*W) + c*(H*W) + (h_start+ph)*W + (w_start+pw);
grad_input[in_idx] = grad_output[out_idx];
```
</details>

<details>
<summary>Hint 3 (Strong): Almost complete forward kernel</summary>

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int H_out = H / 2, W_out = W / 2;
int total = batch * C * H_out * W_out;
if (idx >= total) return;

int ow = idx % W_out;
int oh = (idx / W_out) % H_out;
int c  = (idx / (W_out * H_out)) % C;
int b  = idx / (W_out * H_out * C);

int h_start = oh * 2;
int w_start = ow * 2;

float max_val = -INFINITY;
int max_idx = 0;
for (int ph = 0; ph < 2; ph++) {
    for (int pw = 0; pw < 2; pw++) {
        float val = input[b*(C*H*W) + c*(H*W) + (h_start+ph)*W + (w_start+pw)];
        if (val > max_val) {
            max_val = val;
            max_idx = ph * 2 + pw;
        }
    }
}

output[idx] = max_val;
max_indices[idx] = max_idx;
```
</details>

---

## Testing

Build and run the tests to verify your solution:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target puzzle_08_test
./build/puzzle_08_test
```

The test suite runs 4 tests:

1. **maxpool_forward_4x4**: 4×4→2×2 hardcoded — hand-verifiable max selection and indices
2. **maxpool_backward_grad_routing**: Verifies gradient routes ONLY to max positions
3. **maxpool_lenet_dims**: LeNet Pool1 (24×24×6→12×12×6) and Pool2 (8×8×16→4×4×16) dimensions
4. **maxpool_all_equal**: Edge case where all elements in a window are equal

All 4 tests must pass for the puzzle to be complete.
