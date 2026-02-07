# Puzzle 10: Conv2D Backward Pass — Input Gradient

## Overview

Implement the **backward pass for the input** of a 2D convolution on the GPU.
During training, after computing the forward pass (`output = conv(input, filters)`),
backpropagation needs to flow gradients from the output back to the input.
This puzzle computes `grad_input` — how the loss changes with respect to each
input pixel.

**Why this matters for LeNet:**
When training LeNet, gradients flow backward through each layer. The conv
backward-input pass propagates gradients from feature maps back to the
preceding layer's activations (or the raw image for Conv1):

```
Loss → ... → grad_output (24×24×6) → Conv1 backward → grad_input (28×28×1)
                                           ↑
                                    This is what we're building!

Loss → ... → grad_output (8×8×16)  → Conv2 backward → grad_input (12×12×6)
```

**Connection to Puzzle 07 (Conv Forward):**
The forward pass computes `output = conv(input, W)`.
The backward-input pass reverses this: given `grad_output`, compute
`grad_input` — the gradient that flows to the previous layer.

```
Forward:   input ──[conv with W]──→ output
Backward:  grad_input ←──[conv backward]── grad_output
```

---

## The Math: Deriving grad_input

### Starting point: the forward pass

Recall from Puzzle 07:

```
out[b][k][oh][ow] = bias[k]
                   + Σ_{c}  Σ_{fh}  Σ_{fw}
                       input[b][c][oh+fh][ow+fw] × W[k][c][fh][fw]
```

### Given: upstream gradient

The upstream gradient `grad_output = ∂L/∂output` has the same shape as
the forward output:

```
grad_output: (batch, C_out, H_out, W_out)   — same shape as forward output
```

### Deriving ∂L/∂input

By the chain rule, we need to find every output element that used a
particular input element, and sum up the contributions:

```
∂L/∂input[b][c][h][w] = Σ_k  Σ_{oh,ow where input[b][c][h][w] contributed}
                            grad_output[b][k][oh][ow] × W[k][c][fh][fw]
```

In the forward pass, `input[b][c][h][w]` appears in `out[b][k][oh][ow]`
when `oh + fh = h` and `ow + fw = w`, i.e., when `oh = h - fh` and
`ow = w - fw`. We need `oh` and `ow` to be valid output positions
(0 ≤ oh < H_out, 0 ≤ ow < W_out).

This gives us the **key formula**:

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ∂L/∂input[b][c][h][w] = Σ_{k=0}^{C_out-1}                     │
│                            Σ_{fh=0}^{F-1}                       │
│                              Σ_{fw=0}^{F-1}                     │
│                                grad_output[b][k][h-fh][w-fw]    │
│                                × W[k][c][fh][fw]                │
│                                                                  │
│  where (h-fh) and (w-fw) must be valid output positions:        │
│    0 ≤ h-fh < H_out  AND  0 ≤ w-fw < W_out                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Shape verification

```
grad_input shape = (batch, C_in, H, W)  ✓ same shape as forward input
```

---

## Conceptual View: Full Convolution with Rotated Filter

The grad_input computation can be understood as a **"full" convolution**
of `grad_output` with a **180°-rotated** filter. This is a beautiful
mathematical identity:

### What is a 180° rotation?

Rotating a filter by 180° means reversing both its rows and columns:

```
Original filter W[k][c]:    Rotated filter W_rot[k][c]:

┌───┬───┬───┐               ┌───┬───┬───┐
│ a │ b │ c │               │ i │ h │ g │
├───┼───┼───┤      180°     ├───┼───┼───┤
│ d │ e │ f │    ──────→    │ f │ e │ d │
├───┼───┼───┤               ├───┼───┼───┤
│ g │ h │ i │               │ c │ b │ a │
└───┴───┴───┘               └───┴───┴───┘

W_rot[k][c][fh][fw] = W[k][c][F-1-fh][F-1-fw]
```

### What is "full" padding?

"Full" padding adds (F-1) zeros on all sides of grad_output, so the
convolution can "slide off" the edges:

```
grad_output (3×3):       After full padding with F=3 (pad=2):

┌───┬───┬───┐            ┌───┬───┬───┬───┬───┬───┬───┐
│ a │ b │ c │            │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┤            ├───┼───┼───┼───┼───┼───┼───┤
│ d │ e │ f │   pad=2    │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┤  ──────→   ├───┼───┼───┼───┼───┼───┼───┤
│ g │ h │ i │            │ 0 │ 0 │ a │ b │ c │ 0 │ 0 │
└───┴───┴───┘            ├───┼───┼───┼───┼───┼───┼───┤
                         │ 0 │ 0 │ d │ e │ f │ 0 │ 0 │
                         ├───┼───┼───┼───┼───┼───┼───┤
                         │ 0 │ 0 │ g │ h │ i │ 0 │ 0 │
                         ├───┼───┼───┼───┼───┼───┼───┤
                         │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
                         ├───┼───┼───┼───┼───┼───┼───┤
                         │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
                         └───┴───┴───┴───┴───┴───┴───┘

Padded size: (H_out + 2*(F-1)) × (W_out + 2*(F-1))
           = (H_out + 2F - 2) × (W_out + 2F - 2)
           = H_in × W_in  (when H_out = H_in - F + 1)  ← output matches input size!
```

### The equivalence

```
grad_input = full_conv(grad_output, rotate_180(W))

Equivalently, for each input element:
  ∂L/∂input[b][c][h][w] = Σ_k conv(padded_grad_output, W_rot)[b][c][h][w]
```

### Why does this work?

In the forward pass, `input[b][c][h][w]` was multiplied by `W[k][c][fh][fw]`
to contribute to `output[b][k][h-fh][w-fw]`. Running the gradient backward
means multiplying `grad_output[b][k][oh][ow]` by the same weight, but now
we're accumulating into the input position. The bounds checking and filter
reversal correspond exactly to full-padding the grad_output and convolving
with the rotated filter.

---

## Practical Implementation: Direct Formula

Rather than literally padding and rotating, we implement the gradient
directly using bounds checking:

```
∂L/∂input[b][c][h][w] = Σ_{k=0}^{C_out-1}
                           Σ_{fh=0}^{F-1}
                             Σ_{fw=0}^{F-1}
                               grad_output[b][k][h-fh][w-fw] × W[k][c][fh][fw]

    with bounds check: skip if (h-fh) < 0  or (h-fh) >= H_out
                               (w-fw) < 0  or (w-fw) >= W_out
```

The bounds check is equivalent to the zero-padding: positions where the
filter "slides off" the grad_output simply contribute zero.

### Visualizing which output positions affect input[h][w]

```
Example: 5×5 input, 3×3 filter → 3×3 output (H_out=3, W_out=3)

For input position (2,2) — the center pixel:

  grad_output (3×3)        Filter W (3×3)
  ┌───┬───┬───┐            ┌───┬───┬───┐
  │g00│g01│g02│            │w00│w01│w02│
  ├───┼───┼───┤            ├───┼───┼───┤
  │g10│g11│g12│            │w10│w11│w12│
  ├───┼───┼───┤            ├───┼───┼───┤
  │g20│g21│g22│            │w20│w21│w22│
  └───┴───┴───┘            └───┴───┴───┘

  grad_input[2][2] = g00·w22 + g01·w21 + g02·w20    (fh=2: oh=0)
                   + g10·w12 + g11·w11 + g12·w10    (fh=1: oh=1)
                   + g20·w02 + g21·w01 + g22·w00    (fh=0: oh=2)

  = Σ over all valid (oh,ow) of grad_output[oh][ow] × W[2-oh+oh...] 
  → All 9 output positions contribute! (center pixel is in every receptive field)

For input position (0,0) — top-left corner:

  Only output position (0,0) used input[0][0] (with filter position fh=0, fw=0):

  grad_input[0][0] = g00·w00
  → Only 1 output position contributes! (corner pixel is in only one receptive field)
```

---

## Memory Layout (NCHW)

All tensors use **NCHW** layout — same as Puzzle 07:

```
grad_output: (batch, C_out, H_out, W_out)  — grad_output[b][k][oh][ow]
filters:     (C_out, C_in,  F,     F)      — filters[k][c][fh][fw]
grad_input:  (batch, C_in,  H,     W)      — grad_input[b][c][h][w]

Where: H_out = H - F + 1,  W_out = W - F + 1
```

### Indexing Formulas

```cuda
// grad_output: grad_output[b*(C_out*H_out*W_out) + k*(H_out*W_out) + oh*W_out + ow]
// filters:     filters[k*(C_in*F*F) + c*(F*F) + fh*F + fw]
// grad_input:  grad_input[b*(C_in*H*W) + c*(H*W) + h*W + w]
```

---

## Thread Mapping: One Thread Per Input Element

Each thread computes one element of `grad_input[b][c][h][w]`.
We flatten the 4D input into a 1D grid:

```
Thread mapping:
  total_elements = batch × C_in × H × W
  idx = blockIdx.x * blockDim.x + threadIdx.x

  Decode 4D indices from flat idx:
    w  = idx % W
    h  = (idx / W) % H
    c  = (idx / (W * H)) % C_in
    b  = idx / (W * H * C_in)
```

---

## Kernel Signature

```cuda
__global__ void conv2d_backward_input(const float* grad_output,
                                       const float* filters,
                                       float* grad_input,
                                       int batch, int C_in, int H, int W,
                                       int C_out, int F);
```

**Parameters:**
- `grad_output`: Upstream gradient, shape (batch, C_out, H_out, W_out) in NCHW
- `filters`:     Filter weights, shape (C_out, C_in, F, F) in NCHW
- `grad_input`:  Output gradient, shape (batch, C_in, H, W) in NCHW
- `batch`:       Number of samples in the batch
- `C_in`:        Number of input channels
- `H`, `W`:      Input spatial height and width
- `C_out`:       Number of output channels (number of filters)
- `F`:           Filter spatial size (square: F×F)

**Launch configuration:**
```cuda
int total = batch * C_in * H * W;
int threads = 256;
int blocks = (total + threads - 1) / threads;
conv2d_backward_input<<<blocks, threads>>>(d_grad_output, d_filters, d_grad_input,
                                            batch, C_in, H, W, C_out, F);
```

---

## Step-by-Step Guide

1. **Compute your flat index**: `idx = blockIdx.x * blockDim.x + threadIdx.x`.

2. **Bounds check**: If `idx >= batch * C_in * H * W`, return.

3. **Decode 4D position** from `idx`: extract `b`, `c`, `h`, `w`
   using integer division and modulo.

4. **Compute H_out, W_out**: `H_out = H - F + 1`, `W_out = W - F + 1`.

5. **Initialize accumulator** to 0.

6. **Triple nested loop**: Over output channels `k`, filter rows `fh`,
   filter columns `fw`:
   - Compute `oh = h - fh`, `ow = w - fw`
   - **Bounds check**: if `oh < 0 || oh >= H_out || ow < 0 || ow >= W_out`, skip
   - Accumulate `grad_output[b][k][oh][ow] × filters[k][c][fh][fw]`

7. **Write result** to `grad_input[b][c][h][w]`.

---

## Hints

<details>
<summary>Hint 1 (Mild): The bounds check replaces padding</summary>

Instead of literally zero-padding grad_output, we skip contributions
where the output index would be out of bounds:
```cuda
int oh = h - fh;
int ow = w - fw;
if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
    // accumulate this contribution
}
```
This is mathematically equivalent to padding grad_output with zeros.
</details>

<details>
<summary>Hint 2 (Medium): The NCHW indexing</summary>

```cuda
int go_idx = b * (C_out * H_out * W_out) + k * (H_out * W_out) + oh * W_out + ow;
int f_idx  = k * (C_in * F * F) + c * (F * F) + fh * F + fw;
```
</details>

<details>
<summary>Hint 3 (Strong): Almost complete solution</summary>

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int total = batch * C_in * H * W;
if (idx >= total) return;

int w  = idx % W;
int h  = (idx / W) % H;
int c  = (idx / (W * H)) % C_in;
int b  = idx / (W * H * C_in);

int H_out = H - F + 1;
int W_out = W - F + 1;

float sum = 0.0f;
for (int k = 0; k < C_out; k++) {
    for (int fh = 0; fh < F; fh++) {
        for (int fw = 0; fw < F; fw++) {
            int oh = h - fh;
            int ow = w - fw;
            if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                sum += grad_output[b*(C_out*H_out*W_out) + k*(H_out*W_out) + oh*W_out + ow]
                     * filters[k*(C_in*F*F) + c*(F*F) + fh*F + fw];
            }
        }
    }
}
grad_input[b*(C_in*H*W) + c*(H*W) + h*W + w] = sum;
```
</details>

---

## Testing

Build and run the tests to verify your solution:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target puzzle_10_test
./build/puzzle_10_test
```

The test suite runs 4 tests:

1. **conv_backward_input_tiny**: Single-channel 5×5 input, 3×3 filter — same setup as Puzzle 07's tiny test, hand-verifiable
2. **conv_backward_input_numerical_gradient_check**: Perturb each input by epsilon, verify analytical vs numerical gradients match (relative error < 1e-2)
3. **conv_backward_input_lenet_conv1**: grad_input shape 28×28×1 — LeNet Conv1 input gradient
4. **conv_backward_input_lenet_conv2**: grad_input shape 12×12×6 — LeNet Conv2 input gradient

All 4 tests must pass for the puzzle to be complete.
