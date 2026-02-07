# Puzzle 07: 2D Convolution (Forward Pass)

## Overview

Implement a **2D convolution forward pass** on the GPU — the core
operation of every convolutional neural network. A convolution slides a
small **filter** (kernel) across a 2D input and computes a weighted sum
at each position, producing a feature map that detects local patterns
like edges, corners, and textures.

**Why this matters for LeNet:**
LeNet-5 has two convolutional layers that transform raw pixels into
increasingly abstract feature maps:

```
Input(1×28×28) → Conv1(6×5×5) → 6×24×24 → Pool → 6×12×12
                                                    ↓
                                           Conv2(16×5×5) → 16×8×8
                                           ^^^^^^^^^^^^
                                           This is what we're building!
```

**Connection to Puzzle 03:**
A fully connected layer connects *every* input to *every* output.
A convolution connects only a *local region* (the receptive field)
to each output — this weight sharing is what makes CNNs efficient.

```
FC layer:     every input  →  every output   (global connections)
Conv layer:   local patch  →  one output     (sliding window)
                ↑
          weight sharing — same filter reused at every position
```

---

## What Does a Convolution Do?

A convolution slides a small filter over the input and computes a
**dot product** at each position:

### Sliding Window Animation (3×3 filter on 5×5 input)

**Step 1: Filter at position (0,0)**
```
  Input (5×5)              Filter (3×3)          Output (3×3)
  ┌───┬───┬───┬───┬───┐   ┌───┬───┬───┐        ┌───┬───┬───┐
  │ 1 │ 2 │ 3 │ . │ . │   │ a │ b │ c │        │ ● │   │   │
  ├───┼───┼───┼───┼───┤   ├───┼───┼───┤   →    ├───┼───┼───┤
  │ 4 │ 5 │ 6 │ . │ . │   │ d │ e │ f │        │   │   │   │
  ├───┼───┼───┼───┼───┤   ├───┼───┼───┤        ├───┼───┼───┤
  │ 7 │ 8 │ 9 │ . │ . │   │ g │ h │ i │        │   │   │   │
  ├───┼───┼───┼───┼───┤   └───┴───┴───┘        └───┴───┴───┘
  │ . │ . │ . │ . │ . │
  ├───┼───┼───┼───┼───┤   ● = 1·a + 2·b + 3·c
  │ . │ . │ . │ . │ . │     + 4·d + 5·e + 6·f
  └───┴───┴───┴───┴───┘     + 7·g + 8·h + 9·i
```

**Step 2: Filter slides right → position (0,1)**
```
  Input (5×5)              Filter (3×3)          Output (3×3)
  ┌───┬───┬───┬───┬───┐   ┌───┬───┬───┐        ┌───┬───┬───┐
  │ . │ 2 │ 3 │ 4 │ . │   │ a │ b │ c │        │ ✓ │ ● │   │
  ├───┼───┼───┼───┼───┤   ├───┼───┼───┤   →    ├───┼───┼───┤
  │ . │ 5 │ 6 │ 7 │ . │   │ d │ e │ f │        │   │   │   │
  ├───┼───┼───┼───┼───┤   ├───┼───┼───┤        ├───┼───┼───┤
  │ . │ 8 │ 9 │10 │ . │   │ g │ h │ i │        │   │   │   │
  ├───┼───┼───┼───┼───┤   └───┴───┴───┘        └───┴───┴───┘
  │ . │ . │ . │ . │ . │
  ├───┼───┼───┼───┼───┤   ● = 2·a + 3·b + 4·c
  │ . │ . │ . │ . │ . │     + 5·d + 6·e + 7·f
  └───┴───┴───┴───┴───┘     + 8·g + 9·h + 10·i
```

**Step 3: Filter slides right → position (0,2)**
```
  Input (5×5)              Filter (3×3)          Output (3×3)
  ┌───┬───┬───┬───┬───┐   ┌───┬───┬───┐        ┌───┬───┬───┐
  │ . │ . │ 3 │ 4 │ 5 │   │ a │ b │ c │        │ ✓ │ ✓ │ ● │
  ├───┼───┼───┼───┼───┤   ├───┼───┼───┤   →    ├───┼───┼───┤
  │ . │ . │ 6 │ 7 │ 8 │   │ d │ e │ f │        │   │   │   │
  ├───┼───┼───┼───┼───┤   ├───┼───┼───┤        ├───┼───┼───┤
  │ . │ . │ 9 │10 │11 │   │ g │ h │ i │        │   │   │   │
  ├───┼───┼───┼───┼───┤   └───┴───┴───┘        └───┴───┴───┘
  │ . │ . │ . │ . │ . │
  ├───┼───┼───┼───┼───┤   ● = 3·a + 4·b + 5·c   ← first row done!
  │ . │ . │ . │ . │ . │     + 6·d + 7·e + 8·f     filter drops down
  └───┴───┴───┴───┴───┘     + 9·g + 10·h + 11·i   to next row...
```

**Step 4: Filter drops to position (1,0)**
```
  Input (5×5)              Filter (3×3)          Output (3×3)
  ┌───┬───┬───┬───┬───┐   ┌───┬───┬───┐        ┌───┬───┬───┐
  │ . │ . │ . │ . │ . │   │ a │ b │ c │        │ ✓ │ ✓ │ ✓ │
  ├───┼───┼───┼───┼───┤   ├───┼───┼───┤   →    ├───┼───┼───┤
  │ 4 │ 5 │ 6 │ . │ . │   │ d │ e │ f │        │ ● │   │   │
  ├───┼───┼───┼───┼───┤   ├───┼───┼───┤        ├───┼───┼───┤
  │ 7 │ 8 │ 9 │ . │ . │   │ g │ h │ i │        │   │   │   │
  ├───┼───┼───┼───┼───┤   └───┴───┴───┘        └───┴───┴───┘
  │10 │11 │12 │ . │ . │
  ├───┼───┼───┼───┼───┤   ...and so on until all 3×3 = 9 outputs
  │ . │ . │ . │ . │ . │   are computed.
  └───┴───┴───┴───┴───┘
```

---

## Output Dimension Formula

With no padding and stride = 1, the filter must fit entirely inside the input:

```
H_out = H_in - F + 1
W_out = W_in - F + 1

Where:
  H_in, W_in = input spatial dimensions
  F          = filter size (square: F×F)
```

### Derivation — Why This Formula?

Consider a 1D case: an input of length 7 and a filter of length 3.

```
Position 0:  [X X X] . . . .    filter starts at index 0
Position 1:  . [X X X] . . .    filter starts at index 1
Position 2:  . . [X X X] . .    filter starts at index 2
Position 3:  . . . [X X X] .    filter starts at index 3
Position 4:  . . . . [X X X]    filter starts at index 4  ← last valid
                                                             position
Position 5:  . . . . . [X X ??  ← would go OUT OF BOUNDS!

Last valid start = 7 - 3 = 4
Number of positions = 4 - 0 + 1 = 5 = (7 - 3) + 1
```

The same logic applies independently to height and width in 2D.

### LeNet Dimensions

```
Conv1: input 28×28, filter 5×5  →  output (28 - 5 + 1) × (28 - 5 + 1) = 24×24
Conv2: input 12×12, filter 5×5  →  output (12 - 5 + 1) × (12 - 5 + 1) =  8×8
```

---

## Multi-Channel Convolution

Real convolutions operate on **multi-channel** inputs (e.g., RGB has
3 channels). Each output channel (filter) processes *all* input
channels and sums the results:

```
    MULTI-CHANNEL CONVOLUTION
    ─────────────────────────

    Input: C_in channels               Filters: C_out sets of C_in filters
    ┌─────────┐                         ┌─────────┐ ┌─────────┐
    │ chan 0   │                         │filter 0  │ │filter 1  │ ... C_out filters
    │ (H × W) │                         │(C_in×F×F)│ │(C_in×F×F)│
    ├─────────┤                         └────┬─────┘ └────┬─────┘
    │ chan 1   │                              │            │
    │ (H × W) │       For output channel k:  │            │
    ├─────────┤       ──────────────────────  │            │
    │  ...     │                              ▼            ▼
    ├─────────┤       Σ over C_in channels   out[k]      out[k+1]
    │chan C-1  │       + bias[k]             (H_out×W_out)
    └─────────┘

    out[b][k][h][w] = bias[k]
                    + Σ_{c=0}^{C_in-1}
                        Σ_{fh=0}^{F-1}
                          Σ_{fw=0}^{F-1}
                            input[b][c][h+fh][w+fw] × filter[k][c][fh][fw]
```

### LeNet Conv Layer Specifics

```
Conv1:  C_in=1  → C_out=6    filter shape: (6, 1, 5, 5)    +  6 biases
        1 input channel (grayscale)
        6 output feature maps, each 24×24

Conv2:  C_in=6  → C_out=16   filter shape: (16, 6, 5, 5)   + 16 biases
        6 input channels (from Conv1 + pooling)
        16 output feature maps, each 8×8
```

---

## NCHW Memory Layout

All tensors use **NCHW** (batch, channels, height, width) layout —
the standard for CUDA convolution:

```
NCHW Layout:  data[b][c][h][w] = data[ b*(C*H*W) + c*(H*W) + h*W + w ]

    Dimension order in memory (innermost → outermost):
    ←──────── fastest varying (contiguous) ──────────→
         w          h           c            b
     (width)    (height)   (channels)    (batch)

    Example: input[1][2][3][4] with shape (B=2, C=3, H=8, W=8)
    = data[1 * (3*8*8) + 2 * (8*8) + 3 * 8 + 4]
    = data[192 + 128 + 24 + 4]
    = data[348]
```

### Tensor Shapes (NCHW)

```
Input:   (batch, C_in,  H_in,  W_in)      input[b][c][h][w]
Filter:  (C_out, C_in,  F,     F)         filter[k][c][fh][fw]
Bias:    (C_out)                           bias[k]
Output:  (batch, C_out, H_out, W_out)     output[b][k][oh][ow]
```

### Indexing Formulas

```cuda
// Input:  input[b * (C_in * H * W) + c * (H * W) + h * W + w]
// Filter: filter[k * (C_in * F * F) + c * (F * F) + fh * F + fw]
// Output: output[b * (C_out * H_out * W_out) + k * (H_out * W_out) + oh * W_out + ow]
```

---

## The Math

### Per-element formula

```
out[b][k][oh][ow] = bias[k]
                   + Σ_{c=0}^{C_in-1}
                       Σ_{fh=0}^{F-1}
                         Σ_{fw=0}^{F-1}
                           input[b][c][oh+fh][ow+fw] × filter[k][c][fh][fw]

where:
  b   = batch index            (0 to batch-1)
  k   = output channel index   (0 to C_out-1)
  oh  = output height position (0 to H_out-1)
  ow  = output width position  (0 to W_out-1)
  c   = input channel index    (0 to C_in-1)
  fh  = filter height offset   (0 to F-1)
  fw  = filter width offset    (0 to F-1)
```

---

## Thread Mapping: One Thread Per Output Element

Each thread computes one output element `output[b][k][oh][ow]`.
We flatten the 4D output into a 1D grid:

```
Thread mapping:
  total_outputs = batch × C_out × H_out × W_out
  idx = blockIdx.x * blockDim.x + threadIdx.x

  Decode 4D indices from flat idx:
    ow = idx % W_out
    oh = (idx / W_out) % H_out
    k  = (idx / (W_out * H_out)) % C_out
    b  = idx / (W_out * H_out * C_out)
```

---

## Kernel Signature

```cuda
__global__ void conv2d_forward(const float* input, const float* filters,
                               const float* bias, float* output,
                               int batch, int C_in, int H, int W,
                               int C_out, int F);
```

**Parameters:**
- `input`:   Input tensor, shape (batch, C_in, H, W) in NCHW layout
- `filters`: Filter weights, shape (C_out, C_in, F, F) in NCHW layout
- `bias`:    Bias vector, size C_out
- `output`:  Output tensor, shape (batch, C_out, H_out, W_out) in NCHW layout
- `batch`:   Number of samples in the batch
- `C_in`:    Number of input channels
- `H`, `W`:  Input spatial height and width
- `C_out`:   Number of output channels (number of filters)
- `F`:       Filter spatial size (square: F×F)

**Launch configuration:**
```cuda
int H_out = H - F + 1;
int W_out = W - F + 1;
int total = batch * C_out * H_out * W_out;
int threads = 256;
int blocks = (total + threads - 1) / threads;
conv2d_forward<<<blocks, threads>>>(d_input, d_filters, d_bias, d_output,
                                     batch, C_in, H, W, C_out, F);
```

---

## Step-by-Step Guide

1. **Compute your flat index**: `idx = blockIdx.x * blockDim.x + threadIdx.x`.

2. **Bounds check**: If `idx >= batch * C_out * H_out * W_out`, return.

3. **Decode 4D position** from `idx`: extract `b`, `k`, `oh`, `ow`
   using integer division and modulo.

4. **Initialize accumulator** with `bias[k]`.

5. **Triple nested loop**: Over input channels `c`, filter rows `fh`,
   filter columns `fw` — accumulate `input[b][c][oh+fh][ow+fw] × filter[k][c][fh][fw]`.

6. **Write result** to `output[b][k][oh][ow]`.

---

## Hints

<details>
<summary>Hint 1 (Mild): Decoding 4D indices from a flat index</summary>

The output shape is `(batch, C_out, H_out, W_out)`. To decode:
```cuda
int H_out = H - F + 1;
int W_out = W - F + 1;
int ow = idx % W_out;
int oh = (idx / W_out) % H_out;
int k  = (idx / (W_out * H_out)) % C_out;
int b  = idx / (W_out * H_out * C_out);
```
</details>

<details>
<summary>Hint 2 (Medium): The NCHW indexing</summary>

For input with shape (batch, C_in, H, W):
```cuda
int input_idx  = b * (C_in * H * W) + c * (H * W) + (oh + fh) * W + (ow + fw);
int filter_idx = k * (C_in * F * F) + c * (F * F) + fh * F + fw;
```
</details>

<details>
<summary>Hint 3 (Strong): Almost complete solution</summary>

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int H_out = H - F + 1;
int W_out = W - F + 1;
int total = batch * C_out * H_out * W_out;
if (idx >= total) return;

int ow = idx % W_out;
int oh = (idx / W_out) % H_out;
int k  = (idx / (W_out * H_out)) % C_out;
int b  = idx / (W_out * H_out * C_out);

float sum = bias[k];
for (int c = 0; c < C_in; c++) {
    for (int fh = 0; fh < F; fh++) {
        for (int fw = 0; fw < F; fw++) {
            sum += input[b*(C_in*H*W) + c*(H*W) + (oh+fh)*W + (ow+fw)]
                 * filters[k*(C_in*F*F) + c*(F*F) + fh*F + fw];
        }
    }
}
output[b*(C_out*H_out*W_out) + k*(H_out*W_out) + oh*W_out + ow] = sum;
```
</details>

---

## Testing

Build and run the tests to verify your solution:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target puzzle_07_test
./build/puzzle_07_test
```

The test suite runs 5 tests:

1. **conv_single_3x3**: Single-channel 5×5 input with 3×3 filter — hand-verifiable
2. **conv_lenet_conv1**: 28×28×1 → 24×24×6 — LeNet Conv1 dimensions
3. **conv_lenet_conv2**: 12×12×6 → 8×8×16 — LeNet Conv2 dimensions
4. **conv_batch_processing**: Batch of 8 images through Conv1
5. **conv_output_dimensions**: Verifies H_out and W_out match expected formula

All 5 tests must pass for the puzzle to be complete.
