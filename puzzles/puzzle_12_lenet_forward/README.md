# Puzzle 12: LeNet-5 Forward Pass (Full Network)

## Overview

Assemble the complete LeNet-5 convolutional neural network by chaining
all the individual layer kernels you built in Puzzles 03-08. Given a
28x28 grayscale MNIST image, produce 10 class probabilities.

This puzzle is about **orchestration**: calling the right kernels in
the right order with the right buffer sizes. You already know how each
layer works — now wire them together.

## LeNet-5 Architecture

```
 Input        Conv1       ReLU    Pool1       Conv2       ReLU    Pool2
 28x28x1  -> 24x24x6  -> same -> 12x12x6 -> 8x8x16  -> same -> 4x4x16
   |           |                    |           |                  |
   |       6 filters              2x2 max    16 filters          2x2 max
   |       5x5, s=1              stride 2    5x5, s=1           stride 2
   |                                                               |
   |                                                          Flatten
   |                                                               |
   |            FC1        ReLU       FC2       ReLU       FC3     v
   |         256->120   -> same -> 120->84  -> same ->  84->10 <- 256
   |                                                       |
   |                                                    Softmax
   |                                                       |
   v                                                    10 probs
```

## Complete Data Flow with Dimensions

```
Layer       Operation          Input Shape       Output Shape     Params
----------------------------------------------------------------------
Input       (raw image)        batch x 1x28x28   -                 -
Conv1       conv2d 5x5, K=6   batch x 1x28x28   batch x 6x24x24  156
ReLU1       relu               batch x 6x24x24   batch x 6x24x24  0
Pool1       maxpool 2x2        batch x 6x24x24   batch x 6x12x12  0
Conv2       conv2d 5x5, K=16  batch x 6x12x12   batch x 16x8x8   2416
ReLU2       relu               batch x 16x8x8    batch x 16x8x8   0
Pool2       maxpool 2x2        batch x 16x8x8    batch x 16x4x4   0
Flatten     reshape            batch x 16x4x4    batch x 256      0
FC1         linear 256->120    batch x 256        batch x 120      30840
ReLU3       relu               batch x 120        batch x 120      0
FC2         linear 120->84     batch x 120        batch x 84       10164
ReLU4       relu               batch x 84         batch x 84       0
FC3         linear 84->10      batch x 84         batch x 10       850
Softmax     softmax            batch x 10         batch x 10       0
----------------------------------------------------------------------
Total parameters: 44,426
```

## Dimension Derivations

```
Conv1:  H_out = 28 - 5 + 1 = 24       W_out = 28 - 5 + 1 = 24
Pool1:  H_out = 24 / 2     = 12       W_out = 24 / 2     = 12
Conv2:  H_out = 12 - 5 + 1 = 8        W_out = 12 - 5 + 1 = 8
Pool2:  H_out = 8  / 2     = 4        W_out = 8  / 2     = 4
Flat:   16 x 4 x 4 = 256
```

## Parameter Count Breakdown

```
Conv1 filters:  6 x (1x5x5)  = 150  weights + 6 biases  =   156
Conv2 filters: 16 x (6x5x5)  = 2400 weights + 16 biases =  2,416
FC1:           256 x 120      = 30720 weights + 120 biases= 30,840
FC2:           120 x 84       = 10080 weights + 84 biases = 10,164
FC3:            84 x 10       =   840 weights + 10 biases =    850
                                                    Total = 44,426
```

## Memory Allocation Plan

For a forward pass with batch size B, you need these GPU buffers:

```
Buffer              Size (floats)          Example (B=32)
--------------------------------------------------------------
input               B x 1 x 28 x 28       25,088
conv1_out           B x 6 x 24 x 24       110,592
relu1_out           B x 6 x 24 x 24       110,592
pool1_out           B x 6 x 12 x 12       27,648
pool1_indices       B x 6 x 12 x 12       27,648 (ints)
conv2_out           B x 16 x 8 x 8        32,768
relu2_out           B x 16 x 8 x 8        32,768
pool2_out           B x 16 x 4 x 4        8,192
pool2_indices       B x 16 x 4 x 4        8,192 (ints)
flat                (= pool2_out)          (no copy needed)
fc1_out             B x 120                3,840
relu3_out           B x 120                3,840
fc2_out             B x 84                 2,688
relu4_out           B x 84                 2,688
fc3_out             B x 10                 320
probs               B x 10                 320
--------------------------------------------------------------
Total activations:  ~360K floats (~1.4 MB for B=32)

Parameters (constant across batches):
conv1_w: 150, conv1_b: 6
conv2_w: 2400, conv2_b: 16
fc1_w: 30720, fc1_b: 120
fc2_w: 10080, fc2_b: 84
fc3_w: 840, fc3_b: 10
--------------------------------------------------------------
Total parameters: 44,426 floats (~174 KB)
```

## The Flatten Operation

Flattening is a **reshape**, not a computation. The data in memory
after Pool2 is already contiguous in the right order:

```
Pool2 output (NCHW):  batch x 16 x 4 x 4

In memory for one sample:
  [ch0: 4x4 values][ch1: 4x4 values]...[ch15: 4x4 values]
  = 16 x 16 = 256 contiguous floats

FC1 input:            batch x 256

Same memory, different interpretation! No kernel needed.
Just pass the pool2_out pointer directly to fc_forward.
```

## How to Chain the Kernels

```
// You already have these kernels from previous puzzles:
conv2d_forward(input, filters, bias, output, B, C_in, H, W, C_out, F)
relu_forward(input, output, n)
maxpool_forward(input, output, indices, B, C, H, W)
fc_forward(input, weights, bias, output, B, in_feat, out_feat)
softmax_forward(logits, probs, B, num_classes)

// Chain them:
conv2d_forward(input, conv1_w, conv1_b, conv1_out, B, 1, 28, 28, 6, 5)
relu_forward(conv1_out, relu1_out, B*6*24*24)
maxpool_forward(relu1_out, pool1_out, pool1_idx, B, 6, 24, 24)
conv2d_forward(pool1_out, conv2_w, conv2_b, conv2_out, B, 6, 12, 12, 16, 5)
relu_forward(conv2_out, relu2_out, B*16*8*8)
maxpool_forward(relu2_out, pool2_out, pool2_idx, B, 16, 8, 8)
// pool2_out IS the flattened input — 16*4*4 = 256 per sample
fc_forward(pool2_out, fc1_w, fc1_b, fc1_out, B, 256, 120)
relu_forward(fc1_out, relu3_out, B*120)
fc_forward(relu3_out, fc2_w, fc2_b, fc2_out, B, 120, 84)
relu_forward(fc2_out, relu4_out, B*84)
fc_forward(relu4_out, fc3_w, fc3_b, fc3_out, B, 84, 10)
softmax_forward(fc3_out, probs, B, 10)
```

## Step-by-Step Guide

1. **Allocate all GPU buffers** for parameters and intermediate
   activations (see Memory Allocation Plan above)
2. **Copy parameters** (weights, biases) from host to device
3. **Copy input images** from host to device
4. **Call each kernel in sequence** — see "How to Chain the Kernels"
5. **Copy output probabilities** back to host
6. **Free all GPU buffers**

The student's job is to write the `lenet_forward()` function that
orchestrates steps 1-6.

## Hints

**Hint 1 (Mild):** The flatten step is free — pool2_out is already
a contiguous array of 256 floats per sample. Just pass the pointer
to fc_forward.

**Hint 2 (Medium):** Remember that maxpool_forward needs an extra
`int*` buffer for max_indices. You'll need two of these (one for
Pool1, one for Pool2). These indices are needed later for the
backward pass (Puzzle 13), but you must allocate them here.

**Hint 3 (Strong):** The kernel launch parameters for each layer:
- conv2d/maxpool: 1D grid, threads=256, blocks=(total+255)/256
- relu: 1D grid, threads=256, blocks=(n+255)/256
- fc: 2D grid, blockDim(16,16), gridDim for (out_feat, batch)
- softmax: 1D grid, one thread per sample
