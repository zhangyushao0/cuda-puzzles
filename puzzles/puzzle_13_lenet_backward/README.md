# Puzzle 13: LeNet-5 Backward Pass (Full Backpropagation)

## Overview

Implement the **complete backward pass** for the LeNet-5 network from Puzzle 12.
Given the forward-pass activations (saved during the forward pass) and the
cross-entropy loss gradient, compute gradients for **every learnable parameter**
(conv weights/biases, FC weights/biases) by chaining backward kernels in reverse
order through the entire network.

This is the keystone puzzle — it ties together every backward kernel from
Puzzles 04-10 into one coherent backpropagation pipeline.

## Backward Flow Diagram (Reverse of Forward)

```
Forward:
  Input -> Conv1 -> ReLU1 -> Pool1 -> Conv2 -> ReLU2 -> Pool2
       -> FC1 -> ReLU3 -> FC2 -> ReLU4 -> FC3 -> Softmax -> loss

Backward (reverse order):
  ∂L/∂logits ← Softmax+CE backward (probs - one_hot_labels)
       │
  ∂L/∂FC3_in ← FC3 backward  (also produces ∂L/∂fc3_w, ∂L/∂fc3_b)
       │
  ∂L/∂relu4_in ← ReLU4 backward (mask: fc2_out > 0)
       │
  ∂L/∂FC2_in ← FC2 backward  (also produces ∂L/∂fc2_w, ∂L/∂fc2_b)
       │
  ∂L/∂relu3_in ← ReLU3 backward (mask: fc1_out > 0)
       │
  ∂L/∂FC1_in ← FC1 backward  (also produces ∂L/∂fc1_w, ∂L/∂fc1_b)
       │
  ∂L/∂pool2_out ← (= ∂L/∂flat) unflatten is a no-op
       │
  ∂L/∂relu2_out ← Pool2 backward (routes grad to max positions)
       │
  ∂L/∂conv2_out ← ReLU2 backward (mask: conv2_out > 0)
       │
  ∂L/∂pool1_out ← Conv2 backward input (also produces ∂L/∂conv2_w, ∂L/∂conv2_b)
       │
  ∂L/∂relu1_out ← Pool1 backward (routes grad to max positions)
       │
  ∂L/∂conv1_out ← ReLU1 backward (mask: conv1_out > 0)
       │
  (∂L/∂input) ← Conv1 backward input (also produces ∂L/∂conv1_w, ∂L/∂conv1_b)
```

## Forward Intermediates That Must Be Saved

The backward pass requires data saved during the forward pass:

```
Category         Buffer              Used By              Shape (per batch B)
─────────────────────────────────────────────────────────────────────────────
Layer Inputs     d_input             Conv1 weight grad     B × 1 × 28 × 28
                 pool1_out           Conv2 weight grad     B × 6 × 12 × 12
                 pool2_out (=flat)   FC1 weight grad       B × 256
                 relu3_out           FC2 weight grad       B × 120
                 relu4_out           FC3 weight grad       B × 84

ReLU Masks       conv1_out           ReLU1 backward        B × 6 × 24 × 24
(pre-ReLU vals)  conv2_out           ReLU2 backward        B × 16 × 8 × 8
                 fc1_out             ReLU3 backward        B × 120
                 fc2_out             ReLU4 backward        B × 84

Pool Indices     pool1_indices       Pool1 backward        B × 6 × 12 × 12 (int)
                 pool2_indices       Pool2 backward        B × 16 × 4 × 4  (int)

Softmax Probs    probs               Softmax+CE backward   B × 10
```

## Memory Overhead Discussion

The backward pass requires:

**1. Forward activation storage (already allocated in LeNetActivations):**
All intermediate activations from the forward pass must be kept alive until
the backward pass completes. This is the main memory cost of training vs inference.

**2. Gradient buffers for activations (temporary, backward-pass only):**
```
Buffer                  Size (floats)       Example (B=32)
────────────────────────────────────────────────────────────
grad_fc3_out            B × 10              320
grad_relu4_out          B × 84              2,688
grad_fc2_out            B × 84              2,688
grad_relu3_out          B × 120             3,840
grad_fc1_out            B × 120             3,840
grad_pool2_out          B × 256             8,192
grad_relu2_out          B × 16 × 8 × 8     32,768
grad_conv2_out          B × 16 × 8 × 8     32,768
grad_pool1_out          B × 6 × 12 × 12    27,648
grad_relu1_out          B × 6 × 24 × 24    110,592
grad_conv1_out          B × 6 × 24 × 24    110,592
────────────────────────────────────────────────────────────
Total grad activations: ~336K floats (~1.3 MB for B=32)
```

**3. Parameter gradient buffers (same shape as parameters):**
```
Buffer          Size (floats)
──────────────────────────────
grad_conv1_w    150
grad_conv1_b    6
grad_conv2_w    2,400
grad_conv2_b    16
grad_fc1_w      30,720
grad_fc1_b      120
grad_fc2_w      10,080
grad_fc2_b      84
grad_fc3_w      840
grad_fc3_b      10
──────────────────────────────
Total:          44,426 floats (~174 KB)
```

**Total backward memory ≈ forward activations + ~1.5 MB gradient buffers.**
Training uses roughly 2× the memory of inference due to storing both activations
and their gradients. The parameter gradients are small (~174 KB) compared to
activation gradients (~1.3 MB) because parameters are shared across the batch.

## Backward Kernel Chain (in order of execution)

```
Step  Kernel                    Inputs                          Outputs
─────────────────────────────────────────────────────────────────────────────────
 1    softmax_ce_backward       probs, labels                   grad_fc3_out
 2    fc_backward_weights       grad_fc3_out, relu4_out         grad_fc3_w
 3    fc_backward_bias          grad_fc3_out                    grad_fc3_b
 4    fc_backward_input         grad_fc3_out, fc3_w             grad_relu4_out
 5    relu_backward             grad_relu4_out, fc2_out         grad_fc2_out
 6    fc_backward_weights       grad_fc2_out, relu3_out         grad_fc2_w
 7    fc_backward_bias          grad_fc2_out                    grad_fc2_b
 8    fc_backward_input         grad_fc2_out, fc2_w             grad_relu3_out
 9    relu_backward             grad_relu3_out, fc1_out         grad_fc1_out
10    fc_backward_weights       grad_fc1_out, pool2_out         grad_fc1_w
11    fc_backward_bias          grad_fc1_out                    grad_fc1_b
12    fc_backward_input         grad_fc1_out, fc1_w             grad_pool2_out
13    maxpool_backward          grad_pool2_out, pool2_indices   grad_relu2_out
14    relu_backward             grad_relu2_out, conv2_out       grad_conv2_out
15    conv2d_backward_weights   grad_conv2_out, pool1_out       grad_conv2_w
16    conv2d_backward_bias      grad_conv2_out                  grad_conv2_b
17    conv2d_backward_input     grad_conv2_out, conv2_w         grad_pool1_out
18    maxpool_backward          grad_pool1_out, pool1_indices   grad_relu1_out
19    relu_backward             grad_relu1_out, conv1_out       grad_conv1_out
20    conv2d_backward_weights   grad_conv1_out, d_input         grad_conv1_w
21    conv2d_backward_bias      grad_conv1_out                  grad_conv1_b
```

## Hints

**Hint 1 (Mild):** The unflatten step is free — just like flatten in the
forward pass, it's a reshape. grad_pool2_out is the same pointer as grad_flat.

**Hint 2 (Medium):** `maxpool_backward` requires its `grad_input` buffer
to be **zero-initialized** before the kernel runs (use `cudaMemset`).
The kernel only writes to max positions, leaving other positions untouched.

**Hint 3 (Strong):** The ReLU backward uses the **pre-ReLU** activation
(e.g., `conv1_out`, not `relu1_out`) to determine the mask. The condition
is `input[i] > 0`, matching the forward pass condition.
