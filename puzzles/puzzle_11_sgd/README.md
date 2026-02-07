# Puzzle 11: SGD Optimizer

## Overview

Implement **two** CUDA kernels for **Stochastic Gradient Descent (SGD)**: the
parameter update step and gradient zeroing. This is the final piece of the
training loop — after all forward and backward passes have computed gradients,
SGD uses those gradients to **actually change the weights**.

**Why this matters for LeNet:**
Every training iteration follows this pattern:

```
Forward pass → Loss → Backward pass → SGD update → Zero gradients → Repeat
  (Puzzles 1-7)  (P5)   (Puzzles 4,6)   ^^^^^^^^     ^^^^^^^^^^^^
                                          THIS PUZZLE!
```

**Connection to all previous backward passes:**
Puzzles 4 and 6 computed gradients (dW, db, dX) for ReLU and FC layers.
Those gradients tell us the **direction** of steepest loss increase. SGD
steps in the **opposite direction** to reduce the loss:

```
Puzzle 06 (FC backward): computes dW, db  ──→  SGD uses dW, db to update W, b
Puzzle 04 (ReLU backward): routes gradients ──→  SGD uses accumulated grads
```

---

## The Loss Landscape

Think of the loss as a surface over the space of all possible weight values.
Training = finding the lowest point on this surface:

```
  Loss
   │
 4 ┤  ╲                                     ╱
   │    ╲                                   ╱
 3 ┤     ╲                                 ╱
   │      ╲                               ╱
 2 ┤       ╲           ╱╲                ╱
   │        ╲         ╱  ╲              ╱
 1 ┤         ╲       ╱    ╲           ╱
   │          ╲     ╱      ╲         ╱
 0 ┤           ╲___╱        ╲_______╱
   │              ↑              ↑
   │          local min      global min
   ├──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┤
                  weight value →

  SGD navigates this landscape by following the negative gradient:

  Start here (random init)
       ↓
  w ●─────→ compute loss
       ↑         │
       │         ↓
       │    compute gradient (backward pass)
       │         │
       │         ↓
       └── w = w - lr × grad  ←── THIS IS SGD!
```

### Gradient Descent Intuition

Imagine you're blindfolded on a hilly landscape and want to reach the valley:

1. **Feel the slope** under your feet (= compute gradient)
2. **Step downhill** (= subtract gradient from weights)
3. **Step size** matters (= learning rate)
   - Too big → overshoot the valley, bounce around
   - Too small → take forever to reach the bottom
   - Just right → steady descent to minimum

The "stochastic" in SGD means we estimate the gradient from a **mini-batch**
(subset of training data) rather than the full dataset. This adds noise but
makes each step much faster.

---

## The Update Rule

### Formula

```
w_new = w_old - learning_rate × gradient
```

That's the entire algorithm. For every single parameter in the network:

```
For each parameter w[i] with gradient grad[i]:
    w[i] = w[i] - lr × grad[i]
```

### Element-wise operation

SGD updates each parameter **independently** — no interaction between parameters:

```
Weights:    [ 0.50,  0.30, -0.20,  0.10,  0.80]
Gradients:  [ 0.10, -0.05,  0.20,  0.00, -0.15]
lr = 0.01

Update:     w -= lr × grad

New weights:[ 0.50 - 0.01×0.10,    = 0.499
              0.30 - 0.01×(-0.05), = 0.3005
             -0.20 - 0.01×0.20,    = -0.202
              0.10 - 0.01×0.00,    = 0.10
              0.80 - 0.01×(-0.15)] = 0.8015
```

### Mini-batch SGD in the training loop

```
for each epoch:
    for each mini-batch:
        1. Forward pass:    Y = f(X; W)          ← Puzzles 1-7
        2. Compute loss:    L = loss(Y, target)   ← Puzzle 5
        3. Backward pass:   grad = ∂L/∂W          ← Puzzles 4, 6
        4. Update weights:  W -= lr × grad         ← THIS KERNEL
        5. Zero gradients:  grad = 0               ← THIS KERNEL
```

### Why zero gradients?

Gradients **accumulate** across forward/backward passes. If we don't zero
them before the next mini-batch, the new gradients would be **added** to the
old ones, giving incorrect updates. Zeroing ensures each update uses only
the gradient from the current mini-batch.

---

## Learning Rate: The Most Important Hyperparameter

```
  Loss                                    Loss
   │                                       │
   │  ╲        lr too large                │  ╲        lr just right
   │   ╲  ●←──→●←──→●  (diverge!)         │   ╲
   │    ╲    ╱                             │    ╲  ●
   │     ╲__╱                              │     ╲  ╲●
   │                                       │      ╲___●  (converge!)
   ├────────────────                       ├────────────────

  Loss                                    
   │                                      
   │  ╲        lr too small               
   │   ●                                  
   │    ● ● ● ● ● ● ● (too slow!)       
   │     ╲__╱                              
   │                                       
   ├────────────────                       
```

Typical learning rates: 0.001 to 0.1 (LeNet often uses 0.01)

---

## Kernel Signatures

### Kernel 1: SGD Update

```cuda
__global__ void sgd_update(float* weights, const float* gradients,
                           float learning_rate, int n);
```

**Parameters:**
- `weights`: Parameter array to update **in-place**, size n
- `gradients`: Gradient array (computed by backward pass), size n
- `learning_rate`: Step size (scalar, same for all parameters)
- `n`: Total number of parameters

### Kernel 2: Zero Gradients

```cuda
__global__ void zero_gradients(float* gradients, int n);
```

**Parameters:**
- `gradients`: Gradient array to zero out, size n
- `n`: Total number of elements

**Launch configuration (both kernels):**
```cuda
int threads = 256;
int blocks = (n + threads - 1) / threads;
sgd_update<<<blocks, threads>>>(d_weights, d_gradients, lr, n);
zero_gradients<<<blocks, threads>>>(d_gradients, n);
```

---

## Step-by-Step Guide

### SGD update kernel:

1. **Calculate your index**: `int i = blockIdx.x * blockDim.x + threadIdx.x`
2. **Bounds check**: If `i >= n`, return immediately
3. **Update weight**: `weights[i] = weights[i] - learning_rate * gradients[i]`

### Zero gradients kernel:

1. **Calculate your index**: `int i = blockIdx.x * blockDim.x + threadIdx.x`
2. **Bounds check**: If `i >= n`, return immediately
3. **Zero the gradient**: `gradients[i] = 0.0f`

---

## Hints

<details>
<summary>Hint 1 (Mild): Thread index and bounds</summary>

Same 1D grid pattern as Puzzle 01:
```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= n) return;
```
</details>

<details>
<summary>Hint 2 (Medium): SGD update — one line</summary>

```cuda
weights[i] -= learning_rate * gradients[i];
```
Note: this modifies weights **in-place**. The gradients are read-only.
</details>

<details>
<summary>Hint 3 (Strong): Both kernels — almost complete</summary>

```cuda
// SGD update
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    weights[i] -= learning_rate * gradients[i];
}

// Zero gradients
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    gradients[i] = 0.0f;
}
```
</details>

---

## LeNet-5 Parameter Count

SGD updates **every learnable parameter** in the network. For LeNet-5:

```
Layer          Parameters        Count
─────────────────────────────────────────
Conv1:  6 filters × (5×5×1 + 1)  =    156
Conv2: 16 filters × (5×5×6 + 1)  =  2,416
FC1:   (256→120) weights + bias   = 30,840
FC2:   (120→84)  weights + bias   = 10,164
FC3:   (84→10)   weights + bias   =    850
─────────────────────────────────────────
Total                               44,426
```

One SGD step updates all 44,426 parameters simultaneously on the GPU —
each parameter handled by its own thread, all in parallel.

---

## Testing

Build and run the tests to verify your solution:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target puzzle_11_test
./build/puzzle_11_test
```

The test suite runs 4 tests:

1. **sgd_hardcoded_update**: Small array with known values — verify exact w -= lr*grad
2. **sgd_multi_step_loss_decrease**: Multiple SGD steps on a quadratic loss — verify loss decreases
3. **sgd_gradient_zeroing**: Verify zero_gradients sets all elements to 0.0f
4. **sgd_lenet_param_count**: Update ~44K parameters (LeNet total) — verify at scale

All 4 tests must pass for the puzzle to be complete.
