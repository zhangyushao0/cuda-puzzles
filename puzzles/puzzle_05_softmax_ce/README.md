# Puzzle 05: Softmax + Cross-Entropy Loss

## Overview

Implement **softmax**, **cross-entropy loss**, and their combined **backward pass**
on the GPU. This is the final layer of LeNet-5 (and most classifiers): it converts
raw logits into probabilities, measures how wrong we are, and computes the gradient
to drive learning.

```
FC3 output (logits)  →  Softmax  →  probabilities  →  Cross-Entropy  →  loss
      z[i]                            p[i]                                 L

                    Backward:  ∂L/∂z[i] = p[i] - y[i]   ← elegantly simple!
```

**Why this matters for LeNet:**
After FC3 produces 10 raw scores (one per digit), softmax converts them to
probabilities that sum to 1, and cross-entropy measures the distance between
our predictions and the true label. The backward pass gives us the gradient
to update all the network's weights.

```
FC3 output: [2.1, -0.3, 0.8, ..., 1.5]   ← raw logits (10 values)
     ↓ softmax
Probs:      [0.31, 0.03, 0.08, ..., 0.17]  ← probabilities (sum = 1.0)
     ↓ cross-entropy with label=0
Loss:       1.17                             ← how wrong we are
     ↓ backward
Gradient:   [-0.69, 0.03, 0.08, ..., 0.17]  ← push in the right direction
```

---

## The Critical Problem: Numerical Overflow

### Naive softmax EXPLODES with large logits

The softmax formula involves `exp()`, which grows **incredibly fast**:

```
exp(10)   = 22,026
exp(100)  = 2.69 × 10^43
exp(710)  = 1.70 × 10^308    ← maximum for float64
exp(89)   = 4.49 × 10^38     ← maximum for float32!
exp(1000) = +Infinity         ← OVERFLOW! NaN everywhere!
```

**Before (WRONG — overflows):**
```
logits = [1000, 1001, 1002]

exp(1000) = Inf
exp(1001) = Inf
exp(1002) = Inf

softmax = [Inf/Inf, Inf/Inf, Inf/Inf] = [NaN, NaN, NaN]  ← DISASTER!
```

### The Max-Subtraction Trick (MANDATORY)

The key insight: **subtracting the same constant from all logits doesn't change
the softmax output** but prevents overflow:

```
softmax(z[i]) = exp(z[i]) / Σ exp(z[k])
              = exp(z[i] - m) / Σ exp(z[k] - m)    where m = max(z)
```

**Proof:** Multiply numerator and denominator by `exp(-m)`:
```
exp(z[i]) / Σ exp(z[k])
= exp(z[i]) · exp(-m) / (Σ exp(z[k]) · exp(-m))
= exp(z[i] - m) / Σ exp(z[k] - m)
```

**After (CORRECT — stable):**
```
logits = [1000, 1001, 1002]
max    = 1002

shifted = [1000-1002, 1001-1002, 1002-1002] = [-2, -1, 0]

exp(-2) = 0.135
exp(-1) = 0.368
exp(0)  = 1.000
sum     = 1.503

softmax = [0.090, 0.245, 0.665]  ← valid probabilities, sum ≈ 1.0 ✓
```

The largest shifted value is always 0, so `exp(0) = 1` — no overflow possible!

---

## The Math

### Softmax (Forward)

Converts logits to probabilities for one sample in the batch:

```
Step 1: m = max(z[0], z[1], ..., z[C-1])          — find max logit
Step 2: For each class i:  exp_i = exp(z[i] - m)   — shifted exponentials
Step 3: sum_exp = Σ exp_i                           — normalization constant
Step 4: p[i] = exp_i / sum_exp                      — probabilities

Properties:
  - All p[i] ∈ (0, 1)
  - Σ p[i] = 1.0 (exactly)
  - Largest logit → largest probability
```

### Cross-Entropy Loss

Measures how far our predictions are from the true label:

```
L = -Σ y[i] · log(p[i] + ε)

where:
  y[i] = one-hot label (1 for true class, 0 for others)
  ε = 1e-10  (prevents log(0) = -Infinity)

For one-hot labels, this simplifies to:
  L = -log(p[true_class] + ε)

For a batch of B samples:
  L_batch = (1/B) · Σ_b L_b
```

### Backward Pass: The Beautiful Simplification

The gradient of cross-entropy loss with respect to logits is:

```
∂L/∂z[i] = p[i] - y[i]

That's it! No exp(), no log(), no division.
```

**Why this is elegant:**
- For the true class (y[i] = 1): gradient = p[i] - 1 (negative → push logit UP)
- For wrong classes (y[i] = 0): gradient = p[i] - 0 = p[i] (positive → push logit DOWN)
- The gradient is **exactly the error**: how far each probability is from the target

**Example:**
```
probs  = [0.1,  0.7,  0.2]    ← model thinks class 1
labels = [0.0,  0.0,  1.0]    ← true answer is class 2

grad   = [0.1,  0.7, -0.8]    ← push class 2 up, others down
          ↑      ↑     ↑
       push    push  push
       down    down    up
```

---

## Memory Layout

```
Logits (batch × num_classes):   logits[b * num_classes + c]
Probs  (batch × num_classes):   probs[b * num_classes + c]
Labels (batch × num_classes):   labels[b * num_classes + c]   (one-hot)
Loss   (scalar or batch):       loss[b] per sample, or averaged
Grad   (batch × num_classes):   grad_logits[b * num_classes + c]
```

---

## Thread Mapping

### Softmax Forward: One Thread Per Sample

Each thread processes one sample (all C classes):

```
Thread b computes softmax for sample b:

b = blockIdx.x * blockDim.x + threadIdx.x

1. Find max: loop over C classes
2. Compute exp(z[i] - max): loop over C classes
3. Sum exponentials: loop over C classes
4. Normalize: p[i] = exp_i / sum: loop over C classes
```

### Cross-Entropy Loss: One Thread Per Sample

```
Thread b computes loss for sample b:

L[b] = -Σ_c labels[b][c] * log(probs[b][c] + 1e-10)
```

### Backward: One Thread Per Sample

```
Thread b computes gradient for sample b:

For each class c:
  grad[b][c] = probs[b][c] - labels[b][c]
```

---

## Kernel Signatures

```cuda
// Kernel 1: Softmax forward — logits to probabilities
// MUST use max-subtraction trick for numerical stability
__global__ void softmax_forward(const float* logits, float* probs,
                                int batch, int num_classes);

// Kernel 2: Cross-entropy loss per sample
// Uses log(p + epsilon) for numerical stability
__global__ void cross_entropy_loss(const float* probs, const float* labels,
                                   float* losses, int batch, int num_classes);

// Kernel 3: Combined backward — elegant gradient computation
// grad_logits[b][c] = probs[b][c] - labels[b][c]
__global__ void softmax_ce_backward(const float* probs, const float* labels,
                                    float* grad_logits, int batch,
                                    int num_classes);
```

**Launch configuration (all three kernels):**
```cuda
int threads = 256;
int blocks = (batch + threads - 1) / threads;
kernel<<<blocks, threads>>>(...);
```

---

## Step-by-Step Guide

### Kernel 1: softmax_forward

1. **Calculate sample index**: `b = blockIdx.x * blockDim.x + threadIdx.x`
2. **Bounds check**: `if (b >= batch) return;`
3. **Find max logit** for sample b (loop over classes)
4. **Compute shifted exponentials**: `exp(z[c] - max)` for each class
5. **Sum the exponentials**
6. **Normalize**: `probs[b][c] = exp_c / sum_exp`

### Kernel 2: cross_entropy_loss

1. **Calculate sample index**: Same as above
2. **Bounds check**
3. **Accumulate**: `-labels[b][c] * logf(probs[b][c] + 1e-10f)` over all classes
4. **Write** per-sample loss

### Kernel 3: softmax_ce_backward

1. **Calculate sample index**: Same as above
2. **Bounds check**
3. **For each class c**: `grad[b][c] = probs[b][c] - labels[b][c]`

---

## Hints

<details>
<summary>Hint 1 (Mild): Max-subtraction structure</summary>

```cuda
int b = blockIdx.x * blockDim.x + threadIdx.x;
if (b >= batch) return;

// Step 1: Find max
float max_val = logits[b * num_classes + 0];
for (int c = 1; c < num_classes; c++) {
    max_val = fmaxf(max_val, logits[b * num_classes + c]);
}

// Step 2-3: Compute exp and sum
// ... your code here ...
```
</details>

<details>
<summary>Hint 2 (Medium): Complete softmax loop</summary>

```cuda
float sum_exp = 0.0f;
for (int c = 0; c < num_classes; c++) {
    float exp_val = expf(logits[b * num_classes + c] - max_val);
    probs[b * num_classes + c] = exp_val;  // store temporarily
    sum_exp += exp_val;
}
for (int c = 0; c < num_classes; c++) {
    probs[b * num_classes + c] /= sum_exp;  // normalize
}
```
</details>

<details>
<summary>Hint 3 (Strong): Nearly complete cross-entropy</summary>

```cuda
float loss = 0.0f;
for (int c = 0; c < num_classes; c++) {
    loss -= labels[b * num_classes + c] *
            logf(probs[b * num_classes + c] + 1e-10f);
}
losses[b] = loss;
```
</details>

---

## Testing

Build and run the tests to verify your solution:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target puzzle_05_test
./build/puzzle_05_test
```

The test suite runs 5 tests:

1. **softmax_probs_sum_to_one**: Softmax output sums to 1.0 within tolerance
2. **softmax_numerical_stability**: Logits [1000,1001,1002] produce valid results (no NaN/Inf)
3. **cross_entropy_loss_correctness**: CE loss matches CPU reference
4. **backward_gradient_correctness**: Gradient = probs - labels verified
5. **round_trip_forward_backward**: Full forward→backward pipeline consistency

All 5 tests must pass for the puzzle to be complete.
