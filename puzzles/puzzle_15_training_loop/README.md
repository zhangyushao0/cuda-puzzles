# Puzzle 15: Full Training Loop (LeNet-5)

## Overview

Implement a complete training loop for LeNet-5 on MNIST-like data.
This puzzle combines **every previous puzzle** into an end-to-end training
pipeline: Xavier weight initialization, batched forward pass, loss computation,
full backward pass, and SGD weight updates.

## Training Algorithm Flowchart

```
  ┌─────────────────────────────────────┐
  │  1. Xavier Init (seed=42)           │
  │     W ~ Uniform(-limit, +limit)     │
  │     limit = sqrt(6 / (fan_in+fan_out))│
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │  for epoch = 1..N:                  │
  │    shuffle(dataset)                 │
  │    ┌────────────────────────────────┤
  │    │  for each mini-batch:          │
  │    │                                │
  │    │  ┌─────────────────────────┐   │
  │    │  │ 2. Forward Pass         │   │
  │    │  │    (Puzzle 12)          │   │
  │    │  │    input → probs        │   │
  │    │  └──────────┬──────────────┘   │
  │    │             │                  │
  │    │  ┌──────────▼──────────────┐   │
  │    │  │ 3. Compute Loss         │   │
  │    │  │    (Puzzle 05)          │   │
  │    │  │    cross_entropy(probs, │   │
  │    │  │                 labels) │   │
  │    │  └──────────┬──────────────┘   │
  │    │             │                  │
  │    │  ┌──────────▼──────────────┐   │
  │    │  │ 4. Backward Pass        │   │
  │    │  │    (Puzzles 04-10)      │   │
  │    │  │    dL/dW for all layers │   │
  │    │  └──────────┬──────────────┘   │
  │    │             │                  │
  │    │  ┌──────────▼──────────────┐   │
  │    │  │ 5. SGD Update           │   │
  │    │  │    (Puzzle 11)          │   │
  │    │  │    W -= lr * dW         │   │
  │    │  └──────────┬──────────────┘   │
  │    │             │                  │
  │    │  ┌──────────▼──────────────┐   │
  │    │  │ 6. Zero Gradients       │   │
  │    │  │    (Puzzle 11)          │   │
  │    │  └─────────────────────────┘   │
  │    └────────────────────────────────┤
  │    Print: epoch, avg_loss, accuracy │
  └─────────────────────────────────────┘
```

## Xavier Initialization

Proper weight initialization prevents vanishing/exploding gradients.

```
Xavier Uniform:
  limit = sqrt(6.0 / (fan_in + fan_out))
  W ~ Uniform(-limit, +limit)

For each layer:
  Layer   fan_in   fan_out  limit
  ─────────────────────────────────
  Conv1   1x5x5=25    6    ~0.440
  Conv2   6x5x5=150  16    ~0.190
  FC1     256        120    ~0.126
  FC2     120         84    ~0.171
  FC3      84         10    ~0.253

Biases initialized to 0.
```

## Hyperparameters

```
Learning rate:  0.01
Batch size:     32
Epochs:         10
Seed:           42 (weight init)
```

## Expected Accuracy Curve (Full MNIST)

```
Epoch   Train Loss   Train Acc
──────────────────────────────
  1      ~1.5         ~55%
  2      ~0.6         ~80%
  3      ~0.3         ~90%
  4      ~0.2         ~93%
  5      ~0.15        ~95%
  6      ~0.12        ~96%
  7      ~0.10        ~96%
  8      ~0.08        ~97%
  9      ~0.07        ~97%
 10      ~0.06        ~98%
```

## Backward Pass Details

The full backward pass reverses the forward pass:

```
Forward:  Input → Conv1 → ReLU1 → Pool1 → Conv2 → ReLU2 → Pool2
          → FC1 → ReLU3 → FC2 → ReLU4 → FC3 → Softmax → Loss

Backward: dLoss → dSoftmax+CE → dFC3 → dReLU4 → dFC2 → dReLU3 → dFC1
          → dPool2 → dReLU2 → dConv2 → dPool1 → dReLU1 → dConv1
```

Each backward step computes:
- **Input gradient**: passed to the previous layer
- **Weight gradient**: accumulated for SGD update
- **Bias gradient**: accumulated for SGD update

## Functions to Implement

```cpp
// Initialize weights using Xavier uniform distribution
void xavier_init(LeNetParams& params, unsigned seed);

// Run one training epoch over the dataset
float train_epoch(const float* d_images, const int* d_labels,
                  int num_samples, int batch_size,
                  LeNetParams& params, LeNetGradients& grads,
                  LeNetActivations& act, LeNetBackwardActs& back_act,
                  float learning_rate);

// Evaluate accuracy on a dataset
float evaluate(const float* d_images, const int* d_labels,
               int num_samples, int batch_size,
               LeNetParams& params, LeNetActivations& act);

// Full training loop: init + train + evaluate
void training_loop(const float* d_images, const int* d_labels,
                   int num_samples, int num_epochs,
                   int batch_size, float learning_rate,
                   unsigned seed);
```

## Puzzle Dependencies

This puzzle bundles reference solutions from ALL previous puzzles:
- **Puzzle 03**: FC Forward
- **Puzzle 04**: ReLU Forward + Backward
- **Puzzle 05**: Softmax + Cross-Entropy (Forward + Backward)
- **Puzzle 06**: FC Backward (weights, bias, input)
- **Puzzle 07**: Conv2D Forward
- **Puzzle 08**: MaxPool Forward + Backward
- **Puzzle 09**: Conv2D Backward Weights + Bias
- **Puzzle 10**: Conv2D Backward Input
- **Puzzle 11**: SGD Update + Zero Gradients
- **Puzzle 12**: LeNet Forward Pass orchestration

## Hints

**Hint 1 (Mild):** Xavier init for conv layers uses `fan_in = C_in * F * F`
and `fan_out = C_out * F * F`. For FC layers, `fan_in = in_features` and
`fan_out = out_features`.

**Hint 2 (Medium):** The backward pass through MaxPool requires the
`max_indices` saved during the forward pass. Make sure your forward
pass stores them and your backward pass uses the same buffers.

**Hint 3 (Strong):** The backward pass must zero `grad_input` buffers
for MaxPool backward before the kernel call, since it uses scatter
(atomicAdd-like) writes. Use `cudaMemset(ptr, 0, size)`.
