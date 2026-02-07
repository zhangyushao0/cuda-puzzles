// Puzzle 05: Softmax + Cross-Entropy Loss
//
// Implement three CUDA kernels for the output layer of a classifier:
//   1. softmax_forward:     logits → probabilities (MUST use max-subtraction trick!)
//   2. cross_entropy_loss:  probabilities + labels → per-sample loss
//   3. softmax_ce_backward: probabilities + labels → gradient w.r.t. logits
//
// See README.md for detailed explanation and hints.

#include <cuda_runtime.h>

// TODO: Implement softmax forward pass with numerical stability
//
// Parameters:
//   logits      - raw scores, size (batch × num_classes), row-major
//   probs       - output probabilities, size (batch × num_classes), row-major
//   batch       - number of samples in the batch
//   num_classes - number of classes (e.g., 10 for MNIST digits)
//
// Each thread handles one sample (all classes for that sample).
//
// CRITICAL: You MUST use the max-subtraction trick to prevent overflow!
//   Step 1: Find m = max(logits[b][c]) over all classes c
//   Step 2: Compute exp(logits[b][c] - m) for each class
//   Step 3: Sum the exponentials
//   Step 4: Divide each exponential by the sum to get probabilities
//
// Without max-subtraction, logits like [1000, 1001, 1002] will produce
// exp(1000) = Inf, resulting in NaN output!
//
// Memory indexing (row-major):
//   logits[b][c] = logits[b * num_classes + c]
//   probs[b][c]  = probs[b * num_classes + c]
__global__ void softmax_forward(const float* logits, float* probs,
                                int batch, int num_classes) {
    // TODO: Your implementation here
}

// TODO: Implement cross-entropy loss computation
//
// Parameters:
//   probs       - probabilities from softmax, size (batch × num_classes)
//   labels      - one-hot encoded labels, size (batch × num_classes)
//   losses      - output per-sample losses, size (batch)
//   batch       - number of samples in the batch
//   num_classes - number of classes
//
// Each thread handles one sample.
//
// Formula: L[b] = -Σ_c labels[b][c] * log(probs[b][c] + 1e-10f)
//
// The epsilon (1e-10f) prevents log(0) = -Infinity when a probability
// is exactly zero.
__global__ void cross_entropy_loss(const float* probs, const float* labels,
                                   float* losses, int batch, int num_classes) {
    // TODO: Your implementation here
}

// TODO: Implement backward pass for softmax + cross-entropy
//
// Parameters:
//   probs       - probabilities from softmax, size (batch × num_classes)
//   labels      - one-hot encoded labels, size (batch × num_classes)
//   grad_logits - output gradient w.r.t. logits, size (batch × num_classes)
//   batch       - number of samples in the batch
//   num_classes - number of classes
//
// Each thread handles one sample (all classes for that sample).
//
// The elegant formula:
//   grad_logits[b][c] = probs[b][c] - labels[b][c]
//
// This beautiful simplification comes from combining the softmax and
// cross-entropy derivatives. No exp(), log(), or division needed!
__global__ void softmax_ce_backward(const float* probs, const float* labels,
                                    float* grad_logits, int batch,
                                    int num_classes) {
    // TODO: Your implementation here
}
