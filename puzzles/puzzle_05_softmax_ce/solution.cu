// Puzzle 05: Softmax + Cross-Entropy Loss — Reference Solution
//
// Three kernels for the classifier output layer:
//   1. softmax_forward:     logits → probabilities (with max-subtraction trick)
//   2. cross_entropy_loss:  probs + labels → per-sample loss
//   3. softmax_ce_backward: probs + labels → gradient (probs - labels)

#include <cuda_runtime.h>

// Kernel 1: Softmax forward with max-subtraction for numerical stability
// Each thread handles one sample across all classes.
__global__ void softmax_forward(const float* logits, float* probs,
                                int batch, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    int offset = b * num_classes;

    // Step 1: Find max logit for this sample (prevents exp overflow)
    float max_val = logits[offset];
    for (int c = 1; c < num_classes; c++) {
        max_val = fmaxf(max_val, logits[offset + c]);
    }

    // Step 2-3: Compute shifted exponentials and their sum
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        float exp_val = expf(logits[offset + c] - max_val);
        probs[offset + c] = exp_val;  // store temporarily
        sum_exp += exp_val;
    }

    // Step 4: Normalize to get probabilities
    for (int c = 0; c < num_classes; c++) {
        probs[offset + c] /= sum_exp;
    }
}

// Kernel 2: Cross-entropy loss per sample
// L[b] = -Σ_c labels[b][c] * log(probs[b][c] + epsilon)
__global__ void cross_entropy_loss(const float* probs, const float* labels,
                                   float* losses, int batch, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    int offset = b * num_classes;
    float loss = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        loss -= labels[offset + c] * logf(probs[offset + c] + 1e-10f);
    }
    losses[b] = loss;
}

// Kernel 3: Combined softmax + cross-entropy backward
// The elegant result: grad_logits[b][c] = probs[b][c] - labels[b][c]
__global__ void softmax_ce_backward(const float* probs, const float* labels,
                                    float* grad_logits, int batch,
                                    int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    int offset = b * num_classes;
    for (int c = 0; c < num_classes; c++) {
        grad_logits[offset + c] = probs[offset + c] - labels[offset + c];
    }
}
