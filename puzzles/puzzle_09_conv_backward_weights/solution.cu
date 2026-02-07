// Puzzle 09: Conv2D Backward — Weight & Bias Gradients — Reference Solution
//
// Kernel 1: Weight gradient (correlation between input and grad_output)
//   ∂L/∂W[k][c][fh][fw] = Σ_b Σ_h Σ_w grad_out[b][k][h][w] × input[b][c][h+fh][w+fw]
//
// Kernel 2: Bias gradient (sum of grad_output over batch and spatial dims)
//   ∂L/∂bias[k] = Σ_b Σ_h Σ_w grad_out[b][k][h][w]
//
// Each weight-gradient thread computes one element of ∂L/∂W using a 1D grid.
// Each bias-gradient thread computes one element of ∂L/∂bias.
// Direct nested-loop implementation for educational clarity.

#include <cuda_runtime.h>

__global__ void conv2d_backward_weights(const float* grad_output, const float* input,
                                         float* grad_weights,
                                         int batch, int C_in, int H, int W,
                                         int C_out, int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H - F + 1;
    int W_out = W - F + 1;
    int total = C_out * C_in * F * F;

    if (idx >= total) return;

    // Decode 4D indices from flat index: grad_weights[k][c][fh][fw]
    int fw = idx % F;
    int fh = (idx / F) % F;
    int c  = (idx / (F * F)) % C_in;
    int k  = idx / (F * F * C_in);

    float sum = 0.0f;

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                int go_idx = b * (C_out * H_out * W_out) + k * (H_out * W_out)
                           + h * W_out + w;
                int in_idx = b * (C_in * H * W) + c * (H * W)
                           + (h + fh) * W + (w + fw);
                sum += grad_output[go_idx] * input[in_idx];
            }
        }
    }

    grad_weights[k * (C_in * F * F) + c * (F * F) + fh * F + fw] = sum;
}

__global__ void conv2d_backward_bias(const float* grad_output, float* grad_bias,
                                      int batch, int C_out, int H_out, int W_out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= C_out) return;

    float sum = 0.0f;

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                sum += grad_output[b * (C_out * H_out * W_out) + k * (H_out * W_out)
                                 + h * W_out + w];
            }
        }
    }

    grad_bias[k] = sum;
}
