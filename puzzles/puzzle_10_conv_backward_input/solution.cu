// Puzzle 10: Conv2D Backward Pass — Input Gradient — Reference Solution
//
// ∂L/∂input[b][c][h][w] = Σ_k Σ_fh Σ_fw grad_output[b][k][h-fh][w-fw] × W[k][c][fh][fw]
//
// Each thread computes one grad_input element using a 1D grid.
// Bounds checking replaces the conceptual full-padding of grad_output.

#include <cuda_runtime.h>

__global__ void conv2d_backward_input(const float* grad_output,
                                       const float* filters,
                                       float* grad_input,
                                       int batch, int C_in, int H, int W,
                                       int C_out, int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * C_in * H * W;

    if (idx >= total) return;

    // Decode 4D indices from flat index (NCHW order)
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
                    int go_idx = b * (C_out * H_out * W_out)
                               + k * (H_out * W_out)
                               + oh * W_out + ow;
                    int f_idx  = k * (C_in * F * F)
                               + c * (F * F)
                               + fh * F + fw;
                    sum += grad_output[go_idx] * filters[f_idx];
                }
            }
        }
    }

    grad_input[b * (C_in * H * W) + c * (H * W) + h * W + w] = sum;
}
