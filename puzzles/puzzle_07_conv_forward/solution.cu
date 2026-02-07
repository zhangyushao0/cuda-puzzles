// Puzzle 07: 2D Convolution (Forward Pass) — Reference Solution
//
// out[b][k][oh][ow] = bias[k]
//                    + Σ_c Σ_fh Σ_fw input[b][c][oh+fh][ow+fw] × filter[k][c][fh][fw]
//
// Each thread computes one output element using a 1D grid.
// Direct nested-loop implementation for educational clarity.

#include <cuda_runtime.h>

__global__ void conv2d_forward(const float* input, const float* filters,
                               const float* bias, float* output,
                               int batch, int C_in, int H, int W,
                               int C_out, int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H - F + 1;
    int W_out = W - F + 1;
    int total = batch * C_out * H_out * W_out;

    if (idx >= total) return;

    // Decode 4D indices from flat index (NCHW order)
    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int k  = (idx / (W_out * H_out)) % C_out;
    int b  = idx / (W_out * H_out * C_out);

    float sum = bias[k];

    for (int c = 0; c < C_in; c++) {
        for (int fh = 0; fh < F; fh++) {
            for (int fw = 0; fw < F; fw++) {
                int input_idx  = b * (C_in * H * W) + c * (H * W)
                               + (oh + fh) * W + (ow + fw);
                int filter_idx = k * (C_in * F * F) + c * (F * F)
                               + fh * F + fw;
                sum += input[input_idx] * filters[filter_idx];
            }
        }
    }

    output[b * (C_out * H_out * W_out) + k * (H_out * W_out) + oh * W_out + ow] = sum;
}
