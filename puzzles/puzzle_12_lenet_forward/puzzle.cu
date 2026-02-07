// Puzzle 12: LeNet-5 Forward Pass (Full Network) — Student Template
//
// Chain all individual layer kernels into a complete LeNet-5 forward pass:
//   Input(28x28x1) -> Conv1 -> ReLU -> Pool1 -> Conv2 -> ReLU -> Pool2
//   -> Flatten -> FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> Softmax -> probs
//
// Your job: implement lenet_forward() — allocate nothing, just launch
// the kernels in the correct order with the correct dimensions.
// Buffer allocation is handled by the test harness.

#include <cuda_runtime.h>
#include <cfloat>

// ============================================================
// Bundled reference kernels from previous puzzles
// (Provided so you can focus on orchestration)
// ============================================================

// From Puzzle 07: Conv2D Forward
__global__ void conv2d_forward(const float* input, const float* filters,
                               const float* bias, float* output,
                               int batch, int C_in, int H, int W,
                               int C_out, int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - F + 1;
    int W_out = W - F + 1;
    int total = batch * C_out * H_out * W_out;
    if (idx >= total) return;

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

// From Puzzle 04: ReLU Forward
__global__ void relu_forward(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

// From Puzzle 08: Max Pooling Forward
__global__ void maxpool_forward(const float* input, float* output,
                                int* max_indices,
                                int batch, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H / 2;
    int W_out = W / 2;
    int total = batch * C * H_out * W_out;
    if (idx >= total) return;

    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int c  = (idx / (W_out * H_out)) % C;
    int b  = idx / (W_out * H_out * C);

    int h_start = oh * 2;
    int w_start = ow * 2;

    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int ph = 0; ph < 2; ph++) {
        for (int pw = 0; pw < 2; pw++) {
            int in_idx = b * (C * H * W) + c * (H * W)
                       + (h_start + ph) * W + (w_start + pw);
            float val = input[in_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = ph * 2 + pw;
            }
        }
    }
    output[idx] = max_val;
    max_indices[idx] = max_idx;
}

// From Puzzle 03: FC Forward
__global__ void fc_forward(float* input, float* weights, float* bias,
                           float* output, int batch, int in_features,
                           int out_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (b < batch && j < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[b * in_features + i] * weights[j * in_features + i];
        }
        output[b * out_features + j] = sum + bias[j];
    }
}

// From Puzzle 05: Softmax Forward
__global__ void softmax_forward(const float* logits, float* probs,
                                int batch, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    int offset = b * num_classes;

    float max_val = logits[offset];
    for (int c = 1; c < num_classes; c++) {
        max_val = fmaxf(max_val, logits[offset + c]);
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        float exp_val = expf(logits[offset + c] - max_val);
        probs[offset + c] = exp_val;
        sum_exp += exp_val;
    }

    for (int c = 0; c < num_classes; c++) {
        probs[offset + c] /= sum_exp;
    }
}

// ============================================================
// LeNet-5 parameter and activation structures
// ============================================================

struct LeNetParams {
    float* conv1_w;   // [6 x 1 x 5 x 5] = 150
    float* conv1_b;   // [6]
    float* conv2_w;   // [16 x 6 x 5 x 5] = 2400
    float* conv2_b;   // [16]
    float* fc1_w;     // [120 x 256] = 30720
    float* fc1_b;     // [120]
    float* fc2_w;     // [84 x 120] = 10080
    float* fc2_b;     // [84]
    float* fc3_w;     // [10 x 84] = 840
    float* fc3_b;     // [10]
};

struct LeNetActivations {
    float* conv1_out;      // [B x 6 x 24 x 24]
    float* relu1_out;      // [B x 6 x 24 x 24]
    float* pool1_out;      // [B x 6 x 12 x 12]
    int*   pool1_indices;  // [B x 6 x 12 x 12]
    float* conv2_out;      // [B x 16 x 8 x 8]
    float* relu2_out;      // [B x 16 x 8 x 8]
    float* pool2_out;      // [B x 16 x 4 x 4]
    int*   pool2_indices;  // [B x 16 x 4 x 4]
    float* fc1_out;        // [B x 120]
    float* relu3_out;      // [B x 120]
    float* fc2_out;        // [B x 84]
    float* relu4_out;      // [B x 84]
    float* fc3_out;        // [B x 10]
    float* probs;          // [B x 10]
};

// ============================================================
// Allocate / free helpers (provided)
// ============================================================

void alloc_activations(LeNetActivations& act, int batch) {
    CUDA_CHECK(cudaMalloc(&act.conv1_out,     batch * 6  * 24 * 24 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&act.relu1_out,     batch * 6  * 24 * 24 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&act.pool1_out,     batch * 6  * 12 * 12 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&act.pool1_indices, batch * 6  * 12 * 12 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&act.conv2_out,     batch * 16 * 8  * 8  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&act.relu2_out,     batch * 16 * 8  * 8  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&act.pool2_out,     batch * 16 * 4  * 4  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&act.pool2_indices, batch * 16 * 4  * 4  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&act.fc1_out,       batch * 120 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&act.relu3_out,     batch * 120 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&act.fc2_out,       batch * 84  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&act.relu4_out,     batch * 84  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&act.fc3_out,       batch * 10  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&act.probs,         batch * 10  * sizeof(float)));
}

void free_activations(LeNetActivations& act) {
    cudaFree(act.conv1_out);
    cudaFree(act.relu1_out);
    cudaFree(act.pool1_out);
    cudaFree(act.pool1_indices);
    cudaFree(act.conv2_out);
    cudaFree(act.relu2_out);
    cudaFree(act.pool2_out);
    cudaFree(act.pool2_indices);
    cudaFree(act.fc1_out);
    cudaFree(act.relu3_out);
    cudaFree(act.fc2_out);
    cudaFree(act.relu4_out);
    cudaFree(act.fc3_out);
    cudaFree(act.probs);
}

void alloc_params(LeNetParams& p) {
    CUDA_CHECK(cudaMalloc(&p.conv1_w, 150   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.conv1_b, 6     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.conv2_w, 2400  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.conv2_b, 16    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.fc1_w,   30720 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.fc1_b,   120   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.fc2_w,   10080 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.fc2_b,   84    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.fc3_w,   840   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.fc3_b,   10    * sizeof(float)));
}

void free_params(LeNetParams& p) {
    cudaFree(p.conv1_w);
    cudaFree(p.conv1_b);
    cudaFree(p.conv2_w);
    cudaFree(p.conv2_b);
    cudaFree(p.fc1_w);
    cudaFree(p.fc1_b);
    cudaFree(p.fc2_w);
    cudaFree(p.fc2_b);
    cudaFree(p.fc3_w);
    cudaFree(p.fc3_b);
}

void upload_params(LeNetParams& d, const float* conv1_w, const float* conv1_b,
                   const float* conv2_w, const float* conv2_b,
                   const float* fc1_w, const float* fc1_b,
                   const float* fc2_w, const float* fc2_b,
                   const float* fc3_w, const float* fc3_b) {
    CUDA_CHECK(cudaMemcpy(d.conv1_w, conv1_w, 150   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.conv1_b, conv1_b, 6     * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.conv2_w, conv2_w, 2400  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.conv2_b, conv2_b, 16    * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.fc1_w,   fc1_w,   30720 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.fc1_b,   fc1_b,   120   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.fc2_w,   fc2_w,   10080 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.fc2_b,   fc2_b,   84    * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.fc3_w,   fc3_w,   840   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.fc3_b,   fc3_b,   10    * sizeof(float), cudaMemcpyHostToDevice));
}

// ============================================================
// TODO: Implement the LeNet-5 Forward Pass
// ============================================================
//
// Given:
//   d_input — device pointer to input images [batch x 1 x 28 x 28]
//   p       — device pointers to all network parameters
//   act     — pre-allocated device buffers for all intermediate activations
//   batch   — number of images in the batch
//
// Your job: launch the kernels in the correct order to compute:
//   Input -> Conv1 -> ReLU -> Pool1 -> Conv2 -> ReLU -> Pool2
//   -> Flatten -> FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> Softmax
//
// The final output should be in act.probs [batch x 10].
//
// Kernel launch cheat sheet:
//   conv2d_forward<<<blocks, 256>>>(in, filters, bias, out, B, C_in, H, W, C_out, F)
//   relu_forward<<<blocks, 256>>>(in, out, n)
//   maxpool_forward<<<blocks, 256>>>(in, out, indices, B, C, H, W)
//   fc_forward<<<dim3(grid_x, grid_y), dim3(16, 16)>>>(in, w, b, out, B, in_f, out_f)
//   softmax_forward<<<blocks, 256>>>(logits, probs, B, num_classes)

void lenet_forward(const float* d_input, LeNetParams& p,
                   LeNetActivations& act, int batch) {
    // TODO: Your code here
    // Launch all 12 kernels in sequence (Conv1, ReLU1, Pool1,
    // Conv2, ReLU2, Pool2, FC1, ReLU3, FC2, ReLU4, FC3, Softmax)
    //
    // Hint: The flatten step requires NO kernel — pool2_out is
    // already 256 contiguous floats per sample. Pass pool2_out
    // directly to fc_forward as input.
}
