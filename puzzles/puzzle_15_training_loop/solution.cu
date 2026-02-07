// Puzzle 15: Full Training Loop (LeNet-5) — Reference Solution
//
// Implements end-to-end training for LeNet-5:
//   1. Xavier weight initialization
//   2. Forward pass (Puzzle 12)
//   3. Cross-entropy loss (Puzzle 05)
//   4. Full backward pass (Puzzles 04-10)
//   5. SGD weight update (Puzzle 11)
//
// All kernels from Puzzles 03-11 are bundled below for independent compilation.

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <random>

// ============================================================
// Bundled reference kernels from Puzzles 03-11
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

// From Puzzle 04: ReLU Backward
__global__ void relu_backward(const float* grad_output, const float* input,
                              float* grad_input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
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

// From Puzzle 08: Max Pooling Backward
__global__ void maxpool_backward(const float* grad_output, const int* max_indices,
                                 float* grad_input,
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

    int local_idx = max_indices[idx];
    int ph = local_idx / 2;
    int pw = local_idx % 2;

    int h_start = oh * 2;
    int w_start = ow * 2;

    int in_idx = b * (C * H * W) + c * (H * W)
               + (h_start + ph) * W + (w_start + pw);
    grad_input[in_idx] = grad_output[idx];
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

// From Puzzle 05: Cross-Entropy Loss
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

// From Puzzle 05: Softmax + Cross-Entropy Backward
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

// From Puzzle 06: FC Backward — Weight gradient
__global__ void fc_backward_weights(float* grad_output, float* input,
                                     float* grad_weights, int batch,
                                     int in_features, int out_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < out_features && i < in_features) {
        float sum = 0.0f;
        for (int b = 0; b < batch; b++) {
            sum += grad_output[b * out_features + j] * input[b * in_features + i];
        }
        grad_weights[j * in_features + i] = sum;
    }
}

// From Puzzle 06: FC Backward — Bias gradient
__global__ void fc_backward_bias(float* grad_output, float* grad_bias,
                                  int batch, int out_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < out_features) {
        float sum = 0.0f;
        for (int b = 0; b < batch; b++) {
            sum += grad_output[b * out_features + j];
        }
        grad_bias[j] = sum;
    }
}

// From Puzzle 06: FC Backward — Input gradient
__global__ void fc_backward_input(float* grad_output, float* weights,
                                   float* grad_input, int batch,
                                   int in_features, int out_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (b < batch && i < in_features) {
        float sum = 0.0f;
        for (int jj = 0; jj < out_features; jj++) {
            sum += grad_output[b * out_features + jj] * weights[jj * in_features + i];
        }
        grad_input[b * in_features + i] = sum;
    }
}

// From Puzzle 09: Conv2D Backward — Weight gradient
__global__ void conv2d_backward_weights(const float* grad_output, const float* input,
                                         float* grad_weights,
                                         int batch, int C_in, int H, int W,
                                         int C_out, int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - F + 1;
    int W_out = W - F + 1;
    int total = C_out * C_in * F * F;
    if (idx >= total) return;

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

// From Puzzle 09: Conv2D Backward — Bias gradient
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

// From Puzzle 10: Conv2D Backward — Input gradient
__global__ void conv2d_backward_input(const float* grad_output,
                                       const float* filters,
                                       float* grad_input,
                                       int batch, int C_in, int H, int W,
                                       int C_out, int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * C_in * H * W;
    if (idx >= total) return;

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

// From Puzzle 11: SGD Update
__global__ void sgd_update(float* weights, const float* gradients,
                           float learning_rate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        weights[i] -= learning_rate * gradients[i];
    }
}

// From Puzzle 11: Zero Gradients
__global__ void zero_gradients(float* gradients, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        gradients[i] = 0.0f;
    }
}

// ============================================================
// LeNet-5 structures
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

struct LeNetGradients {
    float* conv1_w;   float* conv1_b;
    float* conv2_w;   float* conv2_b;
    float* fc1_w;     float* fc1_b;
    float* fc2_w;     float* fc2_b;
    float* fc3_w;     float* fc3_b;
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

struct LeNetBackwardActs {
    float* grad_fc3_out;     // [B x 10]
    float* grad_relu4_out;   // [B x 84]
    float* grad_fc2_out;     // [B x 84]
    float* grad_relu3_out;   // [B x 120]
    float* grad_fc1_out;     // [B x 120]
    float* grad_pool2_out;   // [B x 256]
    float* grad_relu2_out;   // [B x 16 x 8 x 8]
    float* grad_conv2_out;   // [B x 16 x 8 x 8]
    float* grad_pool1_out;   // [B x 6 x 12 x 12]
    float* grad_relu1_out;   // [B x 6 x 24 x 24]
    float* grad_conv1_out;   // [B x 6 x 24 x 24]
    float* losses;           // [B]
    float* d_labels_onehot;  // [B x 10]
};

// ============================================================
// Allocation helpers
// ============================================================

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
    cudaFree(p.conv1_w); cudaFree(p.conv1_b);
    cudaFree(p.conv2_w); cudaFree(p.conv2_b);
    cudaFree(p.fc1_w);   cudaFree(p.fc1_b);
    cudaFree(p.fc2_w);   cudaFree(p.fc2_b);
    cudaFree(p.fc3_w);   cudaFree(p.fc3_b);
}

void alloc_gradients(LeNetGradients& g) {
    CUDA_CHECK(cudaMalloc(&g.conv1_w, 150   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv1_b, 6     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv2_w, 2400  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv2_b, 16    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc1_w,   30720 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc1_b,   120   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc2_w,   10080 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc2_b,   84    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc3_w,   840   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc3_b,   10    * sizeof(float)));
}

void free_gradients(LeNetGradients& g) {
    cudaFree(g.conv1_w); cudaFree(g.conv1_b);
    cudaFree(g.conv2_w); cudaFree(g.conv2_b);
    cudaFree(g.fc1_w);   cudaFree(g.fc1_b);
    cudaFree(g.fc2_w);   cudaFree(g.fc2_b);
    cudaFree(g.fc3_w);   cudaFree(g.fc3_b);
}

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
    cudaFree(act.conv1_out);  cudaFree(act.relu1_out);
    cudaFree(act.pool1_out);  cudaFree(act.pool1_indices);
    cudaFree(act.conv2_out);  cudaFree(act.relu2_out);
    cudaFree(act.pool2_out);  cudaFree(act.pool2_indices);
    cudaFree(act.fc1_out);    cudaFree(act.relu3_out);
    cudaFree(act.fc2_out);    cudaFree(act.relu4_out);
    cudaFree(act.fc3_out);    cudaFree(act.probs);
}

void alloc_backward_acts(LeNetBackwardActs& ba, int batch) {
    CUDA_CHECK(cudaMalloc(&ba.grad_fc3_out,    batch * 10  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.grad_relu4_out,  batch * 84  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.grad_fc2_out,    batch * 84  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.grad_relu3_out,  batch * 120 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.grad_fc1_out,    batch * 120 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.grad_pool2_out,  batch * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.grad_relu2_out,  batch * 16 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.grad_conv2_out,  batch * 16 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.grad_pool1_out,  batch * 6  * 12 * 12 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.grad_relu1_out,  batch * 6  * 24 * 24 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.grad_conv1_out,  batch * 6  * 24 * 24 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.losses,          batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ba.d_labels_onehot, batch * 10  * sizeof(float)));
}

void free_backward_acts(LeNetBackwardActs& ba) {
    cudaFree(ba.grad_fc3_out);   cudaFree(ba.grad_relu4_out);
    cudaFree(ba.grad_fc2_out);   cudaFree(ba.grad_relu3_out);
    cudaFree(ba.grad_fc1_out);   cudaFree(ba.grad_pool2_out);
    cudaFree(ba.grad_relu2_out); cudaFree(ba.grad_conv2_out);
    cudaFree(ba.grad_pool1_out); cudaFree(ba.grad_relu1_out);
    cudaFree(ba.grad_conv1_out); cudaFree(ba.losses);
    cudaFree(ba.d_labels_onehot);
}

// Helper: convert integer labels to one-hot on GPU
__global__ void labels_to_onehot(const int* labels, float* onehot,
                                 int batch, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    int offset = b * num_classes;
    for (int c = 0; c < num_classes; c++) {
        onehot[offset + c] = (c == labels[b]) ? 1.0f : 0.0f;
    }
}

// ============================================================
// LeNet-5 Forward Pass (from Puzzle 12)
// ============================================================

void lenet_forward(const float* d_input, LeNetParams& p,
                   LeNetActivations& act, int batch) {
    const int threads = 256;

    { int total = batch * 6 * 24 * 24;
      conv2d_forward<<<(total+threads-1)/threads, threads>>>(
          d_input, p.conv1_w, p.conv1_b, act.conv1_out, batch, 1, 28, 28, 6, 5);
      KERNEL_CHECK(); }

    { int n = batch * 6 * 24 * 24;
      relu_forward<<<(n+threads-1)/threads, threads>>>(act.conv1_out, act.relu1_out, n);
      KERNEL_CHECK(); }

    { int total = batch * 6 * 12 * 12;
      maxpool_forward<<<(total+threads-1)/threads, threads>>>(
          act.relu1_out, act.pool1_out, act.pool1_indices, batch, 6, 24, 24);
      KERNEL_CHECK(); }

    { int total = batch * 16 * 8 * 8;
      conv2d_forward<<<(total+threads-1)/threads, threads>>>(
          act.pool1_out, p.conv2_w, p.conv2_b, act.conv2_out, batch, 6, 12, 12, 16, 5);
      KERNEL_CHECK(); }

    { int n = batch * 16 * 8 * 8;
      relu_forward<<<(n+threads-1)/threads, threads>>>(act.conv2_out, act.relu2_out, n);
      KERNEL_CHECK(); }

    { int total = batch * 16 * 4 * 4;
      maxpool_forward<<<(total+threads-1)/threads, threads>>>(
          act.relu2_out, act.pool2_out, act.pool2_indices, batch, 16, 8, 8);
      KERNEL_CHECK(); }

    { dim3 block(16, 16); dim3 grid((120+15)/16, (batch+15)/16);
      fc_forward<<<grid, block>>>(act.pool2_out, p.fc1_w, p.fc1_b,
                                   act.fc1_out, batch, 256, 120);
      KERNEL_CHECK(); }

    { int n = batch * 120;
      relu_forward<<<(n+threads-1)/threads, threads>>>(act.fc1_out, act.relu3_out, n);
      KERNEL_CHECK(); }

    { dim3 block(16, 16); dim3 grid((84+15)/16, (batch+15)/16);
      fc_forward<<<grid, block>>>(act.relu3_out, p.fc2_w, p.fc2_b,
                                   act.fc2_out, batch, 120, 84);
      KERNEL_CHECK(); }

    { int n = batch * 84;
      relu_forward<<<(n+threads-1)/threads, threads>>>(act.fc2_out, act.relu4_out, n);
      KERNEL_CHECK(); }

    { dim3 block(16, 16); dim3 grid((10+15)/16, (batch+15)/16);
      fc_forward<<<grid, block>>>(act.relu4_out, p.fc3_w, p.fc3_b,
                                   act.fc3_out, batch, 84, 10);
      KERNEL_CHECK(); }

    { softmax_forward<<<(batch+threads-1)/threads, threads>>>(
          act.fc3_out, act.probs, batch, 10);
      KERNEL_CHECK(); }
}

// ============================================================
// LeNet-5 Backward Pass
// ============================================================

void lenet_backward(const float* d_input, LeNetParams& p,
                    LeNetGradients& g, LeNetActivations& act,
                    LeNetBackwardActs& ba, int batch) {
    const int threads = 256;

    // --- Softmax + CE backward: dL/d(fc3_out) ---
    { softmax_ce_backward<<<(batch+threads-1)/threads, threads>>>(
          act.probs, ba.d_labels_onehot, ba.grad_fc3_out, batch, 10);
      KERNEL_CHECK(); }

    // --- FC3 backward: grad_fc3_out -> grad_relu4_out, grad_fc3_w, grad_fc3_b ---
    { dim3 block(16, 16); dim3 grid((84+15)/16, (10+15)/16);
      fc_backward_weights<<<grid, block>>>(ba.grad_fc3_out, act.relu4_out,
                                            g.fc3_w, batch, 84, 10);
      KERNEL_CHECK(); }
    { fc_backward_bias<<<(10+threads-1)/threads, threads>>>(
          ba.grad_fc3_out, g.fc3_b, batch, 10);
      KERNEL_CHECK(); }
    { dim3 block(16, 16); dim3 grid((84+15)/16, (batch+15)/16);
      fc_backward_input<<<grid, block>>>(ba.grad_fc3_out, p.fc3_w,
                                          ba.grad_relu4_out, batch, 84, 10);
      KERNEL_CHECK(); }

    // --- ReLU4 backward ---
    { int n = batch * 84;
      relu_backward<<<(n+threads-1)/threads, threads>>>(
          ba.grad_relu4_out, act.fc2_out, ba.grad_fc2_out, n);
      KERNEL_CHECK(); }

    // --- FC2 backward ---
    { dim3 block(16, 16); dim3 grid((120+15)/16, (84+15)/16);
      fc_backward_weights<<<grid, block>>>(ba.grad_fc2_out, act.relu3_out,
                                            g.fc2_w, batch, 120, 84);
      KERNEL_CHECK(); }
    { fc_backward_bias<<<(84+threads-1)/threads, threads>>>(
          ba.grad_fc2_out, g.fc2_b, batch, 84);
      KERNEL_CHECK(); }
    { dim3 block(16, 16); dim3 grid((120+15)/16, (batch+15)/16);
      fc_backward_input<<<grid, block>>>(ba.grad_fc2_out, p.fc2_w,
                                          ba.grad_relu3_out, batch, 120, 84);
      KERNEL_CHECK(); }

    // --- ReLU3 backward ---
    { int n = batch * 120;
      relu_backward<<<(n+threads-1)/threads, threads>>>(
          ba.grad_relu3_out, act.fc1_out, ba.grad_fc1_out, n);
      KERNEL_CHECK(); }

    // --- FC1 backward ---
    { dim3 block(16, 16); dim3 grid((256+15)/16, (120+15)/16);
      fc_backward_weights<<<grid, block>>>(ba.grad_fc1_out, act.pool2_out,
                                            g.fc1_w, batch, 256, 120);
      KERNEL_CHECK(); }
    { fc_backward_bias<<<(120+threads-1)/threads, threads>>>(
          ba.grad_fc1_out, g.fc1_b, batch, 120);
      KERNEL_CHECK(); }
    { dim3 block(16, 16); dim3 grid((256+15)/16, (batch+15)/16);
      fc_backward_input<<<grid, block>>>(ba.grad_fc1_out, p.fc1_w,
                                          ba.grad_pool2_out, batch, 256, 120);
      KERNEL_CHECK(); }

    // --- Pool2 backward: grad_pool2_out (B x 256) -> grad_relu2_out (B x 16 x 8 x 8) ---
    // Must zero grad_relu2_out first since maxpool_backward scatters
    CUDA_CHECK(cudaMemset(ba.grad_relu2_out, 0,
                          batch * 16 * 8 * 8 * sizeof(float)));
    { int total = batch * 16 * 4 * 4;
      maxpool_backward<<<(total+threads-1)/threads, threads>>>(
          ba.grad_pool2_out, act.pool2_indices,
          ba.grad_relu2_out, batch, 16, 8, 8);
      KERNEL_CHECK(); }

    // --- ReLU2 backward ---
    { int n = batch * 16 * 8 * 8;
      relu_backward<<<(n+threads-1)/threads, threads>>>(
          ba.grad_relu2_out, act.conv2_out, ba.grad_conv2_out, n);
      KERNEL_CHECK(); }

    // --- Conv2 backward ---
    { int total = 16 * 6 * 5 * 5;
      conv2d_backward_weights<<<(total+threads-1)/threads, threads>>>(
          ba.grad_conv2_out, act.pool1_out, g.conv2_w,
          batch, 6, 12, 12, 16, 5);
      KERNEL_CHECK(); }
    { conv2d_backward_bias<<<(16+threads-1)/threads, threads>>>(
          ba.grad_conv2_out, g.conv2_b, batch, 16, 8, 8);
      KERNEL_CHECK(); }
    { int total = batch * 6 * 12 * 12;
      conv2d_backward_input<<<(total+threads-1)/threads, threads>>>(
          ba.grad_conv2_out, p.conv2_w, ba.grad_pool1_out,
          batch, 6, 12, 12, 16, 5);
      KERNEL_CHECK(); }

    // --- Pool1 backward: grad_pool1_out (B x 6 x 12 x 12) -> grad_relu1_out (B x 6 x 24 x 24) ---
    CUDA_CHECK(cudaMemset(ba.grad_relu1_out, 0,
                          batch * 6 * 24 * 24 * sizeof(float)));
    { int total = batch * 6 * 12 * 12;
      maxpool_backward<<<(total+threads-1)/threads, threads>>>(
          ba.grad_pool1_out, act.pool1_indices,
          ba.grad_relu1_out, batch, 6, 24, 24);
      KERNEL_CHECK(); }

    // --- ReLU1 backward ---
    { int n = batch * 6 * 24 * 24;
      relu_backward<<<(n+threads-1)/threads, threads>>>(
          ba.grad_relu1_out, act.conv1_out, ba.grad_conv1_out, n);
      KERNEL_CHECK(); }

    // --- Conv1 backward (weights + bias only, no need for input gradient) ---
    { int total = 6 * 1 * 5 * 5;
      conv2d_backward_weights<<<(total+threads-1)/threads, threads>>>(
          ba.grad_conv1_out, d_input, g.conv1_w,
          batch, 1, 28, 28, 6, 5);
      KERNEL_CHECK(); }
    { conv2d_backward_bias<<<(6+threads-1)/threads, threads>>>(
          ba.grad_conv1_out, g.conv1_b, batch, 6, 24, 24);
      KERNEL_CHECK(); }
}

// ============================================================
// SGD update for all parameters
// ============================================================

void sgd_update_all(LeNetParams& p, LeNetGradients& g, float lr) {
    const int threads = 256;

    auto launch = [&](float* w, const float* gw, int n) {
        sgd_update<<<(n+threads-1)/threads, threads>>>(w, gw, lr, n);
        KERNEL_CHECK();
    };

    launch(p.conv1_w, g.conv1_w, 150);
    launch(p.conv1_b, g.conv1_b, 6);
    launch(p.conv2_w, g.conv2_w, 2400);
    launch(p.conv2_b, g.conv2_b, 16);
    launch(p.fc1_w,   g.fc1_w,   30720);
    launch(p.fc1_b,   g.fc1_b,   120);
    launch(p.fc2_w,   g.fc2_w,   10080);
    launch(p.fc2_b,   g.fc2_b,   84);
    launch(p.fc3_w,   g.fc3_w,   840);
    launch(p.fc3_b,   g.fc3_b,   10);
}

// ============================================================
// Xavier Initialization
// ============================================================

// Helper: fill GPU buffer with Xavier uniform values
static void xavier_fill(float* d_ptr, int n, int fan_in, int fan_out,
                        std::mt19937& gen) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    std::uniform_real_distribution<float> dis(-limit, limit);
    std::vector<float> h_buf(n);
    for (int i = 0; i < n; i++) {
        h_buf[i] = dis(gen);
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, h_buf.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void xavier_init(LeNetParams& params, unsigned seed) {
    std::mt19937 gen(seed);

    // Conv1: 6 filters of 1x5x5, fan_in=1*5*5=25, fan_out=6*5*5=150
    xavier_fill(params.conv1_w, 150, 25, 150, gen);
    CUDA_CHECK(cudaMemset(params.conv1_b, 0, 6 * sizeof(float)));

    // Conv2: 16 filters of 6x5x5, fan_in=6*5*5=150, fan_out=16*5*5=400
    xavier_fill(params.conv2_w, 2400, 150, 400, gen);
    CUDA_CHECK(cudaMemset(params.conv2_b, 0, 16 * sizeof(float)));

    // FC1: 256 -> 120
    xavier_fill(params.fc1_w, 30720, 256, 120, gen);
    CUDA_CHECK(cudaMemset(params.fc1_b, 0, 120 * sizeof(float)));

    // FC2: 120 -> 84
    xavier_fill(params.fc2_w, 10080, 120, 84, gen);
    CUDA_CHECK(cudaMemset(params.fc2_b, 0, 84 * sizeof(float)));

    // FC3: 84 -> 10
    xavier_fill(params.fc3_w, 840, 84, 10, gen);
    CUDA_CHECK(cudaMemset(params.fc3_b, 0, 10 * sizeof(float)));
}

// ============================================================
// Train one epoch
// ============================================================

float train_epoch(const float* d_images, const int* d_labels,
                  int num_samples, int batch_size,
                  LeNetParams& params, LeNetGradients& grads,
                  LeNetActivations& act, LeNetBackwardActs& back_act,
                  float learning_rate) {
    const int threads = 256;
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    float total_loss = 0.0f;

    for (int b = 0; b < num_batches; b++) {
        int start = b * batch_size;
        int cur_batch = batch_size;
        if (start + cur_batch > num_samples) {
            cur_batch = num_samples - start;
        }

        const float* batch_images = d_images + start * 1 * 28 * 28;
        const int* batch_labels = d_labels + start;

        // 1. Forward pass
        lenet_forward(batch_images, params, act, cur_batch);

        // 2. Convert labels to one-hot
        labels_to_onehot<<<(cur_batch+threads-1)/threads, threads>>>(
            batch_labels, back_act.d_labels_onehot, cur_batch, 10);
        KERNEL_CHECK();

        // 3. Compute loss
        cross_entropy_loss<<<(cur_batch+threads-1)/threads, threads>>>(
            act.probs, back_act.d_labels_onehot, back_act.losses, cur_batch, 10);
        KERNEL_CHECK();

        // Sum losses on host
        std::vector<float> h_losses(cur_batch);
        CUDA_CHECK(cudaMemcpy(h_losses.data(), back_act.losses,
                              cur_batch * sizeof(float), cudaMemcpyDeviceToHost));
        float batch_loss = 0.0f;
        for (int i = 0; i < cur_batch; i++) batch_loss += h_losses[i];
        total_loss += batch_loss;

        // 4. Backward pass
        lenet_backward(batch_images, params, grads, act, back_act, cur_batch);

        // 5. SGD update
        sgd_update_all(params, grads, learning_rate);
    }

    return total_loss / (float)num_samples;
}

// ============================================================
// Evaluate accuracy
// ============================================================

float evaluate(const float* d_images, const int* d_labels,
               int num_samples, int batch_size,
               LeNetParams& params, LeNetActivations& act) {
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    int correct = 0;

    for (int b = 0; b < num_batches; b++) {
        int start = b * batch_size;
        int cur_batch = batch_size;
        if (start + cur_batch > num_samples) {
            cur_batch = num_samples - start;
        }

        const float* batch_images = d_images + start * 1 * 28 * 28;
        const int* batch_labels = d_labels + start;

        // Forward pass
        lenet_forward(batch_images, params, act, cur_batch);

        // Copy probs and labels back
        std::vector<float> h_probs(cur_batch * 10);
        std::vector<int> h_labels(cur_batch);
        CUDA_CHECK(cudaMemcpy(h_probs.data(), act.probs,
                              cur_batch * 10 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_labels.data(), batch_labels,
                              cur_batch * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // Count correct predictions
        for (int i = 0; i < cur_batch; i++) {
            int pred = 0;
            float max_prob = h_probs[i * 10];
            for (int c = 1; c < 10; c++) {
                if (h_probs[i * 10 + c] > max_prob) {
                    max_prob = h_probs[i * 10 + c];
                    pred = c;
                }
            }
            if (pred == h_labels[i]) correct++;
        }
    }

    return (float)correct / (float)num_samples;
}

// ============================================================
// Full training loop
// ============================================================

void training_loop(const float* d_images, const int* d_labels,
                   int num_samples, int num_epochs,
                   int batch_size, float learning_rate,
                   unsigned seed) {
    // Allocate everything
    LeNetParams params;
    alloc_params(params);

    LeNetGradients grads;
    alloc_gradients(grads);

    LeNetActivations act;
    alloc_activations(act, batch_size);

    LeNetBackwardActs back_act;
    alloc_backward_acts(back_act, batch_size);

    // Xavier init
    xavier_init(params, seed);

    // Train
    for (int epoch = 1; epoch <= num_epochs; epoch++) {
        float avg_loss = train_epoch(d_images, d_labels, num_samples,
                                      batch_size, params, grads, act,
                                      back_act, learning_rate);
        float accuracy = evaluate(d_images, d_labels, num_samples,
                                   batch_size, params, act);
        printf("Epoch %d: loss=%.4f, accuracy=%.2f%%\n",
               epoch, avg_loss, accuracy * 100.0f);
    }

    // Cleanup
    free_backward_acts(back_act);
    free_activations(act);
    free_gradients(grads);
    free_params(params);
}
