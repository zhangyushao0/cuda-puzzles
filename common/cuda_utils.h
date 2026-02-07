#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// CUDA error checking macro
// Wraps CUDA API calls and prints file:line on error, then exits
#define CUDA_CHECK(err) do { \
  cudaError_t e = (err); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while(0)

// Kernel launch error checking macro
// Checks for launch errors and synchronizes to catch execution errors
#define KERNEL_CHECK() do { \
  CUDA_CHECK(cudaGetLastError()); \
  CUDA_CHECK(cudaDeviceSynchronize()); \
} while(0)

// CUDA timer utility class using cudaEvent timing
class CudaTimer {
private:
  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  bool started;

public:
  CudaTimer() : started(false) {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
  }

  ~CudaTimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }

  void start() {
    CUDA_CHECK(cudaEventRecord(start_event));
    started = true;
  }

  void stop() {
    if (!started) {
      fprintf(stderr, "CudaTimer: stop() called before start()\n");
      return;
    }
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
  }

  float elapsed_ms() {
    if (!started) {
      fprintf(stderr, "CudaTimer: elapsed_ms() called before start()\n");
      return 0.0f;
    }
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    return ms;
  }

  // Convenience method: start, run function, stop, return elapsed time
  template<typename Func>
  static float measure(Func func) {
    CudaTimer timer;
    timer.start();
    func();
    timer.stop();
    return timer.elapsed_ms();
  }
};

#endif // CUDA_UTILS_H
