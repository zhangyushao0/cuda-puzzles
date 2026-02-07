#include "common.h"
#include <cstdio>
#include <cstring>

// Test CUDA error checking macros
TEST_CASE(cuda_check_valid_call) {
  // Test that CUDA_CHECK works with valid calls
  float* d_ptr;
  CUDA_CHECK(cudaMalloc(&d_ptr, 1024));
  CUDA_CHECK(cudaFree(d_ptr));
}

// Test kernel error checking
__global__ void test_kernel() {
  // Simple kernel that does nothing
}

TEST_CASE(kernel_check_valid_launch) {
  test_kernel<<<1, 1>>>();
  KERNEL_CHECK();
}

// Test fill_random produces deterministic output
TEST_CASE(fill_random_deterministic) {
  const int n = 100;
  float arr1[n], arr2[n];
  
  fill_random(arr1, n, 42, 0.0f, 1.0f);
  fill_random(arr2, n, 42, 0.0f, 1.0f);
  
  bool passed = check_array_close(arr1, arr2, n, 1e-6f, 1e-6f);
  if (!passed) {
    throw std::runtime_error("fill_random not deterministic");
  }
}

// Test check_close passes for close values
TEST_CASE(check_close_passes) {
  if (!check_close(1.0f, 1.0001f, 1e-3f, 1e-3f)) {
    throw std::runtime_error("check_close failed for close values");
  }
  if (!check_close(0.0f, 0.0f, 1e-4f, 1e-4f)) {
    throw std::runtime_error("check_close failed for equal values");
  }
}

// Test check_close fails for distant values
TEST_CASE(check_close_fails) {
  if (check_close(1.0f, 2.0f, 1e-4f, 1e-4f)) {
    throw std::runtime_error("check_close passed for distant values");
  }
  if (check_close(0.0f, 1.0f, 1e-4f, 1e-4f)) {
    throw std::runtime_error("check_close passed for very different values");
  }
}

// Test byte swap works correctly
TEST_CASE(byte_swap_correct) {
  uint32_t big_endian = 0x12345678;
  uint32_t little_endian = swap_endian(big_endian);
  
  if (little_endian != 0x78563412) {
    char msg[256];
    snprintf(msg, sizeof(msg), "swap_endian failed: got 0x%08x, expected 0x78563412", little_endian);
    throw std::runtime_error(msg);
  }
  
  // Test round-trip
  uint32_t back = swap_endian(little_endian);
  if (back != big_endian) {
    throw std::runtime_error("swap_endian round-trip failed");
  }
}

// Test test data generator produces correct shapes
TEST_CASE(test_data_shapes) {
  const int n = 10;
  auto images = generate_test_images(n, 123);
  auto labels = generate_test_labels(n, 123);
  
  if (images.size() != (size_t)(n * 28 * 28)) {
    char msg[256];
    snprintf(msg, sizeof(msg), "Wrong image size: %zu, expected %d", images.size(), n * 28 * 28);
    throw std::runtime_error(msg);
  }
  
  if (labels.size() != (size_t)n) {
    char msg[256];
    snprintf(msg, sizeof(msg), "Wrong label count: %zu, expected %d", labels.size(), n);
    throw std::runtime_error(msg);
  }
  
  // Check that values are in valid range
  for (float val : images) {
    if (val < 0.0f || val > 1.0f) {
      throw std::runtime_error("Image value out of range [0, 1]");
    }
  }
  
  for (int label : labels) {
    if (label < 0 || label > 9) {
      throw std::runtime_error("Label out of range [0, 9]");
    }
  }
}

// Test checkerboard pattern
TEST_CASE(checkerboard_pattern) {
  auto image = generate_checkerboard();
  
  if (image.size() != 28 * 28) {
    throw std::runtime_error("Checkerboard wrong size");
  }
  
  // Check corner values (should follow checkerboard pattern)
  if (image[0] != 1.0f) {
    throw std::runtime_error("Checkerboard corner [0,0] should be 1.0");
  }
}

// Test constant image
TEST_CASE(constant_image) {
  auto image = generate_constant_image(0.5f);
  
  if (image.size() != 28 * 28) {
    throw std::runtime_error("Constant image wrong size");
  }
  
  for (float val : image) {
    if (val != 0.5f) {
      throw std::runtime_error("Constant image has non-constant value");
    }
  }
}

// Test CudaTimer
TEST_CASE(cuda_timer) {
  CudaTimer timer;
  timer.start();
  
  // Do some work
  float* d_ptr;
  CUDA_CHECK(cudaMalloc(&d_ptr, 1024 * 1024));
  CUDA_CHECK(cudaMemset(d_ptr, 0, 1024 * 1024));
  CUDA_CHECK(cudaFree(d_ptr));
  
  timer.stop();
  float elapsed = timer.elapsed_ms();
  
  if (elapsed < 0.0f) {
    throw std::runtime_error("Timer returned negative time");
  }
}

// Test array comparison
TEST_CASE(array_comparison) {
  const int n = 10;
  float arr1[n], arr2[n];
  
  for (int i = 0; i < n; i++) {
    arr1[i] = static_cast<float>(i);
    arr2[i] = static_cast<float>(i);
  }
  
  if (!check_array_close(arr1, arr2, n, 1e-4f, 1e-4f)) {
    throw std::runtime_error("Identical arrays not equal");
  }
  
  // Make one element different
  arr2[5] = 100.0f;
  if (check_array_close(arr1, arr2, n, 1e-4f, 1e-4f)) {
    throw std::runtime_error("Different arrays reported as equal");
  }
}

int main() {
  printf("=== Common Utilities Test Suite ===\n");
  printf("Testing CUDA error checking, test framework, and data utilities\n\n");
  
  int result = RUN_ALL_TESTS();
  
  if (result == 0) {
    printf("\n✓ All common utilities verified successfully!\n");
  } else {
    printf("\n✗ Some tests failed. Please check output above.\n");
  }
  
  return result;
}
