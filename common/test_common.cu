#include "common.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstring>

// Test CUDA error checking macros
TEST_CASE("cuda_check_valid_call", "[cuda]") {
  // Test that CUDA_CHECK works with valid calls
  float* d_ptr;
  CUDA_CHECK(cudaMalloc(&d_ptr, 1024));
  CUDA_CHECK(cudaFree(d_ptr));
}

// Test kernel error checking
__global__ void test_kernel() {
  // Simple kernel that does nothing
}

TEST_CASE("kernel_check_valid_launch", "[cuda]") {
  test_kernel<<<1, 1>>>();
  KERNEL_CHECK();
}

// Test fill_random produces deterministic output
TEST_CASE("fill_random_deterministic", "[utils]") {
  const int n = 100;
  float arr1[n], arr2[n];
  
  fill_random(arr1, n, 42, 0.0f, 1.0f);
  fill_random(arr2, n, 42, 0.0f, 1.0f);

  REQUIRE(check_array_close(arr1, arr2, n, 1e-6f, 1e-6f));
}

// Test check_close passes for close values
TEST_CASE("check_close_passes", "[utils]") {
  REQUIRE(check_close(1.0f, 1.0001f, 1e-3f, 1e-3f));
  REQUIRE(check_close(0.0f, 0.0f, 1e-4f, 1e-4f));
}

// Test check_close fails for distant values
TEST_CASE("check_close_fails", "[utils]") {
  REQUIRE_FALSE(check_close(1.0f, 2.0f, 1e-4f, 1e-4f));
  REQUIRE_FALSE(check_close(0.0f, 1.0f, 1e-4f, 1e-4f));
}

// Test byte swap works correctly
TEST_CASE("byte_swap_correct", "[utils]") {
  uint32_t big_endian = 0x12345678;
  uint32_t little_endian = swap_endian(big_endian);

  REQUIRE(little_endian == 0x78563412);

  // Test round-trip
  uint32_t back = swap_endian(little_endian);
  REQUIRE(back == big_endian);
}

// Test test data generator produces correct shapes
TEST_CASE("test_data_shapes", "[data]") {
  const int n = 10;
  auto images = generate_test_images(n, 123);
  auto labels = generate_test_labels(n, 123);

  REQUIRE(images.size() == (size_t)(n * 28 * 28));
  REQUIRE(labels.size() == (size_t)n);

  // Check that values are in valid range
  for (float val : images) {
    REQUIRE(val >= 0.0f);
    REQUIRE(val <= 1.0f);
  }
  
  for (int label : labels) {
    REQUIRE(label >= 0);
    REQUIRE(label <= 9);
  }
}

// Test checkerboard pattern
TEST_CASE("checkerboard_pattern", "[data]") {
  auto image = generate_checkerboard();

  REQUIRE(image.size() == 28 * 28);

  // Check corner values (should follow checkerboard pattern)
  REQUIRE(image[0] == 1.0f);
}

// Test constant image
TEST_CASE("constant_image", "[data]") {
  auto image = generate_constant_image(0.5f);

  REQUIRE(image.size() == 28 * 28);

  for (float val : image) {
    REQUIRE(val == 0.5f);
  }
}

// Test CudaTimer
TEST_CASE("cuda_timer", "[cuda]") {
  CudaTimer timer;
  timer.start();
  
  // Do some work
  float* d_ptr;
  CUDA_CHECK(cudaMalloc(&d_ptr, 1024 * 1024));
  CUDA_CHECK(cudaMemset(d_ptr, 0, 1024 * 1024));
  CUDA_CHECK(cudaFree(d_ptr));
  
  timer.stop();
  float elapsed = timer.elapsed_ms();

  REQUIRE(elapsed >= 0.0f);
}

// Test array comparison
TEST_CASE("array_comparison", "[utils]") {
  const int n = 10;
  float arr1[n], arr2[n];
  
  for (int i = 0; i < n; i++) {
    arr1[i] = static_cast<float>(i);
    arr2[i] = static_cast<float>(i);
  }

  REQUIRE(check_array_close(arr1, arr2, n, 1e-4f, 1e-4f));

  // Make one element different
  arr2[5] = 100.0f;
  REQUIRE_FALSE(check_array_close(arr1, arr2, n, 1e-4f, 1e-4f));
}
