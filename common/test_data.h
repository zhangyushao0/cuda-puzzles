#ifndef TEST_DATA_H
#define TEST_DATA_H

#include <vector>
#include <random>
#include <cmath>

// Generate deterministic synthetic test images (28x28 float arrays)
// Each image is a simple pattern that can be verified
inline std::vector<float> generate_test_images(int n, unsigned int seed = 123) {
  const int img_size = 28 * 28;
  std::vector<float> images(n * img_size);
  
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  
  for (int i = 0; i < n; i++) {
    // Generate a gradient pattern with some randomness
    for (int row = 0; row < 28; row++) {
      for (int col = 0; col < 28; col++) {
        int idx = i * img_size + row * 28 + col;
        
        // Base pattern: radial gradient from center
        float cy = 13.5f;
        float cx = 13.5f;
        float dist = sqrtf((row - cy) * (row - cy) + (col - cx) * (col - cx));
        float base = 1.0f - (dist / 20.0f);
        base = fmaxf(0.0f, fminf(1.0f, base));
        
        // Add deterministic noise based on image index and position
        std::mt19937 local_gen(seed + i * 1000 + row * 28 + col);
        std::uniform_real_distribution<float> noise_dis(-0.1f, 0.1f);
        float noise = noise_dis(local_gen);
        
        images[idx] = fmaxf(0.0f, fminf(1.0f, base + noise));
      }
    }
  }
  
  return images;
}

// Generate deterministic synthetic test labels [0-9]
inline std::vector<int> generate_test_labels(int n, unsigned int seed = 123) {
  std::vector<int> labels(n);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dis(0, 9);
  
  for (int i = 0; i < n; i++) {
    labels[i] = dis(gen);
  }
  
  return labels;
}

// Generate simple test pattern for verification
// Returns a 28x28 image with a simple checkerboard pattern
inline std::vector<float> generate_checkerboard() {
  const int size = 28 * 28;
  std::vector<float> image(size);
  
  for (int row = 0; row < 28; row++) {
    for (int col = 0; col < 28; col++) {
      int idx = row * 28 + col;
      // Checkerboard: alternate 0.0 and 1.0 every 4 pixels
      bool is_white = ((row / 4) + (col / 4)) % 2 == 0;
      image[idx] = is_white ? 1.0f : 0.0f;
    }
  }
  
  return image;
}

// Generate simple constant image for testing
inline std::vector<float> generate_constant_image(float value) {
  return std::vector<float>(28 * 28, value);
}

#endif // TEST_DATA_H
