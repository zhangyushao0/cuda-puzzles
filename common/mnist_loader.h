#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Big-endian to little-endian byte swap for 32-bit integers
inline uint32_t swap_endian(uint32_t val) {
  return ((val >> 24) & 0xff) | 
         ((val >> 8) & 0xff00) | 
         ((val << 8) & 0xff0000) | 
         ((val << 24) & 0xff000000);
}

// MNIST IDX file loader
class MNISTLoader {
public:
  // Load MNIST images from IDX3 format
  // Returns vector of normalized floats [0.0, 1.0]
  static std::vector<float> load_images(const char* path, int& count, int& rows, int& cols) {
    FILE* f = fopen(path, "rb");
    if (!f) {
      fprintf(stderr, "ERROR: Cannot open MNIST image file: %s\n", path);
      fprintf(stderr, "       Please check that the file exists and path is correct.\n");
      exit(1);
    }

    // Read header: magic, count, rows, cols (all big-endian 32-bit integers)
    uint32_t magic, n, r, c;
    fread(&magic, 4, 1, f);
    fread(&n, 4, 1, f);
    fread(&r, 4, 1, f);
    fread(&c, 4, 1, f);

    // Swap from big-endian to little-endian
    magic = swap_endian(magic);
    n = swap_endian(n);
    r = swap_endian(r);
    c = swap_endian(c);

    // Verify magic number for images (0x00000803)
    if (magic != 0x00000803) {
      fprintf(stderr, "ERROR: Invalid MNIST image file magic number: 0x%08x (expected 0x00000803)\n", magic);
      fclose(f);
      exit(1);
    }

    count = n;
    rows = r;
    cols = c;

    // Read pixel data
    int size = n * r * c;
    std::vector<uint8_t> buffer(size);
    fread(buffer.data(), 1, size, f);
    fclose(f);

    // Normalize [0, 255] -> [0.0, 1.0]
    std::vector<float> images(size);
    for (int i = 0; i < size; i++) {
      images[i] = static_cast<float>(buffer[i]) / 255.0f;
    }

    return images;
  }

  // Load MNIST labels from IDX1 format
  // Returns vector of label integers [0-9]
  static std::vector<int> load_labels(const char* path, int& count) {
    FILE* f = fopen(path, "rb");
    if (!f) {
      fprintf(stderr, "ERROR: Cannot open MNIST label file: %s\n", path);
      fprintf(stderr, "       Please check that the file exists and path is correct.\n");
      exit(1);
    }

    // Read header: magic, count (both big-endian 32-bit integers)
    uint32_t magic, n;
    fread(&magic, 4, 1, f);
    fread(&n, 4, 1, f);

    // Swap from big-endian to little-endian
    magic = swap_endian(magic);
    n = swap_endian(n);

    // Verify magic number for labels (0x00000801)
    if (magic != 0x00000801) {
      fprintf(stderr, "ERROR: Invalid MNIST label file magic number: 0x%08x (expected 0x00000801)\n", magic);
      fclose(f);
      exit(1);
    }

    count = n;

    // Read label data
    std::vector<uint8_t> buffer(n);
    fread(buffer.data(), 1, n, f);
    fclose(f);

    // Convert to int
    std::vector<int> labels(n);
    for (int i = 0; i < (int)n; i++) {
      labels[i] = static_cast<int>(buffer[i]);
    }

    return labels;
  }
};

#endif // MNIST_LOADER_H
