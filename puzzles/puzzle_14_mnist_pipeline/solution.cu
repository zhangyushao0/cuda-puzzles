// Puzzle 14: MNIST Data Pipeline — Reference Solution
// Loads IDX binary files, normalizes pixels, creates batches, and shuffles.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

// Big-endian to little-endian byte swap for 32-bit integers
inline uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0x000000ff) |
           ((val >>  8) & 0x0000ff00) |
           ((val <<  8) & 0x00ff0000) |
           ((val << 24) & 0xff000000);
}

// Load MNIST images from IDX3-UBYTE format
// Reads header (big-endian), pixel data, and normalizes to [0.0, 1.0]
void load_mnist_images(const char* path,
                       float** images,
                       int* count,
                       int* rows,
                       int* cols) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open MNIST image file: %s\n", path);
        exit(1);
    }

    // Read 4 big-endian uint32 header fields
    uint32_t magic, n, r, c;
    fread(&magic, 4, 1, f);
    fread(&n, 4, 1, f);
    fread(&r, 4, 1, f);
    fread(&c, 4, 1, f);

    // Byte-swap from big-endian to host (little-endian)
    magic = swap_endian(magic);
    n = swap_endian(n);
    r = swap_endian(r);
    c = swap_endian(c);

    // Verify magic number for IDX3 image files
    if (magic != 0x00000803) {
        fprintf(stderr, "ERROR: Invalid image magic: 0x%08x (expected 0x00000803)\n", magic);
        fclose(f);
        exit(1);
    }

    // Read raw pixel data
    int size = (int)(n * r * c);
    uint8_t* buffer = (uint8_t*)malloc(size);
    fread(buffer, 1, size, f);
    fclose(f);

    // Normalize [0,255] -> [0.0, 1.0]
    float* out = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        out[i] = static_cast<float>(buffer[i]) / 255.0f;
    }
    free(buffer);

    *images = out;
    *count = (int)n;
    *rows = (int)r;
    *cols = (int)c;
}

// Load MNIST labels from IDX1-UBYTE format
// Reads header (big-endian), label bytes, converts to int
void load_mnist_labels(const char* path,
                       int** labels,
                       int* count) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open MNIST label file: %s\n", path);
        exit(1);
    }

    // Read 2 big-endian uint32 header fields
    uint32_t magic, n;
    fread(&magic, 4, 1, f);
    fread(&n, 4, 1, f);

    magic = swap_endian(magic);
    n = swap_endian(n);

    if (magic != 0x00000801) {
        fprintf(stderr, "ERROR: Invalid label magic: 0x%08x (expected 0x00000801)\n", magic);
        fclose(f);
        exit(1);
    }

    // Read label bytes
    uint8_t* buffer = (uint8_t*)malloc(n);
    fread(buffer, 1, n, f);
    fclose(f);

    // Convert to int
    int* out = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < (int)n; i++) {
        out[i] = static_cast<int>(buffer[i]);
    }
    free(buffer);

    *labels = out;
    *count = (int)n;
}

// Create contiguous batches from image data
// Each batch[i] points into the original images array (no copy)
// Only full batches are created; remainder images are dropped
void create_batches(const float* images,
                    int count,
                    int image_size,
                    int batch_size,
                    float*** batches,
                    int* num_batches) {
    int nb = count / batch_size;  // drop remainder
    float** out = (float**)malloc(nb * sizeof(float*));
    for (int i = 0; i < nb; i++) {
        out[i] = const_cast<float*>(images + i * batch_size * image_size);
    }
    *batches = out;
    *num_batches = nb;
}

// Fisher-Yates deterministic shuffle of images + labels together
// Uses std::mt19937 for reproducibility with given seed
void shuffle_data(float* images,
                  int* labels,
                  int count,
                  int image_size,
                  unsigned int seed) {
    std::mt19937 rng(seed);

    // Temp buffer for swapping one image
    std::vector<float> tmp(image_size);

    for (int i = count - 1; i > 0; i--) {
        int j = (int)(rng() % (unsigned int)(i + 1));
        if (i != j) {
            // Swap images
            float* img_i = images + i * image_size;
            float* img_j = images + j * image_size;
            memcpy(tmp.data(), img_i, image_size * sizeof(float));
            memcpy(img_i, img_j, image_size * sizeof(float));
            memcpy(img_j, tmp.data(), image_size * sizeof(float));

            // Swap labels
            int tmp_label = labels[i];
            labels[i] = labels[j];
            labels[j] = tmp_label;
        }
    }
}
