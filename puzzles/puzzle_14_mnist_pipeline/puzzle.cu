// Puzzle 14: MNIST Data Pipeline
// Implement functions to load, normalize, batch, and shuffle MNIST data.
//
// See README.md for the IDX file format specification and hints.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

// =============================================================
// TODO: Implement byte swap for big-endian to little-endian
//
// IDX files store all 32-bit integers in big-endian format.
// x86/x64 CPUs are little-endian, so you must reverse the bytes:
//   Byte 0 <-> Byte 3
//   Byte 1 <-> Byte 2
//
// Use bit-shifts and masks:
//   ((val >> 24) & 0xff)       — move byte 3 to byte 0
//   ((val >>  8) & 0xff00)     — move byte 2 to byte 1
//   ((val <<  8) & 0xff0000)   — move byte 1 to byte 2
//   ((val << 24) & 0xff000000) — move byte 0 to byte 3
// =============================================================
inline uint32_t swap_endian(uint32_t val) {
    // TODO: Your code here
    return val;
}

// =============================================================
// TODO: Load MNIST images from IDX3-UBYTE binary file
//
// Steps:
//   1. Open file in binary read mode ("rb")
//   2. Read 4 uint32_t values: magic, count, rows, cols
//   3. Byte-swap all header values (big-endian -> little-endian)
//   4. Verify magic == 0x00000803
//   5. Allocate uint8 buffer, read all pixel data
//   6. Allocate float output, normalize: pixel / 255.0f
//   7. Set output parameters
// =============================================================
void load_mnist_images(const char* path,
                       float** images,
                       int* count,
                       int* rows,
                       int* cols) {
    // TODO: Your code here
    *images = nullptr;
    *count = 0;
    *rows = 0;
    *cols = 0;
}

// =============================================================
// TODO: Load MNIST labels from IDX1-UBYTE binary file
//
// Steps:
//   1. Open file in binary read mode ("rb")
//   2. Read 2 uint32_t values: magic, count
//   3. Byte-swap both header values
//   4. Verify magic == 0x00000801
//   5. Read uint8 label bytes
//   6. Convert to int array
// =============================================================
void load_mnist_labels(const char* path,
                       int** labels,
                       int* count) {
    // TODO: Your code here
    *labels = nullptr;
    *count = 0;
}

// =============================================================
// TODO: Create batches from image data
//
// Divide the images array into contiguous batches of batch_size.
// Each batch[i] should point into the original images array at
// offset i * batch_size * image_size.
//
// Only create full batches (drop the remainder if count is not
// evenly divisible by batch_size).
//
// Allocate the batches array (array of float pointers).
// =============================================================
void create_batches(const float* images,
                    int count,
                    int image_size,
                    int batch_size,
                    float*** batches,
                    int* num_batches) {
    // TODO: Your code here
    *batches = nullptr;
    *num_batches = 0;
}

// =============================================================
// TODO: Shuffle images and labels together using Fisher-Yates
//
// Use std::mt19937 seeded with the given seed.
// Iterate i from count-1 down to 1:
//   Pick random j in [0, i]
//   Swap image[i] with image[j]  (each image is image_size floats)
//   Swap label[i] with label[j]
//
// This ensures images and labels stay paired after shuffling.
// =============================================================
void shuffle_data(float* images,
                  int* labels,
                  int count,
                  int image_size,
                  unsigned int seed) {
    // TODO: Your code here
}
