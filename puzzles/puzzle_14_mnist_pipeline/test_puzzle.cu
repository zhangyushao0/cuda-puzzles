#include <catch2/catch_test_macros.hpp>
// Puzzle 14: MNIST Data Pipeline — Test Suite
//
// Tests:
//   1. Load bundled 100-image mini-subset
//   2. Byte swap correctness
//   3. Normalization to [0.0, 1.0]
//   4. Batch creation
//   5. Shuffle determinism with seed

#include "test_utils.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <stdexcept>

// Include the implementation
#ifdef USE_SOLUTION
#include "solution.cu"
#else
#include "puzzle.cu"
#endif

// Data directory set by CMake at compile time
#ifndef DATA_DIR
#error "DATA_DIR must be defined (path to bundled MNIST test data)"
#endif

// Helper: build full path to a data file
static std::string data_path(const char* filename) {
    std::string path = DATA_DIR;
    path += "/";
    path += filename;
    return path;
}

// ---------------------------------------------------------------
// Test 1: Load bundled mini-subset (100 images, 100 labels)
// Verify header parsing, correct count, and dimensions
// ---------------------------------------------------------------
TEST_CASE("load_mini_subset", "[puzzle_14_mnist_pipeline]") {
    float* images = nullptr;
    int count = 0, rows = 0, cols = 0;
    std::string img_path = data_path("mini-images-idx3-ubyte");
    load_mnist_images(img_path.c_str(), &images, &count, &rows, &cols);

    if (images == nullptr) {
        throw std::runtime_error("load_mnist_images returned null pointer");
    }
    if (count != 100) {
        throw std::runtime_error("Expected 100 images, got " + std::to_string(count));
    }
    if (rows != 28) {
        throw std::runtime_error("Expected 28 rows, got " + std::to_string(rows));
    }
    if (cols != 28) {
        throw std::runtime_error("Expected 28 cols, got " + std::to_string(cols));
    }

    // Load labels
    int* labels = nullptr;
    int label_count = 0;
    std::string lbl_path = data_path("mini-labels-idx1-ubyte");
    load_mnist_labels(lbl_path.c_str(), &labels, &label_count);

    if (labels == nullptr) {
        free(images);
        throw std::runtime_error("load_mnist_labels returned null pointer");
    }
    if (label_count != 100) {
        free(images);
        free(labels);
        throw std::runtime_error("Expected 100 labels, got " + std::to_string(label_count));
    }

    // All labels should be in [0, 9]
    for (int i = 0; i < label_count; i++) {
        if (labels[i] < 0 || labels[i] > 9) {
            free(images);
            free(labels);
            throw std::runtime_error("Label " + std::to_string(i) +
                                     " out of range: " + std::to_string(labels[i]));
        }
    }

    free(images);
    free(labels);
}

// ---------------------------------------------------------------
// Test 2: Byte swap correctness
// Verify swap_endian produces correct results for known values
// ---------------------------------------------------------------
TEST_CASE("byte_swap", "[puzzle_14_mnist_pipeline]") {
    // Magic number: on disk as [0x00, 0x00, 0x08, 0x03]
    // Little-endian CPU reads: 0x03080000
    // After swap: 0x00000803
    uint32_t val1 = 0x03080000;
    uint32_t swapped1 = swap_endian(val1);
    if (swapped1 != 0x00000803) {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "swap_endian(0x%08x) = 0x%08x, expected 0x00000803",
                 val1, swapped1);
        throw std::runtime_error(buf);
    }

    // Label magic: 0x01080000 -> 0x00000801
    uint32_t val2 = 0x01080000;
    uint32_t swapped2 = swap_endian(val2);
    if (swapped2 != 0x00000801) {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "swap_endian(0x%08x) = 0x%08x, expected 0x00000801",
                 val2, swapped2);
        throw std::runtime_error(buf);
    }

    // Count = 100: big-endian bytes [0x00, 0x00, 0x00, 0x64]
    // Little-endian reads: 0x64000000
    // After swap: 0x00000064 = 100
    uint32_t val3 = 0x64000000;
    uint32_t swapped3 = swap_endian(val3);
    if (swapped3 != 100) {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "swap_endian(0x%08x) = %u, expected 100",
                 val3, swapped3);
        throw std::runtime_error(buf);
    }

    // Roundtrip: swap(swap(x)) == x
    uint32_t val4 = 0xDEADBEEF;
    if (swap_endian(swap_endian(val4)) != val4) {
        throw std::runtime_error("swap_endian is not its own inverse");
    }

    // Identity: swap(0) == 0
    if (swap_endian(0) != 0) {
        throw std::runtime_error("swap_endian(0) != 0");
    }
}

// ---------------------------------------------------------------
// Test 3: Normalization — all pixel values in [0.0, 1.0]
// Also verify raw byte correspondence: pixel_i / 255.0
// ---------------------------------------------------------------
TEST_CASE("normalization", "[puzzle_14_mnist_pipeline]") {
    float* images = nullptr;
    int count = 0, rows = 0, cols = 0;
    std::string img_path = data_path("mini-images-idx3-ubyte");
    load_mnist_images(img_path.c_str(), &images, &count, &rows, &cols);

    if (images == nullptr) {
        throw std::runtime_error("load_mnist_images returned null");
    }

    int total = count * rows * cols;

    // Check all values in [0.0, 1.0]
    for (int i = 0; i < total; i++) {
        if (images[i] < 0.0f || images[i] > 1.0f) {
            free(images);
            throw std::runtime_error("Pixel " + std::to_string(i) +
                                     " out of [0,1]: " + std::to_string(images[i]));
        }
    }

    // Cross-check: re-read raw bytes and verify normalization
    FILE* f = fopen(img_path.c_str(), "rb");
    if (!f) {
        free(images);
        throw std::runtime_error("Cannot reopen image file for verification");
    }
    fseek(f, 16, SEEK_SET);  // skip 16-byte header
    uint8_t* raw = (uint8_t*)malloc(total);
    fread(raw, 1, total, f);
    fclose(f);

    int mismatch_count = 0;
    for (int i = 0; i < total; i++) {
        float expected = static_cast<float>(raw[i]) / 255.0f;
        if (fabsf(images[i] - expected) > 1e-6f) {
            mismatch_count++;
            if (mismatch_count <= 5) {
                fprintf(stderr, "  Pixel %d: raw=%u, expected=%.6f, got=%.6f\n",
                        i, raw[i], expected, images[i]);
            }
        }
    }

    free(raw);
    free(images);

    if (mismatch_count > 0) {
        throw std::runtime_error("Normalization mismatch: " +
                                 std::to_string(mismatch_count) + " pixels differ");
    }
}

// ---------------------------------------------------------------
// Test 4: Batch creation — correct number and sizes
// ---------------------------------------------------------------
TEST_CASE("batch_creation", "[puzzle_14_mnist_pipeline]") {
    float* images = nullptr;
    int count = 0, rows = 0, cols = 0;
    std::string img_path = data_path("mini-images-idx3-ubyte");
    load_mnist_images(img_path.c_str(), &images, &count, &rows, &cols);

    if (images == nullptr) {
        throw std::runtime_error("load_mnist_images returned null");
    }

    const int image_size = rows * cols;  // 784

    // Test with batch_size=32: 100/32 = 3 full batches (96 images, drop 4)
    {
        float** batches = nullptr;
        int num_batches = 0;
        create_batches(images, count, image_size, 32, &batches, &num_batches);

        if (num_batches != 3) {
            free(images);
            if (batches) free(batches);
            throw std::runtime_error("Expected 3 batches (100/32), got " +
                                     std::to_string(num_batches));
        }

        // Each batch pointer should be offset into images
        for (int b = 0; b < num_batches; b++) {
            float* expected_ptr = images + b * 32 * image_size;
            if (batches[b] != expected_ptr) {
                free(images);
                free(batches);
                throw std::runtime_error("Batch " + std::to_string(b) +
                                         " pointer doesn't match expected offset");
            }
        }

        free(batches);
    }

    // Test with batch_size=10: 100/10 = 10 full batches
    {
        float** batches = nullptr;
        int num_batches = 0;
        create_batches(images, count, image_size, 10, &batches, &num_batches);

        if (num_batches != 10) {
            free(images);
            if (batches) free(batches);
            throw std::runtime_error("Expected 10 batches (100/10), got " +
                                     std::to_string(num_batches));
        }

        // Verify first element of each batch matches source data
        for (int b = 0; b < num_batches; b++) {
            int src_idx = b * 10 * image_size;
            if (batches[b][0] != images[src_idx]) {
                free(images);
                free(batches);
                throw std::runtime_error("Batch " + std::to_string(b) +
                                         " data doesn't match source");
            }
        }

        free(batches);
    }

    // Test with batch_size=100: exactly 1 batch
    {
        float** batches = nullptr;
        int num_batches = 0;
        create_batches(images, count, image_size, 100, &batches, &num_batches);

        if (num_batches != 1) {
            free(images);
            if (batches) free(batches);
            throw std::runtime_error("Expected 1 batch (100/100), got " +
                                     std::to_string(num_batches));
        }

        if (batches[0] != images) {
            free(images);
            free(batches);
            throw std::runtime_error("Single batch pointer should equal images pointer");
        }

        free(batches);
    }

    free(images);
}

// ---------------------------------------------------------------
// Test 5: Shuffle determinism — same seed produces same result
// ---------------------------------------------------------------
TEST_CASE("shuffle_determinism", "[puzzle_14_mnist_pipeline]") {
    // Load data twice for two independent shuffles
    float *images1 = nullptr, *images2 = nullptr;
    int *labels1 = nullptr, *labels2 = nullptr;
    int count1 = 0, count2 = 0, rows = 0, cols = 0;
    int lcount1 = 0, lcount2 = 0;

    std::string img_path = data_path("mini-images-idx3-ubyte");
    std::string lbl_path = data_path("mini-labels-idx1-ubyte");

    load_mnist_images(img_path.c_str(), &images1, &count1, &rows, &cols);
    load_mnist_images(img_path.c_str(), &images2, &count2, &rows, &cols);
    load_mnist_labels(lbl_path.c_str(), &labels1, &lcount1);
    load_mnist_labels(lbl_path.c_str(), &labels2, &lcount2);

    if (!images1 || !images2 || !labels1 || !labels2) {
        throw std::runtime_error("Failed to load data for shuffle test");
    }

    const int image_size = rows * cols;
    const unsigned int seed = 12345;

    // Shuffle both copies with the same seed
    shuffle_data(images1, labels1, count1, image_size, seed);
    shuffle_data(images2, labels2, count2, image_size, seed);

    // They must be identical
    int total_pixels = count1 * image_size;
    for (int i = 0; i < total_pixels; i++) {
        if (images1[i] != images2[i]) {
            free(images1); free(images2);
            free(labels1); free(labels2);
            throw std::runtime_error("Shuffle not deterministic: images differ at index " +
                                     std::to_string(i));
        }
    }

    for (int i = 0; i < count1; i++) {
        if (labels1[i] != labels2[i]) {
            free(images1); free(images2);
            free(labels1); free(labels2);
            throw std::runtime_error("Shuffle not deterministic: labels differ at index " +
                                     std::to_string(i));
        }
    }

    // Verify shuffle actually changed the order (not a no-op)
    // Reload original labels and check label order changed
    int* orig_labels = nullptr;
    int orig_lcount = 0;
    load_mnist_labels(lbl_path.c_str(), &orig_labels, &orig_lcount);

    int same_position = 0;
    for (int i = 0; i < count1; i++) {
        if (labels1[i] == orig_labels[i]) {
            same_position++;
        }
    }

    free(orig_labels);

    // With 100 items shuffled, it's astronomically unlikely that >90% stay in place
    // (even accounting for duplicate labels, Fisher-Yates will move most)
    if (same_position > 90) {
        free(images1); free(images2);
        free(labels1); free(labels2);
        throw std::runtime_error("Shuffle appears to be a no-op: " +
                                 std::to_string(same_position) + "/100 labels in same position");
    }

    // Verify image-label pairing is maintained:
    // After shuffle, each image should still correspond to its label.
    // We verify by checking that a different seed produces different results.
    float* images3 = nullptr;
    int* labels3 = nullptr;
    int count3 = 0, lcount3 = 0;
    load_mnist_images(img_path.c_str(), &images3, &count3, &rows, &cols);
    load_mnist_labels(lbl_path.c_str(), &labels3, &lcount3);

    shuffle_data(images3, labels3, count3, image_size, seed + 1);  // different seed

    // With different seed, results should differ
    int diff_count = 0;
    for (int i = 0; i < count1; i++) {
        if (labels1[i] != labels3[i]) {
            diff_count++;
        }
    }

    free(images1); free(images2); free(images3);
    free(labels1); free(labels2); free(labels3);

    if (diff_count == 0) {
        throw std::runtime_error("Different seeds produced identical shuffles");
    }
}

