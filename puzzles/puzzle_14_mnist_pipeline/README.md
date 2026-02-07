# Puzzle 14: MNIST Data Pipeline

## Overview

Build a complete data pipeline for loading, preprocessing, and batching
MNIST handwritten digit images. This puzzle focuses on **CPU-side data
engineering**: parsing binary file formats, byte-order conversion,
normalization, batching, and deterministic shuffling.

These are the exact operations every deep learning framework performs
behind the scenes before data ever reaches a GPU kernel.

## MNIST IDX File Format

The MNIST dataset uses the IDX binary format, a simple format where
all multi-byte integers are stored in **big-endian** byte order.

### IDX3-UBYTE (Images)

```
Offset  Type            Description
------  ----            -----------
0000    uint32 (BE)     Magic number: 0x00000803
0004    uint32 (BE)     Number of images
0008    uint32 (BE)     Number of rows (28)
0012    uint32 (BE)     Number of columns (28)
0016    uint8[N×R×C]    Pixel data (row-major, 0-255)
```

Each image is 28×28 = 784 bytes of unsigned 8-bit pixel values.
Pixels are stored row-major: all columns of row 0, then row 1, etc.

### IDX1-UBYTE (Labels)

```
Offset  Type            Description
------  ----            -----------
0000    uint32 (BE)     Magic number: 0x00000801
0004    uint32 (BE)     Number of labels
0008    uint8[N]        Label data (0-9)
```

### Big-Endian Byte Swap

x86/x64 CPUs are **little-endian**, but IDX files store integers in
**big-endian** format. You must swap bytes for all header integers:

```
uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0x000000ff) |
           ((val >>  8) & 0x0000ff00) |
           ((val <<  8) & 0x00ff0000) |
           ((val << 24) & 0xff000000);
}
```

Example: The magic number `0x00000803` is stored on disk as bytes
`[0x00, 0x00, 0x08, 0x03]`. A little-endian CPU reads this as
`0x03080000`, so byte-swapping produces `0x00000803`.

## Normalization

Raw pixel values are `uint8` in `[0, 255]`. Neural networks expect
floating-point values in `[0.0, 1.0]`:

```
float normalized = static_cast<float>(pixel) / 255.0f;
```

## Batching

Training processes images in **batches** (groups of B images). Given
N total images and batch size B:

```
num_full_batches = N / B
remainder = N % B  (last batch may be smaller, or dropped)
```

Each batch is a contiguous block of `B × 28 × 28` float values.

## Shuffling

To prevent the network from learning order-dependent patterns,
shuffle the dataset before each epoch. Use a **deterministic**
shuffle with a fixed seed for reproducibility:

```
// Fisher-Yates shuffle with seeded PRNG
void shuffle(indices, count, seed) {
    mt19937 rng(seed);
    for (i = count-1; i > 0; i--) {
        j = rng() % (i + 1);
        swap(indices[i], indices[j]);
    }
}
```

## Functions to Implement

```cpp
// Load images from IDX3-UBYTE file, normalize to [0,1]
// Returns: float array of size count × 28 × 28
void load_mnist_images(const char* path,
                       float** images,      // output: allocated array
                       int* count,          // output: number of images
                       int* rows,           // output: image height (28)
                       int* cols);          // output: image width (28)

// Load labels from IDX1-UBYTE file
// Returns: int array of size count
void load_mnist_labels(const char* path,
                       int** labels,        // output: allocated array
                       int* count);         // output: number of labels

// Split data into batches of size batch_size
// Returns: array of batch pointers and batch count
void create_batches(const float* images,
                    int count,
                    int image_size,         // 28*28 = 784
                    int batch_size,
                    float*** batches,       // output: array of batch pointers
                    int* num_batches);      // output: number of batches

// Deterministic shuffle using Fisher-Yates with given seed
void shuffle_data(float* images,
                  int* labels,
                  int count,
                  int image_size,           // 28*28 = 784
                  unsigned int seed);
```

## Bundled Mini-Dataset

The `data/` directory contains a 100-image subset for testing:

```
data/mini-images-idx3-ubyte    (78,416 bytes = 16 + 100×28×28)
data/mini-labels-idx1-ubyte    (108 bytes = 8 + 100)
```

All tests use this bundled subset — no download required.

## Downloading Full MNIST

For training with the full 60,000-image dataset, download from any mirror:

**Mirror 1 — Yann LeCun's website (original):**
```bash
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

**Mirror 2 — GitHub (ossian-mnist):**
```bash
wget https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-images-idx3-ubyte.gz
wget https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-labels-idx1-ubyte.gz
gunzip *.gz
```

**Mirror 3 — Hugging Face:**
```bash
# Via the datasets library:
pip install datasets
python -c "from datasets import load_dataset; load_dataset('mnist')"
```

After downloading, extract and place files in the `data/` directory.

## Step-by-Step Guide

1. **Implement `swap_endian`** — bit-shift and mask to reverse byte order
2. **Implement `load_mnist_images`** — open file, read/swap header, verify
   magic (0x00000803), read pixels, normalize to `[0.0, 1.0]`
3. **Implement `load_mnist_labels`** — open file, read/swap header, verify
   magic (0x00000801), read label bytes, convert to `int`
4. **Implement `create_batches`** — divide image array into contiguous chunks
5. **Implement `shuffle_data`** — Fisher-Yates with `std::mt19937` seeded RNG

## Hints

**Hint 1 (Mild):** Use `fopen` with `"rb"` mode and `fread` for binary I/O.
The entire pixel buffer can be read in one `fread` call.

**Hint 2 (Medium):** For `create_batches`, each batch pointer should point
into the original images array — no copying needed. Just compute offsets:
`batches[i] = images + i * batch_size * image_size`.

**Hint 3 (Strong):** The Fisher-Yates shuffle swaps elements from back to
front. For images, each "element" is `image_size` floats, so you need to
swap `image_size` floats at a time (use `std::swap_ranges` or a temp buffer).
