# LeNet-5 CUDA Puzzles

**Learn CUDA C++ by building a CNN from scratch**

This project teaches GPU programming through 15 progressive puzzles that implement LeNet-5 for MNIST digit classification. You'll learn CUDA fundamentals, neural network mathematics, and backpropagation—all in pure CUDA C++ without external ML libraries.

## Prerequisites

- **NVIDIA GPU**: Compute Capability 7.5+ (Turing or newer)
- **CUDA Toolkit**: 13.1 or later
- **CMake**: 3.18 or later
- **C++17 Compiler**: 
  - Windows: MSVC 2019+ (Visual Studio 2019 or later)
  - Linux: GCC 9+ or Clang 10+

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/cuda-puzzles.git
cd cuda-puzzles

# Configure with native GPU architecture auto-detection
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native

# Build all puzzles
cmake --build build

# Run a single puzzle test (Windows example)
./build/puzzles/puzzle_01_vector_add/Debug/puzzle_01_test_solution.exe

# Run a single puzzle test (Linux example)
./build/puzzles/puzzle_01_vector_add/puzzle_01_test_solution

# Run all tests via CTest
ctest --test-dir build --output-on-failure
```

## Puzzle Index

| # | Puzzle | Concept | Difficulty |
|---|--------|---------|------------|
| 01 | Vector Addition | CUDA basics, thread indexing, grid-stride loop | ⭐ |
| 02 | Matrix Multiplication | GEMM foundation for FC layers, row-major layout | ⭐ |
| 03 | FC Layer Forward | Neural network layer (Y = X·W^T + b) | ⭐⭐ |
| 04 | ReLU Activation | First backward pass, chain rule introduction | ⭐⭐ |
| 05 | Softmax + Cross-Entropy | Numerical stability (max-subtraction trick) | ⭐⭐⭐ |
| 06 | FC Layer Backward | Three gradients (weights, bias, input), numerical gradient checking | ⭐⭐⭐ |
| 07 | Conv2D Forward | Sliding window convolution, NCHW layout | ⭐⭐⭐ |
| 08 | Max Pooling | Gradient routing, saving max indices | ⭐⭐⭐ |
| 09 | Conv2D Backward (Weights) | Weight gradients as correlation | ⭐⭐⭐⭐ |
| 10 | Conv2D Backward (Input) | Input gradients, 180° filter rotation trick | ⭐⭐⭐⭐ |
| 11 | SGD Optimizer | Weight updates (w -= lr*grad), gradient zeroing | ⭐⭐ |
| 12 | LeNet Forward Pass | Full network integration (12 layer orchestration) | ⭐⭐⭐⭐ |
| 13 | LeNet Backward Pass | Complete backpropagation through entire network | ⭐⭐⭐⭐⭐ |
| 14 | MNIST Data Pipeline | IDX format parsing, big-endian byte swap, bundled mini-dataset (100 images) | ⭐⭐ |
| 15 | Full Training Loop | Xavier init, epoch training, overfit tests | ⭐⭐⭐⭐⭐ |

Each puzzle has:
- `README.md`: Problem statement, learning objectives, and hints
- `puzzle.cu`: Starter code with TODO comments
- `solution.cu`: Reference implementation
- `test_puzzle.cu`: Automated test suite

## LeNet-5 Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    LeNet-5 Architecture for MNIST                       ║
║                                                                          ║
║  Input     Conv1      Pool1     Conv2      Pool2    Flatten  FC1  FC2  Out║
║  28×28×1 → 24×24×6 → 12×12×6 → 8×8×16 → 4×4×16 → 256 → 120 → 84 → 10 ║
║                                                                          ║
║  ┌───────┐  ┌───────┐  ┌─────┐  ┌─────┐  ┌───┐  ┌───┐ ┌──┐ ┌──┐ ┌──┐ ║
║  │28 × 28│→ │24 × 24│→ │12×12│→ │8 × 8│→ │4×4│→ │256│→│120│→│84│→│10│ ║
║  │  × 1  │  │  × 6  │  │ × 6 │  │ ×16 │  │×16│  └───┘ └──┘ └──┘ └──┘ ║
║  └───────┘  └───────┘  └─────┘  └─────┘  └───┘                         ║
║       5×5 conv    2×2 pool  5×5 conv  2×2 pool  flatten  FC    FC   FC  ║
║       6 filters   stride 2  16 filters stride 2         +ReLU +ReLU soft║
║                                                                          ║
║  Parameters: ~44,426 total                                               ║
║  Conv1: 6×(5×5×1+1) = 156    FC1: 256×120+120 = 30,840                 ║
║  Conv2: 16×(5×5×6+1)= 2,416  FC2: 120×84+84   = 10,164                ║
║                                FC3: 84×10+10    = 850                    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Layer Details:**
- **Conv1**: 5×5 kernel, 6 output channels, stride=1, padding=0 (28→24)
- **Pool1**: 2×2 max pooling, stride=2 (24→12)
- **Conv2**: 5×5 kernel, 16 output channels, stride=1, padding=0 (12→8)
- **Pool2**: 2×2 max pooling, stride=2 (8→4)
- **FC1**: 256 → 120 with ReLU activation
- **FC2**: 120 → 84 with ReLU activation
- **FC3**: 84 → 10 with softmax activation

## MNIST Dataset

Tests use a bundled 100-image mini-dataset in `data/`. For full training (60,000 images):

**Option 1: Official MNIST**
```bash
cd data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

**Option 2: Amazon S3 Mirror (faster)**
```bash
cd data
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

**On Windows (without wget):** Download files manually from the URLs above and extract with 7-Zip or WinRAR.

## Running Tests

**Individual puzzle:**
```bash
# Build specific test target
cmake --build build --target puzzle_07_test_solution

# Run the test executable (Windows)
./build/puzzles/puzzle_07_conv_forward/Debug/puzzle_07_test_solution.exe

# Run the test executable (Linux)
./build/puzzles/puzzle_07_conv_forward/puzzle_07_test_solution
```

**All tests via CTest:**
```bash
ctest --test-dir build --output-on-failure
```

**Expected output:**
```
Test project C:/repo/cuda-puzzles/build
    Start  1: test_common
1/16 Test  #1: test_common .......................   Passed    0.15 sec
    Start  2: puzzle_01_test_solution
2/16 Test  #2: puzzle_01_test_solution ...........   Passed    0.12 sec
    Start  3: puzzle_02_test_solution
3/16 Test  #3: puzzle_02_test_solution ...........   Passed    0.11 sec
    Start  4: puzzle_03_test_solution
4/16 Test  #4: puzzle_03_test_solution ...........   Passed    0.13 sec
    Start  5: puzzle_04_test_solution
5/16 Test  #5: puzzle_04_test_solution ...........   Passed    0.12 sec
    Start  6: puzzle_05_test_solution
6/16 Test  #6: puzzle_05_test_solution ...........   Passed    0.14 sec
    Start  7: puzzle_06_test_solution
7/16 Test  #7: puzzle_06_test_solution ...........   Passed    0.15 sec
    Start  8: puzzle_07_test_solution
8/16 Test  #8: puzzle_07_test_solution ...........   Passed    0.16 sec
    Start  9: puzzle_08_test_solution
9/16 Test  #9: puzzle_08_test_solution ...........   Passed    0.13 sec
    Start 10: puzzle_09_test_solution
10/16 Test #10: puzzle_09_test_solution ...........   Passed    0.18 sec
    Start 11: puzzle_10_test_solution
11/16 Test #11: puzzle_10_test_solution ...........   Passed    0.17 sec
    Start 12: puzzle_11_test_solution
12/16 Test #12: puzzle_11_test_solution ...........   Passed    0.11 sec
    Start 13: puzzle_12_test_solution
13/16 Test #13: puzzle_12_test_solution ...........   Passed    0.19 sec
    Start 14: puzzle_13_test_solution
14/16 Test #14: puzzle_13_test_solution ...........   Passed    0.21 sec
    Start 15: puzzle_14_test_solution
15/16 Test #15: puzzle_14_test_solution ...........   Passed    0.14 sec
    Start 16: puzzle_15_test_solution
16/16 Test #16: puzzle_15_test_solution ...........   Passed    2.45 sec

100% tests passed, 0 tests failed out of 16

Total Test time (real) =   5.46 sec
```

## Project Structure

```
cuda-puzzles/
├── CMakeLists.txt          # Root build configuration
├── README.md               # This file
├── .gitignore
├── common/                 # Shared utilities and headers
│   ├── CMakeLists.txt
│   ├── cuda_utils.h        # CUDA error checking macros
│   ├── test_utils.h        # Test assertion utilities
│   ├── mnist_loader.h      # MNIST IDX file parser
│   ├── test_data.h         # Deterministic test data generation
│   ├── common.h            # Master include header
│   └── test_common.cu      # Test suite for common utilities
├── data/                   # MNIST dataset (bundled mini-set + optional full dataset)
│   ├── train-images-idx3-ubyte  (download from MNIST mirrors)
│   ├── train-labels-idx1-ubyte  (download from MNIST mirrors)
│   ├── t10k-images-idx3-ubyte   (download from MNIST mirrors)
│   └── t10k-labels-idx1-ubyte   (download from MNIST mirrors)
└── puzzles/                # Individual puzzle implementations
    ├── CMakeLists.txt      # Includes all puzzle subdirectories
    ├── puzzle_01_vector_add/
    │   ├── CMakeLists.txt
    │   ├── README.md       # Problem statement and hints
    │   ├── puzzle.cu       # Starter code with TODOs
    │   ├── solution.cu     # Reference implementation
    │   └── test_puzzle.cu  # Automated test suite
    ├── puzzle_02_matmul/
    │   └── ... (same structure)
    ... (puzzles 03-15 follow same pattern)
    └── puzzle_15_training_loop/
        └── ...
```

## Learning Path

**Recommended Order:**
1. **Puzzles 1-2**: CUDA basics (threads, blocks, memory transfer)
2. **Puzzles 3-6**: Fully connected layers (forward + backward pass)
3. **Puzzles 7-10**: Convolutional layers (forward + backward pass)
4. **Puzzle 11**: Optimizer (SGD with gradient descent)
5. **Puzzles 12-13**: Full network integration (forward + backward)
6. **Puzzle 14**: Data loading (MNIST IDX format)
7. **Puzzle 15**: Training loop (Xavier init, epochs, validation)

**Key Concepts Covered:**
- CUDA thread hierarchy (threads, blocks, grids)
- Memory management (host ↔ device transfers, cudaMalloc/cudaFree)
- Kernel launch configurations (grid stride loops)
- Neural network mathematics (matrix multiplication, convolution, activation functions)
- Backpropagation (chain rule, gradient computation)
- Numerical stability (max-subtraction trick for softmax)
- Weight initialization (Xavier/Glorot initialization)
- Training dynamics (learning rate, epochs, overfitting)

## Implementation Details

**Design Philosophy:**
- **Educational, not optimized**: Naive implementations for clarity (no shared memory tiling, no cuDNN)
- **Pure CUDA C++**: No external ML libraries (PyTorch, TensorFlow, etc.)
- **Direct nested-loop convolution**: No im2col transformation (easier to understand)
- **One-thread-per-output parallelism**: Simple kernel patterns
- **Deterministic testing**: Fixed random seeds for reproducibility

**Memory Layout:**
- **NCHW format**: Batch × Channels × Height × Width
- **Row-major ordering**: Standard C/C++ array layout
- **Contiguous storage**: All tensors are flat arrays

**Testing Framework:**
- **Automated tests**: Every puzzle has comprehensive test coverage
- **Numerical gradient checking**: Validates backward pass implementations
- **Tolerance levels**: 1e-4 for forward pass, 1e-3 for backward pass
- **Exit codes**: 0 if all pass, 1 if any fail

## Troubleshooting

**CMake cannot find CUDA:**
```bash
# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH  # Linux
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH%  # Windows

# Specify CUDA toolkit manually
cmake -B build -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

**Compute capability errors:**
```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Set manually if auto-detection fails
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75  # For Turing (RTX 20-series)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86  # For Ampere (RTX 30-series)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89  # For Ada Lovelace (RTX 40-series)
```

**Tests fail with "out of memory":**
- Your GPU may have insufficient memory (< 4GB)
- Reduce batch size in test files (edit test_puzzle.cu)
- Tests are designed for 6GB+ GPUs

**Build errors on Windows:**
- Ensure Visual Studio 2019+ is installed
- Use "x64 Native Tools Command Prompt for VS 2019"
- CMake may require `-G "Visual Studio 16 2019" -A x64`

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **LeNet-5 Architecture**: Yann LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998)
- **CUDA Programming**: NVIDIA CUDA C++ Programming Guide

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-puzzle`)
3. Commit your changes (`git commit -m "Add new puzzle"`)
4. Push to the branch (`git push origin feature/new-puzzle`)
5. Open a pull request

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Happy CUDA Puzzle Solving! 🚀🧩**
