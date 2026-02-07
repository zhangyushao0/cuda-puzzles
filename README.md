# CUDA Puzzles

LeNet-5 implementation in pure CUDA C++.

## Building

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
```

## Project Structure

```
cuda-puzzles/
├── common/          # Shared utilities and headers
├── data/            # MNIST dataset (generated at runtime)
└── puzzles/         # Individual puzzle implementations
```
