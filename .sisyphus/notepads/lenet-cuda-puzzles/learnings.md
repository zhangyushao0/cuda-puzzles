# Learnings and Conventions

## Project Conventions
- Pure CUDA C++ only (no cuDNN, no PyTorch, no external ML libraries)
- CMake build system (MSVC + nvcc on Windows)
- Minimum sm_75 architecture (CUDA 13.1 dropped older)
- Deterministic seeds: seed=42 for weights, seed=123 for data
- Tolerance: 1e-4 for forward pass, 1e-3 for backward pass
- NCHW memory layout for all tensors

## Code Style
- CUDA_CHECK() wraps every CUDA API call
- KERNEL_CHECK() after every kernel launch
- ASCII diagrams in every README (80 columns max)
- Math formulas (forward AND backward) in every puzzle

## Test Pattern
- Every puzzle: README.md, puzzle.cu, solution.cu, test_puzzle.cu, CMakeLists.txt
- Tests output: "[PASS] test_name" or "[FAIL] test_name: details"
- Exit code 0 if all pass, 1 if any fail
- Final line: "X/Y tests passed"

