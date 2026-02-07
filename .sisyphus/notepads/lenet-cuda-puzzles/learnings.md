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


## Project Infrastructure Setup

### CMake Configuration
- Root CMakeLists.txt:
  - `cmake_minimum_required(VERSION 3.18)` for CUDA support
  - `project(cuda-puzzles LANGUAGES CXX CUDA)` declares CUDA language
  - `CMAKE_CXX_STANDARD 17` and `CMAKE_CUDA_STANDARD 17`
  - `CMAKE_CUDA_ARCHITECTURES` defaults to `native` if not specified
  - Includes comment about sm_75 minimum (CUDA 13.x baseline)
  - Calls `add_subdirectory(common)` and `add_subdirectory(puzzles)`

- common/CMakeLists.txt:
  - Creates INTERFACE library target `cuda_puzzles_common`
  - Sets include directory to `${CMAKE_CURRENT_SOURCE_DIR}`
  - Allows other modules to link against it for header access

- puzzles/CMakeLists.txt:
  - Currently empty placeholder
  - Will add puzzle subdirectories as features are implemented

### Directory Structure
```
cuda-puzzles/
├── CMakeLists.txt         (root build config)
├── README.md              (project overview)
├── .gitignore             (build artifacts, MNIST data)
├── common/
│   └── CMakeLists.txt     (shared headers library)
├── data/                  (empty, for MNIST dataset)
└── puzzles/
    └── CMakeLists.txt     (puzzle subdirectories will go here)
```

### Build System Notes
- Tested with: `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native`
- Output indicates CUDA 13.1.115 with MSVC 19.44
- CMake configuration completes successfully
- Generated Visual Studio 2022 solution files
- CUDA architectures set to `native` for auto-detection

### Git Setup
- Initialized empty repository
- Initial commit: `feat(infra): initialize project with CMake build system`
- Git configured with user: "CUDA Puzzles" <cuda-puzzles@example.com>
- .gitignore includes: build/, *.o, *.exe, *.exp, *.lib, .vs/, MNIST data files

## Common Utilities Infrastructure (Task 2)

### Test Framework Pattern
- Created lightweight test framework with TEST_CASE macro and RUN_ALL_TESTS()
- Macro hygiene: use test_name instead of name to avoid conflicts with struct members
- Test registration uses static constructor pattern for automatic registration
- Output format: "[PASS]" / "[FAIL]" with detailed error messages

### CUDA Error Checking Pattern
- CUDA_CHECK(err): Wraps API calls, prints file:line on error, exits
- KERNEL_CHECK(): Calls cudaGetLastError() + cudaDeviceSynchronize()
- Both use do-while(0) pattern for safe multi-statement macros

### Tolerance Values for Testing
- Default: atol=1e-4, rtol=1e-4 for forward pass
- Formula: |a - b| <= atol + rtol * max(|a|, |b|)
- Use 1e-3 for backward pass (more numerical error)

### Byte Swapping for MNIST
- IDX format is big-endian, Windows is little-endian
- swap_endian: ((val>>24)&0xff) | ((val>>8)&0xff00) | ((val<<8)&0xff0000) | ((val<<24)&0xff000000)
- Magic numbers: 0x00000803 (images), 0x00000801 (labels)

### Test Data Generation
- Use std::mt19937 for deterministic random generation
- Same seed produces identical output across runs
- Seeds: 42 for weights, 123 for test data

## Puzzle 01: Vector Addition (Task 3)

### Puzzle Structure Template (5 files)
Every puzzle follows this exact structure:
1. `README.md` — Concept overview, ASCII diagrams, formula, step-by-step guide, 3-level hints
2. `puzzle.cu` — Template with `// TODO: Your code here` markers, includes `common.h`
3. `solution.cu` — Complete working implementation with CUDA_CHECK/KERNEL_CHECK
4. `test_puzzle.cu` — Test suite using TEST_CASE macro, check_array_close, RUN_ALL_TESTS()
5. `CMakeLists.txt` — Two targets: `puzzle_XX_test` (puzzle.cu) and `puzzle_XX_test_solution` (solution.cu)

### Grid Stride Loop Pattern
```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = tid; i < N; i += stride) {
    C[i] = A[i] + B[i];
}
```
- Handles arbitrary array sizes (not just multiples of block size)
- Each thread processes multiple elements when N > total threads
- This is the standard pattern for element-wise operations

### Test Pattern Details
- Forward-declare kernel in test_puzzle.cu (defined in puzzle.cu or solution.cu)
- Include `test_utils.h` and `cuda_utils.h` separately (not `common.h`) to avoid unnecessary deps
- Test body: allocate host → fill data → cudaMalloc → cudaMemcpy H2D → launch kernel → cudaMemcpy D2H → check_array_close → cleanup
- Throw `std::runtime_error` on failure (framework catches exceptions)
- Use different seeds per array (e.g., seed=42 for A, seed=43 for B) to avoid identical inputs
- Clean up memory even on failure paths before throwing

### CMakeLists.txt Pattern
```cmake
add_executable(puzzle_XX_test test_puzzle.cu puzzle.cu)
target_link_libraries(puzzle_XX_test PRIVATE cuda_puzzles_common)
add_executable(puzzle_XX_test_solution test_puzzle.cu solution.cu)
target_link_libraries(puzzle_XX_test_solution PRIVATE cuda_puzzles_common)
```

### Build Output Location (Windows/MSVC)
- Executables go to: `build/puzzles/puzzle_XX_*/Debug/puzzle_XX_test_solution.exe`
- Must use full path when running: `"C:\repo\cuda-puzzles\build\puzzles\...\Debug\*.exe"`

## Puzzle 02: Matrix Multiplication (Task 4)

### Matmul-to-FC-Layer Connection
- Every fully connected layer is matmul + bias: `Y = X * W^T + b`
- LeNet FC dimensions map directly to matmul: 256x120, 120x84, 84x10
- Understanding naive matmul is prerequisite for Puzzles 03 (FC forward) and 06 (FC backward)

### 2D Grid Pattern for Matrix Operations
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
dim3 blockDim(16, 16);  // 256 threads per block
dim3 gridDim((N + 15) / 16, (M + 15) / 16);
```
- gridDim.x covers columns (N), gridDim.y covers rows (M)
- Bounds check is critical: `if (row < M && col < N)` since grid may exceed matrix

### Row-Major Indexing
- Element [i][j] of an MxN matrix is at index `i * N + j`
- A[i][k] → A[i * K + k]
- B[k][j] → B[k * N + j]
- C[i][j] → C[i * N + j]

### CMake Pattern: #ifdef-based kernel selection
- Used `#ifdef USE_SOLUTION` / `#include "solution.cu"` / `#else` / `#include "puzzle.cu"` in test_puzzle.cu
- CMake adds `target_compile_definitions(... PRIVATE USE_SOLUTION)` for solution target
- Alternative to Puzzle 01's approach of separate .cu files as CMake sources
- Both patterns work; #ifdef is simpler for single-kernel puzzles

### Tolerance for Large Matrix Multiply
- Use atol=1e-3, rtol=1e-3 for larger matrices (64x128, 256x120)
- Accumulated floating-point error in dot products scales with K (inner dimension)
- Default 1e-4 may fail for K=128+ due to FP rounding


## Puzzle 03: Fully Connected Layer Forward Pass (Task 5)

### FC Layer = Matmul + Bias Broadcast
- Core insight: FC forward is just `Y = X · W^T + b`
- Matmul part is identical to Puzzle 02's inner loop
- Only addition is `+ bias[j]` after the dot product
- Weight matrix stored as (out_features × in_features), NOT transposed in memory
- Kernel indexes `weights[j * in_features + i]` which effectively does the transpose

### Thread Mapping (Same as Matmul)
```cuda
int j = blockIdx.x * blockDim.x + threadIdx.x;  // output feature
int b = blockIdx.y * blockDim.y + threadIdx.y;   // batch index
```
- x-dimension maps to output features (columns)
- y-dimension maps to batch samples (rows)
- Each thread computes one output element: output[b][j]

### LeNet FC Layer Dimensions
- FC1: batch×256 → batch×120 (256 inputs from flattened 4×4×16 feature maps)
- FC2: batch×120 → batch×84
- FC3: batch×84 → batch×10 (10 output classes for digits)

### Test Data Strategy
- Use small random values (-0.1 to 0.1) for inputs/weights in large tests
- Keeps accumulated sums reasonable, avoids large FP errors
- Bias can use wider range (-0.5 to 0.5)
- Tolerance: 1e-3 for larger dimensions (same as Puzzle 02)
- Hardcoded test (4→3) uses exact values for hand-verification

### No Activation Here
- FC forward is JUST the linear transform (no ReLU)
- ReLU comes separately in Puzzle 04
- This separation keeps each puzzle focused on one concept

## Puzzle 04: ReLU Forward + Backward (Task 6)

### First Backward Pass Pattern
- This is the first puzzle with TWO kernels: forward and backward
- Backward kernel requires the **original input** (not output) to reconstruct the activation mask
- Both kernels share the same mask condition (`input[i] > 0`) but apply it differently:
  - Forward: masks the activation value
  - Backward: masks the incoming gradient

### Chain Rule Introduction
- `dL/dx = dL/dy × dy/dx` — gradient from above × local derivative
- For ReLU: local derivative is 1 (pass through) or 0 (block)
- This makes ReLU backward a "gradient mask" — simplest possible chain rule application
- Computational graph shows forward (left→right) and backward (right→left) arrows

### Gradient Masking Pattern
- `grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f`
- Negative inputs block both activations AND gradients ("dead neurons")
- At exactly x=0: convention is derivative=0 (subgradient), so gradient is blocked

### Testing Backward Passes
- Test backward separately with hardcoded values (gradient routing test)
- Test forward+backward round-trip with large random arrays (end-to-end)
- Edge case test at x=0 boundary (both forward and backward behavior)
- Use tighter tolerance (1e-6) for element-wise ops since no accumulation error

### Element-wise Kernel Pattern
- 1D grid with 256 threads per block
- `blocks = (n + threads - 1) / threads`
- Standard bounds check: `if (i >= n) return`
- Same pattern reusable for all element-wise activations (sigmoid, tanh, etc.)

## Puzzle 07: Conv2D Forward Pass

### NCHW Indexing Pattern
- `data[b][c][h][w] = data[b*(C*H*W) + c*(H*W) + h*W + w]`
- Filter: `filter[k][c][fh][fw] = filter[k*(C_in*F*F) + c*(F*F) + fh*F + fw]`
- Output: `output[b*(C_out*H_out*W_out) + k*(H_out*W_out) + oh*W_out + ow]`

### 1D Grid for 4D Output
- Flatten output (batch × C_out × H_out × W_out) into 1D thread grid
- Decode 4D indices with modular arithmetic: ow, oh, k, b (innermost to outermost)
- Simpler than 4D grid mapping, avoids wasting threads in degenerate dimensions

### Output Dimension Formula
- `H_out = H_in - F + 1` (no padding, stride=1)
- LeNet Conv1: (28-5)+1=24, Conv2: (12-5)+1=8
- Derivation: last valid start = H_in - F, positions = (H_in - F) - 0 + 1

### Direct Nested-Loop Convolution
- Three inner loops: c (input channels), fh (filter height), fw (filter width)
- Educational clarity > performance (no im2col, no shared memory tiling)
- Initialize accumulator with bias[k] before the loops

### Multi-Channel Accumulation
- Each output channel k has its own set of C_in×F×F filter weights
- Sum over ALL input channels for each output position
- Conv1: 1 input → 6 output (6 filters × 1×5×5 = 150 weights + 6 biases)
- Conv2: 6 input → 16 output (16 filters × 6×5×5 = 2400 weights + 16 biases)

### Test Strategy for Conv
- Hand-verifiable test: all-ones 3×3 filter on sequential 5×5 input → sum of 3×3 patches
- Dimension check test: zero filters + known bias → verifies output is bias-only with correct shape
- Random data tests with CPU reference for LeNet Conv1/Conv2 dimensions
- Batch test: same filters applied to multiple images

## Puzzle 12: LeNet-5 Forward Pass (Full Network) — COMPLETED 4/4

### Kernel Chaining / Orchestration
- The "puzzle" is orchestration, not kernel writing — all 5 kernel types are provided
- 12 kernel launches in sequence: Conv1, ReLU1, Pool1, Conv2, ReLU2, Pool2, FC1, ReLU3, FC2, ReLU4, FC3, Softmax
- Flatten step requires NO kernel — pool2_out memory is already contiguous (16×4×4=256 floats per sample)
- Each kernel's output buffer feeds directly as the next kernel's input buffer

### LeNet-5 Architecture Dimensions
- Input: (B, 1, 28, 28)
- Conv1(5×5, 6 filters): → (B, 6, 24, 24) — 150 weights + 6 biases
- ReLU + Pool(2×2): → (B, 6, 12, 12)
- Conv2(5×5, 16 filters): → (B, 16, 8, 8) — 2400 weights + 16 biases
- ReLU + Pool(2×2): → (B, 16, 4, 4) = (B, 256)
- FC1: 256→120 (30720 weights), FC2: 120→84 (10080 weights), FC3: 84→10 (840 weights)
- Softmax: → (B, 10) probabilities summing to 1.0
- Total parameters: 44,426

### FC Kernel Launch Dimensions
- fc_forward uses 2D grid: dim3 block(16,16), dim3 grid((out_f+15)/16, (batch+15)/16)
- blockIdx.x → output features, blockIdx.y → batch dimension
- Different from 1D kernels (conv, relu, pool, softmax) which use simple blocks/threads

### Struct-Based Buffer Management
- LeNetParams struct: device pointers for all weights/biases (10 pointers)
- LeNetActivations struct: device pointers for all intermediates (14 pointers including pool indices)
- alloc_activations/free_activations pattern cleanly manages 14 GPU allocations
- upload_params copies all host weights to device in one function call
- Pool indices (int*) tracked for potential backward pass reuse

### Testing Strategy for Full Forward Pass
- CPU reference: full lenet_forward_cpu chains all CPU layer functions identically
- Tolerance: atol=1e-4, rtol=1e-3 works for 12-layer chain (no need to relax further)
- Deterministic test: two GPU runs with same seed produce bit-exact results
- Intermediate dimensions test: downloads conv1_out, pool1_out, pool2_out, probs from GPU
- Batch test with 8 samples ensures per-sample independence

## Task 18: Integration Testing and README Completion (Feb 7, 2026)

**What We Did:**
- Created comprehensive top-level README.md (310+ lines) with:
  - Complete puzzle index table (all 15 puzzles with difficulty ratings)
  - ASCII LeNet-5 architecture diagram with parameter counts
  - Prerequisites section with CUDA Toolkit, CMake, GPU requirements
  - Quick start build instructions for Windows/Linux
  - MNIST download instructions with mirror URLs
  - Running tests section (individual + CTest batch)
  - Troubleshooting section for common issues
  - Project structure tree diagram
  - Learning path guidance

- Set up CTest integration:
  - Added `enable_testing()` to root CMakeLists.txt
  - Added `add_test()` calls to common/CMakeLists.txt for test_common
  - Added `add_test()` calls to all 15 puzzle CMakeLists.txt files

- Full build verification:
  - Clean build from scratch: `cmake -B build_clean -DCMAKE_CUDA_ARCHITECTURES=native`
  - Full build succeeded with exit code 0 (warnings about Unicode characters are benign)
  - CTest execution: `ctest --test-dir build_clean -C Debug --output-on-failure`
  - Result: 16/16 tests passed (100% success rate)
  - Total test time: 3.76 seconds

- Spot-checked individual tests:
  - puzzle_01_test_solution.exe: 4/4 tests passed
  - puzzle_08_test_solution.exe: 4/4 tests passed
  - puzzle_15_test_solution.exe: 5/5 tests passed

**CTest Configuration Discovery:**
- On Windows with Visual Studio multi-config generator, CTest requires `-C Debug` flag
- Without it: "Test not available without configuration"
- Linux single-config generators don't need this flag

**Commit Details:**
- Message: `feat: add top-level README and integration test configuration`
- Files changed: 19 (root CMakeLists.txt, README.md, common/CMakeLists.txt, puzzles/CMakeLists.txt, 15 puzzle CMakeLists.txt)
- Lines changed: +611 insertions, -6 deletions
- Commit hash: ce37c5b

**Final Project Stats:**
- Total puzzles: 15 (covering all LeNet-5 operations)
- Total tests: 16 (test_common + 15 puzzle solution tests)
- Total parameter count: ~44,426 (verified in architecture diagram)
- README word count: ~2,500 words
- Documentation completeness: 100%

**Success Metrics Achieved:**
✓ 16/16 tests pass via CTest
✓ README contains exactly 15 puzzle entries
✓ README > 150 lines (310 lines achieved)
✓ Clean build time < 10 minutes (completed in ~2 minutes)
✓ Zero compiler errors (only benign Unicode warnings)
✓ Exit code 0 for all build and test commands

## Project Completion Summary (Final)

### All Tasks Complete: 18/18 (100%)
- Task 1-2: Infrastructure and common utilities ✓
- Tasks 3-14: Puzzles 01-12 ✓
- Tasks 15-17: Puzzles 13-15 (LeNet backward, MNIST pipeline, training loop) ✓
- Task 18: Final integration and top-level README ✓

### All Verification Criteria Met (33/33 checkboxes)
- Definition of Done: 5/5 ✓
- Numbered Tasks: 18/18 ✓
- Final Checklist: 10/10 ✓

### Test Results
- CTest: 16/16 tests passed (100%)
- Total test time: 4.24 seconds
- Zero compiler errors, zero test failures

### Documentation Delivered
- Top-level README: 318 lines with comprehensive guide
- 15 puzzle READMEs: All with ASCII diagrams and math formulas
- Architecture diagram: Full LeNet-5 with parameter counts

### Project Statistics
- Total files created: ~85+
- Total parameters in LeNet: 44,426
- Lines of code: Thousands across 15 puzzles
- Test coverage: 100% (every puzzle has automated tests)

### Build System
- CMake configuration: Clean
- CTest integration: Complete
- Platform: Windows MSVC + CUDA 13.1
- Architecture: sm_75+ (auto-detected)

### Repository State
- Latest commit: ce37c5b "feat: add top-level README and integration test configuration"
- Git status: Clean working directory
- All plan checkboxes: Marked complete

The LeNet-5 CUDA Puzzles project is 100% complete and ready for educational use.
