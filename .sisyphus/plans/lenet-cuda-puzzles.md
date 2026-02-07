# LeNet-Style CNN CUDA Puzzles for MNIST

## TL;DR

> **Quick Summary**: Build a complete LeNet-5 CNN from scratch in pure CUDA C++ through 15 progressive educational puzzles, each with guided documentation (ASCII diagrams, math formulas), a TODO-template kernel, and a comprehensive test harness that verifies correctness automatically.
> 
> **Deliverables**:
> - 15 self-contained CUDA puzzles progressing from basic ops to full LeNet training
> - Each puzzle: `README.md` (guided docs), `puzzle.cu` (TODO template), `solution.cu` (reference), `test_puzzle.cu` (auto-grading)
> - Shared infrastructure: CMake build system, MNIST IDX loader, CUDA error macros, test utilities
> - Full LeNet achieving >95% MNIST accuracy after all puzzles are completed
> 
> **Estimated Effort**: XL (15 puzzles × implementation + docs + tests + infrastructure)
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: Task 1 (infra) → Task 2 (common utils) → Tasks 3-8 (foundation puzzles) → Tasks 9-14 (CNN puzzles) → Task 15 (integration) → Task 16 (final validation)

---

## Context

### Original Request
Build a from-scratch LeNet-style CNN in pure CUDA C++ for MNIST digit classification. Focus on educational clarity with extensive documentation, ASCII diagrams, and math formulas in every kernel. Structure as puzzles where each puzzle contains: fully guided documentation explaining what to do, related knowledge to learn, and comprehensive tests for verifying correctness.

### Interview Summary
**Key Discussions**:
- Repository is completely empty — greenfield project
- Pure CUDA C++ only (no cuDNN, no PyTorch, no external ML libraries)
- Each puzzle must be independently compilable and testable
- Focus on educational clarity over performance
- Windows environment with MSVC + CUDA 13.1 + RTX 5070 (compute 12.0)

**Research Findings**:
- LeNet-5 for 28×28 MNIST: Input→Conv1(5×5,6)→Pool→Conv2(5×5,16)→Pool→FC(256→120)→FC(120→84)→FC(84→10)→Softmax
- All forward/backward gradient formulas documented and verified
- Standard educational progression: foundations → backprop → CNN components → integration
- Numerical gradient checking is standard verification for ML implementations
- CUDA 13.1 minimum sm_75 (dropped sm_50–sm_70)
- im2col + GEMM is industry standard for convolution; direct nested loop is more educational
- MNIST IDX format requires big-endian byte swapping on little-endian systems

### Metis Review
**Identified Gaps** (addressed):
- Build system must be CMake (Makefiles won't work on Windows without g++) → **Resolved: CMake only**
- CUDA 13.1 dropped old architectures → **Resolved: target sm_75 minimum, use `native` detection**
- MNIST data provisioning → **Resolved: bundle tiny test subset + synthetic deterministic data for tests, full download optional**
- Convolution approach ambiguity → **Resolved: direct nested loop (educational clarity) in main puzzles, im2col NOT in scope**
- Floating point determinism → **Resolved: tolerance-based comparison (1e-4 forward, 1e-3 backward)**
- Softmax numerical overflow → **Resolved: max-subtraction trick mandatory, documented in puzzle**
- Puzzle independence → **Resolved: each puzzle includes pre-built reference implementations of its dependencies**
- Activation function choice → **Resolved: ReLU (modern, simpler derivative, better training)**

---

## Work Objectives

### Core Objective
Create 15 progressive CUDA C++ puzzles that teach students how to build a complete LeNet-5 CNN for MNIST digit classification from scratch, with each puzzle being a self-contained learning unit with guided documentation, a kernel to implement, and automated tests.

### Concrete Deliverables
- `CMakeLists.txt` — root build configuration
- `common/` — shared utilities (CUDA error macros, MNIST loader, test framework)
- `puzzles/puzzle_01_vector_add/` through `puzzles/puzzle_15_training_loop/` — 15 puzzle directories
- Each puzzle directory contains:
  - `README.md` — Guided documentation with ASCII diagrams, math formulas, step-by-step instructions
  - `puzzle.cu` — TODO template with clear markers where student writes code
  - `solution.cu` — Complete reference implementation
  - `test_puzzle.cu` — Comprehensive automated test harness
  - `CMakeLists.txt` — Per-puzzle build configuration
- `data/` — Bundled mini MNIST subset (100 images) + synthetic test data generator
- Top-level `README.md` — Project overview, setup instructions, puzzle index

### Definition of Done
- [ ] `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build` succeeds with zero errors
- [ ] Every `puzzle_XX_test` executable returns exit code 0 when run with reference solution
- [ ] Every puzzle README renders correctly in terminal at 80 columns (ASCII art intact)
- [ ] Full LeNet trained on MNIST achieves >95% test accuracy within 10 epochs
- [ ] A student stuck on puzzle N can still attempt puzzle N+1 (independence verified)

### Must Have
- Pure CUDA C++ — no external ML libraries, no cuDNN, no Python dependencies for core puzzles
- ASCII architecture diagrams in every puzzle README
- Math formulas (forward AND backward) in every kernel's documentation
- Automated test harness per puzzle (compile → run → exit code 0/1)
- Deterministic seeds for reproducibility (seed=42 for weights, seed=123 for data)
- CUDA error checking macro used everywhere
- Tolerance-based floating-point comparison (not bit-exact)
- CMake build system (cross-platform compatible, MSVC + nvcc on Windows)
- Big-endian byte swapping in MNIST loader
- Softmax max-subtraction numerical stability trick
- Each puzzle independently compilable and testable

### Must NOT Have (Guardrails)
- ❌ NO shared memory optimization, tiling, or warp-level puzzles — naive one-thread-per-output is fine
- ❌ NO multi-GPU support
- ❌ NO batch normalization, dropout, or modern regularization
- ❌ NO dynamic memory allocation in kernels — all buffers pre-allocated by host
- ❌ NO mixed precision / FP16 / tensor cores
- ❌ NO custom memory allocators or memory pools
- ❌ NO im2col optimization (direct nested-loop convolution for educational clarity)
- ❌ NO Makefiles (won't work on Windows without g++)
- ❌ NO Python dependencies for puzzle execution (Python only for optional utilities)
- ❌ NO explaining basic C++ syntax — assume C++ competency, teach CUDA + ML concepts
- ❌ NO external framework references as primary explanation ("this is what PyTorch does") — math first

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.
> Every criterion is verifiable by running a command.
>
> **FORBIDDEN** — acceptance criteria that require:
> - "User manually tests..."
> - "User visually confirms..."
> - ANY step where a human must perform an action

### Test Decision
- **Infrastructure exists**: NO (greenfield)
- **Automated tests**: YES (tests-after — each puzzle has its own test harness)
- **Framework**: Custom lightweight test framework in `common/test_utils.h` (no GoogleTest dependency — keep it pure CUDA C++)
- **Approach**: Reference output comparison with tolerance + numerical gradient checking

### Test Architecture

Each puzzle's `test_puzzle.cu` follows this pattern:
```cpp
#include "common/test_utils.h"
#include "puzzle.cu"  // Student's implementation (or solution.cu for reference)

int main() {
    // 1. Setup: deterministic inputs (seeded random or hardcoded)
    // 2. Run student's kernel
    // 3. Compare output to pre-computed reference values
    // 4. Print PASS/FAIL per test case with numerical details
    // 5. Return 0 if all pass, 1 if any fail
}
```

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

Every task includes QA scenarios where the executing agent directly verifies the deliverable by:
1. Building with CMake
2. Running the test executable
3. Checking exit code and output
4. Verifying documentation renders correctly

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Project infrastructure (git, CMake, directory structure)
└── (nothing else — foundational)

Wave 2 (After Wave 1):
├── Task 2: Common utilities (CUDA macros, MNIST loader, test framework)
└── (nothing else — everything depends on this)

Wave 3 (After Wave 2 — PARALLEL BATCH):
├── Task 3:  Puzzle 01 - Vector Addition
├── Task 4:  Puzzle 02 - Matrix Multiplication
├── Task 5:  Puzzle 03 - FC Layer Forward
├── Task 6:  Puzzle 04 - ReLU Activation Forward + Backward
├── Task 7:  Puzzle 05 - Softmax + Cross-Entropy Loss
├── Task 8:  Puzzle 06 - FC Layer Backward
├── Task 9:  Puzzle 07 - Conv2D Forward
├── Task 10: Puzzle 08 - Max Pooling Forward + Backward
├── Task 11: Puzzle 09 - Conv2D Backward (Weight Gradients)
├── Task 12: Puzzle 10 - Conv2D Backward (Input Gradients)
├── Task 13: Puzzle 11 - SGD Optimizer + Weight Update
├── Task 14: Puzzle 12 - LeNet Forward Pass (Full Network)
└── Task 15: Puzzle 13 - LeNet Backward Pass (Full Backprop)

Wave 4 (After Wave 3):
├── Task 16: Puzzle 14 - MNIST Data Pipeline
├── Task 17: Puzzle 15 - Full Training Loop
└── Task 18: Final integration testing + top-level README
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2 | None |
| 2 | 1 | 3-15 | None |
| 3 | 2 | None | 4,5,6,7,8,9,10,11,12,13 |
| 4 | 2 | None | 3,5,6,7,8,9,10,11,12,13 |
| 5 | 2 | None | 3,4,6,7,8,9,10,11,12,13 |
| 6 | 2 | None | 3,4,5,7,8,9,10,11,12,13 |
| 7 | 2 | None | 3,4,5,6,8,9,10,11,12,13 |
| 8 | 2 | None | 3,4,5,6,7,9,10,11,12,13 |
| 9 | 2 | None | 3,4,5,6,7,8,10,11,12,13 |
| 10 | 2 | None | 3,4,5,6,7,8,9,11,12,13 |
| 11 | 2 | None | 3,4,5,6,7,8,9,10,12,13 |
| 12 | 2 | None | 3,4,5,6,7,8,9,10,11,13 |
| 13 | 2 | None | 3,4,5,6,7,8,9,10,11,12 |
| 14 | 2 | 16 | 3,4,5,6,7,8,9,10,11,12,13,15 |
| 15 | 2 | 17 | 3,4,5,6,7,8,9,10,11,12,13,14 |
| 16 | 2,14 | 18 | 17 (partial) |
| 17 | 2,15 | 18 | 16 (partial) |
| 18 | 3-17 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1 | `delegate_task(category="quick", load_skills=[], ...)` |
| 2 | 2 | `delegate_task(category="unspecified-high", load_skills=[], ...)` |
| 3 | 3-15 | `delegate_task(category="deep", load_skills=[], ...)` per puzzle (parallel) |
| 4 | 16-18 | `delegate_task(category="deep", load_skills=[], ...)` |

---

## LeNet-5 Architecture Reference (for all puzzles)

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

---

## TODOs

---

### Task 1: Project Infrastructure Setup

- [ ] 1. Project Infrastructure — Git init, directory structure, root CMakeLists.txt

  **What to do**:
  - Initialize git repository (`git init`)
  - Create directory structure:
    ```
    cuda-puzzles/
    ├── CMakeLists.txt          (root build config)
    ├── README.md               (project overview)
    ├── common/                 (shared utilities — created empty, populated in Task 2)
    │   └── CMakeLists.txt
    ├── data/                   (MNIST data — created empty)
    ├── puzzles/                (puzzle directories — created empty)
    │   └── CMakeLists.txt
    └── .gitignore
    ```
  - Write root `CMakeLists.txt`:
    ```cmake
    cmake_minimum_required(VERSION 3.18)
    project(cuda-puzzles LANGUAGES CXX CUDA)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)
    
    # Default to native GPU architecture if not specified
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    
    # Minimum sm_75 (CUDA 13.x dropped sm_50-sm_70)
    # Users with older GPUs need older CUDA toolkit
    
    add_subdirectory(common)
    add_subdirectory(puzzles)
    ```
  - Write `.gitignore` (build/, *.o, *.exe, *.exp, *.lib, .vs/, data/*.idx3-ubyte, data/*.idx1-ubyte)
  - Write `puzzles/CMakeLists.txt` (empty initially, will add subdirectories as puzzles are created)
  - Write `common/CMakeLists.txt` (header-only library target for common utilities)
  - Verify: `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native` configures without errors

  **Must NOT do**:
  - Do NOT add puzzle directories yet (those come in later tasks)
  - Do NOT install any external dependencies (GoogleTest, etc.)
  - Do NOT create any .cu files yet

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple file creation and configuration — no complex logic
  - **Skills**: []
    - No specialized skills needed for directory setup
  - **Skills Evaluated but Omitted**:
    - `git-master`: Not needed — just a simple `git init`, not complex git operations

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (solo)
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - None (greenfield project)

  **Documentation References**:
  - CMake CUDA support: CMake 3.18+ has native CUDA language support via `enable_language(CUDA)` or `project(... LANGUAGES CUDA)`
  - `CMAKE_CUDA_ARCHITECTURES`: Set to `native` for auto-detection, or explicit list like `"75;80;86;89;90;100;120"`

  **External References**:
  - CMake CUDA documentation: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html

  **Acceptance Criteria**:

  - [ ] Directory structure exists as specified above
  - [ ] `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native` exits with code 0
  - [ ] `.gitignore` contains build/, *.o, *.exe entries
  - [ ] `git log --oneline` shows initial commit

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: CMake configures successfully
    Tool: Bash
    Preconditions: CUDA toolkit installed, MSVC available
    Steps:
      1. Run: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
      2. Assert: exit code is 0
      3. Assert: output contains "Build files have been written"
      4. Assert: build/ directory exists
    Expected Result: CMake generates build system without errors
    Failure Indicators: Non-zero exit code, "error" in output
    Evidence: Terminal output captured

  Scenario: Directory structure is complete
    Tool: Bash
    Preconditions: Task completed
    Steps:
      1. Run: dir /s /b CMakeLists.txt (or ls -R on Linux)
      2. Assert: root CMakeLists.txt exists
      3. Assert: common/CMakeLists.txt exists
      4. Assert: puzzles/CMakeLists.txt exists
      5. Assert: .gitignore exists
      6. Assert: data/ directory exists
    Expected Result: All directories and files present
    Failure Indicators: Missing files or directories
    Evidence: Directory listing captured
  ```

  **Commit**: YES
  - Message: `feat(infra): initialize project with CMake build system`
  - Files: `CMakeLists.txt, common/CMakeLists.txt, puzzles/CMakeLists.txt, .gitignore, README.md`
  - Pre-commit: `cmake -B build`

---

### Task 2: Common Utilities (CUDA Macros, Test Framework, MNIST Loader)

- [ ] 2. Common Utilities — cuda_utils.h, test_utils.h, mnist_loader.h

  **What to do**:
  - Create `common/cuda_utils.h`:
    ```cpp
    // CUDA_CHECK(err) macro — wraps every CUDA call, prints file:line on error, exits
    // KERNEL_CHECK() macro — calls cudaGetLastError() + cudaDeviceSynchronize() after kernel launch
    // Timer utility — cudaEvent-based timing for optional performance measurement
    ```
  - Create `common/test_utils.h`:
    ```cpp
    // check_close(got, expected, atol=1e-4, rtol=1e-4) — per-element tolerance check
    // check_array_close(got, expected, n, atol, rtol) — array comparison with detailed diff report
    // print_test_result(name, passed, details) — formatted PASS/FAIL output
    // TEST_CASE(name) macro — lightweight test registration
    // RUN_ALL_TESTS() — runs all registered tests, returns 0 if all pass
    // set_seed(seed) — deterministic curand/stdlib seed setup
    // fill_random(arr, n, seed, min, max) — deterministic random fill on host
    ```
    Output format: `"[PASS] test_name"` or `"[FAIL] test_name: Expected 0.12345, Got 0.12344, Diff 0.00001 (tolerance 0.0001)"`
    Final line: `"X/Y tests passed"` — exit code 0 if all pass, 1 if any fail
  - Create `common/mnist_loader.h`:
    ```cpp
    // MNISTLoader class:
    //   - load_images(path) — reads IDX3 format, handles big-endian byte swap
    //   - load_labels(path) — reads IDX1 format, handles big-endian byte swap
    //   - normalize(images) — scales pixel values from [0,255] to [0.0, 1.0]
    //   - Returns: vector of float arrays (images) and int array (labels)
    //   - Error handling: clear error message if file not found (not segfault)
    //
    // Big-endian byte swap:
    //   uint32_t swap_endian(uint32_t val) {
    //     return ((val>>24)&0xff) | ((val>>8)&0xff00) | ((val<<8)&0xff0000) | ((val<<24)&0xff000000);
    //   }
    ```
  - Create `common/common.h` — single include that pulls in all common headers
  - Create synthetic test data generator in `common/test_data.h`:
    ```cpp
    // generate_test_images(n, seed) — creates n deterministic 28×28 float images
    // generate_test_labels(n, seed) — creates n deterministic labels [0-9]
    // Used by puzzle tests so they don't require MNIST download
    ```
  - Update `common/CMakeLists.txt`:
    ```cmake
    add_library(cuda_puzzles_common INTERFACE)
    target_include_directories(cuda_puzzles_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    ```
  - Write a `common/test_common.cu` that verifies all utilities work:
    - CUDA_CHECK on a valid call → no error
    - CUDA_CHECK on an invalid call → catches error gracefully
    - fill_random produces deterministic output
    - check_close passes for close values, fails for distant values
    - Byte swap works correctly
    - Test data generator produces correct shapes

  **Must NOT do**:
  - Do NOT use GoogleTest or any external test framework
  - Do NOT use cuDNN or any external ML library
  - Do NOT include cuRAND device API (use simple host-side PRNG for deterministic seeds)
  - Do NOT make MNIST loader a hard dependency — tests work with synthetic data

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Core infrastructure that all 15 puzzles depend on — must be robust, well-documented, and thoroughly tested
  - **Skills**: []
    - No specialized skills needed — pure CUDA C++ header files
  - **Skills Evaluated but Omitted**:
    - `playwright`: No browser interaction
    - `frontend-ui-ux`: No UI

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (solo)
  - **Blocks**: Tasks 3-18 (all puzzles depend on common utilities)
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - MNIST IDX format spec: Magic number (4 bytes big-endian), dimensions, raw data. Format: `0x00000803` for images (3 dims: count×rows×cols), `0x00000801` for labels (1 dim: count)
  - CUDA error checking pattern: `#define CUDA_CHECK(err) do { cudaError_t e = (err); if (e != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)`

  **External References**:
  - MNIST file format: http://yann.lecun.com/exdb/mnist/ — IDX file format specification
  - CUDA error handling best practices: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

  **Acceptance Criteria**:

  - [ ] `common/cuda_utils.h` exists with CUDA_CHECK and KERNEL_CHECK macros
  - [ ] `common/test_utils.h` exists with check_close, check_array_close, TEST_CASE, RUN_ALL_TESTS
  - [ ] `common/mnist_loader.h` exists with byte-swap and IDX parsing
  - [ ] `common/test_data.h` exists with synthetic data generators
  - [ ] `cmake --build build --target test_common` compiles without error
  - [ ] `./build/test_common` exits with code 0, output contains "tests passed"

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Common utilities compile and self-test passes
    Tool: Bash
    Preconditions: Task 1 complete, CMake configured
    Steps:
      1. Run: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
      2. Run: cmake --build build --target test_common
      3. Assert: exit code is 0 (compilation succeeds)
      4. Run: ./build/test_common (or .\build\Debug\test_common.exe on Windows)
      5. Assert: exit code is 0
      6. Assert: output contains "tests passed"
      7. Assert: output contains "[PASS]" for each sub-test
    Expected Result: All utility self-tests pass
    Failure Indicators: Compilation errors, non-zero exit code, [FAIL] in output
    Evidence: Terminal output captured

  Scenario: CUDA_CHECK catches invalid operation
    Tool: Bash
    Preconditions: test_common built
    Steps:
      1. Run: ./build/test_common
      2. Assert: output contains test for CUDA error catching
      3. Assert: error handling test reports [PASS]
    Expected Result: Error macro correctly catches and reports CUDA errors
    Evidence: Terminal output captured

  Scenario: Deterministic random generation
    Tool: Bash
    Preconditions: test_common built
    Steps:
      1. Run: ./build/test_common (first run)
      2. Capture output values from fill_random test
      3. Run: ./build/test_common (second run)
      4. Assert: output values are identical between runs
    Expected Result: Same seed produces identical data across runs
    Evidence: Both outputs captured and compared
  ```

  **Commit**: YES
  - Message: `feat(common): add CUDA utils, test framework, MNIST loader, and test data generator`
  - Files: `common/cuda_utils.h, common/test_utils.h, common/mnist_loader.h, common/test_data.h, common/common.h, common/test_common.cu, common/CMakeLists.txt`
  - Pre-commit: `cmake --build build --target test_common && ./build/test_common`

---

### Task 3: Puzzle 01 — Vector Addition (CUDA Warmup)

- [ ] 3. Puzzle 01 — Vector Addition

  **What to do**:
  - Create `puzzles/puzzle_01_vector_add/` with: `README.md`, `puzzle.cu`, `solution.cu`, `test_puzzle.cu`, `CMakeLists.txt`
  - **README.md** must include:
    - Overview: "Your first CUDA kernel! Add two vectors element-wise on the GPU."
    - ASCII diagram of grid/block/thread mapping:
      ```
      Vector A:  [a0, a1, a2, a3, a4, a5, a6, a7, ...]
      Vector B:  [b0, b1, b2, b3, b4, b5, b6, b7, ...]
                  +   +   +   +   +   +   +   +
      Vector C:  [c0, c1, c2, c3, c4, c5, c6, c7, ...]
      
      Thread mapping:
      Block 0:          Block 1:          Block 2:
      [T0,T1,T2,T3] → [T0,T1,T2,T3] → [T0,T1,T2,T3] → ...
       ↓  ↓  ↓  ↓      ↓  ↓  ↓  ↓      ↓  ↓  ↓  ↓
      [c0,c1,c2,c3]   [c4,c5,c6,c7]   [c8,c9,c10,c11]
      ```
    - Math formula: `C[i] = A[i] + B[i], for i = 0, 1, ..., N-1`
    - CUDA concepts: `__global__`, `blockIdx`, `blockDim`, `threadIdx`, grid stride loop
    - Step-by-step guide: (1) Calculate global thread index, (2) Bounds check, (3) Compute sum
    - Hints section (progressive: easy → medium → strong hint)
  - **puzzle.cu**: Template with `// TODO: Implement the vector addition kernel` markers
  - **solution.cu**: Complete working implementation
  - **test_puzzle.cu**: Tests with known inputs:
    - Test 1: Small array (N=8), hardcoded values
    - Test 2: Medium array (N=1024), seeded random
    - Test 3: Large array (N=1,000,000), seeded random
    - Test 4: Edge case — N not divisible by block size
  - **CMakeLists.txt**: Builds test linking against common utils, builds both puzzle and solution variants
  - Update `puzzles/CMakeLists.txt` to add this subdirectory

  **Must NOT do**:
  - Do NOT use shared memory (unnecessary for vector add)
  - Do NOT add performance benchmarking (out of scope)
  - Do NOT explain basic C++ syntax

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: First puzzle establishes the template pattern that ALL subsequent puzzles follow — must be exemplary in documentation quality, code style, and test thoroughness
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `playwright`: No browser interaction
    - `frontend-ui-ux`: No UI

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 4-15)
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `common/test_utils.h` — Use TEST_CASE and check_array_close for test structure
  - `common/cuda_utils.h` — Use CUDA_CHECK and KERNEL_CHECK macros
  - `common/test_data.h` — Use fill_random for deterministic test data

  **Documentation References**:
  - CUDA Programming Guide §2.1: Kernels — `__global__` function declaration
  - CUDA Programming Guide §2.2: Thread Hierarchy — blockIdx, blockDim, threadIdx

  **WHY Each Reference Matters**:
  - `test_utils.h`: Establishes the exact test output format that ALL puzzles must follow
  - `cuda_utils.h`: CUDA_CHECK wraps every cudaMalloc/cudaMemcpy call; KERNEL_CHECK after launch
  - First puzzle sets the documentation standard — every subsequent puzzle README follows this template

  **Acceptance Criteria**:

  - [ ] `cmake --build build --target puzzle_01_test` compiles with exit code 0
  - [ ] `./build/puzzle_01_test` with solution.cu exits with code 0
  - [ ] Output contains "4/4 tests passed"
  - [ ] README.md contains ASCII diagram of thread mapping
  - [ ] README.md contains math formula for vector addition
  - [ ] puzzle.cu contains `// TODO:` markers where student writes code
  - [ ] solution.cu compiles and all tests pass

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Solution passes all tests
    Tool: Bash
    Preconditions: Task 2 complete, common utilities available
    Steps:
      1. Run: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
      2. Run: cmake --build build --target puzzle_01_test_solution
      3. Assert: exit code 0
      4. Run: ./build/puzzle_01_test_solution
      5. Assert: exit code 0
      6. Assert: output contains "4/4 tests passed"
      7. Assert: output contains "[PASS]" for each test
    Expected Result: All 4 test cases pass with reference solution
    Evidence: Terminal output captured

  Scenario: Template compiles but tests fail (student hasn't implemented yet)
    Tool: Bash
    Preconditions: puzzle.cu contains only TODO stubs
    Steps:
      1. Run: cmake --build build --target puzzle_01_test
      2. Assert: compiles (exit code 0 from build)
      3. Run: ./build/puzzle_01_test
      4. Assert: tests fail (exit code 1) — proving tests actually check correctness
    Expected Result: Template compiles but produces wrong results
    Evidence: Terminal output shows [FAIL]

  Scenario: README documentation quality
    Tool: Bash
    Preconditions: README.md created
    Steps:
      1. Read: puzzles/puzzle_01_vector_add/README.md
      2. Assert: contains "# Puzzle 01" or similar header
      3. Assert: contains ASCII art (lines with ┌ or + or → characters)
      4. Assert: contains "C[i] = A[i] + B[i]" or equivalent formula
      5. Assert: contains "TODO" or "Step-by-step" section
      6. Assert: file is < 80 columns wide (check longest line)
    Expected Result: README has all required educational elements
    Evidence: File content captured
  ```

  **Commit**: YES
  - Message: `feat(puzzle-01): add vector addition puzzle with guided docs and tests`
  - Files: `puzzles/puzzle_01_vector_add/*`
  - Pre-commit: `cmake --build build --target puzzle_01_test_solution && ./build/puzzle_01_test_solution`

---

### Task 4: Puzzle 02 — Matrix Multiplication

- [ ] 4. Puzzle 02 — Matrix Multiplication (Naive)

  **What to do**:
  - Create `puzzles/puzzle_02_matmul/` with full puzzle structure
  - **README.md** must include:
    - Why matrix multiplication matters for neural networks (FC layers = matmul)
    - ASCII diagram of matrix dimensions and element computation:
      ```
      A (M×K)        B (K×N)        C (M×N)
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │ a00 a01 │   │ b00 b01 │   │ c00 c01 │
      │ a10 a11 │ × │ b10 b11 │ = │ c10 c11 │
      └─────────┘   └─────────┘   └─────────┘
      
      c[i][j] = Σ_k a[i][k] * b[k][j]
      ```
    - Thread-to-element mapping: each thread computes one output element
    - Row-major memory layout explanation with ASCII
    - Math formula: `C[i,j] = Σ_{k=0}^{K-1} A[i,k] × B[k,j]`
    - Discussion of global memory access patterns
  - **test_puzzle.cu**: Tests:
    - Test 1: 2×2 × 2×2 (hardcoded, hand-verifiable)
    - Test 2: 4×3 × 3×5 (hardcoded)
    - Test 3: 64×128 × 128×64 (seeded random, compared against CPU reference)
    - Test 4: Non-square matrices (256×120, matching FC layer dimensions in LeNet)

  **Must NOT do**:
  - Do NOT use shared memory tiling (that's an optimization)
  - Do NOT use cuBLAS

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Matrix multiplication is the foundation of FC layers — documentation must clearly connect matmul to neural network computation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `puzzles/puzzle_01_vector_add/` — Follow identical file structure and documentation template established in Puzzle 01
  - `common/test_utils.h` — Use check_array_close with atol=1e-4

  **WHY Each Reference Matters**:
  - Puzzle 01 template: Ensures consistent look and feel across all puzzles
  - Matmul dimensions (256×120, 120×84, 84×10) directly correspond to LeNet FC layers — call this out in documentation

  **Acceptance Criteria**:

  - [ ] Solution passes 4/4 tests
  - [ ] README contains row-major memory layout ASCII diagram
  - [ ] README contains matmul formula with summation notation
  - [ ] Tests include LeNet-specific dimensions (256×120)

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Solution passes all tests
    Tool: Bash
    Steps:
      1. Build: cmake --build build --target puzzle_02_test_solution
      2. Run: ./build/puzzle_02_test_solution
      3. Assert: exit code 0, output contains "4/4 tests passed"
    Expected Result: All tests pass
    Evidence: Terminal output

  Scenario: Template compiles but fails tests
    Tool: Bash
    Steps:
      1. Build: cmake --build build --target puzzle_02_test
      2. Run: ./build/puzzle_02_test
      3. Assert: exit code 1 (tests detect unimplemented kernel)
    Expected Result: Correctly detects wrong output
    Evidence: Terminal output shows [FAIL]
  ```

  **Commit**: YES
  - Message: `feat(puzzle-02): add matrix multiplication puzzle`
  - Files: `puzzles/puzzle_02_matmul/*`
  - Pre-commit: `cmake --build build --target puzzle_02_test_solution && ./build/puzzle_02_test_solution`

---

### Task 5: Puzzle 03 — Fully Connected Layer Forward

- [ ] 5. Puzzle 03 — FC Layer Forward Pass

  **What to do**:
  - Create `puzzles/puzzle_03_fc_forward/`
  - **README.md** must include:
    - What a fully connected layer does in a neural network
    - ASCII diagram of a neuron and a full FC layer:
      ```
      Single Neuron:
      x0 ──w0──┐
      x1 ──w1──┤
      x2 ──w2──┼──→ Σ(wi*xi) + b ──→ activation ──→ output
      x3 ──w3──┤
      x4 ──w4──┘
      
      FC Layer (batch):
      Y = X · W^T + b
      
      Input X:  (batch × in_features)    e.g., (32 × 256)
      Weights W: (out_features × in_features) e.g., (120 × 256)
      Bias b:    (out_features)           e.g., (120)
      Output Y:  (batch × out_features)  e.g., (32 × 120)
      ```
    - Math formulas:
      ```
      y[b][j] = Σ_{i=0}^{in-1} x[b][i] × w[j][i] + bias[j]
      ```
    - How this connects to Puzzle 02 (FC forward = matmul + bias)
    - Memory layout for weights and biases
  - **test_puzzle.cu**: Tests:
    - Test 1: Single sample, 4 inputs → 3 outputs (hardcoded, hand-verifiable)
    - Test 2: Batch of 8, 256→120 (LeNet FC1 dimensions)
    - Test 3: Batch of 8, 120→84 (LeNet FC2 dimensions)
    - Test 4: Batch of 8, 84→10 (LeNet output dimensions)

  **Must NOT do**:
  - Do NOT include activation function (that's Puzzle 04)
  - Do NOT include backward pass (that's Puzzle 06)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: First neural network layer puzzle — must bridge matmul concept to ML concept clearly
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `puzzles/puzzle_02_matmul/` — FC forward is matmul + bias addition; reference this connection explicitly
  - `common/test_utils.h` — check_array_close for output comparison

  **WHY Each Reference Matters**:
  - Puzzle 02 establishes matmul understanding; Puzzle 03 shows its direct application in neural networks
  - LeNet dimension tests (256→120, 120→84, 84→10) ensure the implementation works for the actual network

  **Acceptance Criteria**:
  - [ ] Solution passes 4/4 tests
  - [ ] README ASCII art shows neuron diagram
  - [ ] README shows formula `y = Xw^T + b`
  - [ ] Tests use actual LeNet FC dimensions

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Solution passes all tests including LeNet dimensions
    Tool: Bash
    Steps:
      1. Build and run puzzle_03_test_solution
      2. Assert: exit code 0, "4/4 tests passed"
    Evidence: Terminal output

  Scenario: Template fails tests
    Tool: Bash
    Steps:
      1. Build and run puzzle_03_test
      2. Assert: exit code 1
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(puzzle-03): add fully connected forward pass puzzle`
  - Files: `puzzles/puzzle_03_fc_forward/*`

---

### Task 6: Puzzle 04 — ReLU Activation Forward + Backward

- [ ] 6. Puzzle 04 — ReLU Forward and Backward Pass

  **What to do**:
  - Create `puzzles/puzzle_04_relu/`
  - **README.md** must include:
    - What activation functions do and why they're needed (non-linearity)
    - ASCII diagram of ReLU function:
      ```
      Output
        │      ╱
        │     ╱
        │    ╱
        │   ╱
      ──┼──╱────── Input
        │ ╱
        │╱
        │
      
      ReLU(x) = max(0, x)
      ```
    - Forward formula: `y = max(0, x)`
    - Backward formula (chain rule introduction):
      ```
      dy/dx = { 1  if x > 0
              { 0  if x ≤ 0
      
      ∂L/∂x = ∂L/∂y × dy/dx = { ∂L/∂y  if x > 0
                                { 0       if x ≤ 0
      ```
    - **FIRST INTRODUCTION OF CHAIN RULE** — explain with ASCII computational graph:
      ```
      Forward:    x ──→ [ReLU] ──→ y ──→ [Loss] ──→ L
      
      Backward:   ∂L/∂x ←── [dReLU] ←── ∂L/∂y ←── [dLoss] ←── 1
      ```
    - Why ReLU is preferred over sigmoid/tanh (vanishing gradient problem, computation speed)
  - **puzzle.cu**: TWO kernels to implement:
    - `relu_forward(float* input, float* output, int n)`
    - `relu_backward(float* grad_output, float* input, float* grad_input, int n)`
  - **test_puzzle.cu**: Tests:
    - Test 1: Forward — hardcoded values including negatives, zeros, positives
    - Test 2: Backward — verify gradient is 0 for negative inputs, 1 for positive
    - Test 3: Forward + Backward round-trip with random data (seed=42)
    - Test 4: Edge case — input is exactly 0.0 (gradient should be 0)

  **Must NOT do**:
  - Do NOT implement sigmoid, tanh, or other activations
  - Do NOT include in-place operation (keep separate input/output buffers for clarity)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: First backward pass puzzle — chain rule introduction is a critical pedagogical moment
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `common/test_utils.h` — check_close for per-element comparison
  - Puzzle 01 documentation template for ASCII art style

  **WHY Each Reference Matters**:
  - This is the FIRST puzzle with a backward pass — documentation must be extremely clear about chain rule
  - Students first encounter ∂L/∂x notation here — set the standard for all subsequent backward pass puzzles

  **Acceptance Criteria**:
  - [ ] Solution passes 4/4 tests (both forward and backward)
  - [ ] README contains ReLU graph ASCII art
  - [ ] README introduces chain rule with computational graph diagram
  - [ ] Backward kernel correctly produces 0 gradient for x ≤ 0

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Forward and backward both pass
    Tool: Bash
    Steps:
      1. Build and run puzzle_04_test_solution
      2. Assert: exit code 0, "4/4 tests passed"
    Evidence: Terminal output

  Scenario: Gradient correctness verified
    Tool: Bash
    Steps:
      1. Run puzzle_04_test_solution
      2. Assert: output shows gradient=0.0 for negative inputs
      3. Assert: output shows gradient=upstream for positive inputs
    Evidence: Terminal output with gradient values
  ```

  **Commit**: YES
  - Message: `feat(puzzle-04): add ReLU forward and backward pass puzzle`
  - Files: `puzzles/puzzle_04_relu/*`

---

### Task 7: Puzzle 05 — Softmax + Cross-Entropy Loss

- [ ] 7. Puzzle 05 — Softmax + Cross-Entropy Forward and Backward

  **What to do**:
  - Create `puzzles/puzzle_05_softmax_ce/`
  - **README.md** must include:
    - What softmax does (converts logits to probabilities)
    - What cross-entropy loss measures (distance between predicted and true distribution)
    - ASCII diagram of softmax → loss pipeline:
      ```
      Logits z:     [2.0, 1.0, 0.1]
                         │
                    ┌────▼────┐
                    │ Softmax │   softmax(zi) = exp(zi) / Σ exp(zk)
                    └────┬────┘
                         │
      Probabilities p: [0.659, 0.242, 0.099]
                         │
                    ┌────▼─────────┐
                    │ Cross-Entropy │  L = -Σ yi × log(pi)
                    └────┬─────────┘
                         │
      Loss L:          0.418  (if true class = 0)
      ```
    - **CRITICAL: Numerical stability trick**:
      ```
      // WRONG (overflows for large logits):
      softmax(zi) = exp(zi) / Σ exp(zk)
      
      // RIGHT (numerically stable):
      m = max(z)
      softmax(zi) = exp(zi - m) / Σ exp(zk - m)
      ```
    - Backward formula (the elegant simplification):
      ```
      ∂L/∂zi = pi - yi  (probability minus one-hot label)
      
      This is beautifully simple! The gradient of softmax+CE combined
      is just: (predictions - targets)
      ```
    - Why we combine softmax and cross-entropy (numerical stability + simpler gradient)
  - **puzzle.cu**: THREE functions:
    - `softmax_forward(float* logits, float* probs, int batch, int classes)` — with max-subtraction
    - `cross_entropy_loss(float* probs, int* labels, float* loss, int batch, int classes)` — with log(p+eps)
    - `softmax_ce_backward(float* probs, int* labels, float* grad_logits, int batch, int classes)`
  - **test_puzzle.cu**: Tests:
    - Test 1: Softmax — hardcoded logits, verify probabilities sum to 1.0
    - Test 2: Softmax — large logits (test numerical stability: [1000, 1001, 1002])
    - Test 3: Cross-entropy — known loss value for known inputs
    - Test 4: Backward — verify gradient = (probs - one_hot_labels) for a batch
    - Test 5: Full forward-backward round trip

  **Must NOT do**:
  - Do NOT skip the max-subtraction trick (even if tests pass without it on small values, it WILL fail on large logits)
  - Do NOT use log(p) without epsilon protection

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Numerical stability is critical and easy to get wrong — documentation must explain why naive approach fails
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - Puzzle 04 backward pass pattern — builds on chain rule understanding
  - `common/test_utils.h` — check_close with appropriate tolerance for probability values

  **WHY Each Reference Matters**:
  - Max-subtraction trick is MANDATORY — test with logits [1000, 1001, 1002] to prove naive approach fails
  - The ∂L/∂z = p - y formula is one of the most elegant results in ML — document it beautifully

  **Acceptance Criteria**:
  - [ ] Solution passes 5/5 tests
  - [ ] Softmax handles large logits (>100) without NaN/Inf
  - [ ] Probabilities sum to 1.0 (within tolerance)
  - [ ] README explains max-subtraction trick with before/after example

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Numerical stability test
    Tool: Bash
    Steps:
      1. Build and run puzzle_05_test_solution
      2. Assert: test with logits [1000,1001,1002] passes (no NaN/Inf)
      3. Assert: probabilities sum to 1.0 within 1e-5
    Evidence: Terminal output

  Scenario: Gradient simplification verified
    Tool: Bash
    Steps:
      1. Run puzzle_05_test_solution
      2. Assert: backward test shows gradient ≈ (probs - one_hot)
    Evidence: Terminal output with gradient comparison
  ```

  **Commit**: YES
  - Message: `feat(puzzle-05): add softmax + cross-entropy loss puzzle`
  - Files: `puzzles/puzzle_05_softmax_ce/*`

---

### Task 8: Puzzle 06 — FC Layer Backward Pass

- [ ] 8. Puzzle 06 — FC Layer Backward Pass

  **What to do**:
  - Create `puzzles/puzzle_06_fc_backward/`
  - **README.md** must include:
    - Recap of FC forward (from Puzzle 03): `Y = X · W^T + b`
    - Full backward pass derivation with ASCII computational graph:
      ```
      Forward:  X ──→ [× W^T] ──→ [+ b] ──→ Y ──→ ...
      
      Backward: ∂L/∂X ←── ∂L/∂W ←── ∂L/∂b ←── ∂L/∂Y
      
      Given ∂L/∂Y (upstream gradient):
      
      ∂L/∂W = (∂L/∂Y)^T · X    (gradient w.r.t. weights)
      ∂L/∂b = Σ_batch ∂L/∂Y    (gradient w.r.t. bias — sum over batch)
      ∂L/∂X = ∂L/∂Y · W        (gradient w.r.t. input — for chain rule to previous layer)
      ```
    - Detailed dimension analysis:
      ```
      ∂L/∂Y: (batch × out_features)
      X:      (batch × in_features)
      W:      (out_features × in_features)
      
      ∂L/∂W = ∂L/∂Y^T · X        → (out × batch) · (batch × in) = (out × in)  ✓ matches W shape
      ∂L/∂b = sum(∂L/∂Y, axis=0)  → (out)  ✓ matches b shape
      ∂L/∂X = ∂L/∂Y · W           → (batch × out) · (out × in) = (batch × in)  ✓ matches X shape
      ```
    - Why ∂L/∂X matters (passes gradient to previous layer)
  - **puzzle.cu**: THREE kernels:
    - `fc_backward_weights(float* grad_output, float* input, float* grad_weights, ...)`
    - `fc_backward_bias(float* grad_output, float* grad_bias, ...)`
    - `fc_backward_input(float* grad_output, float* weights, float* grad_input, ...)`
  - **test_puzzle.cu**: Tests:
    - Test 1: Hardcoded small example (2 samples, 3→2), verify all three gradients
    - Test 2: Numerical gradient check — perturb each weight by ε, compute (L(w+ε)-L(w-ε))/(2ε), compare to analytical gradient
    - Test 3: LeNet FC1 dimensions (batch=8, 256→120)
    - Test 4: Verify gradient shapes match parameter shapes

  **Must NOT do**:
  - Do NOT combine forward and backward into one kernel
  - Do NOT skip the numerical gradient check test — it's the gold standard for verifying backward passes

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: First full backward pass for a parameterized layer — must get the gradient derivation explanation right
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `puzzles/puzzle_03_fc_forward/` — Forward pass that this backward pass corresponds to
  - `puzzles/puzzle_04_relu/` — Chain rule pattern introduced there, applied here

  **WHY Each Reference Matters**:
  - Puzzle 03's forward pass gives the equations being differentiated
  - Numerical gradient checking pattern will be reused in Conv backward puzzles

  **Acceptance Criteria**:
  - [ ] Solution passes 4/4 tests
  - [ ] Numerical gradient check: relative error < 1e-3 for all parameters
  - [ ] README shows dimension analysis proving gradient shapes are correct
  - [ ] All three gradient kernels (weights, bias, input) work independently

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: All gradients verified including numerical check
    Tool: Bash
    Steps:
      1. Build and run puzzle_06_test_solution
      2. Assert: exit code 0, "4/4 tests passed"
      3. Assert: output shows "numerical gradient check: PASS" with relative error < 1e-3
    Evidence: Terminal output

  Scenario: Gradient shapes are correct
    Tool: Bash
    Steps:
      1. Run puzzle_06_test_solution
      2. Assert: grad_weights shape matches weights shape
      3. Assert: grad_bias shape matches bias shape
      4. Assert: grad_input shape matches input shape
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(puzzle-06): add FC backward pass puzzle with numerical gradient checking`
  - Files: `puzzles/puzzle_06_fc_backward/*`

---

### Task 9: Puzzle 07 — Conv2D Forward Pass

- [ ] 9. Puzzle 07 — 2D Convolution Forward Pass

  **What to do**:
  - Create `puzzles/puzzle_07_conv_forward/`
  - **README.md** must include:
    - What convolution does — feature extraction from spatial data
    - ASCII diagram of a single convolution operation:
      ```
      Input (5×5):          Filter (3×3):         Output element:
      ┌─────────────────┐   ┌───────────┐
      │ 1  2  3  4  5   │   │ 1  0  1   │         1×1 + 2×0 + 3×1
      │ 6  7  8  9  10  │   │ 0  1  0   │       + 6×0 + 7×1 + 8×0
      │ 11 12 13 14 15  │ * │ 1  0  1   │       + 11×1 + 12×0 + 13×1
      │ 16 17 18 19 20  │   └───────────┘       = 1+3+7+11+13 = 35
      │ 21 22 23 24 25  │
      └─────────────────┘
      ```
    - Multi-channel convolution diagram:
      ```
      Input: H×W×C_in     Filters: K × (F×F×C_in)     Output: H'×W'×K
      
      For each output pixel (h, w) and filter k:
      out[h][w][k] = Σ_c Σ_fh Σ_fw input[h+fh][w+fw][c] × filter[k][fh][fw][c] + bias[k]
      ```
    - Output dimension formula: `H_out = (H_in - F + 2*pad) / stride + 1`
    - LeNet Conv1 specific dimensions: 28×28×1 → 5×5 kernel, 6 filters → 24×24×6
    - Memory layout: NCHW (batch, channels, height, width)
    - Thread mapping strategy: one thread per output pixel
  - **puzzle.cu**: Kernel `conv2d_forward(float* input, float* filters, float* bias, float* output, int batch, int C_in, int H, int W, int C_out, int F)`
  - **test_puzzle.cu**: Tests:
    - Test 1: 1×1×5×5 input, 1×1×3×3 filter, hardcoded (hand-computable)
    - Test 2: 1×1×28×28 input, 6×1×5×5 filters (LeNet Conv1 shapes)
    - Test 3: 1×6×12×12 input, 16×6×5×5 filters (LeNet Conv2 shapes)
    - Test 4: Batch of 4, LeNet Conv1 dimensions (verify batch processing)
    - Test 5: Verify output dimensions are exactly correct

  **Must NOT do**:
  - Do NOT implement padding (LeNet uses no padding)
  - Do NOT implement stride > 1 (LeNet conv uses stride=1)
  - Do NOT use im2col — use direct nested-loop convolution for educational clarity
  - Do NOT use shared memory optimization

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Convolution is THE core CNN operation — ASCII diagrams must be exceptional, math must be crystal clear
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `puzzles/puzzle_02_matmul/` — Convolution generalizes matmul to spatial data; reference this conceptual connection
  - `common/cuda_utils.h` — CUDA_CHECK + KERNEL_CHECK

  **WHY Each Reference Matters**:
  - Understanding that "convolution is a kind of structured matrix multiplication" bridges from Puzzle 02 knowledge
  - NCHW memory layout must be consistent across ALL conv/pool puzzles

  **Acceptance Criteria**:
  - [ ] Solution passes 5/5 tests
  - [ ] Output dimensions match: (28-5)/1+1=24 for Conv1, (12-5)/1+1=8 for Conv2
  - [ ] README has sliding window ASCII animation
  - [ ] README derives output dimension formula step-by-step

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: LeNet Conv1 dimensions correct
    Tool: Bash
    Steps:
      1. Build and run puzzle_07_test_solution
      2. Assert: "5/5 tests passed"
      3. Assert: Conv1 test (28×28×1 → 24×24×6) produces correct dimensions
    Evidence: Terminal output

  Scenario: Handcoded example verifiable by human
    Tool: Bash
    Steps:
      1. Run puzzle_07_test_solution
      2. Assert: Test 1 output matches hand-computed 35 (or whatever the expected value is)
    Evidence: Terminal output with exact values
  ```

  **Commit**: YES
  - Message: `feat(puzzle-07): add Conv2D forward pass puzzle`
  - Files: `puzzles/puzzle_07_conv_forward/*`

---

### Task 10: Puzzle 08 — Max Pooling Forward + Backward

- [ ] 10. Puzzle 08 — Max Pooling Forward and Backward Pass

  **What to do**:
  - Create `puzzles/puzzle_08_maxpool/`
  - **README.md** must include:
    - What pooling does — spatial downsampling, translation invariance
    - ASCII diagram:
      ```
      Input (4×4):                    Output (2×2):
      ┌─────┬─────┐                  ┌─────┐
      │ 1 3 │ 2 1 │                  │ 3 2 │
      │ 0 2 │ 1 0 │   max pool 2×2  │     │
      ├─────┼─────┤  ──────────────→ │ 4 3 │
      │ 4 1 │ 3 2 │   stride 2      │     │
      │ 0 1 │ 1 0 │                  └─────┘
      └─────┴─────┘
      
      max_indices (for backward):
      ┌─────┐
      │ 1,0 │ ← position of max in each 2×2 window
      │ 2,0 │
      └─────┘
      ```
    - Forward formula: `y[i][j] = max(x[2i:2i+2, 2j:2j+2])`
    - Backward: gradient routing (only max element gets gradient):
      ```
      Forward: gradient only flows through the winning element
      
      ∂L/∂x[m,n] = { ∂L/∂y[i,j]  if x[m,n] was the max in window (i,j)
                    { 0             otherwise
      ```
    - Why we need to save max indices during forward pass
  - **puzzle.cu**: TWO kernels:
    - `maxpool_forward(float* input, float* output, int* max_indices, int batch, int C, int H, int W)`
    - `maxpool_backward(float* grad_output, int* max_indices, float* grad_input, int batch, int C, int H, int W)`
  - **test_puzzle.cu**: Tests:
    - Test 1: 4×4 → 2×2 hardcoded example (verify max values AND indices)
    - Test 2: Backward — verify only max positions receive gradient
    - Test 3: LeNet dimensions — 24×24×6 → 12×12×6 (Pool1) and 8×8×16 → 4×4×16 (Pool2)
    - Test 4: Edge case — all elements in a pool window are equal

  **Must NOT do**:
  - Do NOT implement average pooling (stick with max pool)
  - Do NOT support variable pool sizes (only 2×2 stride 2)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Gradient routing concept is new — "only the max gets the gradient" must be explained with excellent diagrams
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `puzzles/puzzle_04_relu/` — Similar forward+backward structure, but with index tracking
  - `common/test_utils.h` — Test both floating point values and integer indices

  **WHY Each Reference Matters**:
  - Max pooling backward is conceptually similar to ReLU backward (selective gradient routing)
  - Must save max_indices during forward — this is a new pattern not seen in previous puzzles

  **Acceptance Criteria**:
  - [ ] Solution passes 4/4 tests
  - [ ] Forward correctly identifies max values AND their positions
  - [ ] Backward correctly routes gradient ONLY to max positions
  - [ ] README clearly explains max index tracking with diagram

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Forward and backward pass both correct
    Tool: Bash
    Steps:
      1. Build and run puzzle_08_test_solution
      2. Assert: "4/4 tests passed"
      3. Assert: max indices are correct for hardcoded example
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(puzzle-08): add max pooling forward and backward puzzle`
  - Files: `puzzles/puzzle_08_maxpool/*`

---

### Task 11: Puzzle 09 — Conv2D Backward (Weight Gradients)

- [ ] 11. Puzzle 09 — Conv2D Backward: Weight Gradients

  **What to do**:
  - Create `puzzles/puzzle_09_conv_backward_weights/`
  - **README.md** must include:
    - Recap of Conv2D forward from Puzzle 07
    - Derivation of weight gradient:
      ```
      Given: out[b][k][h][w] = Σ_c Σ_fh Σ_fw in[b][c][h+fh][w+fw] × W[k][c][fh][fw] + bias[k]
      
      ∂L/∂W[k][c][fh][fw] = Σ_b Σ_h Σ_w ∂L/∂out[b][k][h][w] × in[b][c][h+fh][w+fw]
      
      This is a CORRELATION between the input and the upstream gradient!
      ```
    - ASCII diagram showing which input elements contribute to each weight gradient:
      ```
      For weight W[k][c][0][0]:
      
      ∂L/∂out:        Input channel c:
      ┌─────────┐     ┌─────────────┐
      │ g00 g01 │     │ x00 x01 x02 │
      │ g10 g11 │     │ x10 x11 x12 │
      └─────────┘     │ x20 x21 x22 │
                      └─────────────┘
      
      ∂L/∂W[k][c][0][0] = g00×x00 + g01×x01 + g10×x10 + g11×x11
      ```
    - Bias gradient: `∂L/∂bias[k] = Σ_b Σ_h Σ_w ∂L/∂out[b][k][h][w]`
  - **puzzle.cu**: TWO kernels:
    - `conv2d_backward_weights(float* grad_output, float* input, float* grad_weights, ...)`
    - `conv2d_backward_bias(float* grad_output, float* grad_bias, ...)`
  - **test_puzzle.cu**: Tests:
    - Test 1: Tiny example (3×3 input, 2×2 filter) — hand-verifiable weight gradients
    - Test 2: Numerical gradient check (perturb weights, measure loss change)
    - Test 3: LeNet Conv1 dimensions — verify gradient shapes
    - Test 4: LeNet Conv2 dimensions — verify gradient shapes

  **Must NOT do**:
  - Do NOT combine weight gradient and input gradient in one puzzle (input gradient is Puzzle 10)
  - Do NOT skip numerical gradient verification

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Conv backward is the hardest concept in the puzzle set — derivation must be meticulous
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `puzzles/puzzle_07_conv_forward/` — Forward pass being differentiated
  - `puzzles/puzzle_06_fc_backward/` — Numerical gradient check pattern to reuse

  **WHY Each Reference Matters**:
  - Must understand forward convolution to derive weight gradients
  - Numerical gradient check from Puzzle 06 is the same technique applied to conv weights

  **Acceptance Criteria**:
  - [ ] Solution passes 4/4 tests
  - [ ] Numerical gradient check: relative error < 1e-3
  - [ ] README shows correlation interpretation of weight gradient
  - [ ] Gradient shapes match weight shapes (K×C×F×F)

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Weight gradients verified numerically
    Tool: Bash
    Steps:
      1. Build and run puzzle_09_test_solution
      2. Assert: "4/4 tests passed"
      3. Assert: numerical gradient relative error < 1e-3
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(puzzle-09): add Conv2D backward weight gradients puzzle`
  - Files: `puzzles/puzzle_09_conv_backward_weights/*`

---

### Task 12: Puzzle 10 — Conv2D Backward (Input Gradients)

- [ ] 12. Puzzle 10 — Conv2D Backward: Input Gradients

  **What to do**:
  - Create `puzzles/puzzle_10_conv_backward_input/`
  - **README.md** must include:
    - Why we need input gradients (to pass gradient to previous layer in the chain)
    - The "full convolution" trick:
      ```
      ∂L/∂input[b][c][h][w] = Σ_k Σ_fh Σ_fw ∂L/∂out[b][k][h-fh][w-fw] × W[k][c][fh][fw]
      
      This is equivalent to convolving the upstream gradient with 
      a ROTATED (180°) version of the filter, with FULL padding!
      
      Original filter:     Rotated 180°:
      ┌───────┐            ┌───────┐
      │ a b c │            │ i h g │
      │ d e f │     →      │ f e d │
      │ g h i │            │ c b a │
      └───────┘            └───────┘
      ```
    - ASCII diagram showing padding + rotation:
      ```
      Upstream gradient (2×2):      Rotated filter (3×3):
      
      Padded ∂L/∂out:               ┌───────┐
      ┌───────────────────┐         │ w22 w21 w20 │
      │ 0   0   0   0     │         │ w12 w11 w10 │
      │ 0  g00 g01  0     │    *    │ w02 w01 w00 │
      │ 0  g10 g11  0     │         └───────┘
      │ 0   0   0   0     │
      └───────────────────┘
      ```
    - Dimension verification: output shape = input shape ✓
  - **puzzle.cu**: Kernel `conv2d_backward_input(float* grad_output, float* weights, float* grad_input, ...)`
  - **test_puzzle.cu**: Tests:
    - Test 1: Tiny example (same as Puzzle 09) — hand-verifiable
    - Test 2: Numerical gradient check (perturb input, measure loss change)
    - Test 3: LeNet Conv1 dimensions — grad_input shape = 28×28×1
    - Test 4: LeNet Conv2 dimensions — grad_input shape = 12×12×6

  **Must NOT do**:
  - Do NOT implement as actual full-padding convolution (use direct formula for educational clarity)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: The 180° rotation insight is conceptually deep — documentation must build intuition, not just show formulas
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `puzzles/puzzle_09_conv_backward_weights/` — Sister puzzle; together they complete Conv backward
  - `puzzles/puzzle_07_conv_forward/` — Forward pass being differentiated

  **Acceptance Criteria**:
  - [ ] Solution passes 4/4 tests
  - [ ] Numerical gradient check: relative error < 1e-3
  - [ ] README explains 180° rotation trick with diagram
  - [ ] Gradient shape matches input shape

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Input gradients verified numerically
    Tool: Bash
    Steps:
      1. Build and run puzzle_10_test_solution
      2. Assert: "4/4 tests passed"
      3. Assert: numerical gradient relative error < 1e-3
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(puzzle-10): add Conv2D backward input gradients puzzle`
  - Files: `puzzles/puzzle_10_conv_backward_input/*`

---

### Task 13: Puzzle 11 — SGD Optimizer + Weight Update

- [ ] 13. Puzzle 11 — SGD Optimizer and Weight Update Rule

  **What to do**:
  - Create `puzzles/puzzle_11_sgd/`
  - **README.md** must include:
    - What optimization is — adjusting weights to minimize loss
    - Gradient descent intuition with ASCII:
      ```
      Loss
       │  ╲
       │   ╲         ╱
       │    ╲       ╱
       │     ╲     ╱
       │      ╲   ╱
       │       ╲ ╱  ← We want to find this minimum
       │        •
       └──────────────── Weight value
       
       Update rule: w_new = w_old - learning_rate × ∂L/∂w
       
       Move in the OPPOSITE direction of the gradient!
      ```
    - Mini-batch SGD formula:
      ```
      For each parameter θ:
        θ = θ - η × (1/batch_size) × Σ ∂L/∂θ
      
      where η = learning rate (e.g., 0.01)
      ```
    - Why learning rate matters (too high = diverge, too low = slow)
    - Gradient zeroing between batches
  - **puzzle.cu**: TWO kernels:
    - `sgd_update(float* params, float* gradients, float lr, int n)` — applies w -= lr * grad
    - `zero_gradients(float* gradients, int n)` — sets all gradients to 0
  - **test_puzzle.cu**: Tests:
    - Test 1: Hardcoded weight update — verify w_new = w_old - lr * grad
    - Test 2: Multiple steps — verify loss decreases on a simple function (quadratic)
    - Test 3: Verify gradient zeroing works
    - Test 4: LeNet-sized parameter count (verify all ~44K parameters update correctly)

  **Must NOT do**:
  - Do NOT implement momentum, Adam, or other optimizers
  - Do NOT implement learning rate scheduling

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Connects all backward pass puzzles to actual learning — must explain the "why" beautifully
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `puzzles/puzzle_06_fc_backward/` — Gradients computed there are consumed here
  - All backward puzzles (04, 06, 08, 09, 10) — produce the gradients that SGD uses

  **Acceptance Criteria**:
  - [ ] Solution passes 4/4 tests
  - [ ] Weight update is exactly: w -= lr * grad
  - [ ] Gradient zeroing sets all values to 0.0
  - [ ] README has loss landscape ASCII diagram

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: SGD update and gradient zeroing verified
    Tool: Bash
    Steps:
      1. Build and run puzzle_11_test_solution
      2. Assert: "4/4 tests passed"
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(puzzle-11): add SGD optimizer and weight update puzzle`
  - Files: `puzzles/puzzle_11_sgd/*`

---

### Task 14: Puzzle 12 — LeNet Forward Pass (Full Network)

- [ ] 14. Puzzle 12 — Complete LeNet-5 Forward Pass

  **What to do**:
  - Create `puzzles/puzzle_12_lenet_forward/`
  - **README.md** must include:
    - Full architecture diagram (the big ASCII diagram from the architecture reference section)
    - Complete data flow with dimensions at every stage:
      ```
      Input:  (batch × 1 × 28 × 28)     [MNIST image]
        ↓ Conv1: 6 filters of 5×5
      C1:     (batch × 6 × 24 × 24)     [24 = (28-5)/1 + 1]
        ↓ ReLU
      A1:     (batch × 6 × 24 × 24)
        ↓ MaxPool 2×2
      S2:     (batch × 6 × 12 × 12)     [12 = 24/2]
        ↓ Conv2: 16 filters of 5×5
      C3:     (batch × 16 × 8 × 8)      [8 = (12-5)/1 + 1]
        ↓ ReLU
      A3:     (batch × 16 × 8 × 8)
        ↓ MaxPool 2×2
      S4:     (batch × 16 × 4 × 4)      [4 = 8/2]
        ↓ Flatten
      F:      (batch × 256)              [256 = 16×4×4]
        ↓ FC1: 256 → 120
      FC5:    (batch × 120)
        ↓ ReLU
      A5:     (batch × 120)
        ↓ FC2: 120 → 84
      FC6:    (batch × 84)
        ↓ ReLU
      A6:     (batch × 84)
        ↓ FC3: 84 → 10
      FC7:    (batch × 10)
        ↓ Softmax
      Output: (batch × 10)              [class probabilities]
      ```
    - Total parameter count breakdown
    - Memory allocation plan (how much device memory needed for all layers)
    - How to chain individual puzzles' kernels together
  - **puzzle.cu**: One function `lenet_forward(...)` that:
    - Allocates all intermediate buffers
    - Calls each layer's kernel in sequence (from Puzzles 03-08)
    - Includes flatten operation (reshape, no kernel needed — just pointer arithmetic)
    - Returns final logits
  - **Provides reference implementations** of all individual kernels (from previous puzzle solutions) so this puzzle is independently compilable
  - **test_puzzle.cu**: Tests:
    - Test 1: Single image forward pass — verify output is 10 probabilities summing to 1.0
    - Test 2: Batch of 8 images — verify batch processing works
    - Test 3: With deterministic weights (seed=42) — verify exact output probabilities
    - Test 4: Verify intermediate dimensions at each layer boundary

  **Must NOT do**:
  - Do NOT include backward pass (that's Puzzle 13)
  - Do NOT include training logic

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Integration puzzle tying all forward components together — must be meticulously documented
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 16 (Puzzle 14 depends on forward pass)
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `puzzles/puzzle_03_fc_forward/solution.cu` — FC forward kernel reference
  - `puzzles/puzzle_04_relu/solution.cu` — ReLU kernel reference
  - `puzzles/puzzle_05_softmax_ce/solution.cu` — Softmax kernel reference
  - `puzzles/puzzle_07_conv_forward/solution.cu` — Conv2D forward kernel reference
  - `puzzles/puzzle_08_maxpool/solution.cu` — MaxPool forward kernel reference

  **WHY Each Reference Matters**:
  - This puzzle chains ALL previous forward kernels — must include their reference solutions
  - Student's job is the ORCHESTRATION, not re-implementing individual layers
  - Each reference solution is bundled so this puzzle is independently testable

  **Acceptance Criteria**:
  - [ ] Solution passes 4/4 tests
  - [ ] Output is 10 probabilities summing to 1.0 (within 1e-5)
  - [ ] Intermediate dimensions correct at every layer
  - [ ] README has complete architecture diagram with dimensions

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Full forward pass produces valid probabilities
    Tool: Bash
    Steps:
      1. Build and run puzzle_12_test_solution
      2. Assert: "4/4 tests passed"
      3. Assert: output probabilities sum to 1.0 within 1e-5
      4. Assert: all probabilities are between 0.0 and 1.0
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(puzzle-12): add complete LeNet forward pass puzzle`
  - Files: `puzzles/puzzle_12_lenet_forward/*`

---

### Task 15: Puzzle 13 — LeNet Backward Pass (Full Backprop)

- [ ] 15. Puzzle 13 — Complete LeNet-5 Backward Pass

  **What to do**:
  - Create `puzzles/puzzle_13_lenet_backward/`
  - **README.md** must include:
    - Full backward flow diagram (reverse of forward):
      ```
      Forward: Input → Conv1 → ReLU → Pool1 → Conv2 → ReLU → Pool2 → FC1 → ReLU → FC2 → ReLU → FC3 → Softmax → Loss
      
      Backward (reverse order):
      ∂L/∂logits ← Softmax+CE backward (∂L/∂z = p - y)
          ↓
      ∂L/∂FC3_in ← FC3 backward (grad_input = grad_out × W)
          ↓
      ∂L/∂A6 ← ReLU backward (mask out negatives)
          ↓
      ... (continues for each layer in reverse)
      ```
    - Key insight: backward pass is just forward pass in reverse, applying chain rule at each step
    - Which intermediate values need to be SAVED from forward pass:
      ```
      Must save during forward:
      ✓ All layer inputs (for weight gradient computation)
      ✓ ReLU inputs (for backward masking)
      ✓ Max pool indices (for gradient routing)
      ✓ Softmax probabilities (for loss gradient)
      ```
    - Memory overhead of storing intermediates
  - **puzzle.cu**: `lenet_backward(...)` that:
    - Takes loss gradient and all saved forward intermediates
    - Calls each backward kernel in reverse order
    - Computes gradients for ALL parameters (conv weights, conv biases, FC weights, FC biases)
  - **Provides reference implementations** of all backward kernels from previous puzzles
  - **test_puzzle.cu**: Tests:
    - Test 1: Full forward+backward with deterministic weights — verify all gradient shapes
    - Test 2: Numerical gradient check on a SUBSET of parameters (5 random weights per layer)
    - Test 3: Verify gradient flow — loss should decrease after one SGD step
    - Test 4: Verify gradient magnitudes are reasonable (not exploding/vanishing)

  **Must NOT do**:
  - Do NOT include SGD update (that's Puzzle 11)
  - Do NOT include training loop (that's Puzzle 15)
  - Do NOT numerical-gradient-check ALL parameters (too slow) — sample a subset

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Most complex puzzle — chains all backward kernels in correct order with correct intermediate data
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 17 (Training loop depends on backward pass)
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `puzzles/puzzle_12_lenet_forward/solution.cu` — Forward pass that generates intermediates
  - `puzzles/puzzle_06_fc_backward/solution.cu` — FC backward reference
  - `puzzles/puzzle_04_relu/solution.cu` — ReLU backward reference
  - `puzzles/puzzle_08_maxpool/solution.cu` — MaxPool backward reference
  - `puzzles/puzzle_09_conv_backward_weights/solution.cu` — Conv weight gradient reference
  - `puzzles/puzzle_10_conv_backward_input/solution.cu` — Conv input gradient reference
  - `puzzles/puzzle_05_softmax_ce/solution.cu` — Softmax+CE backward reference

  **WHY Each Reference Matters**:
  - All backward kernels are bundled as reference so this puzzle is independently testable
  - Student's job is the correct ORDERING and DATA FLOW, not re-implementing backward kernels

  **Acceptance Criteria**:
  - [ ] Solution passes 4/4 tests
  - [ ] All gradient shapes match parameter shapes
  - [ ] Sampled numerical gradient check: relative error < 1e-2 (relaxed tolerance for deep chain)
  - [ ] One SGD step reduces loss (sanity check)

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Full backward pass verified
    Tool: Bash
    Steps:
      1. Build and run puzzle_13_test_solution
      2. Assert: "4/4 tests passed"
      3. Assert: gradient shapes reported in output match expected
      4. Assert: numerical gradient check passes
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(puzzle-13): add complete LeNet backward pass puzzle`
  - Files: `puzzles/puzzle_13_lenet_backward/*`

---

### Task 16: Puzzle 14 — MNIST Data Pipeline

- [ ] 16. Puzzle 14 — MNIST Data Loading and Batching

  **What to do**:
  - Create `puzzles/puzzle_14_mnist_pipeline/`
  - **README.md** must include:
    - MNIST dataset structure (60K train, 10K test, 28×28 grayscale)
    - IDX file format explanation:
      ```
      IDX3 Format (images):
      ┌─────────────────────────────────────┐
      │ Magic: 0x00000803 (4 bytes, big-endian) │
      │ Count: N          (4 bytes, big-endian) │
      │ Rows:  28         (4 bytes, big-endian) │
      │ Cols:  28         (4 bytes, big-endian) │
      │ Pixel data: N × 28 × 28 bytes (uint8)  │
      └─────────────────────────────────────┘
      ```
    - Big-endian to little-endian conversion:
      ```
      Network byte order (big-endian):    Host byte order (little-endian):
      [0x00][0x00][0x08][0x03]    →     [0x03][0x08][0x00][0x00]
       MSB               LSB             LSB               MSB
      
      swap_endian(val):
        return ((val>>24)&0xff) | ((val>>8)&0xff00) |
               ((val<<8)&0xff0000) | ((val<<24)&0xff000000);
      ```
    - Normalization: pixel values [0,255] → [0.0, 1.0]
    - Mini-batch construction and shuffling
    - Where to download MNIST (multiple mirrors)
  - **puzzle.cu**: Functions to implement:
    - `load_mnist_images(const char* path, float** images, int* count)` — parse IDX3
    - `load_mnist_labels(const char* path, int** labels, int* count)` — parse IDX1
    - `create_batches(float* images, int* labels, int n, int batch_size, ...)` — split into batches
    - `shuffle_data(float* images, int* labels, int n, int seed)` — deterministic shuffle
  - **test_puzzle.cu**: Tests:
    - Test 1: Load bundled mini-subset (100 images) — verify count, dimensions
    - Test 2: Byte swap function — verify known values
    - Test 3: Normalization — verify output range [0.0, 1.0]
    - Test 4: Batch creation — verify batch count and sizes (handle last partial batch)
    - Test 5: Shuffle determinism — same seed → same order
  - **Bundle 100 MNIST images + labels** as binary files in `data/` for testing without full download
  - **Provide download instructions** for full MNIST dataset (with multiple mirror URLs)

  **Must NOT do**:
  - Do NOT require full MNIST download for tests to pass
  - Do NOT implement data augmentation
  - Do NOT implement one-hot encoding on GPU (do it on host)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: File I/O, byte-swapping, and data pipeline — lots of edge cases to handle correctly
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (partially — doesn't depend on puzzle content, only common utils)
  - **Parallel Group**: Wave 3 (can start as soon as Task 2 is done)
  - **Blocks**: Task 18 (final integration needs data)
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `common/mnist_loader.h` — Utility header that this puzzle's solution will implement
  - `common/test_data.h` — Synthetic data generator for fallback testing

  **External References**:
  - MNIST format specification: http://yann.lecun.com/exdb/mnist/
  - Mirror URLs: https://ossci-datasets.s3.amazonaws.com/mnist/ (Amazon S3 mirror)

  **WHY Each Reference Matters**:
  - `mnist_loader.h` defines the API; this puzzle implements it
  - Bundled 100-image subset allows tests to run without internet

  **Acceptance Criteria**:
  - [ ] Solution passes 5/5 tests using bundled mini-dataset
  - [ ] Byte swap correctly converts big-endian to little-endian
  - [ ] Images normalized to [0.0, 1.0] range
  - [ ] Data files in `data/` directory (100 images + labels)
  - [ ] README includes download instructions with multiple mirror URLs

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Load and verify bundled mini-MNIST
    Tool: Bash
    Steps:
      1. Build and run puzzle_14_test_solution
      2. Assert: "5/5 tests passed"
      3. Assert: loaded 100 images of 28×28 dimensions
      4. Assert: pixel values in [0.0, 1.0] range
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(puzzle-14): add MNIST data pipeline puzzle with bundled test data`
  - Files: `puzzles/puzzle_14_mnist_pipeline/*, data/*`

---

### Task 17: Puzzle 15 — Full Training Loop

- [ ] 17. Puzzle 15 — Complete LeNet Training Loop

  **What to do**:
  - Create `puzzles/puzzle_15_training_loop/`
  - **README.md** must include:
    - The complete training algorithm:
      ```
      ┌──────────────────────────────────────────┐
      │           Training Loop                    │
      │                                            │
      │  for epoch = 1 to N:                       │
      │    shuffle(training_data)                  │
      │    for each batch:                         │
      │      ┌─────────────────────────────────┐   │
      │      │ 1. Forward pass (Puzzle 12)     │   │
      │      │    Input → LeNet → Predictions  │   │
      │      │                                 │   │
      │      │ 2. Compute loss (Puzzle 05)     │   │
      │      │    L = CrossEntropy(pred, label) │   │
      │      │                                 │   │
      │      │ 3. Backward pass (Puzzle 13)    │   │
      │      │    Compute all gradients        │   │
      │      │                                 │   │
      │      │ 4. Update weights (Puzzle 11)   │   │
      │      │    w -= lr × grad               │   │
      │      │                                 │   │
      │      │ 5. Zero gradients               │   │
      │      │    Reset all grads to 0         │   │
      │      └─────────────────────────────────┘   │
      │    end batch                               │
      │    Print: epoch, avg_loss, accuracy        │
      │  end epoch                                 │
      └──────────────────────────────────────────┘
      ```
    - Hyperparameters:
      ```
      Learning rate: 0.01
      Batch size: 32
      Epochs: 10
      Weight init: Xavier uniform (seed=42)
      ```
    - Xavier initialization formula:
      ```
      W ~ Uniform(-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out)))
      ```
    - How to compute accuracy:
      ```
      accuracy = count(argmax(predictions) == labels) / total_samples
      ```
    - Expected training curve (approximate):
      ```
      Epoch  | Train Loss | Train Acc | Test Acc
      -------|------------|-----------|--------
        1    |   ~2.0     |   ~50%    |  ~50%
        3    |   ~0.5     |   ~85%    |  ~85%
        5    |   ~0.2     |   ~93%    |  ~93%
       10    |   ~0.1     |   ~97%    |  ~95%+
      ```
  - **puzzle.cu**: Functions:
    - `xavier_init(float* weights, int fan_in, int fan_out, int seed)` — weight initialization
    - `train_epoch(LeNet* net, float* images, int* labels, int n, int batch_size, float lr)` — one epoch
    - `evaluate(LeNet* net, float* images, int* labels, int n)` — compute accuracy
    - `training_loop(...)` — full loop calling train_epoch + evaluate
  - **Provides ALL reference solutions** from all 14 previous puzzles bundled together
  - **test_puzzle.cu**: Tests:
    - Test 1: Xavier init — verify weight distribution is within expected range
    - Test 2: Single batch training step — verify loss decreases
    - Test 3: Overfit on 10 samples — verify near-100% accuracy on those 10 (memorization test)
    - Test 4: If full MNIST available: 3 epochs → accuracy > 90% on test set
    - Test 5: If only mini-dataset: overfit on 100 samples → accuracy > 95%

  **Must NOT do**:
  - Do NOT implement learning rate decay
  - Do NOT implement early stopping
  - Do NOT implement checkpoint saving
  - Do NOT require full MNIST for tests to pass (use mini-dataset fallback)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Capstone puzzle — must tie EVERYTHING together with clear documentation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (partially)
  - **Parallel Group**: Wave 3 (after Task 2)
  - **Blocks**: Task 18
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - ALL previous puzzle solutions — bundled as reference implementations
  - `puzzles/puzzle_12_lenet_forward/solution.cu` — Forward pass
  - `puzzles/puzzle_13_lenet_backward/solution.cu` — Backward pass
  - `puzzles/puzzle_11_sgd/solution.cu` — Weight update
  - `puzzles/puzzle_14_mnist_pipeline/solution.cu` — Data loading
  - `puzzles/puzzle_05_softmax_ce/solution.cu` — Loss computation

  **WHY Each Reference Matters**:
  - Every previous puzzle's solution is a component of the training loop
  - This is the culmination — student assembles all pieces

  **Acceptance Criteria**:
  - [ ] Solution passes 5/5 tests
  - [ ] Xavier init produces weights in expected range
  - [ ] Loss decreases monotonically on overfit test (10 samples)
  - [ ] Overfit test achieves >95% accuracy on 10 memorized samples
  - [ ] README has complete training algorithm diagram

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Overfit on small dataset (primary test)
    Tool: Bash
    Preconditions: Bundled mini-dataset available in data/
    Steps:
      1. Build: cmake --build build --target puzzle_15_test_solution
      2. Run: ./build/puzzle_15_test_solution
      3. Assert: exit code 0
      4. Assert: "5/5 tests passed"
      5. Assert: output shows loss decreasing across steps
      6. Assert: overfit accuracy > 95%
    Expected Result: Training loop works end-to-end
    Evidence: Terminal output with loss/accuracy per step

  Scenario: Xavier initialization distribution
    Tool: Bash
    Steps:
      1. Run puzzle_15_test_solution
      2. Assert: weight init test passes
      3. Assert: weights are within ±sqrt(6/(fan_in+fan_out))
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(puzzle-15): add full training loop puzzle — LeNet on MNIST`
  - Files: `puzzles/puzzle_15_training_loop/*`

---

### Task 18: Final Integration Testing + Top-Level README

- [ ] 18. Final Integration — Build-all test, top-level README, puzzle index

  **What to do**:
  - Update top-level `README.md` with:
    - Project title and description
    - Prerequisites (CUDA toolkit, CMake 3.18+, C++17 compiler)
    - Quick start:
      ```
      git clone ...
      cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
      cmake --build build
      ```
    - Puzzle index table:
      ```
      | # | Puzzle | Concept | Difficulty |
      |---|--------|---------|------------|
      | 01 | Vector Addition | CUDA basics, thread indexing | ⭐ |
      | 02 | Matrix Multiply | GEMM, row-major layout | ⭐ |
      | 03 | FC Forward | Neural network layer | ⭐⭐ |
      | 04 | ReLU | Activation, chain rule intro | ⭐⭐ |
      | 05 | Softmax+CE | Loss function, numerical stability | ⭐⭐⭐ |
      | 06 | FC Backward | Backpropagation | ⭐⭐⭐ |
      | 07 | Conv2D Forward | Convolution, feature maps | ⭐⭐⭐ |
      | 08 | Max Pooling | Downsampling, gradient routing | ⭐⭐⭐ |
      | 09 | Conv Backward (W) | Weight gradients | ⭐⭐⭐⭐ |
      | 10 | Conv Backward (X) | Input gradients, rotation trick | ⭐⭐⭐⭐ |
      | 11 | SGD Optimizer | Weight updates, learning | ⭐⭐ |
      | 12 | LeNet Forward | Full network assembly | ⭐⭐⭐⭐ |
      | 13 | LeNet Backward | Full backpropagation | ⭐⭐⭐⭐⭐ |
      | 14 | MNIST Pipeline | Data loading, batching | ⭐⭐ |
      | 15 | Training Loop | End-to-end training | ⭐⭐⭐⭐⭐ |
      ```
    - How to run individual puzzle tests
    - How to run all tests at once
    - MNIST download instructions
    - Architecture reference diagram
  - Create `puzzles/CMakeLists.txt` with all 15 puzzle subdirectories
  - Verify full build: `cmake -B build && cmake --build build` — zero errors
  - Run ALL puzzle tests and verify all pass
  - Create a `run_all_tests.cmake` or CTest configuration that runs all puzzle tests

  **Must NOT do**:
  - Do NOT add CI configuration (out of scope)
  - Do NOT create additional puzzles beyond 15

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration task requiring careful verification of all 15 puzzles together
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final)
  - **Blocks**: None (this is the final task)
  - **Blocked By**: Tasks 3-17 (all puzzles must be complete)

  **References**:

  **Pattern References**:
  - All 15 puzzle directories — verify they follow consistent structure
  - `common/` — verify all utilities are properly linked
  - Root `CMakeLists.txt` — verify all subdirectories are included

  **Acceptance Criteria**:
  - [ ] `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build` exits with code 0
  - [ ] `ctest --test-dir build --output-on-failure` — ALL 15 puzzle tests pass
  - [ ] Top-level README contains puzzle index table
  - [ ] Top-level README contains build instructions
  - [ ] Top-level README contains architecture diagram
  - [ ] Each puzzle is independently buildable via its CMake target

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Full project builds from clean
    Tool: Bash
    Preconditions: No previous build directory
    Steps:
      1. Run: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
      2. Assert: exit code 0
      3. Run: cmake --build build
      4. Assert: exit code 0, no "error" in output
    Expected Result: Clean build succeeds
    Evidence: Terminal output

  Scenario: All puzzle tests pass
    Tool: Bash
    Preconditions: Build complete
    Steps:
      1. Run: ctest --test-dir build --output-on-failure
      2. Assert: exit code 0
      3. Assert: output shows "15 tests passed" (or similar)
      4. Assert: no test failures
    Expected Result: All 15 puzzle tests pass
    Evidence: CTest output captured

  Scenario: README quality check
    Tool: Bash
    Preconditions: README.md exists
    Steps:
      1. Read README.md
      2. Assert: contains puzzle index table with all 15 entries
      3. Assert: contains cmake build instructions
      4. Assert: contains MNIST download instructions
      5. Assert: contains architecture diagram
    Expected Result: README is comprehensive and complete
    Evidence: File content
  ```

  **Commit**: YES
  - Message: `feat: add top-level README, puzzle index, and integration test configuration`
  - Files: `README.md, puzzles/CMakeLists.txt`
  - Pre-commit: `cmake -B build && cmake --build build && ctest --test-dir build`

---

## Commit Strategy

| After Task | Message | Key Files | Verification |
|------------|---------|-----------|--------------|
| 1 | `feat(infra): initialize project with CMake build system` | CMakeLists.txt, .gitignore | `cmake -B build` |
| 2 | `feat(common): add CUDA utils, test framework, MNIST loader` | common/*.h | `./build/test_common` |
| 3 | `feat(puzzle-01): add vector addition puzzle` | puzzles/puzzle_01_*/* | `./build/puzzle_01_test_solution` |
| 4 | `feat(puzzle-02): add matrix multiplication puzzle` | puzzles/puzzle_02_*/* | `./build/puzzle_02_test_solution` |
| 5 | `feat(puzzle-03): add FC forward pass puzzle` | puzzles/puzzle_03_*/* | `./build/puzzle_03_test_solution` |
| 6 | `feat(puzzle-04): add ReLU forward+backward puzzle` | puzzles/puzzle_04_*/* | `./build/puzzle_04_test_solution` |
| 7 | `feat(puzzle-05): add softmax+cross-entropy puzzle` | puzzles/puzzle_05_*/* | `./build/puzzle_05_test_solution` |
| 8 | `feat(puzzle-06): add FC backward pass puzzle` | puzzles/puzzle_06_*/* | `./build/puzzle_06_test_solution` |
| 9 | `feat(puzzle-07): add Conv2D forward pass puzzle` | puzzles/puzzle_07_*/* | `./build/puzzle_07_test_solution` |
| 10 | `feat(puzzle-08): add max pooling puzzle` | puzzles/puzzle_08_*/* | `./build/puzzle_08_test_solution` |
| 11 | `feat(puzzle-09): add Conv2D backward weights puzzle` | puzzles/puzzle_09_*/* | `./build/puzzle_09_test_solution` |
| 12 | `feat(puzzle-10): add Conv2D backward input puzzle` | puzzles/puzzle_10_*/* | `./build/puzzle_10_test_solution` |
| 13 | `feat(puzzle-11): add SGD optimizer puzzle` | puzzles/puzzle_11_*/* | `./build/puzzle_11_test_solution` |
| 14 | `feat(puzzle-12): add LeNet forward pass puzzle` | puzzles/puzzle_12_*/* | `./build/puzzle_12_test_solution` |
| 15 | `feat(puzzle-13): add LeNet backward pass puzzle` | puzzles/puzzle_13_*/* | `./build/puzzle_13_test_solution` |
| 16 | `feat(puzzle-14): add MNIST data pipeline puzzle` | puzzles/puzzle_14_*/* | `./build/puzzle_14_test_solution` |
| 17 | `feat(puzzle-15): add full training loop puzzle` | puzzles/puzzle_15_*/* | `./build/puzzle_15_test_solution` |
| 18 | `feat: add top-level README and integration tests` | README.md | `ctest --test-dir build` |

---

## Success Criteria

### Verification Commands
```bash
# Full build from clean
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build
# Expected: exit code 0, no errors

# Run all tests
ctest --test-dir build --output-on-failure
# Expected: 15/15 tests passed (+ test_common)

# Individual puzzle test
./build/puzzle_07_test_solution
# Expected: "N/N tests passed", exit code 0

# Training convergence (with full MNIST)
./build/puzzle_15_test_solution --full-mnist
# Expected: >95% test accuracy after 10 epochs
```

### Final Checklist
- [ ] All "Must Have" items present in every puzzle
- [ ] All "Must NOT Have" items absent (no shared memory opt, no cuDNN, etc.)
- [ ] All 15 puzzle READMEs have ASCII diagrams
- [ ] All 15 puzzle READMEs have math formulas
- [ ] All 15 puzzle test harnesses pass with reference solutions
- [ ] All puzzle.cu files have clear `// TODO:` markers
- [ ] Common utilities are shared (not duplicated per puzzle)
- [ ] Build system works on Windows with MSVC + CUDA
- [ ] Deterministic results with seed=42 across all puzzles
- [ ] No puzzle requires internet access for basic testing (bundled test data)
