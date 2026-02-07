# Architectural Decisions

## Build System
- CMake 3.18+ with native CUDA support
- CMAKE_CUDA_ARCHITECTURES=native for auto-detection
- Header-only common library (INTERFACE target)

## Educational Approach
- Direct nested-loop convolution (NOT im2col - more educational)
- Naive one-thread-per-output parallelism (no shared memory optimizations)
- Numerical gradient checking for all backward passes
- Bundled mini-MNIST (100 images) so tests work without internet

## LeNet Architecture
- Input: 28×28×1
- Conv1: 5×5, 6 filters → 24×24×6
- Pool1: 2×2 stride 2 → 12×12×6
- Conv2: 5×5, 16 filters → 8×8×16
- Pool2: 2×2 stride 2 → 4×4×16
- Flatten → 256
- FC1: 256→120, ReLU
- FC2: 120→84, ReLU
- FC3: 84→10, Softmax
- Total params: ~44,426


## Task 18: Integration Testing Strategy (Feb 7, 2026)

**CTest Integration Choice:**
- Added `add_test()` to all puzzle CMakeLists.txt files
- Only register `*_test_solution` targets (not `*_test` targets)
- Rationale: Students work on puzzle.cu, but CTest validates solutions work correctly

**README Content Strategy:**
- Comprehensive documentation over minimalism
- Target audience: CUDA beginners with C++ knowledge
- Include practical troubleshooting section (learned from Windows quirks)
- Provide multiple MNIST download options (official + S3 mirror)

**Multi-Config Generator Handling:**
- Windows Visual Studio generates Debug/Release folders
- CTest requires `-C Debug` flag for VS multi-config generators
- Updated README with Windows-specific executable paths (Debug/ prefix)
- Linux single-config generators don't need configuration flag

**Documentation Completeness Philosophy:**
- Every puzzle listed with difficulty rating (⭐ to ⭐⭐⭐⭐⭐)
- ASCII art architecture diagram (text-based, works in any terminal)
- Copy-paste ready commands (no placeholders like <repo-url>)
- Realistic test output examples (actual CTest output format)
