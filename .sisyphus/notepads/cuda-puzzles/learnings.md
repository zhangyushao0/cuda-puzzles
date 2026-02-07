## Puzzle 09 - Conv2D Backward Weights

### Learnings
- Numerical gradient checks for conv layers suffer from float32 catastrophic cancellation when summing many output values (loss_plus - loss_minus). Using double-precision accumulation for the loss values fixes this.
- Smaller problem dimensions (batch=1, C_in=1, H=5, W=5) work better for numerical gradient checks to keep error manageable.
- The epsilon=1e-3 with rel_tol=1e-3 works well when loss accumulation uses double precision.
- The existing puzzles use NCHW layout consistently, flat 1D thread indexing, and the pattern of `#include "puzzle.cu"` / `#include "solution.cu"` via `USE_SOLUTION` define.
- Windows CUDA build generates many C4819 warnings about code page — these are harmless.
- Executables must be run with full path and quotes on Windows bash.

### Conventions
- File names: puzzle.cu, solution.cu, test_puzzle.cu, CMakeLists.txt, README.md
- Test macro: TEST_CASE(name), main() returns RUN_ALL_TESTS()
- GPU helper function pattern: allocate, copy H->D, launch, sync, copy D->H, free
- fill_random() with unique seeds per test for reproducibility
- CPU reference functions for verification
