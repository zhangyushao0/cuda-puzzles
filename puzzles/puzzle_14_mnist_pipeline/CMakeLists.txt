# Puzzle 14: MNIST Data Pipeline
#
# Targets:
#   puzzle_14_test          - build with student's puzzle.cu
#   puzzle_14_test_solution - build with reference solution.cu

# Path to bundled test data
set(PUZZLE_14_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

# Test with student's implementation
add_executable(puzzle_14_test test_puzzle.cu)
target_link_libraries(puzzle_14_test PRIVATE cuda_puzzles_common Catch2::Catch2WithMain)
target_include_directories(puzzle_14_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(puzzle_14_test PRIVATE DATA_DIR="${PUZZLE_14_DATA_DIR}")

# Test with reference solution
add_executable(puzzle_14_test_solution test_puzzle.cu)
target_link_libraries(puzzle_14_test_solution PRIVATE cuda_puzzles_common Catch2::Catch2WithMain)
target_include_directories(puzzle_14_test_solution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(puzzle_14_test_solution PRIVATE USE_SOLUTION DATA_DIR="${PUZZLE_14_DATA_DIR}")

# Register with CTest
add_test(NAME puzzle_14_test_solution COMMAND puzzle_14_test_solution)
