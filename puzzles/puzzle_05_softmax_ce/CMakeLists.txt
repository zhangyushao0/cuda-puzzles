# Puzzle 05: Softmax + Cross-Entropy Loss
#
# Targets:
#   puzzle_05_test          - build with student's puzzle.cu
#   puzzle_05_test_solution - build with reference solution.cu

# Test with student's implementation
add_executable(puzzle_05_test test_puzzle.cu)
target_link_libraries(puzzle_05_test PRIVATE cuda_puzzles_common Catch2::Catch2WithMain)
target_include_directories(puzzle_05_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Test with reference solution
add_executable(puzzle_05_test_solution test_puzzle.cu)
target_link_libraries(puzzle_05_test_solution PRIVATE cuda_puzzles_common Catch2::Catch2WithMain)
target_include_directories(puzzle_05_test_solution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(puzzle_05_test_solution PRIVATE USE_SOLUTION)

# Register with CTest
add_test(NAME puzzle_05_test_solution COMMAND puzzle_05_test_solution)
