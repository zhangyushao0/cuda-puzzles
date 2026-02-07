# Puzzle 02: Matrix Multiplication
#
# Targets:
#   puzzle_02_test          - build with student's puzzle.cu
#   puzzle_02_test_solution - build with reference solution.cu

# Test with student's implementation
add_executable(puzzle_02_test test_puzzle.cu)
target_link_libraries(puzzle_02_test PRIVATE cuda_puzzles_common)
target_include_directories(puzzle_02_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Test with reference solution
add_executable(puzzle_02_test_solution test_puzzle.cu)
target_link_libraries(puzzle_02_test_solution PRIVATE cuda_puzzles_common)
target_include_directories(puzzle_02_test_solution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(puzzle_02_test_solution PRIVATE USE_SOLUTION)

# Register with CTest
add_test(NAME puzzle_02_test_solution COMMAND puzzle_02_test_solution)
