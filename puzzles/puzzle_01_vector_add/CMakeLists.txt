# Puzzle 01: Vector Addition

# Student version — links puzzle.cu (with TODOs) against tests
add_executable(puzzle_01_test test_puzzle.cu puzzle.cu)
target_link_libraries(puzzle_01_test PRIVATE cuda_puzzles_common Catch2::Catch2WithMain)

# Solution version — links solution.cu against the same tests
add_executable(puzzle_01_test_solution test_puzzle.cu solution.cu)
target_link_libraries(puzzle_01_test_solution PRIVATE cuda_puzzles_common Catch2::Catch2WithMain)

# Register with CTest
add_test(NAME puzzle_01_test_solution COMMAND puzzle_01_test_solution)
