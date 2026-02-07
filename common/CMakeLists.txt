# Common utilities and headers library
add_library(cuda_puzzles_common INTERFACE)
target_include_directories(cuda_puzzles_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Test executable to verify all utilities work
add_executable(test_common test_common.cu)
target_link_libraries(test_common PRIVATE cuda_puzzles_common Catch2::Catch2WithMain)

# Register with CTest
add_test(NAME test_common COMMAND test_common)
