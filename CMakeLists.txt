cmake_minimum_required(VERSION 3.18)
project(cuda-puzzles LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default CUDA architecture to native if not specified
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Note: Minimum sm_75 (CUDA 13.x dropped sm_50-sm_70)
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Enable CTest
enable_testing()

# Fetch Catch2 v3 for testing
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# Add subdirectories
add_subdirectory(common)
add_subdirectory(puzzles)
