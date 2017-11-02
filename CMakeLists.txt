cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
# project(cuda_morphology LANGUAGES CXX CUDA)
project(cuda_morphology LANGUAGES CXX)

# find_package(CUDA QUIET REQUIRED)
find_package(CUDA)

set(CMAKE_BUILD_TYPE release)

message(STATUS "")
message(STATUS "  C/C++:")
message(STATUS "    C++ flags (Release):"    ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "    C++ flags (Debug):"      ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "    C flags (Release):"      ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE})
message(STATUS "    C flags (Debug):"        ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG})
message(STATUS "")

add_subdirectory(examples)
add_subdirectory(src)
