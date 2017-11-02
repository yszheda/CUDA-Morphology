set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --std=c++11)

set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -G)
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_DEBUG} -O3 --use_fast_math)

message(STATUS "")
message(STATUS "  CUDA C/C++:")
message(STATUS "    NVCC flags (Release):"    ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_RELEASE})
message(STATUS "    NVCC flags (Debug):"      ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_DEBUG})
message(STATUS "")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

cuda_add_library(cu_morphology STATIC morphology.cu)

install(TARGETS cu_morphology ARCHIVE DESTINATION lib)
install(FILES morphology.h DESTINATION include)
