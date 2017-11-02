include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

add_executable(test_morphology test.cpp)
target_link_libraries(test_morphology cu_morphology ${OpenCV_LIBS})
set_property(TARGET test_morphology PROPERTY CXX_STANDARD 11)
