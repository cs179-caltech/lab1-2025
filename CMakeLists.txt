cmake_minimum_required(VERSION 3.23)
project(lab1 LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
set(CMAKE_CUDA_ARCHITECTURES native)

set(LAB_SOURCE_FILES
    src/blur.cpp
    src/blur.cu
    src/blur.cuh
    src/cuda_header.cuh)

add_executable(noaudio-blur "${LAB_SOURCE_FILES}")
add_executable(audio-blur "${LAB_SOURCE_FILES}")

target_compile_definitions(noaudio-blur PRIVATE -DAUDIO_ON=0)
target_compile_definitions(audio-blur PRIVATE -DAUDIO_ON=1)

target_link_libraries(noaudio-blur CUDA)
target_link_libraries(audio-blur CUDA)
