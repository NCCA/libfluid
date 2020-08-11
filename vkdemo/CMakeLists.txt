cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Name of the project with Language
project(vkdemo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

find_package(Vulkan REQUIRED FATAL_ERROR)
find_package(glfw3 REQUIRED FATAL_ERROR)

# build the executable
add_executable(
    vkdemo
    src/main.cpp
    src/VulkanBaseApp.cpp
    src/VulkanBaseApp.h
    src/SineWaveSimulation.cu
    src/SineWaveSimulation.h
    src/linmath.h
)

include_directories(vkdemo ${VULKAN_INCLUDE_DIRS} include ../common/include $ENV{CUDA_SAMPLES_PATH}/Common)
target_link_libraries(vkdemo Vulkan::Vulkan glfw)

target_compile_options(vkdemo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-m64 -gencode arch=compute_75,code=sm_75>)

