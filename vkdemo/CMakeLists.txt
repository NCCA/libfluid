cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Name of the project with Language
project(vkdemo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_HOST_COMPILER g++)
set(SHADERS_PATH ${PROJECT_SOURCE_DIR}/shaders)

# Set the default VK SDK path (this will change depending on the NCCA lab build
set(VK_SDK_PATH_DEFAULT /public/devel/VulkanSDK/1.2.131.2/x86_64)

# Check if an alternative VK path has been set in the environment variable
IF(DEFINED ENV{VK_SDK_PATH})
 # In this case use the environment variable
 MESSAGE(STATUS "VK_SDK_PATH defined as $ENV{VK_SDK_PATH}. Using this path.")
 SET(VK_SDK_PATH $ENV{VK_SDK_PATH})
ELSE()
 # Otherwise use the default VK path
 MESSAGE(STATUS "VK_SDK_PATH not defined. Using NCCA lab build default ${VK_SDK_PATH_DEFAULT}.")
 SET(VK_SDK_PATH ${VK_SDK_PATH_DEFAULT})
ENDIF()

list(APPEND CMAKE_PREFIX_PATH ${VK_SDK_PATH})


# Search for the Vulkan SDK package
find_package(Vulkan REQUIRED FATAL_ERROR)


# We'll have to assume that glfw3 has been installed in the default system directory
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
    src/VulkanCudaSineWave.h
)

# Note that VULKAN_INCLUDE_DIRS is automatically populated by find_package
include_directories(vkdemo ${VULKAN_INCLUDE_DIRS} include ../common/include $ENV{CUDA_SAMPLES_PATH}/Common ../common/include ../libfluid/include)

target_link_libraries(vkdemo Vulkan::Vulkan glfw -L${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -lfluid)

target_compile_options(vkdemo PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL: -m64 --std=c++11 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75>")

target_compile_options(vkdemo PRIVATE -DSHADERS_PATH="${SHADERS_PATH}")

