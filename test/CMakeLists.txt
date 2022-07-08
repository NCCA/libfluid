cmake_minimum_required(VERSION 3.8)

# Name of the project with Language
project(fluidtest LANGUAGES CXX)

# We need to find the CUDA package which will define variables for includes and libs.
find_package(CUDA)

# Build with Cxx11
set(CMAKE_CXX_STANDARD 14)

# Set the include directories
include_directories(BEFORE ../common/include ../libfluid/include ${CUDA_INCLUDE_DIRS} ../deps/tclap/include)

# build the executable
add_executable(
    fluidtest
    src/main.cpp
    src/myfluidsystem.cpp
    src/myfluidsystem.h
)

target_link_libraries(fluidtest -L${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} -lfluid)
