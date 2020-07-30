cmake_minimum_required(VERSION 3.0)

# Name of the project with Language
project(libfluid)

# find the cuda 10.1 package, currently required for building purposes
find_package(CUDA 10 REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc=true -ccbin $ENV{CMAKE_CUDA_HOST_COMPILER} --compiler-options -fno-strict-aliasing --compiler-options -fPIC -use_fast_math  -gencode arch=compute_75,code=sm_75" )

include_directories(include ../common/include $ENV{CUDA_SAMPLES_PATH}/Common)

# build a library
cuda_add_library(fluid 
            cudasrc/fluidintegrator.cu
            cudasrc/fluidintegrator.cuh
            cudasrc/fluidkernel.cu
            cudasrc/fluidkernel.cuh
            cudasrc/fluidparams.cu
            cudasrc/fluidparams.cuh
            cudasrc/fluidphysics.cu
            cudasrc/fluidphysics.cuh
            cudasrc/fluidsystem.cu
            cudasrc/fluidutil.cu
            cudasrc/fluidutil.cuh
            cudasrc/hash.cu
            cudasrc/hash.cuh
            cudasrc/random.cu
            cudasrc/random.cuh
            include/fluidsystem.h
            SHARED
            )