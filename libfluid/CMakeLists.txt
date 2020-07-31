cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)

#include(CTest)

include_directories(include ../common/include $ENV{CUDA_SAMPLES_PATH}/Common)

add_library(fluid SHARED
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
)

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(fluid PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(fluid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link in CURAND
find_library(LIBCURAND curand)
target_link_libraries(fluid ${LIBCURAND})
