cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)

# Note that the CUDA_SAMPLES_PATH needs to be set as an environment variable
include_directories(include ../common/include $ENV{CUDA_SAMPLES_PATH}/Common)

# Uncomment the line below to get verbose compilation information
#set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA)

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

# Request that fluid is built with -std=c++11
target_compile_features(fluid PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the 
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(fluid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link in CUDA and CURAND
target_link_libraries(fluid ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

