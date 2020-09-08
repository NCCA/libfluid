cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)

include_directories(include ../common/include $ENV{CUDA_SAMPLES_PATH}/Common)

set(CMAKE_VERBOSE_MAKEFILE ON)

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

# Set the default CURAND LIB path (this will change depending on the NCCA lab build)
set(CURAND_LIB_PATH_DEFAULT /usr/local/cuda-9.0/lib64)

# Check if an alternative CURAND path has been set in the environment variable
IF(DEFINED ENV{CURAND_LIB_PATH})
 # In this case use the environment variable
 MESSAGE(STATUS "CURAND_LIB_PATH defined as $ENV{CURAND_LIB_PATH}. Using this path.")
 SET(CURAND_LIB_PATH $ENV{CURAND_LIB_PATH})
ELSE()
 # Otherwise use the default path
 MESSAGE(STATUS "CURAND_LIB_PATH not defined. Using NCCA lab build default ${CURAND_LIB_PATH_DEFAULT}.")
 SET(CURAND_LIB_PATH ${CURAND_LIB_PATH_DEFAULT})
ENDIF()

# Append this search path to the CMAKE prefix path
list(APPEND CMAKE_PREFIX_PATH ${CURAND_LIB_PATH})

# Now find the library to help with the linking
find_library(LIBCURAND curand)

# Link in CURAND
target_link_libraries(fluid ${LIBCURAND})


#target_compile_options(fluid PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-m64 -gencode arch=compute_75,code=sm_75>)
# set_property(TARGET fluid PROPERTY CUDA_ARCHITECTURES 75)

#target_compile_options(fluid PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -m64 --std=c++11>)
#target_compile_options(fluid PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_75,code=sm_75>")
#target_compile_options(fluid PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_75,code=compute_75>")
#target_link_options(fluid PRIVATE $<DEVICE_LINK:-arch=sm_75>)
#add_link_options($<DEVICE_LINK:-arch=sm_75>)

#target_link_options(fluid PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -m64 --std=c++11>)
#target_link_options(fluid PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_75,code=sm_75>")
#target_link_options(fluid PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_75,code=compute_75>")

