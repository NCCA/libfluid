cmake_minimum_required(VERSION 3.5)

# Name of the project with Language
project(libfluid)
find_package(CUDA 9.0 REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ccbin g++ --compiler-options -fno-strict-aliasing --compiler-options -fPIC -use_fast_math --std=c++11" )

#set_target_properties( libfluid
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#set_target_properties(libfluid PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(libfluid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set to C++ 11
set(CMAKE_CXX_STANDARD 11)

include_directories(include)
include_directories(../common/include)
# for helper_math.h may have to alter for your own install of cuda samples
include_directories(/usr/local/cuda/samples/common/inc/)

set(SOURCES 
cudasrc/fluidintegrator.cu   
cudasrc/fluidparams.cu    
cudasrc/fluidsystem.cu  
cudasrc/fluidutil.cu    
cudasrc/random.cu
cudasrc/fluidkernel.cu       
cudasrc/fluidphysics.cu   
cudasrc/hash.cu
include/fluidsystem.h
)



cuda_add_library(libfluid SHARED ${SOURCES})

