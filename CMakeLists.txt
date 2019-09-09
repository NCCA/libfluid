cmake_minimum_required(VERSION 3.8)
project(LibFluidAll LANGUAGES CXX CUDA)

add_library(fluid STATIC
libfluid/cudasrc/fluidintegrator.cu  libfluid/cudasrc/fluidsystem.cu
libfluid/cudasrc/fluidkernel.cu      libfluid/cudasrc/fluidutil.cu
libfluid/cudasrc/fluidparams.cu      libfluid/cudasrc/hash.cu
libfluid/cudasrc/fluidphysics.cu     libfluid/cudasrc/random.cu
)
target_include_directories(fluid PUBLIC libfluid/include/ 
                                        common/include/
                                        /usr/local/cuda-9.0/samples/common/inc/
)
target_compile_features(fluid PUBLIC cxx_std_11)
set_target_properties( fluid 
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
add_executable(fluidTest  test/src/main.cpp  
                          test/src/myfluidsystem.cpp  
                          test/src/myfluidsystem.h)
include_directories(libfluid/include/ /usr/local/cuda-9.0/include/ common/include)
link_directories(${CMAKE_SOURCE_DIR})
target_link_libraries(fluidTest fluid curand)
