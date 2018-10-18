cmake_minimum_required(VERSION 3.5)

# Name of the project with Language
project(libfluid)
# must go before find package
set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA 9.0 REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc=true --compiler-options -fno-strict-aliasing --compiler-options -fPIC -use_fast_math --std=c++11 -gencode arch=compute_62,code=sm_62" )

set(libfluid PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set to C++ 11
set(CMAKE_CXX_STANDARD 11)

include_directories(include)
include_directories(../common/include)
# for helper_math.h may have to alter for your own install of cuda samples
include_directories(/usr/local/cuda/samples/common/inc/)

CUDA_COMPILE( FLUIDSRC 
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

set( PUBLIC_HEADER "include/fluidsystem.h")
set(CUDALINKFLAGS -L/usr/local/cuda-9.0/targets/aarch64-linux/lib/ -lcuda -lcudart -lcublas -lcurand  -lcublas_device)
cuda_add_library(fluid SHARED ${FLUIDSRC} OPTIONS "  ${CUDA_LIBRARIES} £{CUDALINKFLAGS}" )
target_link_libraries(fluid ${CUDA_LIBRARIES} -L/usr/local/cuda-9.0/targets/aarch64-linux/lib/ -lcuda -lcudart -lcublas -lcurand  -lcublas_device ${FLUIDLIB} )

install(TARGETS fluid LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/../lib)
install( FILES  ${PUBLIC_HEADER} DESTINATION ${CMAKE_SOURCE_DIR}/../include/ )

