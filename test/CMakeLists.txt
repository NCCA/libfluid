cmake_minimum_required(VERSION 3.1)
project(fluidtest)
# must go before find_package
set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA 9.0 REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  --compiler-options -fno-strict-aliasing --compiler-options -fPIC -use_fast_math --std=c++11 -gencode arch=compute_62,code=sm_62" )

set(SOURCES   
src/main.cpp  
src/myfluidsystem.cpp 
)
# Set to C++ 11
set(CMAKE_CXX_STANDARD 11)

include_directories(../include)
include_directories(../common/include)
# for jetson
include_directories(/usr/local/cuda-9.0/targets/aarch64-linux/include/)
# for helper_math.h may have to alter for your own install of cuda samples
include_directories(/usr/local/cuda/samples/common/inc/)
set ( FLUIDLIB -L../lib/ -lfluid  )

# add exe and link libs that must be after the other defines
add_executable(fluidtest ${SOURCES})
target_link_libraries(fluidtest ${FLUIDLIB} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cudart_static_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} )



#-L/usr/local/cuda-9.0/targets/aarch64-linux/lib/ -lcuda -lcudart -lcublas -lcurand  -lcublas_device ${FLUIDLIB} )


#-L/usr/local/cuda-9.0/targets/aarch64-linux/lib/  -lcuda )








