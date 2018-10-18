cmake_minimum_required(VERSION 3.5)

# Name of the project with Language
project(libfluid)
# must go before find package
set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA 9.0 REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc=true --compiler-options -fno-strict-aliasing --compiler-options -fPIC -use_fast_math  -gencode arch=compute_62,code=sm_62" )

set(libfluid PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set to C++ 11
set(CMAKE_CXX_STANDARD 11)
include_directories(include)
include_directories(../common/include)
# for helper_math.h may have to alter for your own install of cuda samples
include_directories(/usr/local/cuda/samples/common/inc/)

set(PROJECT_SRC
        cudasrc/fluidintegrator.cu   
        cudasrc/fluidparams.cu    
        cudasrc/fluidsystem.cu  
        cudasrc/fluidutil.cu    
        cudasrc/random.cu
        cudasrc/fluidkernel.cu       
        cudasrc/fluidphysics.cu   
        cudasrc/hash.cu
)

set(CMAKE_CXX_FLAGS "-g -std=c++11")
set(compile_options "-std=c++11")

set(seperable_state ${CUDA_SEPARABLE_COMPILATION})
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)

set(BUILD_SHARED_LIBS OFF)

set(CUDA_SEPARABLE_COMPILATION ON)


#cuda_wrap_srcs(fluid OBJ
#               cuda_compiled_sources
#               ${PROJECT_SRC}
#               ${GPGPU_BUILD_TYPE}
#               OPTIONS "${compile_options}"
#               )

CUDA_COMPILE(cuda_compiled_sources ${PROJECT_SRC} OPTIONS )
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Compute the file name of the intermediate link file used for separable
# compilation.
cuda_compute_separable_compilation_object_file_name(link_file
                                                    fluid
                                                    ${cuda_compiled_sources} 
)


cuda_link_separable_compilation_objects("${link_file}"
                                        fluid
                                        "${compile_options}"
                                        "${cuda_compiled_sources}"
OPTIONS ""
)



set(CUDA_SEPARABLE_COMPILATION ${seperable_state})
set(cache_state ${CUDA_NVCC_FLAGS_CACHE})


#target_link_libraries(fluid
#  LINK_PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY}
#  )


cuda_add_library(fluid SHARED
            ${cuda_compiled_sources}
            ${PROJECT_SRC}
            ${link_file}
            )





#set( PUBLIC_HEADER "include/fluidsystem.h")


#target_link_libraries(fluid ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} )


#install(TARGETS fluid LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/../lib)
#install( FILES  ${PUBLIC_HEADER} DESTINATION ${CMAKE_SOURCE_DIR}/../include/ )

