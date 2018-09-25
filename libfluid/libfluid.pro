# Read shared environment settings from the common include file
include(../common.pri)

# This is set to build a dynamic library
TEMPLATE = lib

# Use this directory to store all the intermediate objects
OBJECTS_DIR = obj

# Set this up as the installation directory for our library
TARGET = $$LIB_INSTALL_DIR/fluid

# Set the C++ flags for this compilation when using the host compiler
QMAKE_CXXFLAGS += -std=c++11 -fPIC 
HEADERS+=cudasrc/*.cuh

# Directories
INCLUDEPATH += ${CUDA_SAMPLES}/common/inc ${PWD}/../common/include include ${CUDA_PATH}/include ${CUDA_PATH}/include/cuda 

# Link with the following libraries
linux:LIBS += -L/usr/lib/x86_64-linux-gnu -L${CUDA_PATH}/lib64 -L${CUDA_PATH}/lib64/nvidia -L/usr/lib64/nvidia -lcudadevrt -lcuda -lcudart -lcurand -lcublas -lcublas_device

# LIBS += -L/usr/lib/x86_64-linux-gnu -lcuda -lcudart -lcudadevrt -lcurand
macx:LIBS = -L/usr/local/cuda/lib/ -lcudadevrt -lcuda -lcudart -lcurand

# Use the following path for nvcc created object files
CUDA_OBJECTS_DIR = cudaobj
 
# CUDA_DIR - the directory of cuda such that CUDA\<version-number\ contains the bin, lib, src and include folders
# Set this environment variable yourself.
CUDA_DIR=${CUDA_PATH}
isEmpty(CUDA_DIR) {
    message(CUDA_DIR not set - set this to the base directory of your local CUDA install (on the labs this should be /usr))
}
 
## CUDA_SOURCES - the source (generally .cu) files for nvcc. No spaces in path names
CUDA_SOURCES += cudasrc/*.cu

## CUDA_INC - all includes needed by the cuda files (such as CUDA\<version-number\include)
CUDA_INC += $$join(INCLUDEPATH,' -I','-I',' ')
 
# nvcc flags ("-Xptxas -v" option is always useful, "-D_DEBUG" for tons of debug info)
#NVCC_DEBUG_FLAGS =
NVCC_DEBUG_FLAGS += -D_DEBUG -g -G  -DTHRUST_DEBUG
#NVCC_DEBUG_FLAGS += -Xptxas -v
# New added by Jon
GENCODE=$$system(../findCudaArch.sh)
NVCCFLAGS =  -I../src/	-m64 $$NVCC_DEBUG_FLAGS $$GENCODE --compiler-options -fno-strict-aliasing --compiler-options -fPIC -use_fast_math --std=c++11 #--ptxas-options=-v
#message($$NVCCFLAGS)
# Define the path and binary for nvcc
NVCCBIN = $$CUDA_DIR/bin/nvcc

#prepare intermediate cuda compiler
cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.o
cuda.commands = $$NVCCBIN $$NVCCFLAGS -dc $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
 
#Set our variable out. These obj files need to be used to create the link obj file
#and used in our final gcc compilation
cuda.variable_out = CUDA_OBJ
cuda.variable_out += OBJECTS
cuda.clean = $$CUDA_OBJECTS_DIR/*.o
# Note that cuda objects are linked separately into one obj, so these intermediate objects are not included in the final link
cuda.CONFIG = no_link 
QMAKE_EXTRA_COMPILERS += cuda
 
# Prepare the linking compiler step (combine tells us that the compiler will combine all the input files)
cudalink.input = CUDA_OBJ
cudalink.CONFIG = combine
cudalink.output = $$OBJECTS_DIR/cuda_link.o
 
# Tweak arch according to your hw's compute capability
cudalink.commands = $$NVCCBIN $$NVCCFLAGS $$CUDA_INC -dlink ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} $$LIBS
cudalink.dependency_type = TYPE_C
cudalink.depend_command = $$NVCCBIN $$NVCCFLAGS -M $$CUDA_INC ${QMAKE_FILE_NAME}

# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cudalink

# Set up the post install script to copy the headers into the appropriate directory
includeinstall.commands = mkdir -p $$INC_INSTALL_DIR && cp include/*.h $$INC_INSTALL_DIR
QMAKE_EXTRA_TARGETS += includeinstall
POST_TARGETDEPS += includeinstall
