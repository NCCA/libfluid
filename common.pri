LIB_INSTALL_DIR=$$PWD/lib
BIN_INSTALL_DIR=$$PWD/bin
INC_INSTALL_DIR=$$PWD/include

linux:QMAKE_CXX = $$(HOST_COMPILER)
macx:QMAKE_CXX=clang++
#QMAKE_CXXFLAGS += -D_DEBUG

INCLUDEPATH += ${CUDA_SAMPLES_PATH}/common/inc ${PWD}/../common/include include ${CUDA_PATH}/include ${CUDA_PATH}/include/cuda

