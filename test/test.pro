include(../common.pri)
TEMPLATE=app
TARGET=$$BIN_INSTALL_DIR/test

SOURCES += src/main.cpp src/myfluidsystem.cpp

# INCLUDEPATH += $${PWD}/../common/include $$INC_INSTALL_DIR ${CUDA_PATH}/include ${CUDA_PATH}/include/cuda ${CUDA_PATH}/samples/common/inc 
INCLUDEPATH += /usr/include/cuda
INCLUDEPATH += $$INC_INSTALL_DIR
OBJECTS_DIR = obj

QMAKE_CXXFLAGS += -std=c++14 -Wall -Wextra -pedantic
macx:CONFIG -= app_bundle
linux:LIBS += -L${CUDA_PATH}/lib64 -L${CUDA_PATH}/lib64/nvidia -L$$LIB_INSTALL_DIR -L/usr/lib/x86_64-linux-gnu -lfluid -lcuda -lcudart -lcudadevrt -lcurand
QMAKE_RPATHDIR += ../lib
message($$QMAKE_RPATHDIR)
macx:LIBS += -L/usr/local/cuda/lib/ -lcudadevrt -lcuda -lcudart -lcurand  -L../lib -lfluid
