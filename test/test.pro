include(../common.pri)
TEMPLATE=app
TARGET=$$BIN_INSTALL_DIR/test

SOURCES += src/main.cpp src/myfluidsystem.cpp

INCLUDEPATH += $$INC_INSTALL_DIR
OBJECTS_DIR = obj

# Set the version of g++ to be compatible with CUDA
QMAKE_CXX = $$(HOST_COMPILER)

# This may or may not be necessary for mac compilation
macx:CONFIG -= app_bundle

# Compilation now depends entirely on pkg-config
LIBS += $$system(pkg-config --silence-errors --libs cuda-8.0 cudart-8.0 curand-8.0 cublas-8.0) -lcublas_device -L$$LIB_INSTALL_DIR -lfluid

QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic $$system(pkg-config --silence-errors --cflags cuda-8.0 cudart-8.0 curand-8.0 cublas-8.0)

QMAKE_RPATHDIR += $$LIB_INSTALL_DIR
message($$QMAKE_RPATHDIR)
