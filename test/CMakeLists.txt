cmake_minimum_required(VERSION 3.8)

# Name of the project with Language
project(test)

set(CMAKE_CXX_STANDARD 11)

include_directories(../common/include ../libfluid/include)

# build the executable
add_executable(
    test
    src/main.cpp
    src/myfluidsystem.cpp
    src/myfluidsystem.h
)

target_link_libraries(test -L${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -lfluid)

