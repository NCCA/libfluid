cmake_minimum_required(VERSION 3.0)
project(libfluid_all)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)

add_subdirectory(libfluid)
add_subdirectory(test)
