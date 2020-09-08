cmake_minimum_required(VERSION 3.8)
project(libfluid_all)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin) 

add_subdirectory(libfluid)
add_subdirectory(test)
#add_subdirectory(vkdemo)
