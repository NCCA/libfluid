# libfluid
An implementation of weakly compressible SPH in CUDA.

Requirements:
- CUDA Samples. Get them using the following:
  git clone https://github.com/NVIDIA/cuda-samples.git
  Set the environment variable CUDA_SAMPLES_PATH to the root directory of cuda-samples

Compilation on NCCA lab build:
1. Enter parent directory of the project.
2. Create a build directory and go into it: "mkdir build && cd build"
3. Call CMAKE to create the make files: "cmake -DCMAKE_INSTALL_PREFIX=.. .."
4. Assuming this worked build the project: "make" (note avoid -j8 as this tends to get stuck in CUDA projects)
5. Assuming this worked, go to your executable: "cd ../bin"
6. Create a directory for the geometry to get dumped to: "mkdir geo"
7. Call the executable setting the path to the library in the process: "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib ./test 100000 32"
8. While it is executing, you can open the geometry files using Houdini Files->Import->Geometry to view it.
