#!/bin/bash

# based on discussion here https://stackoverflow.com/questions/35485087/determining-which-gencode-compute-arch-values-i-need-for-nvcc-within-cmak

source="$(cat << EOF
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
int main()
{
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop,0);
int v = prop.major * 10 + prop.minor;
printf("%d",v);
}
EOF
)"

# dump source to nvcc (needs to be in the path), 
exe=$(mktemp -dt cudaXXXXXXXXXX)
echo "$source"  | nvcc -I/usr/include/cuda -x c++ - -o  $exe/cudaVersion

$exe/cudaVersion
rm -rf $exe
