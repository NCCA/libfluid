#ifndef HASH_CUH
#define HASH_CUH

// Basic includes for debugging output mainly
#include <iostream>
#include <stdio.h>
#include <sys/time.h>

// My own include function to generate some randomness
#include "fluidutil.cuh"
#include "fluidkernel.cuh"
#include "fluidparams.cuh"
#include "helper_math.h"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>

/**
 * The functor class to create a point hash using Thrust
 */
struct PointHashOperator
{
    /// This structure maintains the cell occupancy count (this is the method of Hoetzlein, which avoids the extra cell occupancy check)
    uint *cellOcc;

    /// Construct a new operator
    PointHashOperator(uint *_cellOcc);

    /// The operator functor. Should be called with thrust::transform and a zip_iterator
    __device__ uint operator()(const float3 &pt);
};

#endif // HASH_CUH
