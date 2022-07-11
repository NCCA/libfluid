#ifndef FLUIDUTIL_H
#define FLUIDUTIL_H

#include "fluidsystem.h"

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

// A hash value that indicates the point is outside the grid
__device__ int3 grid_from_point(const float3 pt);
__device__ uint cell_from_grid(const uint3 grid);
__device__ uint cell_from_grid(const int3 grid);
__device__ uint3 grid_from_cell(const uint cell);

__device__ float sqrdist(const float3 &A, const float3 &B);
__device__ float dist(const float3 &A, const float3 &B);

/**
 * @brief An operator (functor) to calculate a colour value from a density.
 *
 */
struct DensityColourOperator
{
    typedef thrust::tuple<
        const float &, // density
        float3 &       // resulting colour
        >
        Tuple;

    /// Construct a new operator
    DensityColourOperator() {}

    /// The operator functor. Should be called with thrust::transform and a zip_iterator
    __device__ void operator()(Tuple t);
};

/// Return the signed distance to the boundary
__device__ float boundarySDF(const float3 &pos);
__device__ float3 boundaryNormalSDF(const float3 &pos);

#endif // FLUIDUTIL_H
