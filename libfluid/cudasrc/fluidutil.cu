#include "fluidutil.cuh"
#include "fluidparams.cuh"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include "helper_math.h"
#include <cublas_v2.h>

/**
 * Compute a grid position from a point. Note that the grid position can be negative in this case
 * if the point is outside [0,1] range. Sanity checks will need to be performed elsewhere.
 */
__device__ int3 grid_from_point(const float3 pt)
{
    return make_int3(floor(pt.x * params.m_res), floor(pt.y * params.m_res), floor(pt.z * params.m_res));
}
/**
 * Compute a cell index from a grid. In this case we make the grid a unsigned int so no bounds checking 
 * applies.
 */
__device__ uint cell_from_grid(const uint3 grid)
{
    return grid.x + grid.y * params.m_res + grid.z * params.m_res2;
}
/**
 * Compute a cell index from a grid. In this case we make the grid an int and apply bounds checking.
 */
__device__ uint cell_from_grid(const int3 grid)
{
    // Test to see if all of the points are inside the grid (I don't think CUDA can do lazy evaluation (?))
    bool isInside = (grid.x >= 0) && (grid.x < params.m_res) &&
                    (grid.y >= 0) && (grid.y < params.m_res) &&
                    (grid.z >= 0) && (grid.z < params.m_res);

    // Write out the hash value if the point is within range [0,1], else write NULL_HASH
    return (isInside) ? cell_from_grid(make_uint3(grid.x, grid.y, grid.z)) : NULL_HASH;
}

__device__ uint3 grid_from_cell(const uint cell)
{
    uint3 ret_val;
    ret_val.z = float(cell) * params.m_h2;
    ret_val.y = float(cell - ret_val.z * params.m_res2) * params.m_h;
    ret_val.x = (cell - ret_val.z * params.m_res2 - ret_val.y * params.m_res);
    return ret_val;
}

/// A virtual device function which determines a distance and squared distance between two input points
__device__ float sqrdist(const float3 &A, const float3 &B)
{
    float3 AB = A - B;
    return dot(AB, AB);
}

__device__ float dist(const float3 &A, const float3 &B)
{
    return sqrt(sqrdist(A, B));
}

/**
 * The operator() allows this class to be used as a functor. This outputs a colour based on how much the density
 * varies from the rest density. If within the range [0,rd] the colour will vary between blue and green. If in the 
 * range [1, 2] or anything greater, the colour will range between green and red. 
 */
__device__ void DensityColourOperator::operator()(Tuple t)
{
    float ratio = thrust::get<0>(t) * params.m_invRestDensity;
    float3 colour = make_float3(1.0f, 0.0f, 0.0f);
    if (ratio < 1.0f)
    {
        colour = make_float3(0.0f, ratio, 1.0f - ratio);
    }
    else if (ratio < 2.0f)
    {
        colour = make_float3(ratio - 1.0f, 2.0f - ratio, 0.0f);
    }
    thrust::get<1>(t) = colour;
}

/**
 * Return the signed distance to the boundary.
 */
__device__ float boundarySDF(const float3 &pos)
{

    // The default SDF is centered at the origin, so we need to transform the point as our cube centre is at [0.5,0.5,0.5]. The top corner
    // is also given by [.5,.5,.5]. The final result is the distance of the vector from pos transformed to the top right quadrant to the corner
    // of the cube. Note that the distance is SIGNED - negative means inside the cube, positive means outside.
    float3 b = make_float3(0.5f);

    float3 zero = make_float3(0.0f);
    float3 d = fabs(pos - b) - b;
    float d_box = fminf(fmaxf(d.x, fmaxf(d.y, d.z)), 0.0) + length(fmaxf(d, zero));

    // The result below is for a sphere bounding the fluid
    float r = 0.5f; // The radius of the sphere
    float d_sphere = length(pos - make_float3(0.5f)) - r;

    return d_box;
}

__device__ float3 boundaryNormalSDF(const float3 &pos)
{
    float eps = 0.0001f;

    // Assume that the normal evaluated at this point in the SDF is close enough to the surface normal (a mistake?)
    return normalize(make_float3(boundarySDF(make_float3(pos.x + eps, pos.y, pos.z)) - boundarySDF(make_float3(pos.x - eps, pos.y, pos.z)),
                                 boundarySDF(make_float3(pos.x, pos.y + eps, pos.z)) - boundarySDF(make_float3(pos.x, pos.y - eps, pos.z)),
                                 boundarySDF(make_float3(pos.x, pos.y, pos.z + eps)) - boundarySDF(make_float3(pos.x, pos.y, pos.z - eps))));
}

/**
 * Routine to 
 * Lifted from http://stackoverflow.com/questions/27094612/cublas-matrix-inversion-from-device
 */
__device__ void squareMatrixInverse3(float *A, float *C)
{
    // Create a cublas handle (not sure what this does)
    cublasHandle_t hdl;
    cublasStatus_t status = cublasCreate_v2(&hdl);

    int info = 0;  // Info is set on completion of cublas execution
    int batch = 1; // Not sure what the batch does - should this be threadIdx?
    int p[3];      // Pivot vector, size 3

    // Call the getrf routine
    // DGETRF computes an LU factorization of a general M-by-N matrix A using partial pivoting with row interchanges.
    // http://www.netlib.org/clapack/old/double/dgetrf.c
    status = cublasSgetrfBatched(hdl, 3, &A, 3, p, &info, batch);
    __syncthreads();

    // Call getri routine
    // DGETRI computes the inverse of a matrix using the LU factorization computed by DGETRF.
    // http://www.netlib.org/clapack/old/double/dgetri.c
    status = cublasSgetriBatched(hdl, 3, (const float **)&A, 3, p, &C, 3, &info, batch);
    __syncthreads();

    // Clean up the handle (not sure if this has an overhead)
    cublasDestroy_v2(hdl);
}