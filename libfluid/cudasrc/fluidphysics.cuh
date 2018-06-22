#ifndef FLUIDPHYSICS_CUH
#define FLUIDPHYSICS_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>

/// This CUDA kernel computes the density and pressure based on the point positions and the occupancy data computed previously
__global__ void computeDensity(
    float *pressure,
    float *soundSpeed,
    float *density,
    const float3 *points,
    const uint *cellOcc,
    const uint *scatterAddress);

/// This CUDA kernel computes the Force (pressure gradient) based on the pressure, density and occupancy data computed previously
__global__ void computePressureGradient(
    float3 *force,
    const float *pressure,
    const float *density,
    const float3 *points,
    const uint *cellOcc,
    const uint *scatterAddress);

/// This CUDA kernel computes the normals accordint to the formula of Akinci et al. https://dl.acm.org/citation.cfm?id=2508395
__global__ void computeNormals(
    float3 *normals,
    const float *density,
    const float3 *points,
    const uint *cellOcc,
    const uint *scatterAddress);

__global__ void computeAllForces(
    float3 *pressureForce,
    float3 *tensionForce,
    float3 *adhesionForce,
    float3 *viscosityForce,
    const float *soundSpeed,
    const float *pressure,
    const float *density,
    const float3 *points,
    const float3 *normals,
    const float3 *velocity,
    const uint *cellOcc,
    const uint *scatterAddress);

/// This CUDA kernel effectively "smooths" the velocity field based on the XSPH method
__global__ void correctVelocityXSPH(float3 *velocity,
                                    const float *density,
                                    const float3 *points,
                                    const uint *cellOcc,
                                    const uint *scatterAddress);

#endif