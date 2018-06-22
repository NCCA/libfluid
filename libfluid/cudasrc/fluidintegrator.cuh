#ifndef FLUIDINTEGRATOR_H
#define FLUIDINTEGRATOR_H

// My own include function to generate some randomness
#include "fluidutil.cuh"
#include "helper_math.h"

// For the CUDA runtime routines (prefixed with "cuda_")
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

/**
 * A basic leapfrog integrator, derived from the Wikipedia page! It is however used in many SPH implementations,
 * including Fluidsv3, because of it's stability.
 */
struct LeapfrogIntegratorOperator
{
    /// The tuple is <x, x_dot, x_ddot, forces...>. Note that for leapfrog we need to store the acceleration.
    typedef thrust::tuple<float3 &,       // Position
                          float3 &,       // Velocity
                          float3 &,       // Acceleration
                          const float3 &, // Pressure Force
                          const float3 &, // Surface Tension Force
                          const float3 &, // Adhesion Force
                          const float3 &  // Viscosity Force
                          >
        Tuple; // STATIC/DYNAMIC

    /// Construct an empty operator
    LeapfrogIntegratorOperator();

    /// The operator functor. Should be called with thrust::transform and a zip_iterator
    __device__ void operator()(Tuple t);

    /// Perform a basic boundary check to correct the position and velocity
    __device__ void boundaryCheckSDF(float3 & /*pos*/, float3 & /*vel*/);
};

#endif // FLUIDINTEGRATOR_H
