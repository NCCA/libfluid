#include "fluidintegrator.cuh"
#include "fluidparams.cuh"

/**
 * Currently the constructor does nothing
 */
LeapfrogIntegratorOperator::LeapfrogIntegratorOperator()
{
}

__device__ void LeapfrogIntegratorOperator::boundaryCheckSDF(float3 &pos, float3 &vel)
{
    // Return the SIGNED distance to the boundary. Note that means that if you're within the boundary this is negative, outside is positive.
    float d = boundarySDF(pos);
    if (d > -params.m_particleSize)
    {
        // Assume that the normal evaluated at this point in the SDF is close enough to the surface normal (a mistake?)
        float3 normal = boundaryNormalSDF(pos);
        pos -= (d + params.m_particleSize) * normal;

        // This is the reflection of the velocity component. Note that this is currently damped
        // by m_vdamp. If 1.0 this is a pure reflection, if 0.0 this means the orthogonal
        // component is effectively zeroed.
        vel -= (1.0f + params.m_vdamp) * dot(normal, vel) * normal;
    }
}

/**
 * The operator() allows this class to be used as a functor. This effectively performs the integration given
 * the last time steps position, velocity and acceleration and the current timesteps forces. 
 * \param The tuple is defined in the corresponding header.
 */
__device__ void LeapfrogIntegratorOperator::operator()(Tuple t)
{
    // Compute the velocity by blending between the last two versions
    float3 pos = thrust::get<0>(t);
    float3 vel = thrust::get<1>(t);
    float3 acc = thrust::get<2>(t);

    // Combine the force terms to be used for integration
    float3 force = thrust::get<3>(t) + thrust::get<4>(t) + thrust::get<5>(t) + thrust::get<6>(t);

    float3 acc_next = force * params.m_mass + params.m_gravity;

    // Sanity check on our acceleration to check it's within our limits
    float sqr_mag = dot(acc_next, acc_next);
    if (sqr_mag > params.m_accLimit2)
    {
        acc_next *= params.m_accLimit / sqrt(sqr_mag);
    }

    // Leapfrog integration as per https://en.wikipedia.org/wiki/Leapfrog_integration
    pos += vel * params.m_dt + 0.5f * acc * params.m_dtdt;
    vel += 0.5f * (acc + acc_next) * params.m_dt;

    // Sanity check on our velocity to check it's within the tolerable limits
    sqr_mag = dot(vel, vel);
    if (sqr_mag > params.m_velLimit2)
    {
        vel *= params.m_velLimit / sqrt(sqr_mag);
    }

    // Check for boundary collision against all the sides
    boundaryCheckSDF(pos, vel);

    // Write out the final positions and velocity
    thrust::get<0>(t) = pos;
    thrust::get<1>(t) = vel;
    thrust::get<2>(t) = acc_next;
}
