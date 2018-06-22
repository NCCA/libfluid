#include "fluidphysics.cuh"
#include "fluidkernel.cuh"
#include "fluidparams.cuh"
#include "fluidutil.cuh"
#include "helper_math.h"

// This function will perform the grid operation, whether it is pressure or density or whatever
__global__ void computeDensity(float *pressure,
                               float *soundSpeed,
                               float *density,
                               const float3 *points,
                               const uint *cellOcc,
                               const uint *scatterAddress)
{
    // Compute the grid cell index from the block (not thread) id - this is because each block
    // is processing a different cell - the threadIdx.x is the index of the particle in that cell
    unsigned int gridCellIdx = cell_from_grid(blockIdx);
    // This is the value that is returned
    float sum = 0.0f;

    // Make sure we only execute threads where we have particles
    if (threadIdx.x < cellOcc[gridCellIdx])
    {
        uint thisPointIdx = scatterAddress[gridCellIdx] + threadIdx.x;
        float3 thisPoint = points[thisPointIdx];
        // Now we must iterate over the neighboring cells
        int i, j, k, threadInBlockIdx;
        uint otherGridCellIdx, otherPointIdx;
        // Note that all the block checks will be the same so there should be no branching in each block
        for (i = ((blockIdx.x == 0) ? 0 : -1); i <= ((blockIdx.x == (gridDim.x - 1)) ? 0 : 1); ++i)
        {
            for (j = ((blockIdx.y == 0) ? 0 : -1); j <= ((blockIdx.y == (gridDim.y - 1)) ? 0 : 1); ++j)
            {
                for (k = ((blockIdx.z == 0) ? 0 : -1); k <= ((blockIdx.z == (gridDim.z - 1)) ? 0 : 1); ++k)
                {
                    // Calculate the index of the other grid cell
                    otherGridCellIdx = cell_from_grid(make_uint3(blockIdx.x + i, blockIdx.y + j, blockIdx.z + k));
                    //printf("gridCellIdx=%d, otherGridCellIdx=%d\n",gridCellIdx,otherGridCellIdx);
                    // Now iterate over all particles in this neighbouring cell
                    for (threadInBlockIdx = 0; threadInBlockIdx < cellOcc[otherGridCellIdx]; ++threadInBlockIdx)
                    {
                        // Determine the index of the neighbouring point in that cell
                        otherPointIdx = scatterAddress[otherGridCellIdx] + threadInBlockIdx;
                        float rr = sqrdist(thisPoint, points[otherPointIdx]);
                        if ((otherPointIdx != thisPointIdx) && (rr <= params.m_h2))
                        {
                            // Calculate the weighted sum
                            sum += params.m_mass * poly6Kernel(sqrt(rr));
                        }
                    }
                }
            }
        }
        // Assign the density as the sum. If there are no nearby particles assign the rest density.
        float d = fabs(boundarySDF(points[thisPointIdx]));
        float correctedDensity = sum;
        if (d < params.m_h)
        {
            //float mass = params.m_restDensity * params.m_h2 * 1.0472f * (3.0 * params.m_h - d);
            correctedDensity += params.m_restDensity * (1.0f - boundaryDensity(d));
        }

        density[thisPointIdx] = (sum == 0.0f) ? params.m_restDensity : correctedDensity;

        // Correct for the boundary density here based on Fujisawa and Miura "An Efficient Boundary Handling with a Modified Density
        // Calculation for SPH" available here: https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.12754
        /*
        float d = fabs(boundarySDF(points[thisPointIdx]));
        if (d < params.m_h) {
            
            density[thisPointIdx] += (boundaryDensity(d));
        }
        */

        // This is the formulation based on WCSPH: MONAGHAN, J. 2005. Smoothed particle hydrodynamics. Rep. Prog. Phys. 68, 1703–1759
        // This is often referred to as the Tait Equation of State. The alternative formulation (from Muller) corresponds to the Ideal Gas law, with gamma=1.
        // If there are no nearby points, set the pressure to zero (this might not be physically justified)
        pressure[thisPointIdx] = (correctedDensity == 0.0f) ? 0.0f : (params.m_gasConstant * params.m_restDensity * params.m_invGamma) * (powf(correctedDensity * params.m_invRestDensity, params.m_gamma) - 1.0f);

        // Compute the speed of sound passing through this fluid (which is the derivative of pressure with respect to density)
        // This was derived from Eq 62 in http://ephyslab.uvigo.es/publica/documents/file_8Gomez-Gesteira_et_al_2010_JHR_SI.pdf
        soundSpeed[thisPointIdx] = params.m_gasConstant * powf(correctedDensity * params.m_invRestDensity, params.m_gamma - 1.0f);
    }
}

// This function will perform the grid operation, whether it is pressure or density or whatever
__global__ void computePressureGradient(float3 *force,
                                        const float *pressure,
                                        const float *density,
                                        const float3 *points,
                                        const uint *cellOcc,
                                        const uint *scatterAddress)
{
    // Compute the grid cell index from the block (not thread) id - this is because each block
    // is processing a different cell - the threadIdx.x is the index of the particle in that cell
    uint gridCellIdx = cell_from_grid(blockIdx);

    // This is the value that is returned
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);

    // Make sure we only execute threads where we have particles
    if (threadIdx.x < cellOcc[gridCellIdx])
    {
        uint thisPointIdx = scatterAddress[gridCellIdx] + threadIdx.x;

        // Now we must iterate over the neighboring cells
        int i, j, k, threadInBlockIdx;
        uint otherGridCellIdx, otherPointIdx;

        // Note that all the block checks will be the same so there should be no branching in each block
        for (i = ((blockIdx.x == 0) ? 0 : -1); i <= ((blockIdx.x == (gridDim.x - 1)) ? 0 : 1); ++i)
        {
            for (j = ((blockIdx.y == 0) ? 0 : -1); j <= ((blockIdx.y == (gridDim.y - 1)) ? 0 : 1); ++j)
            {
                for (k = ((blockIdx.z == 0) ? 0 : -1); k <= ((blockIdx.z == (gridDim.z - 1)) ? 0 : 1); ++k)
                {
                    // Calculate the index of the other grid cell
                    otherGridCellIdx = cell_from_grid(make_uint3(blockIdx.x + i, blockIdx.y + j, blockIdx.z + k));
                    // Now iterate over all particles in this neighbouring cell
                    for (threadInBlockIdx = 0; threadInBlockIdx < cellOcc[otherGridCellIdx]; ++threadInBlockIdx)
                    {
                        // Determine the index of the neighbouring point in that cell
                        otherPointIdx = scatterAddress[otherGridCellIdx] + threadInBlockIdx;
                        float3 rvec = points[otherPointIdx] - points[thisPointIdx];
                        float r2 = dot(rvec, rvec);
                        if ((otherPointIdx != thisPointIdx) &&         // Check the point indices aren't the same
                            (density[otherPointIdx] > params.m_eps) && // Avoid divide by zero
                            (r2 <= params.m_h2) &&                     // Don't bother if outside smoothing length
                            (r2 > params.m_eps))                       // Avoid divide by zero
                        {
                            // Calculate the weighted sum
                            float r = sqrt(r2);
                            float inv_r = 1.0f / r;
                            // This is the pressure gradient formulation from Muller - I've found it to be unstable
                            // float coeff = params.m_mass * gradSpikyKernel(r) *
                            //      (pressure[otherPointIdx] + pressure[thisPointIdx]) / (2.0f * density[otherPointIdx]);

                            // This would be the alternative (more common) pressure gradient term
                            float coeff = params.m_mass * gradSpikyKernel(r) *
                                          (pressure[otherPointIdx] / (density[thisPointIdx] * density[thisPointIdx]) +
                                           pressure[thisPointIdx] / (density[otherPointIdx] * density[otherPointIdx]));

                            sum += coeff * rvec * inv_r;
                        }
                    }
                }
            }
        }
        force[thisPointIdx] = sum;
    }
}

// This function will perform the grid operation, whether it is pressure or density or whatever
__global__ void computeNormals(float3 *normals,
                               const float *density,
                               const float3 *points,
                               const uint *cellOcc,
                               const uint *scatterAddress)
{
    // Compute the grid cell index from the block (not thread) id - this is because each block
    // is processing a different cell - the threadIdx.x is the index of the particle in that cell
    uint gridCellIdx = cell_from_grid(blockIdx);

    // This is the value that is returned
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);

    // Make sure we only execute threads where we have particles
    if (threadIdx.x < cellOcc[gridCellIdx])
    {
        uint thisPointIdx = scatterAddress[gridCellIdx] + threadIdx.x;

        // Now we must iterate over the neighboring cells
        int i, j, k, threadInBlockIdx;
        uint otherGridCellIdx, otherPointIdx;

        // Note that all the block checks will be the same so there should be no branching in each block
        for (i = ((blockIdx.x == 0) ? 0 : -1); i <= ((blockIdx.x == (gridDim.x - 1)) ? 0 : 1); ++i)
        {
            for (j = ((blockIdx.y == 0) ? 0 : -1); j <= ((blockIdx.y == (gridDim.y - 1)) ? 0 : 1); ++j)
            {
                for (k = ((blockIdx.z == 0) ? 0 : -1); k <= ((blockIdx.z == (gridDim.z - 1)) ? 0 : 1); ++k)
                {
                    // Calculate the index of the other grid cell
                    otherGridCellIdx = cell_from_grid(make_uint3(blockIdx.x + i, blockIdx.y + j, blockIdx.z + k));
                    // Now iterate over all particles in this neighbouring cell
                    for (threadInBlockIdx = 0; threadInBlockIdx < cellOcc[otherGridCellIdx]; ++threadInBlockIdx)
                    {
                        // Determine the index of the neighbouring point in that cell
                        otherPointIdx = scatterAddress[otherGridCellIdx] + threadInBlockIdx;
                        float3 rvec = points[thisPointIdx] - points[otherPointIdx];
                        float r2 = dot(rvec, rvec);
                        if ((otherPointIdx != thisPointIdx) &&         // Check the point indices aren't the same
                            (density[otherPointIdx] > params.m_eps) && // Avoid divide by zero
                            (r2 <= params.m_h2) &&                     // Don't bother if outside smoothing length
                            (r2 > params.m_eps))                       // Avoid divide by zero
                        {
                            // Calculate the weighted sum
                            float r = sqrt(r2);

                            // This is the unlabelled formula for normal calculate as per Akinci et al.
                            sum += (params.m_mass * gradPoly6Kernel(r) * rvec) / (density[otherPointIdx] * r);
                        }
                    }
                }
            }
        }
        // Scale the un-normalised normal by the smoothing length as it is presented in their paper
        normals[thisPointIdx] = params.m_h * sum;
    }
}

/**
 * This combined kernel computes all forces applied to the particles, including pressure gradient, viscosity and surface tension.
 */
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
    const uint *scatterAddress)
{
    // Compute the grid cell index from the block (not thread) id - this is because each block
    // is processing a different cell - the threadIdx.x is the index of the particle in that cell
    uint gridCellIdx = cell_from_grid(blockIdx);

    // This is the value that is returned
    float3 pressureSum = make_float3(0.0f, 0.0f, 0.0f);
    float3 viscositySum = make_float3(0.0f, 0.0f, 0.0f);
    float3 tensionSum = make_float3(0.0f, 0.0f, 0.0f);

    // Make sure we only execute threads where we have particles
    if (threadIdx.x < cellOcc[gridCellIdx])
    {
        uint thisPointIdx = scatterAddress[gridCellIdx] + threadIdx.x;

        // Now we must iterate over the neighboring cells
        int i, j, k, threadInBlockIdx;
        uint otherGridCellIdx, otherPointIdx;

        // Note that all the block checks will be the same so there should be no branching in each block
        for (i = ((blockIdx.x == 0) ? 0 : -1); i <= ((blockIdx.x == (gridDim.x - 1)) ? 0 : 1); ++i)
        {
            for (j = ((blockIdx.y == 0) ? 0 : -1); j <= ((blockIdx.y == (gridDim.y - 1)) ? 0 : 1); ++j)
            {
                for (k = ((blockIdx.z == 0) ? 0 : -1); k <= ((blockIdx.z == (gridDim.z - 1)) ? 0 : 1); ++k)
                {
                    // Calculate the index of the other grid cell
                    otherGridCellIdx = cell_from_grid(make_uint3(blockIdx.x + i, blockIdx.y + j, blockIdx.z + k));
                    // Now iterate over all particles in this neighbouring cell
                    for (threadInBlockIdx = 0; threadInBlockIdx < cellOcc[otherGridCellIdx]; ++threadInBlockIdx)
                    {
                        // Determine the index of the neighbouring point in that cell
                        otherPointIdx = scatterAddress[otherGridCellIdx] + threadInBlockIdx;
                        float3 rvec = points[otherPointIdx] - points[thisPointIdx];
                        float r2 = dot(rvec, rvec);
                        if ((otherPointIdx != thisPointIdx) &&         // Check the point indices aren't the same
                            (density[otherPointIdx] > params.m_eps) && // Avoid divide by zero
                            (r2 <= params.m_h2) &&                     // Don't bother if outside smoothing length
                            (r2 > params.m_eps))                       // Avoid divide by zero
                        {
                            // Calculate the weighted sum
                            float r = sqrt(r2);
                            float inv_r = 1.0f / r;

                            // Calculate the components of the viscosity term
                            float3 u_ij = velocity[thisPointIdx] - velocity[otherPointIdx];
                            float ux = dot(u_ij, -rvec);
                            float Pi_ij = 0.0f;

                            if (ux < 0.0f)
                            {
                                float mu_ij = (params.m_h * ux) / (r2 + 0.01 * params.m_h2);
                                Pi_ij = (-params.m_viscosity * (soundSpeed[thisPointIdx] + soundSpeed[otherPointIdx]) * mu_ij) / (density[thisPointIdx] + density[otherPointIdx]);
                            }
                            float midTerm = pressure[otherPointIdx] / (density[thisPointIdx] * density[thisPointIdx]) + pressure[thisPointIdx] / (density[otherPointIdx] * density[otherPointIdx]);
                            float3 commonTerm = params.m_mass * gradSpikyKernel(r) * rvec * inv_r;

                            // Note that the terms are shared between the pressure and viscosity terms so these are computed together
                            pressureSum += commonTerm * midTerm;
                            viscositySum += commonTerm * (midTerm + Pi_ij);

                            // Cohesion and curvature forces defined by Akinci in
                            // https://cg.informatik.uni-freiburg.de/publications/siggraphasia2013/2013_SIGGRAPHASIA_surface_tension_adhesion.pdf
                            float3 cohesion = -params.m_mass * params.m_mass * cohesionKernel(r) * rvec * inv_r;
                            float3 curvature = -params.m_surfaceTension * params.m_mass * (normals[thisPointIdx] - normals[otherPointIdx]);
                            tensionSum += (2.0f * params.m_restDensity * (cohesion + curvature)) / (density[thisPointIdx] + density[thisPointIdx]);
                        }
                    }
                }
            }
        }
        // Write out the final force terms here
        pressureForce[thisPointIdx] = pressureSum;
        viscosityForce[thisPointIdx] = params.m_viscosity * viscositySum;
        tensionForce[thisPointIdx] = params.m_surfaceTension * tensionSum;

        /*
        // Use this to compute the pressure contribution from the boundary
        float d = fabs(boundarySDF(points[thisPointIdx]));
        if (d < params.m_h) {
            // Note that the normal for the SDF by default points out of the surface (rather than in)
            float3 normal = boundaryNormalSDF(points[thisPointIdx]);

            // Compute the mass for the spherical cap defined by the intersection of the boundary and the
            // sphere about the particle with radius h. This is the product of the density and the volume 
            // of the spherical cap, defined here https://en.wikipedia.org/wiki/Spherical_cap.
            float mass = params.m_restDensity * params.m_h2 * 1.0472f * (3.0 * params.m_h - d);

            // Compute the pressure gradient contribution from the boundary surface represented by the spherical cap 
            // Note that this is given by the standard equation for pressure with the integrated pressure gradient 
            // constribution included, except the term 
            // pressure[otherPointIdx]/density[thisPointIdx]^2 = 0 as we'll assume the pressure is zero at the boundary.
            float3 pressureCorrect = -mass * pressure[thisPointIdx] * params.m_invRestDensity * params.m_invRestDensity * 
                                           boundaryPressureGrad(d) * normal;
            //printf("pressure before=[%f,%f,%f], pressure correct=[%f,%f,%f]\n",
            //    pressureForce[thisPointIdx].x, pressureForce[thisPointIdx].y, pressureForce[thisPointIdx].z,
            //    pressureCorrect.x, pressureCorrect.y, pressureCorrect.z);
            pressureForce[thisPointIdx] += pressureCorrect;
        
        } 
        */

        // Adhesion force involves the fluid particles sticking to the bounding planes
        adhesionForce[thisPointIdx] = make_float3(0.0f, 0.0f, 0.0f);

        float3 pos = points[thisPointIdx];
        float z = -fminf(boundarySDF(pos), 0.0f) * 0.25f * params.m_invParticleSize;

        if (z < 1.0f)
        {
            // Note that the normal for the SDF by default points out of the surface (rather than in)
            float3 normal = boundaryNormalSDF(pos);

            float denom = 1.0f / powf(1.0f - z, 3.0f);

            // Compute the adhesion force based on the equation from https://en.wikipedia.org/wiki/Adhesion which inteprets as
            // F_adhesion = (-pi * C * rho_i * rho_j * da) / (24 * z^3)
            // We'll let our adhesion constant be (-pi * C * rho_j) / 24 to reduce the number of multiplications. Currently it is unclear
            // what the area should be - could be h or it could be particle size?
            // This is the van Der Waals forces of the fluid in contact with the boundary.
            //adhesionForce[thisPointIdx] = ((params.m_adhesion * density[thisPointIdx] * params.m_invRestDensity) / (z * z * z) ) * normal;
            adhesionForce[thisPointIdx] = (params.m_adhesion * density[thisPointIdx] * params.m_invRestDensity * denom) * normal;
        }
    }
}

// This function will perform the grid operation, whether it is pressure or density or whatever
__global__ void correctVelocityXSPH(float3 *velocity,
                                    const float *density,
                                    const float3 *points,
                                    const uint *cellOcc,
                                    const uint *scatterAddress)
{
    // Compute the grid cell index from the block (not thread) id - this is because each block
    // is processing a different cell - the threadIdx.x is the index of the particle in that cell
    unsigned int gridCellIdx = cell_from_grid(blockIdx);

    // This is the value that is returned
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);

    // Make sure we only execute threads where we have particles
    if (threadIdx.x < cellOcc[gridCellIdx])
    {
        uint thisPointIdx = scatterAddress[gridCellIdx] + threadIdx.x;
        float3 thisPoint = points[thisPointIdx];

        // Now we must iterate over the neighboring cells
        int i, j, k, threadInBlockIdx;
        uint otherGridCellIdx, otherPointIdx;

        // Note that all the block checks will be the same so there should be no branching in each block
        for (i = ((blockIdx.x == 0) ? 0 : -1); i <= ((blockIdx.x == (gridDim.x - 1)) ? 0 : 1); ++i)
        {
            for (j = ((blockIdx.y == 0) ? 0 : -1); j <= ((blockIdx.y == (gridDim.y - 1)) ? 0 : 1); ++j)
            {
                for (k = ((blockIdx.z == 0) ? 0 : -1); k <= ((blockIdx.z == (gridDim.z - 1)) ? 0 : 1); ++k)
                {
                    // Calculate the index of the other grid cell
                    otherGridCellIdx = cell_from_grid(make_uint3(blockIdx.x + i, blockIdx.y + j, blockIdx.z + k));
                    // Now iterate over all particles in this neighbouring cell
                    for (threadInBlockIdx = 0; threadInBlockIdx < cellOcc[otherGridCellIdx]; ++threadInBlockIdx)
                    {
                        // Determine the index of the neighbouring point in that cell
                        otherPointIdx = scatterAddress[otherGridCellIdx] + threadInBlockIdx;
                        float rr = sqrdist(thisPoint, points[otherPointIdx]);
                        if ((otherPointIdx != thisPointIdx) && // Check they're not the same point
                            (rr <= params.m_h2) &&             // Check if outside smoothing length
                            ((density[thisPointIdx] + density[otherPointIdx]) > params.m_eps))
                        {
                            // Calculate the weighted sum
                            float r = sqrt(rr);
                            float coeff = poly6Kernel(r) * ((2.0f * params.m_mass) / (density[thisPointIdx] + density[otherPointIdx]));
                            sum += coeff * (velocity[otherPointIdx] - velocity[thisPointIdx]);
                        }
                    }
                }
            }
        }
        // Update the velocity based on the calculation above
        velocity[thisPointIdx] += params.m_xsphScale * sum;
    }
}

/**
 * Simple function that estimates distance to boundary using the magnitude of the computed normal and
 * computes the density correction accordingly.
 */
//__global__ void correctDensity(float *density,
//const float3 *normals) {

//}
