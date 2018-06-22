// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <thrust/device_vector.h>
#include "fluidsystem.h"
#include "hash.cuh"
//#include "random.cuh"
#include "fluidintegrator.cuh"
#include "fluidparams.cuh"
#include "fluidphysics.cuh"
#include <random>

/**
 * Construct an uninitialised fluid system
 */
FluidSystem::FluidSystem() : m_isInit(false)
{
    // Note that there are no smart pointers in CUDA. m_params needs to be forward declared due to circular dependencies, which means it has to be a pointer.
    m_params = new FluidParams();
}

/**
 * Delete everything.
 */
FluidSystem::~FluidSystem()
{
    clear();
    delete m_params;
}

/**
 * Delete all the data associated with the system.
 */
void FluidSystem::clear()
{
    // Clear up all the basic problem components
    m_Points.clear();
    m_Normals.clear();
    m_Velocity.clear();
    m_Acceleration.clear();
    m_PressureForce.clear();
    m_TensionForce.clear();
    m_AdhesionForce.clear();
    m_ViscosityForce.clear();
    m_Hash.clear();
    m_CellOcc.clear();
    m_ScatterAddress.clear();
    m_Density.clear();
    m_Pressure.clear();
}

void FluidSystem::createFluidBlock(const uint & firstPtIdx,
                                   const uint & numPoints,
                                   const float &min_x, const float &min_y, const float &min_z,
                                   const float &max_x, const float &max_y, const float &max_z)
{

    // Sanity check
    if ((firstPtIdx + numPoints) > m_params->getNumParticles())
    {
        std::cerr << "FluidSystem::createFluidBlock() - attempting to make " << numPoints << " points but you only have enough space for " << m_params->getNumParticles() - firstPtIdx << "\n!";
        return;
    }

    // I need to do this rather than passing float3's to the function as make_float3 may not be defined on host compiled code.
    float3 minCorner = make_float3(min_x, min_y, min_z);
    float3 maxCorner = make_float3(max_x, max_y, max_z);

    // Calculate the size of the particles based on the bounds of the box
    float3 diff = maxCorner - minCorner;
    float volume = diff.x * diff.y * diff.z;

    // Compute the average volume for each particle
    float partVolume = volume / float(numPoints);

    // Determine the ideal particle radius
    float approxPartDiameter = cbrt(partVolume);

    // Now we need to correct the particle radius. This is done by dividing the volume in each dimension by the
    // approximate diameter to get the "resolution" in that dimension. This is rounded up to get the upper bound partRes.
    float3 tmp = diff / approxPartDiameter;
    //float3 partRes = make_float3(ceilf(tmp.x), ceilf(tmp.y), ceilf(tmp.z));
    float3 partRes = tmp + make_float3(0.5f);

    // The upper bound is then used to determine the radius in each case, the maximum of which defines the particle diameter
    // that we'll use.
    float partDiameter = min(min(diff.x / partRes.x, diff.y / partRes.y), diff.z / partRes.z);

    // The mass is just going to be the optimal mass, rather than the actual mass for each particle in the volume as they are
    // packed. This might cause some slight adjustments during the first couple of time steps of the simulation.
    m_params->setMass(partVolume * m_params->getRestDensity());

    // The radius of a particle is half the diameter (did I need a comment for this?)
    m_params->setParticleSize(partDiameter * 0.5f);

    // Make our points in a boring loop (I can do this in parallel obviously)
    uint cnt = 0;
    float3 pos;
    for (pos.x = minCorner.x + 0.5f * partDiameter; pos.x <= maxCorner.x; pos.x += partDiameter)
    {
        for (pos.y = minCorner.y + 0.5f * partDiameter; pos.y <= maxCorner.y; pos.y += partDiameter)
        {
            for (pos.z = minCorner.z + 0.5f * partDiameter; pos.z <= maxCorner.z; pos.z += partDiameter)
            {
                if (cnt < numPoints)
                {
                    m_Points[firstPtIdx + cnt] = pos;
                    cnt++;
                }
            }
        }
    }
}

/**
 * Default initialiser for the fluid problem
 */
void FluidSystem::setup(const uint &_numPoints, const uint &_res)
{
    // Clear the data structures
    init(_numPoints, _res);

    m_params->setAdhesion(0.0f);
    m_params->setViscosity(0.0f);
    m_params->setSurfaceTension(0.0f);
    m_params->setVDamp(0.5f);

    // Create a default slab of fluid for fun and profit
    createFluidBlock(0,              //firstPtIdx,
                     _numPoints,     //numPoints,
                     0.1, 0.1, 0.1,  //minCorner,
                     0.9, 0.9, 0.9); //maxCorner)
}

/**
 * Export the data to the CPU, presumably to dump in some sort of output file for use in an external
 * DCC. Note that there are serious performance implications to doing this during an advance loop.
 * It should be possible to improve the performance substantially with cuda streams.
 */
void FluidSystem::exportToData(std::vector<float3> &_points,
                               std::vector<float3> &_colour,
                               std::vector<float3> &_velocity)
{
    // Calculate the colour of this particle based on the density
    thrust::device_vector<float3> Colour(m_Density.size());
    DensityColourOperator dc;
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(m_Density.begin(),
                                                                  Colour.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(m_Density.end(),
                                                                  Colour.end())),

                     dc);
    // Copy the colour across to the CPU (thrust should interleave the writes)
    thrust::copy(Colour.begin(), Colour.end(), _colour.begin());

    // Copy the velocity and point data across to the CPU
    thrust::copy(m_Points.begin(), m_Points.end(), _points.begin());
    thrust::copy(m_Velocity.begin(), m_Velocity.end(), _velocity.begin());
}

/**
 * In case you want to load up your data from an external source. I've not tested this, but have no
 * reason to suspect it doesn't work.
 */
void FluidSystem::setupFromData(const std::vector<float3> &_points,
                                const std::vector<float3> &_velocity,
                                const uint &_res)
{
    // Clear the data structures
    init(_points.size(), _res);

    // Copy the data over using thrust (hopefully)
    thrust::copy(_points.begin(), _points.end(), m_Points.begin());
    thrust::copy(_velocity.begin(), _velocity.end(), m_Velocity.begin());
}

/** 
 * Resize all of the vectors to initialise this class. Note that the rest density is currently unknown:
 * unless you want the simulation to explode you might want to set the rest density using the initRestDensity()
 * function, which determines the mean density of the current configuration.
 */
void FluidSystem::init(const uint &_numPoints, const uint &_res)
{
    // Clear away all existing memory
    clear();

    // Sets the resolution - also sets the smoothing length and the kernel constants
    m_params->setRes(_res);

    // The number of particles in the simulation
    m_params->setNumParticles(_numPoints);

    // A gas constant. The value of 100.0f came from Fluids v3 I think. It is a stiffness parameter for the pressure.
    // This seems to have the biggest impact on the springiness of the fluid, with high values preventing it from
    // converging.
    m_params->setGasConstant(100.0f);

    // Velocity damping ranges between 0 and 1 and determines how bouncy particles are when they hit the boundary.
    m_params->setVDamp(0.0f);

    // This value ranges from 1.0 (Muller et al) to 7.0 (Monaghan et al).
    // The power it being computed regardless so there is no difference in performance currently.
    m_params->setGamma(7.0f);

    // The first value is the velocity limit, second is acceleration limit. Fluids v3 used [5,150].
    m_params->setLimits(5.0f, 150.0f);

    // The initial DT is just a dummy time step - it will be changed when advance is called.
    m_params->setDT(1.0f);

    // 0.5 is the default XSPH blending parameter defined for the dam break simulation. Values between 0 (off) and 1 are common.
    m_params->setXSPHScale(0.5f);

    // This determines tolerance for boundary collisions. It should be low (and can be zero I think)
    m_params->setParticleSize(1.0f);

    // Pretty standard formulation for the rest density of water - the actual density is 997kg/m^3.
    m_params->setRestDensity(1000.0f);

    // The mass of the particles tends to introduce numerical imprecision so lets keep it at a constant 1 for each particle.
    m_params->setMass(1.0f);

    // Resize the structures for the problem
    m_Points.resize(m_params->getNumParticles(), make_float3(0.0f, 0.0f, 0.0f));
    m_Normals.resize(m_params->getNumParticles(), make_float3(0.0f, 1.0f, 0.0f));
    m_Velocity.resize(m_params->getNumParticles(), make_float3(0.0f, 0.0f, 0.0f));
    m_Acceleration.resize(m_params->getNumParticles(), make_float3(0.0f, 0.0f, 0.0f));
    m_PressureForce.resize(m_params->getNumParticles(), make_float3(0.0f, 0.0f, 0.0f));
    m_TensionForce.resize(m_params->getNumParticles(), make_float3(0.0f, 0.0f, 0.0f));
    m_AdhesionForce.resize(m_params->getNumParticles(), make_float3(0.0f, 0.0f, 0.0f));
    m_ViscosityForce.resize(m_params->getNumParticles(), make_float3(0.0f, 0.0f, 0.0f));
    m_Hash.resize(m_params->getNumParticles(), 0);
    m_CellOcc.resize(m_params->getNumCells(), 0);
    m_ScatterAddress.resize(m_params->getNumCells(), 0);
    m_Density.resize(m_params->getNumParticles(), 0.0f);
    m_SoundSpeed.resize(m_params->getNumParticles(), 0.0f);
    m_Pressure.resize(m_params->getNumParticles(), 0.0f);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Thrust allocation failed, error " << cudaGetErrorString(err) << "\n";
        exit(0);
    }

    // Set our flag to indicate that we are ready to start simulating
    m_isInit = true;
}

/**
 * Move the fluid solver forward one step. This is the meat of the FluidSystem class, and is generally responsible for the heavy
 * lifting.
 * \param full_dt The total time step.
 * \param substeps The amount of substeps to perform. The actual dt for each substep is full_dt / substeps.
 */
void FluidSystem::advance(const float &full_dt, const uint &substeps)
{
    if (!m_isInit)
        return;

    // Raw pointer casting needed for kernel calls
    float3 *Points_ptr = thrust::raw_pointer_cast(&m_Points[0]);
    float3 *Normals_ptr = thrust::raw_pointer_cast(&m_Normals[0]);
    float3 *Velocity_ptr = thrust::raw_pointer_cast(&m_Velocity[0]);
    uint *CellOcc_ptr = thrust::raw_pointer_cast(&m_CellOcc[0]);
    uint *ScatterAddress_ptr = thrust::raw_pointer_cast(&m_ScatterAddress[0]);
    float *Density_ptr = thrust::raw_pointer_cast(&m_Density[0]);
    float *Pressure_ptr = thrust::raw_pointer_cast(&m_Pressure[0]);
    float *SoundSpeed_ptr = thrust::raw_pointer_cast(&m_SoundSpeed[0]);
    float3 *PressureForce_ptr = thrust::raw_pointer_cast(&m_PressureForce[0]);
    float3 *ViscosityForce_ptr = thrust::raw_pointer_cast(&m_ViscosityForce[0]);
    float3 *AdhesionForce_ptr = thrust::raw_pointer_cast(&m_AdhesionForce[0]);
    float3 *TensionForce_ptr = thrust::raw_pointer_cast(&m_TensionForce[0]);

    // Perform substep iterations
    float dt;
    float part_dt = full_dt / float(substeps);

    // Set up the parameters and synchronise on the GPU if necessary
    m_params->setDT(part_dt);
    m_params->sync();
    m_params->debugPrint();

    for (dt = part_dt; dt <= full_dt; dt += part_dt)
    {
        // Perform a point hash operation using thrust
        thrust::fill(m_CellOcc.begin(), m_CellOcc.end(), 0);                      // clear the existing list
        PointHashOperator pop(CellOcc_ptr);                                       // create the operator to use
        thrust::transform(m_Points.begin(), m_Points.end(), m_Hash.begin(), pop); // apply the operator

        // We've calculated the point hash, now we need to sort the data for cache coherence
        thrust::sort_by_key(m_Hash.begin(),
                            m_Hash.end(),
                            thrust::make_zip_iterator(thrust::make_tuple(m_Points.begin(),
                                                                         m_Velocity.begin(),
                                                                         m_Acceleration.begin())));

        // Calculate the memory scatter addresses
        thrust::exclusive_scan(m_CellOcc.begin(), m_CellOcc.end(), m_ScatterAddress.begin());

        // Here we need to partition the problem and execute the GPU kernel
        uint maxCellOcc = thrust::reduce(m_CellOcc.begin(), m_CellOcc.end(), 0, thrust::maximum<unsigned int>());

        // Determine the parameters for GPU execution based on the cell with the most particles
        uint blockSize = 32 * ceil(maxCellOcc / 32.0f);
        dim3 gridSize(m_params->getRes(), m_params->getRes(), m_params->getRes());

        // Helpful to have this information, as it may affect your grid resolution
        std::cout << "maxCellOcc=" << maxCellOcc << ", blockSize=" << blockSize << ", gridSize=" << m_params->getRes() << "^3\n";

        // Compute the density and pressure in one step
        computeDensity<<<gridSize, blockSize>>>(
            Pressure_ptr,      // The particle pressure, float, size numPoints
            SoundSpeed_ptr,    // The speed of sound calculation, float, size numPoints
            Density_ptr,       // The particle density, float, size numPoints
            Points_ptr,        // The points positions, float3, size numPoints
            CellOcc_ptr,       // The cell occupancy for each cell, uint, size numCells
            ScatterAddress_ptr // The scatter addresses for the start of each cell, uint, size numCells
        );
        cudaThreadSynchronize();

        // Determine the surface normals (used for lots of things, including surface tension and adhesion)
        computeNormals<<<gridSize, blockSize>>>(
            Normals_ptr,         //float3 *normals
            Density_ptr,         //const float *density
            Points_ptr,          //const float3 *points
            CellOcc_ptr,         //const uint *cellOcc
            ScatterAddress_ptr); //const uint *scatterAddress
        cudaThreadSynchronize();

        // Calculate the pressure, surface tension, adhesion and viscosity forces in a single step
        computeAllForces<<<gridSize, blockSize>>>(
            PressureForce_ptr,
            TensionForce_ptr,
            ViscosityForce_ptr,
            AdhesionForce_ptr,
            SoundSpeed_ptr,
            Pressure_ptr,
            Density_ptr,
            Points_ptr,
            Normals_ptr,
            Velocity_ptr,
            CellOcc_ptr,
            ScatterAddress_ptr);
        cudaThreadSynchronize();

        // Perform the integration using our leapfrog integrator
        LeapfrogIntegratorOperator leapfrog;

        // Set a tuple consisting of the points, velocity, force, acceleration to the gpu
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(m_Points.begin(),
                                                                      m_Velocity.begin(),
                                                                      m_Acceleration.begin(),
                                                                      m_PressureForce.begin(),
                                                                      m_TensionForce.begin(),
                                                                      m_AdhesionForce.begin(),
                                                                      m_ViscosityForce.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(m_Points.end(),
                                                                      m_Velocity.end(),
                                                                      m_Acceleration.end(),
                                                                      m_PressureForce.end(),
                                                                      m_TensionForce.end(),
                                                                      m_AdhesionForce.end(),
                                                                      m_ViscosityForce.end())),
                         leapfrog);

        // Apply XSPH to smooth the velocity between time steps
        correctVelocityXSPH<<<gridSize, blockSize>>>(
            Velocity_ptr,
            Density_ptr,
            Points_ptr,
            CellOcc_ptr,
            ScatterAddress_ptr);
        cudaThreadSynchronize();
    }
}

/// Set the rest density (for something other than water, which is 998 kg/m^3)
void FluidSystem::setRestDensity(const float &_restDensity)
{
    m_params->setRestDensity(_restDensity);
}

/// Set the size of the particles (radius)
void FluidSystem::setParticleSize(const float &_particleSize)
{
    m_params->setParticleSize(_particleSize);
}

/// This has been described as a stiffness term for pressure. Fluidsv3 uses 100, so do I.
void FluidSystem::setGasConstant(const float &_gasConstant)
{
    m_params->setGasConstant(_gasConstant);
}

/// Set the velocity and acceleration limits (fiddly)
void FluidSystem::setLimits(const float &_velLimit, const float &_accLimit)
{
    m_params->setLimits(_velLimit, _accLimit);
}

/// Set the level of velocity smoothing due to the XSPH method. Typically 0.5 works.
void FluidSystem::setXSPHScale(const float &_xsphScale)
{
    m_params->setXSPHScale(_xsphScale);
}

/// Set the coefficient for surface tension. Somewhere between 0 and 1 would be a good choice.
void FluidSystem::setSurfaceTension(const float &_surfaceTension)
{
    m_params->setSurfaceTension(_surfaceTension);
}

/// Set the coefficient for viscocity. Somewhere between 0 and 1 would be a good choice.
void FluidSystem::setViscosity(const float &_viscosity)
{
    m_params->setViscosity(_viscosity);
}

/// Set the coefficient for adhesion forces. Somewhere between 0 and 1 would be a good choice.
void FluidSystem::setAdhesion(const float &_adhesion)
{
    m_params->setAdhesion(_adhesion);
}

/// Set how bouncy the fluid will be when it hits a boundary. Somewhere between 0 and 1 would be a good choice.
void FluidSystem::setVDamp(const float &_vdamp)
{
    m_params->setVDamp(_vdamp);
}

/// A coefficient used for determining the density. Ranges from 1 (Muller) to 7 (Monaghan). No performance improvements for choosing 1!
void FluidSystem::setGamma(const float &_gamma)
{
    m_params->setGamma(_gamma);
}

/// Set the particle mass. This is the same for all particles in this implementation. Should be based on initial density and volume.
void FluidSystem::setMass(const float &_mass)
{
    m_params->setMass(_mass);
}
