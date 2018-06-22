#ifndef FLUIDSYSTEM_H
#define FLUIDSYSTEM_H

#include <vector>
#include <thrust/device_vector.h>

// This file includes the default parameters for the fluid system
#include "fluiddefines.h"

// Forward declaration for this class so we can create a member and include this file in g++ compiled code
class FluidParams;

/**
 *
 */
class FluidSystem {
  public:
    typedef enum {
      STATIC=1,
      DYNAMIC=2
    } ParticleState;

    /// Construct an empty fluid system
    FluidSystem();

    /// Destruct our fluid system
    virtual ~FluidSystem();

    /// Initialise a relatively standard dambreak simulation
    virtual void setup(const uint &_numPoints, const uint &_res);

    /// Initialise from data generated externally
    void setupFromData(const std::vector<float3>& /*points*/,
                       const std::vector<float3>& /*velocity*/,
                       const uint &_res = 32);

    /// Dump the data from the current frame to an output vector
    void exportToData(std::vector<float3>& /*points*/,
                      std::vector<float3>& /*colour*/,
                      std::vector<float3>& /*velocity*/);

    /// Progress the simulation
    void advance(const float& /*dt*/ = DEFAULT_TIMESTEP, const uint & /*substeps*/ = 10);

    /// Set the rest density (for something other than water, which is 998 kg/m^3)
    void setRestDensity(const float& _restDensity);

    /// Set the size of the particles (radius)
    void setParticleSize(const float& _particleSize);

    /// This has been described as a stiffness term for pressure. Fluidsv3 uses 100, so do I.
    void setGasConstant(const float& _gasConstant);

    /// Set the velocity and acceleration limits (fiddly)
    void setLimits(const float& _velLimit, const float& _accLimit);

    /// Set the level of velocity smoothing due to the XSPH method. Typically 0.5 works.
    void setXSPHScale(const float& _xsphScale);

    /// Set the coefficient for surface tension. Somewhere between 0 and 1 would be a good choice.
    void setSurfaceTension(const float &_surfaceTension);    

    /// Set the coefficient for viscocity. Somewhere between 0 and 1 would be a good choice.
    void setViscosity(const float &_viscosity);

    /// Set the coefficient for adhesion forces. Somewhere between 0 and 1 would be a good choice.
    void setAdhesion(const float &_adhesion);

    /// Set how bouncy the fluid will be when it hits a boundary. Somewhere between 0 and 1 would be a good choice.
    void setVDamp(const float &_vdamp);

    /// A coefficient used for determining the density. Ranges from 1 (Muller) to 7 (Monaghan). No performance improvements for choosing 1!
    void setGamma(const float& _gamma);

    /// Set the particle mass. This is the same for all particles in this implementation. Should be based on initial density and volume.
    void setMass(const float& _mass);

  protected:
    /// Keep track of whether the simulation is ready to start
    bool m_isInit;

    /// Perform the initialisation of this class by setting up all the memory for advance()
    void init(const uint &_numPoints, const uint &_res);

    /// Clear away all the vector data
    void clear();

    /// The points, velocity and acceleration are stored on the GPU and reused between frames
    thrust::device_vector<float3> m_Points, m_Normals, m_Velocity, m_Acceleration;

    /// The different forces at work on our particles in the simulation
    thrust::device_vector<float3> m_PressureForce, m_ViscosityForce, m_TensionForce, m_AdhesionForce;

    /// Individual point hash for each point - length numPoints
    thrust::device_vector<uint> m_Hash;

    /// Cell occupancy count for each cell - length numCells = res^3
    thrust::device_vector<uint> m_CellOcc;

    /// Store the scatter addresses to find the start position of all the cells in GPU memory. Size numCells
    thrust::device_vector<uint> m_ScatterAddress;

    /// The density computed for each particle. Length numPoints
    thrust::device_vector<float> m_Density;

    /// Compute the speed of sound for the viscosity calculation
    thrust::device_vector<float> m_SoundSpeed;

    /// The pressure computed per particle. Length = numPoints
    thrust::device_vector<float> m_Pressure;  

    /// Construct a block of fluid particles to be used in the simulation
    void createFluidBlock(const uint & /*firstPtIdx*/, 
                          const uint &/*numPoints*/, 
                          const float &min_x, const float &min_y, const float &min_z,
                          const float &max_x, const float &max_y, const float &max_z);

  private:
    /// This object maintains the fluid parameters and copies them to the CUDA constant "params"
    FluidParams *m_params;
};

#endif // FLUIDSYSTEM_H
