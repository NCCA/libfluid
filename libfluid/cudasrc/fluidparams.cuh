#ifndef FLUIDPARAMS_CUH
#define FLUIDPARAMS_CUH

#include "fluiddefines.h"
// added by jon
#include "helper_math.h"
/**
 * This is the structure that will be copied across to the GPU. Note that dynamic initialisation is not supported
 * by __constant__ structs so we will need to set these in the manager class FluidParams.
 * Much of these constants are precomputed inversions, squares and other computationally expensive things to avoid
 * computing these in parallel where possible.
 * Note also that this currently has a pretty small footprint, on account of the small amount of collision geometry.
 * The total constant memory is usually around 64kb, which means that many more boundary planes could be supported
 * (although this will slow down the integrator quite a bit).
 */
struct FluidParamData
{
    /// The resolution of the grid - can be set when operator is created
    uint m_res, m_res2, m_numCells;

    /// The number of points being simulated
    uint m_numParticles;

    /// Fluid parameters needed for the density and pressure calculation
    float m_restDensity, m_invRestDensity, m_gasConstant, m_gamma, m_invGamma;

    /// The mass of each particle
    float m_mass, m_mass2, m_invMass, m_particleSize, m_invParticleSize;

    /// The number time step and squared time step
    float m_dt, m_dtdt;

    /// The smoothing length and precomputed versions and inversions
    float m_h, m_halfh, m_h2, m_h3, m_h4, m_h5, m_h6, m_h8, m_h9, m_invh, m_invh2, m_invh3, m_invh6, m_invh9, m_one_minus_h;

    /// The scale of the velocity smoothing as a result of XSPH
    float m_xsphScale;

    /// The coefficient of surface tension (probably somewhere between 0 and 1 is feasible)
    float m_surfaceTension;

    /// The coefficient of the viscosity term (my guess is somewhere between 0 and 1 is feasible)
    float m_viscosity;

    /// The coefficient of adhesion - this has a physical meaning, and is the product of C (some unknown particle-particle coefficient) and rho_j - the boundary material density.
    float m_adhesion;

    /// Some limits on the acceleration and velocity to ensure the simulation doesn't blow up (based on Fluids v3)
    float m_velLimit, m_accLimit, m_velLimit2, m_accLimit2;

    // The stuff below is used in the boundary calculations
    /// The velocity damping used for boundary collisions in the range [0,1] (1.0f is full reflection, 0.0f will zero the velocity component)
    float m_vdamp;

    /// This is a constant (in kg/m^3) representing the density of the boundary. Glass is 2.4-2.8 * 10^3 kg/m^3.
    float m_boundaryDensity, m_boundaryDensity2, m_invBoundaryDensity;

    /// Are we using gravity?
    float3 m_gravity;

    /// Precomputed coefficients for the various kernels (I took this insiration from Fluids v3)
    float m_poly6Coeff, m_gradPoly6Coeff, m_lapPoly6Coeff;
    float m_spikyCoeff, m_gradSpikyCoeff;
    float m_lapViscosityCoeff;
    float m_cohesionCoeff;

    /// An epsilon for all calculations
    float m_eps;
};

// This is an external reference to the params constant that will be available on the GPU. The actual declaration is in fluidparams.cu
extern __constant__ FluidParamData params;

/**
 * This class manages the initialisation of the fluid parameter struct on the host side
 */
class FluidParams
{
  public:
    /// Set the Fluid parameters in the constructor
    FluidParams(const uint &_res = 1,
                const uint &_numParticles = 1,
                const float &_restDensity = DEFAULT_REST_DENSITY,
                const float &_gasConstant = DEFAULT_GAS_CONSTANT,
                const float &_gamma = 1.0f,
                const float &_mass = 1.0f,
                const float &_xsphScale = 0.5f,
                const float &_surfaceTension = DEFAULT_SURFACE_TENSION_FORCE,
                const float &_viscosity = DEFAULT_VISCOSITY_FORCE,
                const float &_adhesion = DEFAULT_ADHESION_FORCE,
                const float &_velLimit = 5.0f,
                const float &_accLimit = 150.0f,
                const float &_vdamp = 1.0f,
                const float &_particleSize = 1.0f,
                const float &_dt = DEFAULT_TIMESTEP,
                const float &_boundaryDensity = DEFAULT_BOUNDARY_DENSITY,
                const float3 &_gravity = make_float3(0.0f, -9.8f, 0.0f));

    /// Copy the data from the host the GPU (there is no point coming back as it is constant in the GPU context)
    void sync();

    /// Various setting functions to allow properties to be modified directly
    void setRes(const uint &_res);
    void setNumParticles(const uint &_numParticles);
    void setRestDensity(const float &_restDensity);
    void setParticleSize(const float &_particleSize);
    void setGasConstant(const float &_gasConstant);
    void setLimits(const float &_velLimit, const float &_accLimit);
    void setXSPHScale(const float &_xsphScale);
    void setSurfaceTension(const float &_surfaceTension);
    void setViscosity(const float &_viscosity);
    void setAdhesion(const float &_adhesion);
    void setVDamp(const float &_vdamp);
    void setGamma(const float &_gamma);
    void setMass(const float &_mass);
    void setDT(const float &_dt);
    void setBoundaryDensity(const float &_boundaryDensity);

    /// Various getting functions in case these things need to be accessed in the host code
    uint getRes() const { return m_data.m_res; }
    uint getNumParticles() const { return m_data.m_numParticles; }
    uint getNumCells() const { return m_data.m_numCells; }
    float getRestDensity() const { return m_data.m_restDensity; }
    float getParticleSize() const { return m_data.m_particleSize; }
    float getXSPHScale() const { return m_data.m_xsphScale; }
    float getVDamp() const { return m_data.m_vdamp; }
    float getSurfaceTension() const { return m_data.m_surfaceTension; }
    float getViscosity() const { return m_data.m_viscosity; }
    float getAdhesion() const { return m_data.m_adhesion; }
    float getGasConstant() const { return m_data.m_gasConstant; }
    float getGamma() const { return m_data.m_gamma; }
    float getMass() const { return m_data.m_mass; }
    float getDT() const { return m_data.m_dt; }
    float getH() const { return m_data.m_h; }

    void debugPrint() const;

  private:
    /// This is the local (host) copy of the parameters which will get copied to the "params" symbol on sync()
    FluidParamData m_data;

    /// Apply this flag if something changes in the structure and we feel we need to re-synchronise the data with the version on the GPU
    bool m_dirty;
};

#endif //FLUIDPARAMS_CUH
