#include "fluidparams.cuh"
#include <iostream>
#include <limits>

/// This declaration creates a single (global) instance of params which can be accessed from any kernel
__constant__ FluidParamData params;

/**
 * Constructor
 */
FluidParams::FluidParams(const uint &_res,
                         const uint &_numParticles,
                         const float &_restDensity,
                         const float &_gasConstant,
                         const float &_gamma,
                         const float &_mass,
                         const float &_xsphScale,
                         const float &_surfaceTension,
                         const float &_viscosity,
                         const float &_adhesion,
                         const float &_velLimit,
                         const float &_accLimit,
                         const float &_vdamp,
                         const float &_particleSize,
                         const float &_dt,
                         const float &_boundaryDensity,
                         const float3 &_gravity) : m_dirty(true)
{
    // Call the local setter functions to avoid code duplication
    setRes(_res);
    setNumParticles(_numParticles);
    setRestDensity(_restDensity);
    setGasConstant(_gasConstant);
    setGamma(_gamma);
    setMass(_mass);
    setLimits(_velLimit, _accLimit);
    setSurfaceTension(_surfaceTension);
    setViscosity(_viscosity);
    setAdhesion(_adhesion);
    setXSPHScale(_xsphScale);
    setVDamp(_vdamp);
    setParticleSize(_particleSize);
    setDT(_dt);
    setBoundaryDensity(_boundaryDensity);

    // These two parameters are unlikely to be changed during the simulation, but this can be easily modified in future
    m_data.m_gravity = _gravity;
    m_data.m_eps = std::numeric_limits<float>::min();
}

void FluidParams::setRes(const uint &_res)
{
    if ((_res == 0) || (m_data.m_res == _res))
        return;

    // Update the procomputed constants for the resolution
    m_data.m_res = _res;
    m_data.m_res2 = _res * _res;
    m_data.m_numCells = _res * _res * _res;

    // Compute the smoothing length as a function of the resolution (there are lots of precomputed values used in the solver)
    float _h = 1.0f / float(_res);
    m_data.m_h = _h;
    m_data.m_halfh = 0.5f * _h;
    m_data.m_h2 = _h * _h;
    m_data.m_h3 = _h * _h * _h;
    m_data.m_h4 = _h * _h * _h * _h;
    m_data.m_h5 = m_data.m_h3 * m_data.m_h2;
    m_data.m_h6 = m_data.m_h3 * m_data.m_h3;
    m_data.m_h8 = m_data.m_h5 * m_data.m_h3;
    m_data.m_h9 = m_data.m_h6 * m_data.m_h3;
    m_data.m_invh = 1.0f / _h;
    m_data.m_invh2 = 1.0f / m_data.m_h2;
    m_data.m_invh3 = 1.0f / m_data.m_h3;
    m_data.m_invh6 = 1.0f / m_data.m_h6;
    m_data.m_invh9 = 1.0f / (m_data.m_h6 * m_data.m_h3);
    m_data.m_one_minus_h = 1.0f - _h;

    // Set up all the kernel coefficients
    m_data.m_poly6Coeff = 315.0f / (64.0f * CUDART_PI_F * powf(m_data.m_h, 9.0f));
    m_data.m_gradPoly6Coeff = -945.0f / (32.0f * CUDART_PI_F * powf(m_data.m_h, 9.0f));
    m_data.m_lapPoly6Coeff = -945.0f / (32.0f * CUDART_PI_F * powf(m_data.m_h, 9.0f));
    m_data.m_spikyCoeff = 15.0f / (CUDART_PI_F * powf(m_data.m_h, 6.0f));
    m_data.m_gradSpikyCoeff = -45.0f / (CUDART_PI_F * powf(m_data.m_h, 6.0f));
    m_data.m_lapViscosityCoeff = 45.0f / (CUDART_PI_F * powf(m_data.m_h, 6.0f));
    m_data.m_cohesionCoeff = 32.0f / (CUDART_PI_F * powf(m_data.m_h, 9.0f));
    // Flag this as dirty, meaning it needs to be resynchronised on the GPU
    m_dirty = true;
}

void FluidParams::setNumParticles(const uint &_numParticles)
{
    if (m_data.m_numParticles == _numParticles)
        return;
    m_data.m_numParticles = _numParticles;
    m_dirty = true;
}

void FluidParams::setRestDensity(const float &_restDensity)
{
    if (m_data.m_restDensity == _restDensity)
        return;
    m_data.m_restDensity = _restDensity;
    m_data.m_invRestDensity = (_restDensity == 0.0f) ? 1.0f : 1.0f / _restDensity;
    m_dirty = true;
}

void FluidParams::setGamma(const float &_gamma)
{
    if (m_data.m_gamma == _gamma)
        return;
    m_data.m_gamma = _gamma;
    m_data.m_invGamma = (_gamma == 0.0f) ? 1.0f : 1.0f / _gamma;
    m_dirty = true;
}

void FluidParams::setXSPHScale(const float &_xsphScale)
{
    if (m_data.m_xsphScale == _xsphScale)
        return;
    m_data.m_xsphScale = _xsphScale;
    m_dirty = true;
}

void FluidParams::setSurfaceTension(const float &_surfaceTension)
{
    if (m_data.m_surfaceTension == _surfaceTension)
        return;
    m_data.m_surfaceTension = _surfaceTension;
    m_dirty = true;
}

void FluidParams::setViscosity(const float &_viscosity)
{
    if (m_data.m_viscosity == _viscosity)
        return;
    m_data.m_viscosity = _viscosity;
    m_dirty = true;
}

void FluidParams::setAdhesion(const float &_adhesion)
{
    if (m_data.m_adhesion == _adhesion)
        return;
    m_data.m_adhesion = _adhesion;
    m_dirty = true;
}

void FluidParams::setLimits(const float &_velLimit, const float &_accLimit)
{
    if ((m_data.m_velLimit2 != _velLimit * _velLimit) || (m_data.m_accLimit2 != _accLimit * _accLimit))
    {
        m_data.m_velLimit = _velLimit;
        m_data.m_accLimit = _accLimit;
        m_data.m_velLimit2 = _velLimit * _velLimit;
        m_data.m_accLimit2 = _accLimit * _accLimit;
        m_dirty = true;
    }
}

void FluidParams::setGasConstant(const float &_gasConstant)
{
    if (m_data.m_gasConstant == _gasConstant)
        return;
    m_data.m_gasConstant = _gasConstant;
    m_dirty = true;
}

void FluidParams::setMass(const float &_mass)
{
    if (m_data.m_mass == _mass)
        return;
    m_data.m_mass = _mass;
    m_data.m_mass2 = _mass * _mass;
    m_data.m_invMass = 1.0f / _mass;
    m_dirty = true;
}

void FluidParams::setBoundaryDensity(const float &_boundaryDensity)
{
    if (m_data.m_boundaryDensity == _boundaryDensity)
        return;
    m_data.m_boundaryDensity = _boundaryDensity;
    m_data.m_boundaryDensity2 = _boundaryDensity * _boundaryDensity;
    m_data.m_invBoundaryDensity = 1.0f / _boundaryDensity;
    m_dirty = true;
}

void FluidParams::setParticleSize(const float &_particleSize)
{
    if (m_data.m_particleSize == _particleSize)
        return;
    m_data.m_particleSize = _particleSize;
    m_data.m_invParticleSize = 1.0f / _particleSize;
    m_dirty = true;
}

void FluidParams::setDT(const float &_dt)
{
    if (m_data.m_dt == _dt)
        return;
    m_data.m_dt = _dt;
    m_data.m_dtdt = _dt * _dt;
    m_dirty = true;
}

void FluidParams::setVDamp(const float &_vdamp)
{
    if (m_data.m_vdamp == _vdamp)
        return;
    m_data.m_vdamp = _vdamp;
    m_dirty = true;
}

/**
 * Copy the data object stored in this class to the constant symbol
 */
void FluidParams::sync()
{
    if (m_dirty)
    {
        cudaError_t err = cudaMemcpyToSymbol(params, &m_data, sizeof(FluidParamData));
        if (err != cudaSuccess)
        {
            std::cerr << "Copy to symbol params (size=" << sizeof(FluidParams) << ") failed! Reason: " << cudaGetErrorString(err) << "\n";
            exit(0);
        }
        m_dirty = false;
    }
}

void FluidParams::debugPrint() const
{
    std::cout << "FluidParams:\n";
    std::cout << "res: " << m_data.m_res << ", numCells: " << m_data.m_numCells << ", numParticles: " << m_data.m_numParticles << "\n";
    std::cout << "restDensity: " << m_data.m_restDensity << ", gasConstant: " << m_data.m_gasConstant << ", mass: " << m_data.m_mass << " invMass: " << m_data.m_invMass << "\n";
    std::cout << "dt: " << m_data.m_dt << ", h: " << m_data.m_h << ", m_vdamp: " << m_data.m_vdamp << ", particleSize: " << m_data.m_particleSize << "\n";
    std::cout << "Adhesion: " << m_data.m_adhesion << ", Viscosity: " << m_data.m_viscosity << ", SurfaceTension: " << m_data.m_surfaceTension << "\n";
    std::cout << "kernel coefficients: " << m_data.m_poly6Coeff << ", " << m_data.m_gradPoly6Coeff << ", " << m_data.m_lapPoly6Coeff << "\n";
}
