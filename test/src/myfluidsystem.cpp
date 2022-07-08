#include "myfluidsystem.h"

/**
 * Set up my own custom fluid simulation scenario based on the double dam break
 */
void MyFluidSystem::setupDoubleDamBreak(const uint &_numPoints,
                                        const uint &_res,
                                        const float &_adhesion,
                                        const float &_viscosity,
                                        const float &_stension,
                                        const float &_vdamp) noexcept
{
    // Clear the data structures
    init(_numPoints, _res);

    setAdhesion(_adhesion);
    setViscosity(_viscosity);
    setSurfaceTension(_stension);
    setVDamp(_vdamp);

    createFluidBlock(0,                // const uint &firstPtIdx,
                     _numPoints / 2,   // const uint &numPoints,
                     0.0, 0.0, 0.0,    // const vec3 &minCorner,
                     0.33, 0.9, 0.33); // const vec3 &maxCorner)

    createFluidBlock(_numPoints / 2,              // const uint &firstPtIdx,
                     _numPoints - _numPoints / 2, // const uint &numPoints,
                     0.66f, 0.0f, 0.66f,          // const vec3 &minCorner,
                     1.0f, 0.9f, 1.0f);           // const vec3 &maxCorner)
}