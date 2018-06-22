#include "myfluidsystem.h"

/**
 * Set up my own custom fluid simulation scenario based on the double dam break
 */
void MyFluidSystem::setup(const uint &_numPoints, const uint &_res)
{
    // Clear the data structures
    init(_numPoints, _res);

    setAdhesion(0.1f);
    setViscosity(0.2f);
    setSurfaceTension(1.0f);
    setVDamp(0.5f);

    createFluidBlock(0,                //const uint &firstPtIdx,
                     _numPoints / 2,   //const uint &numPoints,
                     0.0, 0.0, 0.0,    //const vec3 &minCorner,
                     0.33, 0.9, 0.33); //const vec3 &maxCorner)

    createFluidBlock(_numPoints / 2,              //const uint &firstPtIdx,
                     _numPoints - _numPoints / 2, //const uint &numPoints,
                     0.66f, 0.0f, 0.66f,          //const vec3 &minCorner,
                     1.0f, 0.9f, 1.0f);           //const vec3 &maxCorner)
}