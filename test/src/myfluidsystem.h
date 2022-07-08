#ifndef MYFLUIDSYSTEM_H
#define MYFLUIDSYSTEM_H

#include "fluidsystem.h"

/**
 * Create my own fluid system with parameters which differ from the default
 */
class MyFluidSystem : public FluidSystem
{
public:
    /// Constructor is a simple pass through
    MyFluidSystem() : FluidSystem() {}

    /// A destructor
    ~MyFluidSystem() {}

    /// Set up a custom problem - we'll make it a double dam break
    void setupDoubleDamBreak(const uint &_numPoints = 10000,
                             const uint &_res = 32,
                             const float &_adhesion = 0.1f,
                             const float &_viscosity = 0.2f,
                             const float &_stension = 1.0f,
                             const float &_vdamp = 0.5f) noexcept;
};

#endif // MYFLUIDSYSTEM_H
