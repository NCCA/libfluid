#ifndef MYFLUIDSYSTEM_H
#define MYFLUIDSYSTEM_H

#include "fluidsystem.h"

/**
 * Create my own fluid system with parameters which differ from the default
 */
class MyFluidSystem : public FluidSystem {
    public:
        /// Constructor is a simple pass through
        MyFluidSystem() : FluidSystem() {}

        /// A destructor
        ~MyFluidSystem() {}

        /// Set up a custom problem - we'll make it a double dam break
        void setup(const uint &_numPoints, const uint &_res);
};

#endif //MYFLUIDSYSTEM_H
