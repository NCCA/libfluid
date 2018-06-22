#ifndef FLUIDDEFINES_H
#define FLUIDDEFINES_H

#define DEFAULT_TIMESTEP 0.01f
#define DEFAULT_REST_DENSITY 1.0f
#define DEFAULT_GAS_CONSTANT 1.0f

/// Arbitrary number of boundary planes defined for this problem. This can be increased but eventually you'll run out of constant memory
#define MAX_BOUNDING_PLANES 8

/// Define the null hash in case the particle manages to make it's way out of the bounding grid
#define NULL_HASH UINT_MAX 

/// Define this here unless it was defined elsewhere
#ifndef CUDART_PI_F             
    #define CUDART_PI_F 3.141592654f
#endif

/// Defines for the default force coefficients
#define DEFAULT_SURFACE_TENSION_FORCE 1.0f
#define DEFAULT_VISCOSITY_FORCE 0.1f
#define DEFAULT_ADHESION_FORCE 1.0f

/// Define the default boundary density for glass which is 2600
#define DEFAULT_BOUNDARY_DENSITY 2600

#endif //FLUIDDEFINES_H