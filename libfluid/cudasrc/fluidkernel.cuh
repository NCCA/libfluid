#ifndef FLUIDKERNEL_CUH
#define FLUIDKERNEL_CUH

/// The poly6 kernels (Desbrun)
__device__ float poly6Kernel(const float r);
__device__ float gradPoly6Kernel(const float r);
__device__ float lapPoly6Kernel(const float r);

/// The spiky kernel (Desbrun)
__device__ float spikyKernel(const float r);
__device__ float gradSpikyKernel(const float r);

/// The viscosity kernel
__device__ float lapViscosityKernel(const float r);

/// The cohesion / surface tension force
__device__ float cohesionKernel(const float r);

/// For this kernel, it returns a "lump" of integrated geometry for the boundary
__device__ float boundaryDensity(const float d);
__device__ float boundaryPressureGrad(const float d);

#endif // FLUIDKERNEL_CUH
