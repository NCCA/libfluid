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

int MyFluidSystem::initCuda(uint8_t  *vkDeviceUUID, size_t UUID_SIZE)
{
    int current_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the GPU which is selected by Vulkan
    while (current_device < device_count) {
        cudaGetDeviceProperties(&deviceProp, current_device);

        if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
            // Compare the cuda device UUID with vulkan UUID
            int ret = memcmp((void*)&deviceProp.uuid, vkDeviceUUID, UUID_SIZE);
            if (ret == 0)
            {
                checkCudaErrors(cudaSetDevice(current_device));
                checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
                printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
                 current_device, deviceProp.name, deviceProp.major,
                 deviceProp.minor);

                return current_device;
            }

        } else {
          devices_prohibited++;
        }

        current_device++;
    }

    if (devices_prohibited == device_count) {
        fprintf(stderr,
                "CUDA error:"
                " No Vulkan-CUDA Interop capable GPU found.\n");
        exit(EXIT_FAILURE);
    }

    return -1;
}