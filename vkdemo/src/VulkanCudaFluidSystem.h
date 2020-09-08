#ifndef NDEBUG
#define ENABLE_VALIDATION (false)
#else
#define ENABLE_VALIDATION (true)
#endif

#include "VulkanBaseApp.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include "linmath.h"

#include "myfluidsystem.h"
#include <helper_cuda.h>

typedef float vec3[3];


class VulkanCudaFluidSystem : public VulkanBaseApp
{

    typedef struct UniformBufferObject_st {
        mat4x4 modelViewProj;
    } UniformBufferObject;

    VkBuffer m_posBuffer, m_velBuffer, m_indexBuffer;
    VkDeviceMemory m_posMemory, m_velMemory, m_indexMemory;
    MyFluidSystem m_sim;
    VkSemaphore m_vkWaitSemaphore, m_vkSignalSemaphore;
    cudaStream_t m_stream;
    cudaExternalSemaphore_t m_cudaWaitSemaphore, m_cudaSignalSemaphore;
    cudaExternalMemory_t m_cudaVertMem;
    float *m_cudaPosMap, *m_cudaVelMap;
    size_t m_numPoints;
    using chrono_tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
    chrono_tp m_lastTime;
    size_t m_lastFrame;
public:
    VulkanCudaFluidSystem(size_t _numPoints, size_t _resolution) :
        VulkanBaseApp("vulkanCudaSineWave", ENABLE_VALIDATION),
        m_posBuffer(VK_NULL_HANDLE),
        m_velBuffer(VK_NULL_HANDLE),        
        m_indexBuffer(VK_NULL_HANDLE),
        m_posMemory(VK_NULL_HANDLE),
        m_velMemory(VK_NULL_HANDLE),
        m_indexMemory(VK_NULL_HANDLE),
        m_ubo(),
        m_sim(),
        m_stream(0),
        m_vkWaitSemaphore(VK_NULL_HANDLE),
        m_vkSignalSemaphore(VK_NULL_HANDLE),
        m_cudaWaitSemaphore(),
        m_cudaSignalSemaphore(),
        m_cudaVertMem(),
        m_cudaPosMap(nullptr),
        m_cudaVelMap(nullptr),
        m_lastFrame(0),
        m_numPoints(_numPoints) {
        
            // Start the fluid simulation
            m_sim.setup(_numPoints, _resolution);

            // Add our compiled vulkan shader files
	
            //char* vertex_shader_path = sdkFindFilePath("sinewave.vert", SHADERS_PATH);
            //char* fragment_shader_path = sdkFindFilePath("sinewave.frag", SHADERS_PATH);
            //m_shaderFiles.push_back(std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, vertex_shader_path));
            //m_shaderFiles.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader_path));
	        char fragShader[256], vertShader[256];
	        sprintf(vertShader,"%s/%s",SHADERS_PATH,"sinewave.vert");
	        sprintf(fragShader,"%s/%s",SHADERS_PATH,"sinewave.frag");
	        m_shaderFiles.push_back(std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, vertShader));
            m_shaderFiles.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, fragShader));
        }
        ~VulkanCudaSineWave() {
            // Make sure there's no pending work before we start tearing down
            checkCudaErrors(cudaStreamSynchronize(m_stream));

            if (m_vkSignalSemaphore != VK_NULL_HANDLE) {
                checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaSignalSemaphore));
                vkDestroySemaphore(m_device, m_vkSignalSemaphore, nullptr);
            }
            if (m_vkWaitSemaphore != VK_NULL_HANDLE) {
                checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaWaitSemaphore));
                vkDestroySemaphore(m_device, m_vkWaitSemaphore, nullptr);
            }

            if (m_velBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(m_device, m_velBuffer, nullptr);
            }
            if (m_velMemory != VK_NULL_HANDLE) {
                vkFreeMemory(m_device, m_velMemory, nullptr);
            }

            if (m_posBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(m_device, m_posBuffer, nullptr);
            }
            if (m_posMemory != VK_NULL_HANDLE) {
                vkFreeMemory(m_device, m_posMemory, nullptr);
            }
            if (m_cudaPosMap) {
                checkCudaErrors(cudaDestroyExternalMemory(m_cudaVertMem));
            } 
            // Looks like I need another cudaVertMem

            if (m_indexBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(m_device, m_indexBuffer, nullptr);
            }
            if (m_indexMemory != VK_NULL_HANDLE) {
                vkFreeMemory(m_device, m_indexMemory, nullptr);
            }
        }

    void fillRenderingCommandBuffer(VkCommandBuffer& commandBuffer) {
        VkBuffer vertexBuffers[] = { m_posBuffer, m_velBuffer };
        VkDeviceSize offsets[] = { 0, 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, sizeof(vertexBuffers) / sizeof(vertexBuffers[0]), vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        // Must investigate this line below
        vkCmdDrawIndexed(commandBuffer, (uint32_t)((m_sim.getWidth() - 1) * (m_sim.getHeight() - 1) * 6), 1, 0, 0, 0);
    }

    void getVertexDescriptions(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc) {
        bindingDesc.resize(2);
        attribDesc.resize(2);

        bindingDesc[0].binding = 0;
        bindingDesc[0].stride = sizeof(vec3);
        bindingDesc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        bindingDesc[1].binding = 1;
        bindingDesc[1].stride = sizeof(vec3);
        bindingDesc[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        attribDesc[0].binding = 0;
        attribDesc[0].location = 0;
        attribDesc[0].format = VK_FORMAT_R32_SFLOAT;
        attribDesc[0].offset = 0;

        attribDesc[1].binding = 1;
        attribDesc[1].location = 1;
        attribDesc[1].format = VK_FORMAT_R32G32_SFLOAT;
        attribDesc[1].offset = 0;
    }

    void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info) {
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        info.primitiveRestartEnable = VK_FALSE;
    }

    void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait, std::vector< VkPipelineStageFlags>& waitStages) const {
        if (m_currentFrame != 0) {
            // Have vulkan wait until cuda is done with the vertex buffer before rendering
            // We don't do this on the first frame, as the wait semaphore hasn't been initialized yet
            wait.push_back(m_vkWaitSemaphore);
            // We want to wait until all the pipeline commands are complete before letting cuda work
            waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
    }

    void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const {
        // Add this semaphore for vulkan to signal once the vertex buffer is ready for cuda to modify
        signal.push_back(m_vkSignalSemaphore);
    }

    void initVulkanApp() {
        int cuda_device = -1;

        // Select cuda device where vulkan is running.
        cuda_device = m_sim.initCuda(m_vkDeviceUUID, VK_UUID_SIZE);
        if (cuda_device == -1)
        {
            printf("Error: No CUDA-Vulkan interop capable device found\n");
            exit(EXIT_FAILURE);
        }

        m_sim.initCudaLaunchConfig(cuda_device);

        // Create the cuda stream we'll be using
        checkCudaErrors(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

        const size_t nVerts = m_numVerts;
        const size_t nInds = (m_sim.getWidth() - 1) * (m_sim.getHeight() - 1) * 6;

        // Create the height map cuda will write to
        createExternalBuffer(nVerts * sizeof(float),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             m_heightBuffer, m_heightMemory);

        // Create the vertex buffer that will hold the xy coordinates for the grid
        createBuffer(nVerts * sizeof(vec2),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     m_xyBuffer, m_xyMemory);

        // Create the index buffer that references from both buffers above
        createBuffer(nInds * sizeof(uint32_t),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     m_indexBuffer, m_indexMemory);

        // Import the height map into cuda and retrieve a device pointer to use
        importCudaExternalMemory((void **)&m_cudaHeightMap, m_cudaVertMem, m_heightMemory, nVerts * sizeof(*m_cudaHeightMap), getDefaultMemHandleType());
        // Set the height map to use in the simulation
        m_sim.initSimulation(m_cudaHeightMap);

        {
            // Set up the initial values for the vertex buffers with Vulkan
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = std::max(nVerts * sizeof(vec2), nInds * sizeof(uint32_t));
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(m_device, stagingMemory, 0, stagingSz, 0, &stagingBase);

            memset(stagingBase, 0, nVerts * sizeof(float));
            copyBuffer(m_heightBuffer, stagingBuffer, nVerts * sizeof(float));

            for (size_t y = 0; y < m_sim.getHeight(); y++) {
                for (size_t x = 0; x < m_sim.getWidth(); x++) {
                    vec2 *stagedVert = (vec2 *)stagingBase;
                    stagedVert[y * m_sim.getWidth() + x][0] = (2.0f * x) / (m_sim.getWidth() - 1) - 1;
                    stagedVert[y * m_sim.getWidth() + x][1] = (2.0f * y) / (m_sim.getHeight() - 1) - 1;
                }
            }
            copyBuffer(m_xyBuffer, stagingBuffer, nVerts * sizeof(vec2));

            {
                uint32_t *indices = (uint32_t *)stagingBase;
                for (size_t y = 0; y < m_sim.getHeight() - 1; y++) {
                    for (size_t x = 0; x < m_sim.getWidth() - 1; x++) {
                        indices[0] = (uint32_t)((y + 0) * m_sim.getWidth() + (x + 0));
                        indices[1] = (uint32_t)((y + 1) * m_sim.getWidth() + (x + 0));
                        indices[2] = (uint32_t)((y + 0) * m_sim.getWidth() + (x + 1));
                        indices[3] = (uint32_t)((y + 1) * m_sim.getWidth() + (x + 0));
                        indices[4] = (uint32_t)((y + 1) * m_sim.getWidth() + (x + 1));
                        indices[5] = (uint32_t)((y + 0) * m_sim.getWidth() + (x + 1));
                        indices += 6;
                    }
                }
            }
            copyBuffer(m_indexBuffer, stagingBuffer, nInds * sizeof(uint32_t));

            vkUnmapMemory(m_device, stagingMemory);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            vkFreeMemory(m_device, stagingMemory, nullptr);
        }

        // Create the semaphore vulkan will signal when it's done with the vertex buffer
        createExternalSemaphore(m_vkSignalSemaphore, getDefaultSemaphoreHandleType());
        // Create the semaphore vulkan will wait for before using the vertex buffer
        createExternalSemaphore(m_vkWaitSemaphore, getDefaultSemaphoreHandleType());
        // Import the semaphore cuda will use -- vulkan's signal will be cuda's wait
        importCudaExternalSemaphore(m_cudaWaitSemaphore, m_vkSignalSemaphore, getDefaultSemaphoreHandleType());
        // Import the semaphore cuda will use -- cuda's signal will be vulkan's wait
        importCudaExternalSemaphore(m_cudaSignalSemaphore, m_vkWaitSemaphore, getDefaultSemaphoreHandleType());
    }

    void importCudaExternalMemory(void **cudaPtr, cudaExternalMemory_t& cudaMem, VkDeviceMemory& vkMem, VkDeviceSize size, VkExternalMemoryHandleTypeFlagBits handleType) {
        cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};

        if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
            externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
            externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
            externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        }
        else {
            throw std::runtime_error("Unknown handle type requested!");
        }

        externalMemoryHandleDesc.size = size;

#ifdef _WIN64
        externalMemoryHandleDesc.handle.win32.handle = (HANDLE)getMemHandle(vkMem, handleType);
#else
        externalMemoryHandleDesc.handle.fd = (int)(uintptr_t)getMemHandle(vkMem, handleType);
#endif

        checkCudaErrors(cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc));

        cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
        externalMemBufferDesc.offset = 0;
        externalMemBufferDesc.size = size;
        externalMemBufferDesc.flags = 0;

        checkCudaErrors(cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem, &externalMemBufferDesc));
    }

    void importCudaExternalSemaphore(cudaExternalSemaphore_t& cudaSem, VkSemaphore& vkSem, VkExternalSemaphoreHandleTypeFlagBits handleType) {
        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};

        if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
            externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
            externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
            externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
        }
        else {
            throw std::runtime_error("Unknown handle type requested!");
        }

#ifdef _WIN64
        externalSemaphoreHandleDesc.handle.win32.handle = (HANDLE)getSemaphoreHandle(vkSem, handleType);
#else
        externalSemaphoreHandleDesc.handle.fd = (int)(uintptr_t)getSemaphoreHandle(vkSem, handleType);
#endif

        externalSemaphoreHandleDesc.flags = 0;

        checkCudaErrors(cudaImportExternalSemaphore(&cudaSem, &externalSemaphoreHandleDesc));
    }

    VkDeviceSize getUniformSize() const {
        return sizeof(UniformBufferObject);
    }

    void updateUniformBuffer(uint32_t imageIndex) {
        {
            mat4x4 view, proj;
            vec3 eye = { 1.75f, 1.75f, 1.25f };
            vec3 center = { 0.0f, 0.0f, -0.25f };
            vec3 up = { 0.0f, 0.0f, 1.0f };

            mat4x4_perspective(proj, (float)degreesToRadians(45.0f), m_swapChainExtent.width / (float)m_swapChainExtent.height, 0.1f, 10.0f);
            proj[1][1] *= -1.0f;        // Flip y axis

            mat4x4_look_at(view, eye, center, up);
            mat4x4_mul(m_ubo.modelViewProj, proj, view);
        }

        void *data;
        vkMapMemory(m_device, m_uniformMemory[imageIndex], 0, getUniformSize(), 0, &data);
        memcpy(data, &m_ubo, sizeof(m_ubo));
        vkUnmapMemory(m_device, m_uniformMemory[imageIndex]);
    }

    std::vector<const char *> getRequiredExtensions() const {
        std::vector<const char *> extensions;
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
        return extensions;
    }

    std::vector<const char *> getRequiredDeviceExtensions() const {
        std::vector<const char *> extensions;
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN64
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif /* _WIN64 */
        return extensions;
    }

    void drawFrame() {
        static chrono_tp startTime = std::chrono::high_resolution_clock::now();

        chrono_tp currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        if (m_currentFrame == 0) {
            m_lastTime = startTime;
        }

        float frame_time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - m_lastTime).count();

        cudaExternalSemaphoreWaitParams waitParams = {};
        waitParams.flags = 0;
        waitParams.params.fence.value = 0;

        cudaExternalSemaphoreSignalParams signalParams = {};
        signalParams.flags = 0;
        signalParams.params.fence.value = 0;

        // Have vulkan draw the current frame...
        VulkanBaseApp::drawFrame();
        // Wait for vulkan to complete it's work
        checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaWaitSemaphore, &waitParams, 1, m_stream));
        // Now step the simulation
        m_sim.stepSimulation(time, m_stream);
        // Signal vulkan to continue with the updated buffers
        checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_cudaSignalSemaphore, &signalParams, 1, m_stream));

        // Output a naive measurement of the frames per second every five seconds
        if (frame_time > 5) {
            std::cout << "Average FPS (over "
                      << std::fixed << std::setprecision(2) << frame_time
                      << " seconds): "
                      << std::fixed << std::setprecision(2)
                      << ((m_currentFrame - m_lastFrame) / frame_time)
                      << std::endl;
            m_lastFrame = m_currentFrame;
            m_lastTime = currentTime;
        }
    }
};