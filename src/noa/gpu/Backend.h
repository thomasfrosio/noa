/**
 * @file gpu/Backend.h
 * @brief Define the GPU backend.
 * @author Thomas - ffyr2w
 * @date 17/01/2021
 */
#pragma once

namespace Noa::CUDA {}
namespace Noa::OpenCL {}

#ifdef NOA_BUILD_CUDA
namespace Noa { namespace GPU = Noa::CUDA; }
#elif NOA_BUILD_OPENCL
namespace Noa { namespace GPU = Noa::OpenCL; }
#else
#error "GPU backend not defined"
#endif

namespace Noa::GPU {
    enum class Backend {
        CUDA = 1,
        OpenCL = 2
    };

    inline constexpr Backend getBackend() noexcept {
#ifdef NOA_BUILD_CUDA
        return Backend::CUDA;
#elif NOA_BUILD_OPENCL
        return Backend::OpenCL;
#else
    #error "GPU backend is not defined"
#endif
    }
}
