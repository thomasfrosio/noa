/**
 * @file noa/Backend.h
 * @brief Define the GPU backend.
 * @author Thomas - ffyr2w
 * @date 17/01/2021
 */
#pragma once

namespace Noa::CUDA {}
namespace Noa::OpenCL {}

// Functions and types on the GPU backend should be called using the Noa::GPU alias,
// as opposed to the explicit namespace (Noa::CUDA or Noa::OpenCL).

#ifdef NOA_BUILD_CUDA
namespace Noa { namespace GPU = CUDA; }
namespace Noa::CUDA {
    enum class Backend {
        CUDA = 1,
        OpenCL = 2
    };

    inline constexpr Backend getBackend() noexcept {
        return Backend::CUDA;
    }
}
#elif NOA_BUILD_OPENCL
namespace Noa { namespace GPU = OpenCL; }
namespace Noa::OpenCL {
    enum class Backend {
        CUDA = 1,
        OpenCL = 2
    };

    inline constexpr Backend getBackend() noexcept {
        return Backend::OpenCL;
    }
}
#else
#error "GPU backend not defined"
#endif
