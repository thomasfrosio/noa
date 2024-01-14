#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Types.hpp"

namespace noa::cuda {
    // Gets the latest version of CUDA supported by the driver. Format 1000 * major + 10 * minor.
    NOA_IH i32 version_driver() {
        i32 version;
        NOA_THROW_IF(cudaDriverGetVersion(&version));
        return version;
    }

    // Gets the CUDA runtime version. Format 1000 * major + 10 * minor.
    NOA_IH i32 version_runtime() {
        i32 version;
        NOA_THROW_IF(cudaRuntimeGetVersion(&version));
        return version;
    }
}
#endif
