#pragma once

#include "noa/gpu/cuda/Error.hpp"
#include "noa/gpu/cuda/Runtime.hpp"

namespace noa::cuda {
    // Gets the latest version of CUDA supported by the driver. Format 1000 * major + 10 * minor.
    inline i32 version_driver() {
        i32 version;
        check(cudaDriverGetVersion(&version));
        return version;
    }

    // Gets the CUDA runtime version. Format 1000 * major + 10 * minor.
    inline i32 version_runtime() {
        i32 version;
        check(cudaRuntimeGetVersion(&version));
        return version;
    }
}
