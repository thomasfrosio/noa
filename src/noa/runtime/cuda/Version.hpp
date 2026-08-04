#pragma once

#include "noa/runtime/cuda/Error.hpp"
#include "noa/runtime/cuda/Runtime.hpp"

namespace noa::cuda {
    // Gets the latest version of CUDA supported by the driver. Format 1000 * major + 10 * minor.
    inline auto version_driver() -> i32 {
        i32 version;
        check(cudaDriverGetVersion(&version));
        return version;
    }

    // Gets the CUDA runtime version. Format 1000 * major + 10 * minor.
    inline auto version_runtime() -> i32 {
        i32 version;
        check(cudaRuntimeGetVersion(&version));
        return version;
    }
}
