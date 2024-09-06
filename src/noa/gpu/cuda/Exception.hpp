#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/Exception.hpp"
#include "noa/core/utils/Strings.hpp"
#include "noa/gpu/cuda/Runtime.hpp"

namespace noa::cuda {
    using noa::check; // bring the check functions to this namespace.

    /// Throws an Exception if the result is not cudaSuccess.
    constexpr void check(cudaError_t result, const std::source_location& location = std::source_location::current()) {
        if (result == cudaSuccess) {
            /*do nothing*/
        } else {
            panic_at_location(location, "{}: {}", cudaGetErrorName(result), cudaGetErrorString(result));
        }
    }

    /// Formats the CUDA error to a human-readable string.
    inline std::string error2string(cudaError_t result) {
        return fmt::format("{}: {}", cudaGetErrorName(result), cudaGetErrorString(result));
    }
}
#endif
