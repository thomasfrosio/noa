#pragma once

#include "noa/runtime/core/Error.hpp"
#include "noa/runtime/core/Strings.hpp"
#include "noa/runtime/cuda/Runtime.hpp"

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
    inline auto error2string(cudaError_t result) -> std::string {
        return fmt::format("{}: {}", cudaGetErrorName(result), cudaGetErrorString(result));
    }
}
