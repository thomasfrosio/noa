#pragma once

#include "noa/base/Error.hpp"
#include "noa/base/Strings.hpp"
#include "noa/runtime/cuda/Runtime.hpp"

namespace noa::cuda {
    using noa::check; // bring the check functions to this namespace.

    /// Throws an Exception if the result is not cudaSuccess.
    template<typename T> requires std::same_as<std::decay_t<T>, cudaError_t>
    constexpr void check(T&& result, const std::source_location& location = std::source_location::current()) {
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
