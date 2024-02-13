#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include <cufft.h>
#include <exception>
#include "noa/gpu/cuda/Exception.hpp"

namespace noa::cuda {
    /// Formats the cuFFT result to a human-readable string.
    std::string error2string(cufftResult_t result);

    /// Throws an Exception if the result is not CUFFT_SUCCESS.
    constexpr void check(cufftResult_t result, const std::source_location& location = std::source_location::current()) {
        if (result == cufftResult_t::CUFFT_SUCCESS) {
            /*do nothing*/
        } else {
            panic_at_location(location, "cuFFT failed with error: {}", error2string(result));
        }
    }
}
#endif
