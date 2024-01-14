#pragma once

#include "noa/core/Config.hpp"
#include "noa/gpu/cuda/Exception.hpp"

#if defined(NOA_IS_OFFLINE)
#include <cufft.h>
#include <exception>

namespace noa::cuda {
    // Formats the cufft result to a human-readable string.
    std::string to_string(cufftResult_t result);

    inline void throw_if(cufftResult_t result, const char* file, const char* function, int line) {
        if (result != cufftResult_t::CUFFT_SUCCESS)
            std::throw_with_nested(noa::Exception(file, function, line, to_string(result)));
    }
}
#endif
