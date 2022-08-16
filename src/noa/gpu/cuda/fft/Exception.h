#pragma once

#include <cufft.h>
#include <exception>

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Exception.h"

namespace noa::cuda {
    // Formats the cufft result to a human-readable string.
    std::string toString(cufftResult_t result);

    NOA_IH void throwIf(cufftResult_t result, const char* file, const char* function, int line) {
        if (result != cufftResult_t::CUFFT_SUCCESS)
            std::throw_with_nested(noa::Exception(file, function, line, toString(result)));
    }
}
