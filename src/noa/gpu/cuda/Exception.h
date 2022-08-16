#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <exception>

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/string/Format.h"

// Ideally, we could overload the ostream<< or add a fmt::formatter, however cudaError_t is not defined in the
// noa namespace and because of ADL (argument-dependent lookup) we would have to use the global namespace, which is
// likely to break the ODR (one definition rule) since users might define their own overload.
// In C++20, none of this would matter since we don't have to use macro to catch the file/function/line.

namespace noa::cuda {
    // Formats the CUDA error result to a human-readable string.
    inline std::string toString(cudaError_t result) {
        return string::format("{}: {}", cudaGetErrorName(result), cudaGetErrorString(result));
    }

    // Throws a nested noa::Exception if cudaError_t =! cudaSuccess.
    inline void throwIf(cudaError_t result, const char* file, const char* function, int line) {
        if (result != cudaSuccess)
            std::throw_with_nested(noa::Exception(file, function, line, toString(result)));
    }
}
