#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/string/Format.hpp"

#if defined(NOA_IS_OFFLINE)
#include <cuda_runtime_api.h>
#include <string>
#include <exception>

// Ideally, we could overload the ostream<< or add a fmt::formatter, however cudaError_t is not defined in the
// noa namespace and because of ADL (argument-dependent lookup) we would have to use the global namespace, which is
// likely to break the ODR (one definition rule) since users might define their own overload.

namespace noa::cuda {
    // Formats the CUDA error result to a human-readable string.
    inline std::string error2string(cudaError_t result) {
        return fmt::format("{}: {}", cudaGetErrorName(result), cudaGetErrorString(result));
    }

    // Throws a nested noa::Exception if cudaError_t =! cudaSuccess.
    inline void throw_if(cudaError_t result, const char* file, const char* function, int line) {
        if (result != cudaSuccess)
            std::throw_with_nested(noa::Exception(file, function, line, error2string(result)));
    }
}
#endif
