#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <exception>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/util/string/Format.h"

// Ideally, we could overload the ostream<< or add a fmt::formatter, however cudaError_t is not defined in the
// Noa namespace and because of ADL we would have to use the global namespace, which is likely to break the ODR.

namespace Noa::CUDA {
    /// Formats the CUDA error @a result to a human readable string.
    NOA_IH std::string toString(cudaError_t result) {
        return String::format("Errno::{}: {}", cudaGetErrorName(result), cudaGetErrorString(result));
    }

    /// Throws a nested Noa::Exception if cudaError_t =! cudaSuccess.
    NOA_IH void throwIf(cudaError_t result, const char* file, const char* function, int line) {
        if (result != cudaSuccess)
            std::throw_with_nested(Noa::Exception(file, function, line, toString(result)));
    }
}

/// Launch the @a kernel and throw any error that might have occurred during or before launch.
#define NOA_CUDA_LAUNCH(blocks, threads, shared_mem, stream_id, kernel, ...) \
do {                                                                         \
    kernel<<<blocks, threads, shared_mem, stream_id>>>(__VA_ARGS__);         \
    NOA_THROW_IF(cudaPeekAtLastError());                                     \
} while (false)
