#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <exception>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/util/string/Format.h"

namespace Noa {
    NOA_IH std::ostream& operator<<(std::ostream& os, cudaError_t error) {
        os << String::format("Errno::{}: {}", cudaGetErrorName(error), cudaGetErrorString(error));
        return os;
    }

    /// Throws a nested Noa::Exception if cudaError_t =! cudaSuccess.
    template<>
    NOA_IH void throwIf<cudaError_t>(cudaError_t result, const char* file, const char* function, const int line) {
        if (result != cudaSuccess)
            std::throw_with_nested(Noa::Exception(file, function, line, result));
    }
}

/// Launch the @a kernel and throw any error that might have occurred during launch.
#define NOA_CUDA_LAUNCH(blocks, threads, shared_mem, stream_id, kernel, ...) \
do {                                                                         \
    kernel<<<blocks, threads, shared_mem, stream_id>>>(__VA_ARGS__);         \
    NOA_THROW_IF(cudaPeekAtLastError());                                     \
} while (false)
