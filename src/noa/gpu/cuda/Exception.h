#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <exception>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/util/string/Format.h"

namespace Noa::CUDA {
    NOA_FH std::string toString(cudaError_t error) {
        return String::format("Errno::{}: {}", cudaGetErrorName(error), cudaGetErrorString(error));
    }

    /**
     * Throw a nested @c Noa::Exception if cudaError_t =! cudaSuccess.
     * @note    As the result of this function being defined in NOA::CUDA, the macro NOA_THROW_IF, defined in
     *          noa/Exception.h, will now call this function when used within NOA::CUDA.
     */
    NOA_FH void throwIf(cudaError_t error, const char* file, const char* function, const int line) {
        if (error != cudaSuccess)
            std::throw_with_nested(Noa::Exception(file, function, line, toString(error)));
    }
}

/// Launch the @a kernel and throw any error that might have occurred during launch.
#define NOA_CUDA_LAUNCH(blocks, threads, shared_mem, stream_id, kernel, ...) \
kernel<<<blocks, threads, shared_mem, stream_id>>>(__VA_ARGS__); NOA_THROW_IF(cudaPeekAtLastError())
