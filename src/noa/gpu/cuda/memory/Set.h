#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Memory {
    /**
     * Initializes or sets device memory to a value.
     * @tparam T            (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
     * @param[out] array    Array on the device.
     * @param elements      Number of elements to set.
     * @param value         The value to assign.
     * @param stream        Stream on which to enqueue this function.
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void set(T* array, size_t elements, T value, Stream& stream);
}
