/// \file noa/gpu/cuda/memory/Set.h
/// \brief Set to value.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::memory::details {
    template<typename T>
    NOA_HOST void set(T* array, size_t elements, T value, Stream& stream);
}

namespace noa::cuda::memory {
    /// Initializes or sets device memory to a value.
    /// \details This function will either call cudaMemsetAsync if \p value is 0, or will launch a kernel.
    /// \tparam T            (u)char, (u)short, (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[out] array    Array on the device.
    /// \param elements      Number of elements to set.
    /// \param value         The value to assign.
    /// \param stream        Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void set(T* array, size_t elements, T value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if (value == T{0}) {
            NOA_THROW_IF(cudaMemsetAsync(array, 0, elements * sizeof(T), stream.id()));
        } else {
            details::set(array, elements, value, stream);
        }
    }
}
