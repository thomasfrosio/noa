/// \file noa/gpu/cuda/memory/Set.h
/// \brief Set to value.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory::details {
    template<typename T>
    NOA_HOST void set(T* array, size_t elements, T value, Stream& stream);

    template<typename T>
    NOA_HOST void set(T* array, size_t array_pitch, size3_t shape, T value, Stream& stream);
}

namespace noa::cuda::memory {
    /// Initializes or sets device memory to a value.
    /// \details This function will either call cudaMemsetAsync if \p value is 0, or will launch a custom kernel.
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

    /// Initializes or sets device memory to a value.
    /// \details This function will either call cudaMemset2DAsync if \p value is 0, or will launch a custom kernel.
    ///
    /// \tparam CHECK_CONTIGUOUS    Filling a contiguous block of memory is often more efficient. If true, the function
    ///                             checks whether or not the data is contiguous and if so performs one contiguous memset.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[out] array           Array on the device.
    /// \param array_pitch          Pitch, in elements, of \p array.
    /// \param shape                Logical {fast, medium, slow} shape of \p array.
    /// \param value                The value to assign.
    /// \param stream               Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous with respect to the host and may return before completion.
    /// \note The order of the last 2 dimensions of \p shape does not matter, but the number of total rows does.
    /// \note Padded regions are NOT modified.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void set(T* array, size_t array_pitch, size3_t shape, T value, Stream& stream) {
        NOA_PROFILE_FUNCTION();

        if constexpr (CHECK_CONTIGUOUS) {
            if (shape.x == array_pitch)
                return set(array, elements(shape), value, stream);
        }
        if (value == T{0}) {
            NOA_THROW_IF(cudaMemset2DAsync(array, array_pitch * sizeof(T), 0,
                                           shape.x, rows(shape), stream.id()));
        } else {
            details::set(array, array_pitch, shape, value, stream);
        }
    }

    /// Initializes or sets device memory to a value. Batched version.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void set(T* array, size_t array_pitch, size3_t shape, size_t batches, T value, Stream& stream) {
        set<CHECK_CONTIGUOUS>(array, array_pitch, {shape.x, rows(shape), batches}, value, stream);
    }
}
