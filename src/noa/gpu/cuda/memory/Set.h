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
    void set(const shared_t<T[]>& src, size_t elements, T value, Stream& stream);

    template<typename T>
    void set(const shared_t<T[]>& src, size4_t stride, size4_t shape, T value, Stream& stream);
}

namespace noa::cuda::memory {
    /// Sets an array with a given value.
    /// \tparam T               Any data type.
    /// \param[out] src         On the \b device. The beginning of range to set.
    /// \param elements         Number of elements to set.
    /// \param value            The value to assign.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void set(const shared_t<T[]>& src, size_t elements, T value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if (value == T{0}) {
            NOA_THROW_IF(cudaMemsetAsync(src.get(), 0, elements * sizeof(T), stream.id()));
            stream.attach(src);
        } else {
            details::set(src, elements, value, stream);
        }
    }

    /// Sets an array with a given value.
    /// \tparam CHECK_CONTIGUOUS    Filling a contiguous block of memory is often more efficient. If true, the function
    ///                             checks whether or not the data is contiguous and if so performs one contiguous memset.
    /// \tparam T                   Any data type.
    /// \param[out] src             On the \b device. The beginning of range to set.
    /// \param stride               Rightmost strides, in elements.
    /// \param shape                Rightmost shape to set.
    /// \param value                The value to assign.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<bool CHECK_CONTIGUOUS = true, typename T>
    NOA_IH void set(const shared_t<T[]>& src, size4_t stride, size4_t shape, T value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if constexpr (CHECK_CONTIGUOUS) {
            if (all(indexing::isContiguous(stride, shape)))
                return set(src, shape.elements(), value, stream);
        }
        details::set(src, stride, shape, value, stream);
    }
}
