/// \file noa/gpu/cuda/memory/Set.h
/// \brief Set to value.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/string/Format.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory::details {
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void set(T* src, size_t elements, T value, Stream& stream);

    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void set(const shared_t<T[]>& src, size4_t strides, size4_t shape, T value, Stream& stream);
}

// TODO Add nvrtc to support any type.

namespace noa::cuda::memory {
    /// Sets an array with a given value.
    /// \tparam T               Any data type.
    /// \param[out] src         On the \b device. The beginning of range to set.
    /// \param elements         Number of elements to set.
    /// \param value            The value to assign.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    ///       One must make sure \p src stays valid until completion.
    template<typename T>
    inline void set(T* src, size_t elements, T value, Stream& stream) {
        if constexpr (traits::is_data_v<T>) {
            if (value == T{0})
                NOA_THROW_IF(cudaMemsetAsync(src, 0, elements * sizeof(T), stream.id()));
            else
                details::set(src, elements, value, stream);
        } else if constexpr (traits::is_floatX_v<T> ||
                             traits::is_intX_v<T> ||
                             traits::is_floatXX_v<T>) {
            if (all(value == T{0})) {
                NOA_THROW_IF(cudaMemsetAsync(src, 0, elements * sizeof(T), stream.id()));
            } else {
                NOA_THROW("Setting an array of {} with a value other than {} is not currently allowed",
                          string::human<T>(), T{0});
            }
        }
    }

    /// Sets an array with a given value.
    /// \tparam T               Any data type.
    /// \param[out] src         On the \b device. The beginning of range to set.
    /// \param elements         Number of elements to set.
    /// \param value            The value to assign.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    inline void set(const shared_t<T[]>& src, size_t elements, T value, Stream& stream) {
        set(src.get(), elements, value, stream);
        stream.attach(src);
    }

    /// Sets an array with a given value.
    /// \tparam SWAP_LAYOUT     Swap the memory layout to optimize the \p dst writes.
    ///                         If false, assume rightmost order is the fastest order.
    /// \tparam T               Any data type.
    /// \param[out] src         On the \b device. The beginning of range to set.
    /// \param strides          Strides, in elements.
    /// \param shape            Shape to set.
    /// \param value            The value to assign.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<bool SWAP_LAYOUT = true, typename T>
    inline void set(const shared_t<T[]>& src, size4_t strides, size4_t shape, T value, Stream& stream) {
        if constexpr (SWAP_LAYOUT) {
            const size4_t order = indexing::order(strides, shape);
            shape = indexing::reorder(shape, order);
            strides = indexing::reorder(strides, order);
        }

        if constexpr (!traits::is_data_v<T>) {
            NOA_CHECK(indexing::areContiguous(strides, shape),
                      "Setting a non-contiguous array of {} is currently not allowed", string::human<T>());
            return set(src, shape.elements(), value, stream);
        } else {
            if (indexing::areContiguous(strides, shape))
                return set(src, shape.elements(), value, stream);
            details::set(src, strides, shape, value, stream);
        }
    }
}
