#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/string/Format.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::memory::details {
    template<typename T>
    constexpr bool is_valid_set_v =
            traits::is_restricted_numeric_v<T> || std::is_same_v<T, bool> ||
            traits::is_vecX_v<T> || traits::is_matXX_v<T>;

    template<typename T, typename = std::enable_if_t<traits::is_restricted_numeric_v<T> || std::is_same_v<T, bool>>>
    void set(T* src, i64 elements, T value, Stream& stream);

    template<typename T, typename = std::enable_if_t<traits::is_restricted_numeric_v<T> || std::is_same_v<T, bool>>>
    void set(T* src, const Strides4<i64>& strides, const Shape4<i64>& shape, T value, Stream& stream);
}

// TODO Add nvrtc to support any type.

namespace noa::cuda::memory {
    // Sets an array, that is on the device, with a given value.
    // One must make sure src stays valid until completion.
    template<typename T, typename = std::enable_if_t<details::is_valid_set_v<T>>>
    void set(T* src, i64 elements, T value, Stream& stream) {
        if constexpr (traits::is_restricted_numeric_v<T> || std::is_same_v<T, bool>) {
            if (value == T{0})
                NOA_THROW_IF(cudaMemsetAsync(src, 0, static_cast<size_t>(elements) * sizeof(T), stream.id()));
            else
                details::set(src, elements, value, stream);
        } else if constexpr (traits::is_vecX_v<T> || traits::is_matXX_v<T>) {
            if (noa::all(value == T{0})) {
                NOA_THROW_IF(cudaMemsetAsync(src, 0, static_cast<size_t>(elements) * sizeof(T), stream.id()));
            } else {
                NOA_THROW("Setting an array of {} with a value other than {} is not currently allowed",
                          string::human<T>(), T{0});
            }
        }
    }

    // Sets an array with a given value.
    template<typename T, typename = std::enable_if_t<details::is_valid_set_v<T>>>
    void set(T* src, Strides4<i64> strides, Shape4<i64> shape, T value, Stream& stream) {
        const auto order = noa::indexing::order(strides, shape);
        shape = noa::indexing::reorder(shape, order);
        strides = noa::indexing::reorder(strides, order);

        if constexpr (!traits::is_restricted_numeric_v<T> && !std::is_same_v<T, bool>) {
            NOA_CHECK(noa::indexing::are_contiguous(strides, shape),
                      "Setting a non-contiguous array of {} is currently not allowed", string::human<T>());
            return set(src, shape.elements(), value, stream);
        } else {
            if (noa::indexing::are_contiguous(strides, shape))
                return set(src, shape.elements(), value, stream);
            details::set(src, strides, shape, value, stream);
        }
    }
}
