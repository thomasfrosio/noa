#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory::details {
    template<typename T>
    constexpr bool is_valid_arange_v =
            traits::is_any_v<T, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t> ||
            traits::is_float_v<T> || traits::is_complex_v<T>;
}

namespace noa::cuda::memory {
    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any data type.
    /// \param[in,out] src      On the \b device. Array with evenly spaced values.
    /// \param elements         Number of elements to set.
    /// \param start            Start of interval.
    /// \param step             Spacing between values.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<details::is_valid_arange_v<T>>>
    void arange(const shared_t<T[]>& src, size_t elements, T start, T step, Stream& stream);

    /// Returns evenly spaced values within a given interval, starting from 0, with a step of 1.
    /// \tparam T               Any data type.
    /// \param[in,out] src      On the \b device. Array with evenly spaced values.
    /// \param elements         Number of elements to set.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<details::is_valid_arange_v<T>>>
    NOA_IH void arange(const shared_t<T[]>& src, size_t elements, Stream& stream) {
        arange(src, elements, T(0), T(1), stream);
    }

    /// Returns evenly spaced values within a given interval.
    /// \tparam T               Any data type.
    /// \param[in,out] src      On the \b device. Array with evenly spaced values.
    /// \param stride           Rightmost strides, in elements, of \p src.
    /// \param shape            Rightmost shape of \p src.
    /// \param start            Start of interval.
    /// \param step             Spacing between values.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<details::is_valid_arange_v<T>>>
    void arange(const shared_t<T[]>& src, size4_t stride, size4_t shape, T start, T step, Stream& stream);

    /// Returns evenly spaced values within a given interval, starting from 0, with a step of 1.
    /// \tparam T               Any data type.
    /// \param[in,out] src      On the \b device. Array with evenly spaced values.
    /// \param stride           Rightmost strides, in elements, of \p src.
    /// \param shape            Rightmost shape of \p src.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<details::is_valid_arange_v<T>>>
    NOA_IH void arange(const shared_t<T[]>& src, size4_t stride, size4_t shape, Stream& stream) {
        arange(src, stride, shape, T(0), T(1), stream);
    }
}
