#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory::details {
    template<typename T, typename U>
    constexpr bool is_valid_cast_v =
            (traits::is_any_v<T, bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, half_t, float, double> &&
             traits::is_any_v<U, bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, half_t, float, double>) ||
            (traits::is_complex_v<T> && traits::is_complex_v<U>);
}

namespace noa::cuda::memory {
    /// Casts one array to another type.
    /// \tparam T               Any data type.
    /// \tparam U               Any data type. If \p T is complex, \p U should be complex as well.
    /// \param[in] input        On the \b device. Array to convert.
    /// \param[out] output      On the \b device. Converted array.
    /// \param elements         Number of elements to convert.
    /// \param clamp            Whether the values should be clamp within the \p U range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_cast_v<T, U>>>
    void cast(const shared_t<T[]>& input,
              const shared_t<U[]>& output,
              size_t elements, bool clamp, Stream& stream);

    /// Casts one array to another type.
    /// \tparam T               Any data type.
    /// \tparam U               Any data type. If \p T is complex, \p U should be complex as well.
    /// \param[in] input        On the \b device. Array to convert.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b device. Converted array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param clamp            Whether the values should be clamp within the \p U range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_cast_v<T, U>>>
    void cast(const shared_t<T[]>& input, size4_t input_stride,
              const shared_t<U[]>& output, size4_t output_stride,
              size4_t shape, bool clamp, Stream& stream);
}
