#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory {
    /// Casts one array to another type.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \tparam U               (u)int32_t, (u)int64_t, half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    ///                         If \p T is complex, \p U should be complex as well.
    /// \param[in] input        On the \b device. Array to convert.
    /// \param[out] output      On the \b device. Converted array.
    /// \param elements         Number of elements to convert.
    /// \param clamp            Whether the values should be clamp within the \p U range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U>
    NOA_HOST void cast(const T* input, U* output, size_t elements, bool clamp, Stream& stream);

    /// Casts one array to another type.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \tparam U               (u)int32_t, (u)int64_t, half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    ///                         If \p T is complex, \p U should be complex as well.
    /// \param[in] input        On the \b device. Array to convert.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b device. Converted array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param clamp            Whether the values should be clamp within the \p U range.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U>
    NOA_HOST void cast(const T* input, size4_t input_stride, U* output, size4_t output_stride,
                       size4_t shape, bool clamp, Stream& stream);
}
