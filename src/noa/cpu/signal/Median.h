/// \file noa/cpu/signal/Median.h
/// \brief Median filters.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::signal::details {
    template<typename T>
    constexpr bool is_valid_median_v =
            traits::is_float_v<T> || traits::is_any_v<T, uint32_t, uint64_t, int32_t, int64_t>;
}

namespace noa::cpu::signal {
    /// Computes the median filter using a 1D window.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b host. Array to filter.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Filtered array.
    /// \param input_stride     Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO or BORDER_REFLECT.
    /// \param window_size      Number of elements to consider for the computation of the median.
    ///                         This corresponds to the innermost dimension.
    ///                         Only odd numbers are supported. If 1, a copy is performed.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note With \c BORDER_REFLECT, the innermost dimension should be >= than ``window_size/2 + 1``.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median1(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride,
                 size4_t shape, BorderMode border_mode, size_t window_size, Stream& stream);

    /// Computes the median filter using a 2D square window.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b host. Array to filter.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Filtered array.
    /// \param input_stride     Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window_size      Number of elements to consider for the computation of the median, for each dimension.
    ///                         This corresponds to the second and innermost dimension.
    ///                         Only odd numbers are supported. If 1, no filter is applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note With \c BORDER_REFLECT, the second and innermost dimensions should be >= than ``window_size/2 + 1``.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median2(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride,
                 size4_t shape, BorderMode border_mode, size_t window_size, Stream& stream);

    /// Computes the median filter using a 3D cubic window.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b host. Array to filter.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Filtered array.
    /// \param input_stride     Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window_size      Number of elements to consider for the computation of the median, for each dimension.
    ///                         Only odd numbers are supported. If 1, no filter is applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note With \c BORDER_REFLECT, each dimension should be >= than ``window_size/2 + 1``.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median3(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride,
                 size4_t shape, BorderMode border_mode, size_t window_size, Stream& stream);
}
