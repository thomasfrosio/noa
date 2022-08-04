/// \file noa/gpu/cuda/signal/Median.h
/// \brief Median filters.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::signal::details {
    template<typename T>
    constexpr bool is_valid_median_v =
            traits::is_float_v<T> || traits::is_any_v<T, uint32_t, uint64_t, int32_t, int64_t>;
}

namespace noa::cuda::signal {
    /// Computes the median filter using a 1D window.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b device. Array to filter.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b device. Filtered array.
    /// \param input_strides    BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window_size      Number of elements to consider for the computation of the median.
    ///                         This corresponds to the width of \p shape.
    ///                         Only odd numbers from 1 to 21 are supported. If 1, a copy is performed.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function runs asynchronously relative to the host and may return before completion.
    /// \note With \c BORDER_REFLECT, the width should be >= than `window_size/2 + 1`.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median1(const shared_t<T[]>& input, size4_t input_strides,
                 const shared_t<T[]>& output, size4_t output_strides,
                 size4_t shape, BorderMode border_mode, size_t window_size, Stream& stream);

    /// Computes the median filter using a 2D square window.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b device. Array to filter.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b device. Filtered array.
    /// \param input_strides    BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window_size      Number of elements to consider for the computation of the median, for each dimension.
    ///                         This corresponds to the height and width dimensions of \p shape.
    ///                         Only odd numbers from 1 to 11 are supported. If 1, no filter is applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function runs asynchronously relative to the host and may return before completion.
    /// \note With \c BORDER_REFLECT, the height and width should be >= than `window_size/2 + 1`.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median2(const shared_t<T[]>& input, size4_t input_strides,
                 const shared_t<T[]>& output, size4_t output_strides,
                 size4_t shape, BorderMode border_mode, size_t window_size, Stream& stream);

    /// Computes the median filter using a 3D cubic window.
    /// \tparam T               (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b device. Array to filter.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b device. Filtered array.
    /// \param input_strides    BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window_size      Number of elements to consider for the computation of the median.
    ///                         This corresponds to the depth, height and width dimension.
    ///                         Only odd numbers from 1 to 5 are supported. If 1, no filter is applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function runs asynchronously relative to the host and may return before completion.
    /// \note With \c BORDER_REFLECT, the depth, height and width should be >= than `window_size/2 + 1`.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median3(const shared_t<T[]>& input, size4_t input_strides,
                 const shared_t<T[]>& output, size4_t output_strides,
                 size4_t shape, BorderMode border_mode, size_t window_size, Stream& stream);
}
