#pragma once

#include "noa/unified/Array.h"

namespace noa::signal::details {
    template<typename T>
    constexpr bool is_valid_median_v =
            traits::is_float_v<T> || traits::is_any_v<T, uint32_t, uint64_t, int32_t, int64_t>;
}

namespace noa::signal {
    /// Computes the median filter using a 1D window.
    /// \tparam T           (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param window_size  Number of elements to consider for the computation of the median.
    ///                     This corresponds to the width dimension.
    ///                     Only odd numbers are supported. On the GPU, this is limited to 21.
    /// \param border_mode  Border mode used for the "implicit padding". Either BORDER_ZERO or BORDER_REFLECT.
    ///                     With BORDER_REFLECT, the width should be >= than ``window_size/2 + 1``.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median1(const Array<T>& input, const Array<T>& output,
                 dim_t window_size, BorderMode border_mode = BORDER_REFLECT);

    /// Computes the median filter using a 2D square window.
    /// \tparam T           (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param window_size  Number of elements to consider for the computation of the median, for each dimension.
    ///                     This corresponds to the height and width dimension.
    ///                     Only odd numbers are supported. On the GPU, this is limited to 11.
    /// \param border_mode  Border mode used for the "implicit padding". Either BORDER_ZERO or BORDER_REFLECT.
    ///                     With BORDER_REFLECT, the height and width should be >= than ``window_size/2 + 1``.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median2(const Array<T>& input, const Array<T>& output,
                 dim_t window_size, BorderMode border_mode = BORDER_REFLECT);

    /// Computes the median filter using a 3D cubic window.
    /// \tparam T           (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param window_size  Number of elements to consider for the computation of the median, for each dimension.
    ///                     This corresponds to the depth, height and width dimensions.
    ///                     Only odd numbers are supported. On the GPU, this is limited to 5.
    /// \param border_mode  Border mode used for the "implicit padding". Either BORDER_ZERO or BORDER_REFLECT.
    ///                     With BORDER_REFLECT, the depth, height and width should be >= than ``window_size/2 + 1``.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median3(const Array<T>& input, const Array<T>& output,
                 dim_t window_size, BorderMode border_mode = BORDER_REFLECT);
}

#define NOA_UNIFIED_MEDIAN_
#include "noa/unified/signal/Median.inl"
#undef NOA_UNIFIED_MEDIAN_
