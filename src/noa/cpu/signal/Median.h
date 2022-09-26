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
    // Computes the median filter using a 1D window.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median1(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides,
                 dim4_t shape, BorderMode border_mode, dim_t window_size, Stream& stream);

    // Computes the median filter using a 2D square window.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median2(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides,
                 dim4_t shape, BorderMode border_mode, dim_t window_size, Stream& stream);

    // Computes the median filter using a 3D cubic window.
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median3(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides,
                 dim4_t shape, BorderMode border_mode, dim_t window_size, Stream& stream);
}
