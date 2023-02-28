#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::signal::details {
    template<typename T>
    constexpr bool is_valid_median_v = noa::traits::is_real_v<T> || noa::traits::is_any_v<T, u32, u64, i32, i64>;
}

namespace noa::cuda::signal {
    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median_filter_1d(const T* input, const Strides4<i64>& input_strides,
                          T* output, const Strides4<i64>& output_strides,
                          const Shape4<i64>& shape, BorderMode border_mode, i64 window_size, Stream& stream);

    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median_filter_2d(const T* input, Strides4<i64> input_strides,
                          T* output, Strides4<i64> output_strides,
                          Shape4<i64> shape, BorderMode border_mode, i64 window_size, Stream& stream);

    template<typename T, typename = std::enable_if_t<details::is_valid_median_v<T>>>
    void median_filter_3d(const T* input, Strides4<i64> input_strides,
                          T* output, Strides4<i64> output_strides,
                          Shape4<i64> shape, BorderMode border_mode, i64 window_size, Stream& stream);
}
