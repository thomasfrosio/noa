#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::signal::details {
    template<typename T, typename U>
    constexpr bool is_valid_conv_v =
            traits::is_float_v<T> && (std::is_same_v<T, U> || (std::is_same_v<T, half_t> && std::is_same_v<U, float>));
}

namespace noa::cpu::signal {
    // If half_t, the filter can be float.
    // Filter should have an odd number of elements for each dimension.

    // 1D convolution along the width.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve1(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   const shared_t<U[]>& filter, dim_t filter_size, Stream& stream);

    // 2D convolution along the height and width.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve2(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   const shared_t<U[]>& filter, dim2_t filter_shape, Stream& stream);

    // 3D convolution along the depth, height and width.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve3(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   const shared_t<U[]>& filter, dim3_t filter_shape, Stream& stream);

    // Separable convolutions with filter0 (depth), then filter1 (height), then filter2 (width).
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  const shared_t<U[]>& filter0, dim_t filter0_size,
                  const shared_t<U[]>& filter1, dim_t filter1_size,
                  const shared_t<U[]>& filter2, dim_t filter2_size, Stream& stream,
                  const shared_t<T[]>& tmp = nullptr, dim4_t tmp_strides = {});

    // ND convolution. ND filter should be in the rightmost order.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  const shared_t<U[]>& filter, dim3_t filter_shape, Stream& stream);
}
