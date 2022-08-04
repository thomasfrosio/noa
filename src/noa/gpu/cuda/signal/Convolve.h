/// \file noa/gpu/cuda/signal/Convolve.h
/// \brief Real space convolutions.
/// \author Thomas - ffyr2w
/// \date 22 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::signal::details {
    template<typename T, typename U>
    constexpr bool is_valid_conv_v = traits::is_float_v<T> && std::is_same_v<T, U>;
}

namespace noa::cuda::signal {
    /// 1D convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Same as \p T.
    /// \param[in] input        On the \b device. Array to convolve.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b device. Convolved array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param[in] filter       On the \b host or \b device. 1D filter.
    /// \param filter_size      Width, in elements, of \p filter. It should be an odd number and be at most 1032 bytes.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p input and \p output should not overlap.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \warning This function modifies a per-device state. As such, there should be no concurrent calls from
    ///          different streams sharing the same device.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve1(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                   const shared_t<U[]>& filter, size_t filter_size, Stream& stream);

    /// 2D convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Same as \p T.
    /// \param[in] input        On the \b device. Array to convolve.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b device. Convolved array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param[in] filter       On the \b host or \b device. 2D filter.
    /// \param filter_shape     HW shape of \p filter. It should be an odd number and be at most 1032 bytes.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p input and \p output should not overlap.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \warning This function modifies a per-device state. As such, there should be no concurrent calls from
    ///          different streams sharing the same device.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve2(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                   const shared_t<U[]>& filter, size2_t filter_shape, Stream& stream);

    /// 3D convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Same as \p T.
    /// \param[in] input        On the \b device. Array to convolve.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b device. Convolved array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param[in] filter       On the \b host or \b device. 3D filter.
    /// \param filter_shape     DHW shape of \p filter. It should be an odd number and be at most 1032 bytes.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p input and \p output should not overlap.
    /// \note This function is optimized for 3x3x3 and 5x5x5 filters.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \warning This function modifies a per-device state. As such, there should be no concurrent calls from
    ///          different streams sharing the same device.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve3(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                   const shared_t<U[]>& filter, size3_t filter_shape, Stream& stream);

    /// Separable convolutions. \p input is convolved with \p filter0, then \p filter1, then \p filter2.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Same as \p T.
    /// \param[in] input        On the \b device. Input array to convolve.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b device. Output convolved array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param[in] filter0      On the \b host or \b device. Applied along the depth dimension of \p shape.
    /// \param filter0_size     Size, in elements, of \p filter0. Should be an odd number and be at most 1032 bytes.
    /// \param[in] filter1      On the \b host or \b device. Applied along the height dimension of \p shape.
    /// \param filter1_size     Size, in elements, of \p filter1. Should be an odd number and be at most 1032 bytes.
    /// \param[in] filter2      On the \b host or \b device. Applied along the width dimension of \p shape.
    /// \param filter2_size     Size, in elements, of \p filter2. Should be an odd number and be at most 1032 bytes.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \param[in,out] tmp      Iff more than one convolution is performed, a temporary array of the same shape
    ///                         as \p input is required. In this case and if nullptr, this temporary array is allocated.
    /// \param tmp_strides      BDHW strides, in elements, of \p tmp.
    ///
    /// \note \p input and \p output should not overlap.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note If a filter is nullptr, the convolution in the corresponding dimension is not applied and it goes
    ///       directly to the next filter, if any. Filters can be equal to each other.
    /// \warning This function modifies a per-device state. As such, there should be no concurrent calls from
    ///          different streams sharing the same device.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve(const shared_t<T[]>& input, size4_t input_strides,
                  const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                  const shared_t<U[]>& filter0, size_t filter0_size,
                  const shared_t<U[]>& filter1, size_t filter1_size,
                  const shared_t<U[]>& filter2, size_t filter2_size, Stream& stream,
                  const shared_t<T[]>& tmp = nullptr, size4_t tmp_strides = {});

    /// ND convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        On the \b device. Input array to convolve.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b device. Output convolved array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param[in] filter       On the \b host or \b device. ND filter. It should be in the rightmost order.
    /// \param filter_shape     DHW shape of \p filter.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note \p input and \p output should not overlap.
    /// \warning This function modifies a per-device state. As such, there should be no concurrent calls from
    ///          different streams sharing the same device.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve(const shared_t<T[]>& input, size4_t input_strides,
                  const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                  const shared_t<U[]>& filter, size3_t filter_shape, Stream& stream);
}
