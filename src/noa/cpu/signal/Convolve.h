/// \file noa/cpu/signal/Convolve.h
/// \brief Real space convolutions.
/// \author Thomas - ffyr2w
/// \date 22 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/math/Ewise.h"

namespace noa::cpu::signal::details {
    template<typename T, typename U>
    constexpr bool is_valid_conv_v =
            traits::is_float_v<T> && (std::is_same_v<T, U> || (std::is_same_v<T, half_t> && std::is_same_v<U, float>));
}

namespace noa::cpu::signal {
    /// 1D convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        On the \b host. Array to convolve.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Convolved array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param[in] filter       On the \b host. 1D filter.
    /// \param filter_size      Width, in elements, of \p filter. It should be an odd number.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve1(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                   const shared_t<U[]>& filter, size_t filter_size, Stream& stream);

    /// 2D convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        On the \b host. Array to convolve.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Convolved array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param[in] filter       On the \b host. 2D filter in the rightmost order.
    /// \param filter_shape     HW shape of \p filter. It should be two odd numbers.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve2(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                   const shared_t<U[]>& filter, size2_t filter_shape, Stream& stream);

    /// 3D convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        On the \b host. Array to convolve.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Convolved array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param[in] filter       On the \b host. 3D filter. It should be in the rightmost order.
    /// \param filter_shape     DHW shape of \p filter. It should be three odd numbers.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    void convolve3(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                   const shared_t<U[]>& filter, size3_t filter_shape, Stream& stream);

    /// Separable convolutions. \p input is convolved with \p filter0, then \p filter1, then \p filter2.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        On the \b host. Input array to convolve.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Output convolved array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param[in] filter0      On the \b host. Applied along the depth dimension of \p shape.
    /// \param filter0_size     Size, in elements, of \p filter0. Should be an odd number.
    /// \param[in] filter1      On the \b host. Applied along the height dimension of \p shape.
    /// \param filter1_size     Size, in elements, of \p filter1. Should be an odd number.
    /// \param[in] filter2      On the \b host. Applied along the width dimension of \p shape.
    /// \param filter2_size     Size, in elements, of \p filter2. Should be an odd number.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \param[in,out] tmp      Iff more than one convolution is performed, a temporary array of the same shape
    ///                         as \p input is required. In this case and if nullptr, this temporary array is allocated.
    /// \param tmp_strides      BDHW strides, in elements, of \p tmp.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note If a filter is nullptr, the convolution in the corresponding dimension is not applied and it goes
    ///       directly to the next filter, if any. Filters can be equal to each other.
    /// \note \p input and \p output should not overlap.
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
    /// \param[in] input        On the \b host. Input array to convolve.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Output convolved array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param[in] filter       On the \b host. ND filter. It should be in the rightmost order.
    /// \param filter_shape     DHW shape of \p filter.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_conv_v<T, U>>>
    inline void convolve(const shared_t<T[]>& input, size4_t input_strides,
                         const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                         const shared_t<U[]>& filter, size3_t filter_shape, Stream& stream) {
        NOA_ASSERT(all(filter_shape >= 1));
        const size_t ndim = filter_shape.ndim();

        // If there's a single dimension, use separable convolution kernels:
        if (ndim == 1 || (ndim == 3 && filter_shape[1] == 1 && filter_shape[2])) {
            if (all(filter_shape == 1)) {
                math::ewise(input, input_strides, static_cast<T>(filter[0]),
                            output, output_strides, shape,
                            noa::math::multiply_t{}, stream);
            } else if (filter_shape[2] > 1) {
                convolve<T, U>(input, input_strides, output, output_strides, shape,
                               nullptr, 0, nullptr, 0, filter, filter_shape[2], stream);
            } else if (filter_shape[1] > 1) {
                convolve<T, U>(input, input_strides, output, output_strides, shape,
                               nullptr, 0, filter, filter_shape[1], nullptr, 0, stream);
            } else {
                convolve<T, U>(input, input_strides, output, output_strides, shape,
                               filter, filter_shape[0], nullptr, 0, nullptr, 0, stream);
            }
            return;
        } else if (ndim == 2) {
            return convolve2(input, input_strides, output, output_strides,
                             shape, filter, {filter_shape[1], filter_shape[2]}, stream);
        } else {
            return convolve3(input, input_strides, output, output_strides,
                             shape, filter, filter_shape, stream);
        }
    }
}
