/// \file noa/cpu/filter/Convolve.h
/// \brief Real space convolutions.
/// \author Thomas - ffyr2w
/// \date 22 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/cpu/memory/PtrHost.h"

namespace noa::cpu::filter {
    /// 1D convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        On the \b host. Array to convolve.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Convolved array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param[in] filter       On the \b host. 1D filter.
    /// \param filter_size      Size, in elements, of \p filter. It should be an odd number from 1 to 129.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U>
    NOA_HOST void convolve1(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                            size4_t shape, const U* filter, size_t filter_size, Stream& stream);

    /// 2D convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        On the \b host. Array to convolve.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Convolved array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param[in] filter       On the \b host. 2D filter.
    /// \param filter_shape     Rightmost shape of \p filter. It should be two odd numbers from 1 to 17.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U>
    NOA_HOST void convolve2(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                            size4_t shape, const U* filter, size2_t filter_shape, Stream& stream);

    /// 3D convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        On the \b host. Array to convolve.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Convolved array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param[in] filter       On the \b host. 3D filter.
    /// \param filter_shape     Rightmost shape of \p filter. It should be three odd numbers from 1 to 5.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U>
    NOA_HOST void convolve3(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                            size4_t shape, const U* filter, size3_t filter_shape, Stream& stream);

    /// ND convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        On the \b host. Input array to convolve.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Output convolved array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param[in] filter       On the \b host. ND filter.
    /// \param filter_shape     Rightmost shape of \p filter.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U>
    NOA_IH void convolve(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                         size4_t shape, const U* filter, size3_t filter_shape, Stream& stream) {
        size_t dim = filter_shape.ndim();
        NOA_ASSERT(dim && dim <= 3);
        switch (dim) {
            case 1:
                return convolve1(input, input_stride, output, output_stride,
                                 shape, filter, filter_shape[2], stream);
            case 2:
                return convolve2(input, input_stride, output, output_stride,
                                 shape, filter, {filter_shape[1], filter_shape[2]}, stream);
            case 3:
                return convolve3(input, input_stride, output, output_stride,
                                 shape, filter, filter_shape, stream);
            default:
                break;
        }
    }

    /// Separable convolutions. \p input is convolved with \p filter0, then \p filter1, then \p filter2.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        On the \b host. Input array to convolve.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Output convolved array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param[in] filter0      On the \b host. Applied along the outermost dimension of \p shape.
    /// \param filter0_size     Size, in elements, of \p filter0. Should be an odd number from 1 to 129.
    /// \param[in] filter1      On the \b host. Applied along the second-most dimension of \p shape.
    /// \param filter1_size     Size, in elements, of \p filter1. Should be an odd number from 1 to 129.
    /// \param[in] filter2      On the \b host. Applied along the innermost dimension of \p shape.
    /// \param filter2_size     Size, in elements, of \p filter2. Should be an odd number from 1 to 129.
    /// \param[in,out] tmp      If more than one convolution is performed, it should be an array of the same shape
    ///                         as \p input. Otherwise, it is ignored and nullptr can be passed.
    /// \param tmp_stride       Rightmost strides, in elements, of \p tmp.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note If a filter is nullptr, the convolution in the corresponding dimension is not applied and it goes
    ///       directly to the next filter, if any. Filters can be equal to each other.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U>
    NOA_HOST void convolve(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                           const U* filter0, size_t filter0_size,
                           const U* filter1, size_t filter1_size,
                           const U* filter2, size_t filter2_size,
                           T* tmp, size4_t tmp_stride, Stream& stream);

    /// Separable convolutions. \p input is convolved with \p filter0, then \p filter1, then \p filter2.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] input        Input array to convolve.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      Output convolved array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param[in] filter0      On the \b host. Applied along the outermost dimension of \p shape.
    /// \param filter0_size     Size, in elements, of \p filter0. Should be an odd number from 1 to 129.
    /// \param[in] filter1      On the \b host. Applied along the second-most dimension of \p shape.
    /// \param filter1_size     Size, in elements, of \p filter1. Should be an odd number from 1 to 129.
    /// \param[in] filter2      On the \b host. Applied along the innermost dimension of \p shape.
    /// \param filter2_size     Size, in elements, of \p filter2. Should be an odd number from 1 to 129.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream will be synchronized when this function returns.
    ///
    /// \note If a filter is nullptr, the convolution in the corresponding dimension is not applied and it goes
    ///       directly to the next filter, if any. Filters can be equal to each other. If more than one convolution
    ///       is performed, a temporary array of the same shape as \p input is allocated.
    /// \note \p input and \p output should not overlap.
    template<typename T, typename U>
    NOA_IH void convolve(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                         const U* filter0, size_t filter0_size,
                         const U* filter1, size_t filter1_size,
                         const U* filter2, size_t filter2_size, Stream& stream) {
        memory::PtrHost<T> tmp;
        int count = 0;
        if (filter0)
            count += 1;
        if (filter1)
            count += 1;
        if (filter2)
            count += 1;
        if (count > 1)
            tmp.reset(shape.elements());
        convolve(input, input_stride, output, output_stride, shape,
                 filter0, filter0_size, filter1, filter1_size, filter2, filter2_size, tmp.get(), shape, stream);
        stream.synchronize();
    }
}
