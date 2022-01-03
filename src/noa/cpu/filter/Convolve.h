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
    /// \param[in] inputs       On the \b host. Array to convolve. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Convolved array. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter       On the \b host. Filter corresponding to the first dimension of \p shape.
    /// \param filter_size      Size, in elements, of \p filter.
    ///                         It should be an odd number from 1 to 129. If 1, a simple copy is computed.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T, typename U>
    NOA_HOST void convolve1(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                            size3_t shape, size_t batches, const U* filter, size_t filter_size, Stream& stream);

    /// 2D convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] inputs       On the \b host. Array to convolve. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Convolved array. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter       On the \b host. 2D filter corresponding to the first two dimensions of \p shape.
    /// \param filter_shape     Physical {fast, medium} shape of \p filter.
    ///                         It should be two odd numbers from 1 to 17. If 1, a simple copy is computed.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T, typename U>
    NOA_HOST void convolve2(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                            size3_t shape, size_t batches, const U* filter, size2_t filter_shape, Stream& stream);

    /// 3D convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] inputs       On the \b host. Array to convolve. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Convolved array. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter       On the \b host. 3D filter corresponding to \p shape.
    /// \param filter_shape     Physical {fast, medium, slow} shape of \p filter.
    ///                         It should be three odd numbers from 1 to 5. If 1, a simple copy is computed.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T, typename U>
    NOA_HOST void convolve3(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                            size3_t shape, size_t batches, const U* filter, size3_t filter_shape, Stream& stream);

    /// ND convolution.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] inputs       On the \b host. Input arrays to convolve. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Output convolved arrays. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter       On the \b host. ND filter corresponding to \p shape.
    /// \param filter_shape     Physical {fast, medium, slow} shape of \p filter.
    ///                         The dimensionality of the convolution is determined by `ndim(filter_shape)`.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T, typename U>
    NOA_IH void convolve(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                         size3_t shape, size_t batches, const U* filter, size3_t filter_shape, Stream& stream) {
        size_t dim = ndim(filter_shape);
        NOA_ASSERT(dim && dim <= 3);
        switch (dim) {
            case 1U:
                return convolve1(inputs, input_pitch, outputs, output_pitch,
                                 shape, batches, filter, filter_shape.x, stream);
            case 2U:
                return convolve2(inputs, input_pitch, outputs, output_pitch,
                                 shape, batches, filter, {filter_shape.x, filter_shape.y}, stream);
            case 3U:
                return convolve3(inputs, input_pitch, outputs, output_pitch,
                                 shape, batches, filter, filter_shape, stream);
            default:
                break;
        }
    }

    /// Separable convolutions. \p inputs is convolved with \p filter0, then \p filter1, then \p filter2.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] inputs       On the \b host. Input arrays to convolve. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Output convolved arrays. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter0      On the \b host. Corresponds to the 1st dimension of \p shape.
    ///                         Can be equal to \p filter1 or \p filter2.
    /// \param filter0_size     Size, in elements, of \p filter0. Should be an odd number from 3 to 129.
    /// \param[in] filter1      On the \b host. Corresponds to the 2nd dimension of \p shape.
    ///                         Can be equal to \p filter0 or \p filter2.
    /// \param filter1_size     Size, in elements, of \p filter1. Should be an odd number from 3 to 129.
    /// \param[in] filter2      On the \b host. Corresponds to the 3rd dimension of \p shape.
    ///                         Can be equal to \p filter0 or \p filter1.
    /// \param filter2_size     Size, in elements, of \p filter2. Should be an odd number from 3 to 129.
    /// \param[in,out] tmp      If more than one convolution is performed (see note below), it should be an array
    ///                         of the same shape as \p inputs (ignoring \p batches). Otherwise, it is ignored
    ///                         and nullptr can be passed.
    /// \param tmp_pitch        Pitch, in elements, of \p tmp.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note If a filter is nullptr, the convolution in the corresponding dimension is not applied and it goes
    ///       directly to the next filter, if any.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T, typename U>
    NOA_HOST void convolve(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                           size3_t shape, size_t batches,
                           const U* filter0, size_t filter0_size,
                           const U* filter1, size_t filter1_size,
                           const U* filter2, size_t filter2_size,
                           T* tmp, size3_t tmp_pitch, Stream& stream);

    /// Separable convolutions. \p inputs is convolved with \p filter0, then \p filter1, then \p filter2.
    /// \tparam T               half_t, float, double.
    /// \tparam U               Should be the same as \p T, except if \p T is half_t. In this case, \p U can be float.
    /// \param[in] inputs       Input arrays to convolve. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     Output convolved arrays. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter0      On the \b host. Corresponds to the 1st dimension of \p shape.
    ///                         Can be equal to \p filter1 or \p filter2.
    /// \param filter0_size     Size, in elements, of \p filter0. Should be an odd number from 3 to 129.
    /// \param[in] filter1      On the \b host. Corresponds to the 2nd dimension of \p shape.
    ///                         Can be equal to \p filter0 or \p filter2.
    /// \param filter1_size     Size, in elements, of \p filter1. Should be an odd number from 3 to 129.
    /// \param[in] filter2      On the \b host. Corresponds to the 3rd dimension of \p shape.
    ///                         Can be equal to \p filter0 or \p filter1.
    /// \param filter2_size     Size, in elements, of \p filter2. Should be an odd number from 3 to 129.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream will be synchronized when this function returns.
    ///
    /// \note If a filter is nullptr, the convolution in the corresponding dimension is not applied and it goes
    ///       directly to the next filter, if any. If more than one convolution is performed, a temporary array
    ///       of the same shape as \p inputs (ignoring \p batches) is allocated.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T, typename U>
    NOA_IH void convolve(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                         size3_t shape, size_t batches,
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
            tmp.reset(elements(shape));
        convolve(inputs, input_pitch, outputs, output_pitch, shape, batches,
                 filter0, filter0_size, filter1, filter1_size, filter2, filter2_size, tmp.get(), shape, stream);
        stream.synchronize();
    }
}
