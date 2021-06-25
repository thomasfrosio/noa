/// \file noa/gpu/cuda/filter/Convolve.h
/// \brief Real space convolutions.
/// \author Thomas - ffyr2w
/// \date 22 Jun 2021

#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::filter {
    /// 1D convolution.
    /// \tparam T               float, double.
    /// \param[in] inputs       Input arrays to convolve. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \a inputs.
    /// \param[out] outputs     Output convolved arrays. One per batch.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs and \a outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param filter           Filter corresponding to the first dimension of \a shape.
    /// \param filter_size      Size, in elements, of \a filter.
    ///                         It should be an odd number from 1 to 129. If 1, a simple copy is computed.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note \a filter can be on the host or device memory. They will be copied to constant device memory anyway.
    template<typename T>
    NOA_HOST void convolve1(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                            size3_t shape, uint batches, const T* filter, uint filter_size, Stream& stream);

    /// 2D convolution.
    /// \tparam T               float, double.
    /// \param[in] inputs       Input arrays to convolve. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \a inputs.
    /// \param[out] outputs     Output convolved arrays. One per batch.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs and \a outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param filter           2D filter corresponding to the first two dimensions of \a shape.
    /// \param filter_shape     Physical {fast, medium} shape of \a filter.
    ///                         It should be two odd numbers from 1 to 17. If 1, a simple copy is computed.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note \a filter can be on the host or device memory. They will be copied to constant device memory anyway.
    template<typename T>
    NOA_HOST void convolve2(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                            size3_t shape, uint batches, const T* filter, uint2_t filter_shape, Stream& stream);

    /// 3D convolution.
    /// \tparam T               float, double.
    /// \param[in] inputs       Input arrays to convolve. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \a inputs.
    /// \param[out] outputs     Output convolved arrays. One per batch.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs and \a outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param filter           3D filter corresponding to \a shape.
    /// \param filter_shape     Physical {fast, medium, slow} shape of \a filter.
    ///                         It should be three odd numbers from 1 to 5. If 1, a simple copy is computed.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note \a filter can be on the host or device memory. They will be copied to constant device memory anyway.
    /// \note This function has template instantiations (i.e. is optimized) for 3x3x3 and 5x5x5 filters.
    template<typename T>
    NOA_HOST void convolve3(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                            size3_t shape, uint batches, const T* filter, uint3_t filter_shape, Stream& stream);

    /// ND convolution.
    /// \tparam T               float, double.
    /// \param[in] inputs       Input arrays to convolve. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \a inputs.
    /// \param[out] outputs     Output convolved arrays. One per batch.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs and \a outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param filter           ND filter corresponding to \a shape.
    /// \param filter_shape     Physical {fast, medium, slow} shape of \a filter.
    ///                         The dimensionality of the convolution is determined by `getNDim(filter_shape)`.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note \a filter can be on the host or device memory. They will be copied to constant device memory anyway.
    template<typename T>
    NOA_IH void convolve(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                         size3_t shape, uint batches, const T* filter, uint3_t filter_shape, Stream& stream) {
        uint ndim = getNDim(filter_shape);
        switch (ndim) {
            case 1U:
                convolve1(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, filter, filter_shape.x, stream);
                break;
            case 2U:
                convolve2(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, filter,
                          {filter_shape.x, filter_shape.y}, stream);
                break;
            case 3U:
                convolve3(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, filter, filter_shape, stream);
                break;
            default:
                NOA_THROW("DEV: getNDim(filter_shape) returned {}", ndim);
        }
    }

    /// Separable convolutions. \a inputs is convolved with \a filter1, then \a filter2, then \a filter3.
    /// \tparam T               float, double.
    /// \param[in] inputs       Input arrays to convolve. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \a inputs.
    /// \param[out] outputs     Output convolved arrays. One per batch.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs and \a outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter1      Filter corresponding to the first dimension of \a shape. Can be equal to \a filter2|3.
    /// \param filter1_size     Size, in elements, of \a filter1. Should be an odd number from 3 to 129.
    /// \param[in] filter2      Filter corresponding to the second dimension of \a shape. Can be equal to \a filter1|3.
    /// \param filter2_size     Size, in elements, of \a filter2. Should be an odd number from 3 to 129.
    /// \param[in] filter3      Filter corresponding to the third dimension of \a shape. Can be equal to \a filter1|2.
    /// \param filter3_size     Size, in elements, of \a filter3. Should be an odd number from 3 to 129.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \param[in,out] tmp      If more than one convolution is performed (see note below), it should be an array
    ///                         of the same shape as \a inputs (accounting for \a batches). Otherwise, it is ignored
    ///                         and nullptr can be passed.
    /// \param tmp_pitch        Pitch, in elements, of \a tmp. It is ignored if \a tmp is not used.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note If a filter is nullptr, the convolution in the corresponding dimension is not applied and it goes
    ///       directly to the next filter, if any.
    /// \note \a filter1, \a filter2 and \a filter3 can be on the host or device memory. They will be copied to
    ///       constant device memory anyway.
    template<typename T>
    NOA_HOST void convolve(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                           size3_t shape, uint batches,
                           const T* filter1, uint filter1_size,
                           const T* filter2, uint filter2_size,
                           const T* filter3, uint filter3_size,
                           Stream& stream,
                           T* tmp, size_t tmp_pitch);

    /// Separable convolutions. \a inputs is convolved with \a filter1, then \a filter2, then \a filter3.
    /// \tparam T               float, double.
    /// \param[in] inputs       Input arrays to convolve. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \a inputs.
    /// \param[out] outputs     Output convolved arrays. One per batch.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs and \a outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter1      Filter corresponding to the first dimension of \a shape. Can be equal to \a filter2|3.
    /// \param filter1_size     Size, in elements, of \a filter1. Should be an odd number from 3 to 129.
    /// \param[in] filter2      Filter corresponding to the second dimension of \a shape. Can be equal to \a filter1|3.
    /// \param filter2_size     Size, in elements, of \a filter2. Should be an odd number from 3 to 129.
    /// \param[in] filter3      Filter corresponding to the third dimension of \a shape. Can be equal to \a filter1|2.
    /// \param filter3_size     Size, in elements, of \a filter3. Should be an odd number from 3 to 129.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    ///
    /// \note If a filter is nullptr, the convolution in the corresponding dimension is not applied and it goes
    ///       directly to the next filter, if any. If more than one convolution is performed, a temporary array
    ///       of the same shape as \a inputs (accounting for \a batches) is allocated on the device.
    /// \note \a filter1, \a filter2 and \a filter3 can be on the host or device memory. They will be copied to
    ///       constant device memory anyway.
    template<typename T>
    NOA_HOST void convolve(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                           size3_t shape, uint batches,
                           const T* filter1, uint filter1_size,
                           const T* filter2, uint filter2_size,
                           const T* filter3, uint filter3_size,
                           Stream& stream);
}
