/// \file noa/cpu/filter/Convolve.h
/// \brief Real space convolutions.
/// \author Thomas - ffyr2w
/// \date 22 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/cpu/memory/PtrHost.h"

namespace noa::filter {
    /// 1D convolution.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Input arrays to convolve. One per batch.
    /// \param[out] outputs     On the \b host. Output convolved arrays. One per batch.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter       On the \b host. Filter corresponding to the first dimension of \p shape.
    /// \param filter_size      Size, in elements, of \p filter.
    ///                         It should be an odd number from 1 to 129. If 1, a simple copy is computed.
    template<typename T>
    NOA_HOST void convolve1(const T* inputs, T* outputs, size3_t shape, uint batches,
                            const T* filter, uint filter_size);

    /// 2D convolution.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Input arrays to convolve. One per batch.
    /// \param[out] outputs     On the \b host. Output convolved arrays. One per batch.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter       On the \b host. 2D filter corresponding to the first two dimensions of \p shape.
    /// \param filter_shape     Physical {fast, medium} shape of \p filter.
    ///                         It should be two odd numbers from 1 to 17. If 1, a simple copy is computed.
    template<typename T>
    NOA_HOST void convolve2(const T* inputs, T* outputs, size3_t shape, uint batches,
                            const T* filter, uint2_t filter_shape);

    /// 3D convolution.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Input arrays to convolve. One per batch.
    /// \param[out] outputs     On the \b host. Output convolved arrays. One per batch.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter       On the \b host. 3D filter corresponding to \p shape.
    /// \param filter_shape     Physical {fast, medium, slow} shape of \p filter.
    ///                         It should be three odd numbers from 1 to 5. If 1, a simple copy is computed.
    template<typename T>
    NOA_HOST void convolve3(const T* inputs, T* outputs, size3_t shape, uint batches,
                            const T* filter, uint3_t filter_shape);

    /// ND convolution.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Input arrays to convolve. One per batch.
    /// \param[out] outputs     On the \b host. Output convolved arrays. One per batch.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in] filter       On the \b host. ND filter corresponding to \p shape.
    /// \param filter_shape     Physical {fast, medium, slow} shape of \p filter.
    ///                         The dimensionality of the convolution is determined by `getNDim(filter_shape)`.
    template<typename T>
    NOA_IH void convolve(const T* inputs, T* outputs, size3_t shape, uint batches,
                         const T* filter, uint3_t filter_shape) {
        uint ndim = getNDim(filter_shape);
        switch (ndim) {
            case 1U:
                convolve1(inputs, outputs, shape, batches, filter, filter_shape.x);
                break;
            case 2U:
                convolve2(inputs, outputs, shape, batches, filter, {filter_shape.x, filter_shape.y});
                break;
            case 3U:
                convolve3(inputs, outputs, shape, batches, filter, filter_shape);
                break;
            default:
                NOA_THROW("DEV: getNDim(filter_shape) returned {}", ndim);
        }
    }

    /// Separable convolutions. \p inputs is convolved with \p filter0, then \p filter1, then \p filter2.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Input arrays to convolve. One per batch.
    /// \param[out] outputs     On the \b host. Output convolved arrays. One per batch.
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
    /// \note If a filter is nullptr, the convolution in the corresponding dimension is not applied and it goes
    ///       directly to the next filter, if any.
    template<typename T>
    NOA_HOST void convolve(const T* inputs, T* outputs, size3_t shape, uint batches,
                           const T* filter0, uint filter0_size,
                           const T* filter1, uint filter1_size,
                           const T* filter2, uint filter2_size,
                           T* tmp);

    /// Separable convolutions. \p inputs is convolved with \p filter0, then \p filter1, then \p filter2.
    /// \tparam T               float, double.
    /// \param[in] inputs       Input arrays to convolve. One per batch.
    /// \param[out] outputs     Output convolved arrays. One per batch.
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
    ///
    /// \note If a filter is nullptr, the convolution in the corresponding dimension is not applied and it goes
    ///       directly to the next filter, if any. If more than one convolution is performed, a temporary array
    ///       of the same shape as \p inputs (ignoring \p batches) is allocated.
    template<typename T>
    NOA_IH void convolve(const T* inputs, T* outputs, size3_t shape, uint batches,
                         const T* filter0, uint filter0_size,
                         const T* filter1, uint filter1_size,
                         const T* filter2, uint filter2_size) {
        memory::PtrHost<T> tmp;
        int count = 0;
        if (filter0)
            count += 1;
        if (filter1)
            count += 1;
        if (filter2)
            count += 1;
        if (count > 1)
            tmp.reset(getElements(shape));
        convolve(inputs, outputs, shape, batches,
                 filter0, filter0_size, filter1, filter1_size, filter2, filter2_size, tmp.get());
    }
}
