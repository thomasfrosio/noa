/// \file noa/gpu/cuda/fourier/Resize.h
/// \brief Fourier crop/pad functions.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::fourier {
    /// CUDA version of noa::fourier::crop. The same features and restrictions apply to this function.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       Input array. Should be not-centered, not-redundant and contiguous.
    /// \param inputs_pitch     Pitch of \a in, in number of elements.
    /// \param inputs_shape     Logical {fast, medium, slow} shape of \a inputs, in complex elements.
    /// \param[out] outputs     Output array. Will be not-centered, not-redundant and contiguous.
    /// \param outputs_pitch    Pitch of \a out, in number of elements.
    /// \param outputs_shape    Logical {fast, medium, slow} shape of \a outputs, in complex elements.
    ///                         All dimensions should be less or equal than the dimensions of \a inputs_shape.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \see noa::fourier::crop for more details.
    /// \note \a inputs and \a outputs should not overlap.
    /// \note This function runs asynchronously with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void crop(const T* inputs, size_t inputs_pitch, size3_t inputs_shape,
                       T* outputs, size_t outputs_pitch, size3_t outputs_shape,
                       uint batches, Stream& stream);

    /// CUDA version of noa::fourier::crop. See overload above.
    template<typename T>
    NOA_IH void crop(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape,
                     uint batches, Stream& stream) {
        crop(inputs, inputs_shape.x / 2 + 1, inputs_shape,
             outputs, outputs_shape.x / 2 + 1, outputs_shape, batches, stream);
    }

    /// CUDA version of noa::fourier::cropFull. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void cropFull(const T* inputs, size_t inputs_pitch, size3_t inputs_shape,
                           T* outputs, size_t outputs_pitch, size3_t outputs_shape,
                           uint batches, Stream& stream);

    /// CUDA version of noa::fourier::cropFull. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void cropFull(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape,
                         uint batches, Stream& stream) {
        cropFull(inputs, inputs_shape.x, inputs_shape, outputs, outputs_shape.x, outputs_shape, batches, stream);
    }

    /// CUDA version of noa::fourier::pad. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void pad(const T* inputs, size_t inputs_pitch, size3_t inputs_shape,
                      T* outputs, size_t outputs_pitch, size3_t outputs_shape,
                      uint batches, Stream& stream);

    /// CUDA version of noa::fourier::pad. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void pad(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape,
                    uint batches, Stream& stream) {
        pad(inputs, inputs_shape.x / 2 + 1, inputs_shape,
            outputs, outputs_shape.x / 2 + 1, outputs_shape, batches, stream);
    }

    /// CUDA version of noa::fourier::padFull. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void padFull(const T* inputs, size_t inputs_pitch, size3_t inputs_shape,
                          T* outputs, size_t outputs_pitch, size3_t outputs_shape,
                          uint batches, Stream& stream);

    /// CUDA version of noa::fourier::padFull. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void padFull(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape,
                        uint batches, Stream& stream) {
        padFull(inputs, inputs_shape.x, inputs_shape, outputs, outputs_shape.x, outputs_shape, batches, stream);
    }
}
