/// \file noa/gpu/cuda/filter/Median.h
/// \brief Median filters.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::filter {
    /// Computes the median filter using a 1D window.
    /// \tparam T               int, uint, float or double
    /// \param[in] inputs       On the \b device. Input array with data to filter. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \a inputs.
    /// \param[out] outputs     On the \b device. Output array where the filtered data is stored. One per batch.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape, in elements, of \a inputs and \a outputs.
    /// \param batches          Number of batches to compute.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window           Number of elements to consider for the computation of the median.
    ///                         This corresponds to the first dimension of \a shape.
    ///                         Only odd numbers from 1 to 21 are supported. If 1, a copy is performed.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function runs asynchronously relative to the host and may return before completion.
    /// \note If \a border_mode is BORDER_REFLECT, the first dimension should be larger or equal than window/2 + 1.
    /// \note \a inputs and \a outputs should not overlap.
    template<typename T>
    NOA_HOST void median1(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch, size3_t shape,
                          uint batches, BorderMode border_mode, uint window, Stream& stream);

    /// Computes the median filter using a 1D window. Version for contiguous layouts.
    template<typename T>
    NOA_IH void median1(const T* inputs, T* outputs, size3_t shape, uint batches,
                        BorderMode border_mode, uint window, Stream& stream) {
        median1(inputs, shape.x, outputs, shape.x, shape, batches, border_mode, window, stream);
    }

    /// Computes the median filter using a 2D square window.
    /// \tparam T               int, uint, float or double
    /// \param[in] inputs       On the \b device. Input array with data to filter. One per batch.
    /// \param in_pitch         Pitch, in elements, of \a inputs.
    /// \param[out] outputs     On the \b device. Output array where the filtered data is stored. One per batch.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param shape            Shape {fast, medium, slow} of \a inputs and \a outputs (excluding the batch), in elements.
    /// \param batches          Number of batches.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window           Number of elements to consider for the computation of the median, for each dimension.
    ///                         This corresponds to the first and second dimension of \a shape.
    ///                         Only odd numbers from 1 to 11 are supported. If 1, no filter is applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function runs asynchronously relative to the host and may return before completion.
    /// \note If \a border_mode is BORDER_REFLECT, the first two dimensions should be larger or equal than window/2 + 1.
    /// \note \a inputs and \a outputs should not overlap.
    template<typename T>
    NOA_HOST void median2(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch, size3_t shape,
                          uint batches, BorderMode border_mode, uint window, Stream& stream);

    /// Computes the median filter using a 2D square window. Version for contiguous layouts.
    template<typename T>
    NOA_IH void median2(const T* inputs, T* outputs, size3_t shape, uint batches,
                        BorderMode border_mode, uint window, Stream& stream) {
        median2(inputs, shape.x, outputs, shape.x, shape, batches, border_mode, window, stream);
    }

    /// Computes the median filter using a 3D cubic window.
    /// \tparam T               int, uint, float or double
    /// \param[in] inputs       On the \b device. Input array with data to filter. One per batch.
    /// \param in_pitch         Pitch, in elements, of \a inputs.
    /// \param[out] outputs     On the \b device. Output array where the filtered data is stored. One per batch.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param shape            Shape {fast, medium, slow} of \a inputs and \a outputs (excluding the batch), in elements.
    /// \param batches          Number of batches.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window           Number of elements to consider for the computation of the median, for each dimension.
    ///                         Only odd numbers from 1 to 5 are supported. If 1, no filter is applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function runs asynchronously relative to the host and may return before completion.
    /// \note If \a border_mode is BORDER_REFLECT, each dimension should be larger or equal than window/2 + 1.
    /// \note \a inputs and \a outputs should not overlap.
    /// \throw If \a border_mode or \a window is not supported.
    template<typename T>
    NOA_HOST void median3(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch, size3_t shape,
                          uint batches, BorderMode border_mode, uint window, Stream& stream);

    /// Computes the median filter using a 3D cubic window. Version for contiguous layouts.
    template<typename T>
    NOA_IH void median3(const T* inputs, T* outputs, size3_t shape, uint batches,
                        BorderMode border_mode, uint window, Stream& stream) {
        median3(inputs, shape.x, outputs, shape.x, shape, batches, border_mode, window, stream);
    }
}
