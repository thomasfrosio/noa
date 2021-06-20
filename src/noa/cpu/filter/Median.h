/// \file noa/cpu/filter/Median.h
/// \brief Median filters.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

namespace noa::filter {
    /// Computes the median filter using a 1D window.
    /// \tparam T               int, uint, float or double.
    /// \param[in] in           Input array with data to filter. One per batch.
    /// \param[out] out         Output array where the filtered data is stored. One per batch.
    /// \param shape            Shape {fast, medium, slow} of \a in and \a out (excluding the batch), in elements.
    /// \param batches          Number of batches.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_MIRROR.
    /// \param window           Number of elements to consider for the computation of the median.
    ///                         This corresponds to the first dimension of \a shape.
    ///                         Only odd numbers are supported. If 1, a copy is performed.
    /// \note If \a border_mode is BORDER_MIRROR, \a shape.x should be `>= window/2 + 1`.
    /// \note \a in and \a out should not overlap.
    /// \throw If \a border_mode is not supported.
    template<typename T>
    NOA_HOST void median1(const T* in, T* out, size3_t shape, uint batches, BorderMode border_mode, uint window);

    /// Computes the median filter using a 2D square window.
    /// \tparam T               int, uint, float or double.
    /// \param[in] in           Input array with data to filter. One per batch.
    /// \param[out] out         Output array where the filtered data is stored. One per batch.
    /// \param shape            Shape {fast, medium, slow} of \a in and \a out (excluding the batch), in elements.
    /// \param batches          Number of batches.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_MIRROR.
    /// \param window           Number of elements to consider for the computation of the median, for each dimension.
    ///                         This corresponds to the first and second dimension of \a shape.
    ///                         Only odd numbers are supported. If 1, no filter is applied.
    /// \note If \a border_mode is BORDER_MIRROR, \a shape.x and \a shape.y should be `>= window/2 + 1`.
    /// \note \a in and \a out should not overlap.
    /// \throw If \a border_mode is not supported.
    template<typename T>
    NOA_HOST void median2(const T* in, T* out, size3_t shape, uint batches, BorderMode border_mode, uint window);

    /// Computes the median filter using a 3D cubic window.
    /// \tparam T               int, uint, float or double.
    /// \param[in] in           Input array with data to filter. One per batch.
    /// \param[out] out         Output array where the filtered data is stored. One per batch.
    /// \param shape            Shape {fast, medium, slow} of \a in and \a out (excluding the batch), in elements.
    /// \param batches          Number of batches.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_MIRROR.
    /// \param window           Number of elements to consider for the computation of the median, for each dimension.
    ///                         Only odd numbers are supported. If 1, no filter is applied.
    /// \note If \a border_mode is BORDER_MIRROR, `all(shape >= window/2 + 1)`.
    /// \note \a in and \a out should not overlap.
    /// \throw If \a border_mode is not supported.
    template<typename T>
    NOA_HOST void median3(const T* in, T* out, size3_t shape, uint batches, BorderMode border_mode, uint window);
}
