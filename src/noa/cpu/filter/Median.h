/// \file noa/cpu/filter/Median.h
/// \brief Median filters.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::filter {
    /// Computes the median filter using a 1D window.
    /// \tparam T               int, uint, float or double.
    /// \param[in] in           On the \b host. Input array with data to filter. One per batch.
    /// \param[out] out         On the \b host. Output array where the filtered data is stored. One per batch.
    /// \param shape            Shape {fast, medium, slow} of \p in and \p out (excluding the batch), in elements.
    /// \param batches          Number of batches.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window           Number of elements to consider for the computation of the median.
    ///                         This corresponds to the first dimension of \p shape.
    ///                         Only odd numbers are supported. If 1, a copy is performed.
    /// \note If \p border_mode is BORDER_REFLECT, the first dimension should be larger or equal than window/2 + 1.
    /// \note \p in and \p out should not overlap.
    template<typename T>
    NOA_HOST void median1(const T* in, T* out, size3_t shape, uint batches, BorderMode border_mode, uint window);

    /// Computes the median filter using a 2D square window.
    /// \tparam T               int, uint, float or double.
    /// \param[in] in           On the \b host. Input array with data to filter. One per batch.
    /// \param[out] out         On the \b host. Output array where the filtered data is stored. One per batch.
    /// \param shape            Shape {fast, medium, slow} of \p in and \p out (excluding the batch), in elements.
    /// \param batches          Number of batches.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window           Number of elements to consider for the computation of the median, for each dimension.
    ///                         This corresponds to the first and second dimension of \p shape.
    ///                         Only odd numbers are supported. If 1, no filter is applied.
    /// \note If \p border_mode is BORDER_REFLECT, the first two dimensions should be larger or equal than window/2 + 1.
    /// \note \p in and \p out should not overlap.
    template<typename T>
    NOA_HOST void median2(const T* in, T* out, size3_t shape, uint batches, BorderMode border_mode, uint window);

    /// Computes the median filter using a 3D cubic window.
    /// \tparam T               int, uint, float or double.
    /// \param[in] in           On the \b host. Input array with data to filter. One per batch.
    /// \param[out] out         On the \b host. Output array where the filtered data is stored. One per batch.
    /// \param shape            Shape {fast, medium, slow} of \p in and \p out (excluding the batch), in elements.
    /// \param batches          Number of batches.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window           Number of elements to consider for the computation of the median, for each dimension.
    ///                         Only odd numbers are supported. If 1, no filter is applied.
    /// \note If \p border_mode is BORDER_REFLECT, each dimension should be larger or equal than window/2 + 1.
    /// \note \p in and \p out should not overlap.
    template<typename T>
    NOA_HOST void median3(const T* in, T* out, size3_t shape, uint batches, BorderMode border_mode, uint window);
}
