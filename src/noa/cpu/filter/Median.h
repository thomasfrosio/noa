/// \file noa/cpu/filter/Median.h
/// \brief Median filters.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu::filter {
    /// Computes the median filter using a 1D window.
    /// \tparam T               int, uint, float or double.
    /// \param[in] inputs       On the \b host. Array to filter. One per batch.
    /// \param[out] outputs     On the \b host. Filtered array. One per batch.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, in \p T elements.
    /// \param batches          Number of contiguous batches to filter.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO or BORDER_REFLECT.
    /// \param window_size      Number of elements to consider for the computation of the median.
    ///                         This corresponds to the first dimension of \p shape.
    ///                         Only odd numbers are supported. If 1, a copy is performed.
    /// \note With \c BORDER_REFLECT, the first dimension should be larger or equal than `window_size/2 + 1`.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T>
    NOA_HOST void median1(const T* inputs, T* outputs, size3_t shape, size_t batches,
                          BorderMode border_mode, size_t window_size);

    /// Computes the median filter using a 2D square window.
    /// \tparam T               int, uint, float or double.
    /// \param[in] inputs       On the \b host. Array to filter. One per batch.
    /// \param[out] outputs     On the \b host. Filtered array. One per batch.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, in \p T elements.
    /// \param batches          Number of contiguous batches to filter.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window_size      Number of elements to consider for the computation of the median, for each dimension.
    ///                         This corresponds to the first and second dimension of \p shape.
    ///                         Only odd numbers are supported. If 1, no filter is applied.
    /// \note With \c BORDER_REFLECT, the first two dimensions should be larger or equal than `window_size/2 + 1`.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T>
    NOA_HOST void median2(const T* inputs, T* outputs, size3_t shape, size_t batches,
                          BorderMode border_mode, size_t window_size);

    /// Computes the median filter using a 3D cubic window.
    /// \tparam T               int, uint, float or double.
    /// \param[in] inputs       On the \b host. Array to filter. One per batch.
    /// \param[out] outputs     On the \b host. Filtered array. One per batch.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, in \p T elements.
    /// \param batches          Number of contiguous batches to filter.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_REFLECT.
    /// \param window_size      Number of elements to consider for the computation of the median, for each dimension.
    ///                         Only odd numbers are supported. If 1, no filter is applied.
    /// \note With \c BORDER_REFLECT, each dimension should be larger or equal than `window_size/2 + 1`.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T>
    NOA_HOST void median3(const T* inputs, T* outputs, size3_t shape, size_t batches,
                          BorderMode border_mode, size_t window_size);
}
