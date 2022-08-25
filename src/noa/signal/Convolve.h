#pragma once

#include "noa/Array.h"

namespace noa::signal {
    /// 1D convolution.
    /// \tparam T           half_t, float, double.
    /// \param[in] input    Array to convolve.
    /// \param[out] output  Convolved array. Should not overlap with \p input.
    /// \param[in] filter   1D, 2D or 3D C-contiguous filter.
    ///                     Dimensions should have an odd number of elements.
    ///                     Dimensions don't have to have the same size.
    ///                     The same ND filter is applied to every output batch.
    ///
    /// \note If the output is on the GPU:\n
    ///         - \p filter can be on any device, including the CPU.\n
    ///         - \p filter size on each dimension is limited to 129 (1D), 17 (2D) and 5 (3D) elements.
    ///         - This function modifies the GPU state via the usage of constant memory. As such,
    ///           there should be no concurrent calls from different streams associated to the same GPU.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void convolve(const Array<T>& input, const Array<T>& output, const Array<T>& filter);

    /// Separable convolutions. \p input is convolved with \p filter1, then \p filter2, then \p filter3.
    /// \tparam T           half_t, float, double.
    /// \param[in] input    Input array to convolve.
    /// \param[out] output  Output convolved array. Should not overlap with \p input.
    /// \param[in] filter1  1D filter with an odd number of elements applied along the depth dimension.
    /// \param[in] filter2  1D filter with an odd number of elements applied along the height dimension.
    /// \param[in] filter3  1D filter with an odd number of elements applied along the width dimension.
    /// \param[in,out] tmp  Temporary array. If only one dimension is filtered, this is ignored. Otherwise,
    ///                     \p tmp should 1) be an array of the same shape as \p output, or 2) be an empty array,
    ///                     in which case a temporary array will be allocated internally.
    ///
    /// \note Filters can be empty. In these cases, the convolution in the corresponding dimension is not applied
    ///       and it goes directly to the next filter, if any. Filters can be equal to each other.
    /// \note If the output is on the GPU:\n
    ///         - The filters can be on any device, including the CPU.\n
    ///         - Filters should be 1D contiguous vectors with a maximum size of 1032 bytes.
    ///         - This function modifies the GPU state via the usage of constant memory. As such,
    ///           there should be no concurrent calls from different streams associated to the same GPU.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void convolve(const Array<T>& input, const Array<T>& output,
                  const Array<T>& filter1, const Array<T>& filter2, const Array<T>& filter3,
                  const Array<T>& tmp = Array<T>{});
}

#define NOA_UNIFIED_CONVOLVE_
#include "noa/signal/details/Convolve.inl"
#undef NOA_UNIFIED_CONVOLVE_
