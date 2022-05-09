#pragma once

#include "noa/unified/Array.h"

namespace noa::filter {
    /// 1D convolution.
    /// \tparam T,U         half_t, float, double.
    /// \param[in] input    Array to convolve.
    /// \param[out] output  Convolved array. Should not overlap with \p input.
    /// \param[in] filter   1D, 2D or 3D filter.
    ///                     Dimensions should have an odd number of elements.
    ///                     Dimensions don't have to have the same size.
    ///                     The same ND filter is applied to every output batch.
    ///
    /// \note If the output is on the GPU:\n
    ///         - \p U should be equal to \p T.\n
    ///         - \p filter can be on any device, including the CPU.\n
    ///         - \p filter size on each dimension is limited to 129 (1D), 17 (2D) and 5 (3D) elements.
    ///         - This function modifies the GPU state via the usage of constant memory. As such,
    ///           there should be no concurrent calls from different streams associated to the same GPU.
    template<typename T, typename U,
             typename = std::enable_if_t<traits::is_float_v<T> && traits::is_almost_same_v<T, U>>>
    void convolve(const Array<T>& input, const Array<T>& output, const Array<U>& filter);

    /// Separable convolutions. \p input is convolved with \p filter0, then \p filter1, then \p filter2.
    /// \tparam T,U         half_t, float, double.
    /// \param[in] input    Input array to convolve.
    /// \param[out] output  Output convolved array. Should not overlap with \p input.
    /// \param[in] filter1  1D filter with an odd number of elements applied along the third-most dimension.
    /// \param[in] filter2  1D filter with an odd number of elements applied along the second-most dimension.
    /// \param[in] filter3  1D filter with an odd number of elements applied along the innermost dimension.
    /// \param[in,out] tmp  Temporary array. If only one dimension is filtered, this is ignored. Otherwise,
    ///                     \p tmp should 1) be an array of the same shape as \p output, or 2) be an empty array,
    ///                     in which case a temporary array will be allocated internally.
    ///
    /// \note Filters can be empty. In these cases, the convolution in the corresponding dimension is not applied
    ///       and it goes directly to the next filter, if any. Filters can be equal to each other.
    /// \note If the output is on the GPU:\n
    ///         - \p U should be equal to \p T.\n
    ///         - The filters can be on any device, including the CPU.\n
    ///         - Filters should be 1D contiguous row vectors. The filter size is limited to 1032 bytes.
    ///         - This function modifies the GPU state via the usage of constant memory. As such,
    ///           there should be no concurrent calls from different streams associated to the same GPU.
    template<typename T, typename U,
             typename = std::enable_if_t<traits::is_float_v<T> && traits::is_almost_same_v<T, U>>>
    void convolve(const Array<T>& input, const Array<T>& output,
                  const Array<U>& filter1, const Array<U>& filter2, const Array<U>& filter3,
                  const Array<T>& tmp = Array<T>{});
}

#define NOA_UNIFIED_CONVOLVE_
#include "noa/unified/signal/Convolve.inl"
#undef NOA_UNIFIED_CONVOLVE_
