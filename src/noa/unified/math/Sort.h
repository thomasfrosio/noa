#pragma once

#include "noa/unified/Array.h"

namespace noa::math {
    /// Sorts an array, in-place.
    /// \tparam T               Any restricted scalar.
    ///                         On the GPU, only (u)int16_t, (u)int32_t, (u)int64_t, half_t, float and double are supported.
    /// \param[in,out] array    Array to sort.
    /// \param ascending        Whether to sort in ascending or descending order.
    /// \param axis             Axis along which to sort. The default is -1, which sorts along the first non-empty
    ///                         dimension in the rightmost order. Otherwise, it should be from 0 to 3, included.
    /// \note All the sort algorithms make temporary copies of the data when sorting along any but the last axis.
    ///       Consequently, sorting along the last axis is faster and uses less space than sorting along any other axis.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_scalar_v<T>>>
    void sort(const Array<T>& array, bool ascending = true, int axis = -1);

    // TODO Add sort by keys.
}

#define NOA_UNIFIED_SORT_
#include "noa/unified/math/Sort.inl"
#undef NOA_UNIFIED_SORT_
