#pragma once

#include "noa/unified/Array.h"

namespace noa::geometry::bspline {
    /// Applies a prefilter to \a input so that the cubic B-spline values will pass through the sample data.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input array.
    /// \param[out] output  Output array. Can be equal to \a input.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T,float, double, cfloat_t, cdouble_t>>>
    void prefilter(const Array<T>& input, const Array<T>& output);
}

#define NOA_UNIFIED_PREFILTER_
#include "noa/unified/geometry/Prefilter.inl"
#undef NOA_UNIFIED_PREFILTER_
