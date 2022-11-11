#pragma once

#include "noa/unified/Array.h"

namespace noa::geometry::bspline {
    /// Applies a prefilter to \a input so that the cubic B-spline values will pass through the sample data.
    /// \details Without prefiltering, cubic B-spline filtering results in smoothened images. This is caused by the
    ///          fact that the cubic B-spline filtering yields a function that does not pass through its coefficients.
    ///          In order to wind up with a cubic B-spline interpolated image that passes through the original samples,
    ///          we need to pre-filter the input.
    /// \tparam Value       float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input array.
    /// \param[out] output  Output array. Can be equal to \a input.
    ///
    /// \see http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm
    /// \see http://www.dannyruijters.nl/cubicinterpolation/ for more details.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value,float, double, cfloat_t, cdouble_t>>>
    void prefilter(const Array<Value>& input, const Array<Value>& output);
}

#define NOA_UNIFIED_PREFILTER_
#include "noa/unified/geometry/Prefilter.inl"
#undef NOA_UNIFIED_PREFILTER_
