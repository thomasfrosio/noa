#pragma once

#include "noa/Array.h"

namespace noa::geometry::bspline {
    /// Applies a prefilter to \a input so that the cubic B-spline values will pass through the sample data.
    /// \details Without prefiltering, cubic B-spline filtering results in smoothened images. This is caused by the
    ///          fact that the cubic B-spline filtering yields a function that does not pass through its coefficients.
    ///          In order to wind up with a cubic B-spline interpolated image that passes through the original samples,
    ///          we need to pre-filter the input.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input array.
    /// \param[out] output  Output array. Can be equal to \a input.
    ///
    /// \see http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm
    /// \see http://www.dannyruijters.nl/cubicinterpolation/ for more details.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T,float, double, cfloat_t, cdouble_t>>>
    void prefilter(const Array<T>& input, const Array<T>& output);
}

#define NOA_UNIFIED_PREFILTER_
#include "noa/geometry/details/Prefilter.inl"
#undef NOA_UNIFIED_PREFILTER_
