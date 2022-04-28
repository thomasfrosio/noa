#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry::bspline {
    /// Applies a prefilter to \a input so that the cubic B-spline values will pass through the sample data.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] input        On the \b host.Input array.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] output      On the \b host. Output array. Can be equal to \a input.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape, in elements, of \a input and \a output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm
    /// \see http://www.dannyruijters.nl/cubicinterpolation/
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void prefilter(const shared_t<T[]>& input, size4_t input_stride,
                   const shared_t<T[]>& output, size4_t output_stride,
                   size4_t shape, Stream& stream);
}
