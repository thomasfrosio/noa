#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry::bspline {
    // Applies a prefilter to "input" so that the cubic B-spline values will pass through the sample data.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T,float, double, cfloat_t, cdouble_t>>>
    void prefilter(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides,
                   size4_t shape, Stream& stream);
}
