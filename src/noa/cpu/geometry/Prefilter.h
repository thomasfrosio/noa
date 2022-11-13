#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry::bspline {
    // Applies a prefilter to "input" so that the cubic B-spline values will pass through the sample data.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, double, cfloat_t, cdouble_t>>>
    void prefilter(const Value* input, dim4_t input_strides,
                   Value* output, dim4_t output_strides,
                   dim4_t shape, dim_t threads);

    // Applies a prefilter to "input" so that the cubic B-spline values will pass through the sample data.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, double, cfloat_t, cdouble_t>>>
    void prefilter(const shared_t<Value[]>& input, dim4_t input_strides,
                   const shared_t<Value[]>& output, dim4_t output_strides,
                   dim4_t shape, Stream& stream);
}
