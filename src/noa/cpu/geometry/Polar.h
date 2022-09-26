#pragma once

#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry {
    // Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void cartesian2polar(const shared_t<T[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         const shared_t<T[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter, Stream& stream);

    // Transforms 2D array(s) from (log-)polar to cartesian coordinates.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void polar2cartesian(const shared_t<T[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         const shared_t<T[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter, Stream& stream);
}
