#pragma once

#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry {
    // Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, double, cfloat_t, cdouble_t>>>
    void cartesian2polar(const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter, Stream& stream);

    // Transforms 2D array(s) from (log-)polar to cartesian coordinates.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, double, cfloat_t, cdouble_t>>>
    void polar2cartesian(const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter, Stream& stream);

    // Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, cfloat_t>>>
    void cartesian2polar(const shared_t<cudaArray>& array,
                         const shared_t<cudaTextureObject_t>& cartesian,
                         InterpMode cartesian_interp, dim4_t cartesian_shape,
                         const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, Stream& stream);

    // Transforms 2D array(s) from (log-)polar to cartesian coordinates.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, cfloat_t>>>
    void polar2cartesian(const shared_t<cudaArray>& array,
                         const shared_t<cudaTextureObject_t>& polar,
                         InterpMode polar_interp, dim4_t polar_shape,
                         const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, Stream& stream);
}
