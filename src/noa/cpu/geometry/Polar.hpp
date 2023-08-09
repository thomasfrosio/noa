#pragma once

#include "noa/core/Types.hpp"

namespace noa::cpu::geometry {
    template<typename Value, typename = std::enable_if_t<nt::is_any_v<Value, f32, f64, c32, c64>>>
    void cartesian2polar(const Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
                         Value* polar, Strides4<i64> polar_strides, Shape4<i64> polar_shape,
                         Vec2<f32> cartesian_center,
                         const Vec2<f32>& radius_range, bool radius_range_endpoint,
                         const Vec2<f32>& angle_range, bool angle_range_endpoint,
                         InterpMode interp, i64 threads);

    template<typename Value, typename = std::enable_if_t<nt::is_any_v<Value, f32, f64, c32, c64>>>
    void polar2cartesian(const Value* polar, Strides4<i64> polar_strides, Shape4<i64> polar_shape,
                         Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
                         Vec2<f32> cartesian_center,
                         const Vec2<f32>& radius_range, bool radius_range_endpoint,
                         const Vec2<f32>& angle_range, bool angle_range_endpoint,
                         InterpMode interp, i64 threads);
}
