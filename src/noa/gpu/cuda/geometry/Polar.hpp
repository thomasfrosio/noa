#pragma once

#include "noa/core/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::geometry {
    template<typename Value, typename = std::enable_if_t<noa::traits::is_any_v<Value, f32, f64, c32, c64>>>
    void cartesian2polar(const Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
                         Value* polar, Strides4<i64> polar_strides, Shape4<i64> polar_shape,
                         Vec2<f32> cartesian_center, const Vec2<f32>& radius_range, const Vec2<f32>& angle_range,
                         bool log, InterpMode interp_mode, Stream& stream);

    template<typename Value, typename = std::enable_if_t<noa::traits::is_any_v<Value, f32, f64, c32, c64>>>
    void polar2cartesian(const Value* polar, Strides4<i64> polar_strides, Shape4<i64> polar_shape,
                         Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
                         Vec2<f32> cartesian_center, const Vec2<f32>& radius_range, const Vec2<f32>& angle_range,
                         bool log, InterpMode interp_mode, Stream& stream);

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, c32>>>
    void cartesian2polar(cudaArray* array,
                         cudaTextureObject_t cartesian,
                         InterpMode cartesian_interp, const Shape4<i64>& cartesian_shape,
                         Value* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
                         const Vec2<f32>& cartesian_center, const Vec2<f32>& radius_range, const Vec2<f32>& angle_range,
                         bool log, Stream& stream);

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, c32>>>
    void polar2cartesian(cudaArray* array,
                         cudaTextureObject_t polar,
                         InterpMode polar_interp, const Shape4<i64>& polar_shape,
                         Value* cartesian, const Strides4<i64>& cartesian_strides, const Shape4<i64>& cartesian_shape,
                         const Vec2<f32>& cartesian_center, const Vec2<f32>& radius_range, const Vec2<f32>& angle_range,
                         bool log, Stream& stream);
}
