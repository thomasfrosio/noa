#pragma once

#include "noa/core/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::geometry::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename Value>
    constexpr bool is_valid_polar_xform_v = noa::traits::is_any_v<Value, f32, c32, f64, c64> && REMAP == HC2FC;
}

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename Value, typename = std::enable_if_t<details::is_valid_polar_xform_v<REMAP, Value>>>
    void cartesian2polar(
            const Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
            Value* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, const Vec2<f32>& angle_range,
            bool log, InterpMode interp, Stream& stream);

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, c32>>>
    void cartesian2polar(
            cudaArray* array, cudaTextureObject_t cartesian,
            InterpMode cartesian_interp, const Shape4<i64>& cartesian_shape,
            Value* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, const Vec2<f32>& angle_range,
            bool log, Stream& stream);
}
