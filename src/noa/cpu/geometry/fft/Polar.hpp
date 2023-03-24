#pragma once

#include "noa/core/Types.hpp"

namespace noa::cpu::geometry::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename Value>
    constexpr bool is_valid_polar_xform_v = noa::traits::is_any_v<Value, f32, f64, c32, c64> && REMAP == HC2FC;
}

namespace noa::cpu::geometry::fft {
    template<noa::fft::Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_polar_xform_v<REMAP, Value>>>
    void cartesian2polar(
            const Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
            Value* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, const Vec2<f32>& angle_range,
            bool log, InterpMode interp_mode, i64 threads);
}
