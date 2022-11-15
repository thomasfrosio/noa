#pragma once

#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename Value>
    constexpr bool is_valid_polar_xform_v =
            traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> && REMAP == HC2FC;
}

namespace noa::cpu::geometry::fft {
    using Remap = noa::fft::Remap;

    // Transforms 2D FFT(s) to (log-)polar coordinates.
    template<Remap REMAP, typename Value,
             typename = std::enable_if_t<details::is_valid_polar_xform_v<REMAP, Value>>>
    void cartesian2polar(const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream);
}
