#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry::details {
    template<int NDIM, typename T, typename M>
    constexpr bool is_valid_shift_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            ((NDIM == 2 && traits::is_any_v<M, float2_t, shared_t<float2_t[]>>) ||
             (NDIM == 3 && traits::is_any_v<M, float3_t, shared_t<float3_t[]>>));
}

namespace noa::cpu::geometry {
    // Applies one or multiple 2D shifts.
    template<typename T, typename S, typename = std::enable_if_t<details::is_valid_shift_v<2, T, S>>>
    void shift2D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                 const S& shifts, InterpMode interp_mode, BorderMode border_mode,
                 T value, bool prefilter, Stream& stream);

    // Applies one or multiple 3D shifts.
    template<typename T, typename S, typename = std::enable_if_t<details::is_valid_shift_v<3, T, S>>>
    void shift3D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                 const S& shifts, InterpMode interp_mode, BorderMode border_mode,
                 T value, bool prefilter, Stream& stream);
}
