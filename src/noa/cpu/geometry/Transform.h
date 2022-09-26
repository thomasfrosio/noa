#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/geometry/Symmetry.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry::details {
    template<int NDIM, typename T, typename M>
    constexpr bool is_valid_transform_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            ((NDIM == 2 && traits::is_any_v<M, float23_t, float33_t, shared_t<float23_t[]>, shared_t<float33_t[]>>) ||
             (NDIM == 3 && traits::is_any_v<M, float34_t, float44_t, shared_t<float34_t[]>, shared_t<float44_t[]>>));
}

namespace noa::cpu::geometry {
    // Applies one or multiple 2D affine transforms.
    template<typename T, typename MAT, typename = std::enable_if_t<details::is_valid_transform_v<2, T, MAT>>>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const MAT& matrices, InterpMode interp_mode, BorderMode border_mode,
                     T value, bool prefilter, Stream& stream);

    // Applies one or multiple 3D affine transforms.
    template<typename T, typename MAT, typename = std::enable_if_t<details::is_valid_transform_v<3, T, MAT>>>
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const MAT& matrices, InterpMode interp_mode, BorderMode border_mode,
                     T value, bool prefilter, Stream& stream);
}

// -- Apply symmetry -- //
namespace noa::cpu::geometry {
    using Symmetry = ::noa::geometry::Symmetry;

    // Shifts, then rotates/scales and applies the symmetry on the 2D input array.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream);

    // Shifts, then rotates/scales and applies the symmetry on the 3D input array.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream);
}
