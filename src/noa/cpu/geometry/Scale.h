#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/geometry/Transform.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Transform.h"

namespace noa::cpu::geometry::details {
    template<int NDIM, typename T, typename S, typename C>
    constexpr bool is_valid_scale_v =
            traits::is_any_v<T, float, cfloat_t, double, cdouble_t> &&
            ((NDIM == 2 && traits::is_any_v<S, float2_t, shared_t<float2_t[]>> && traits::is_any_v<C, float2_t, shared_t<float2_t[]>>) ||
             (NDIM == 3 && traits::is_any_v<S, float3_t, shared_t<float3_t[]>> && traits::is_any_v<C, float3_t, shared_t<float3_t[]>>));
}

namespace noa::cpu::geometry {
    // Applies one or multiple 2D stretching/shrinking.
    template<typename T, typename S, typename C, typename = std::enable_if_t<details::is_valid_scale_v<2, T, S, C>>>
    void scale2D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                 const S& scaling_factors, const C& scaling_centers,
                 InterpMode interp_mode, BorderMode border_mode, T value, bool prefilter, Stream& stream) {

        constexpr bool SINGLE_SCALING = traits::is_float2_v<S>;
        constexpr bool SINGLE_CENTER = traits::is_float2_v<C>;

        auto getInvertTransform_ = [=](size_t index) {
            float2_t scaling_factor;
            if constexpr (SINGLE_SCALING)
                scaling_factor = scaling_factors;
            else
                scaling_factor = scaling_factors.get()[index];

            float2_t scaling_center;
            if constexpr (SINGLE_CENTER)
                scaling_center = scaling_centers;
            else
                scaling_center = scaling_centers.get()[index];

            return float23_t(noa::geometry::translate(scaling_center) *
                             float33_t(noa::geometry::scale(1.f / scaling_factor)) *
                             noa::geometry::translate(-scaling_center));
        };

        if ((SINGLE_SCALING && SINGLE_CENTER) || output_shape[0] == 1) {
            transform2D(input, input_strides, input_shape, output, output_strides, output_shape,
                        getInvertTransform_(0), interp_mode, border_mode, value, prefilter, stream);
        } else {
            memory::PtrHost<float23_t> inv_transforms(output_shape[0]);
            for (size_t i = 0; i < output_shape[0]; ++i)
                inv_transforms[i] = getInvertTransform_(i);
            transform2D(input, input_strides, input_shape, output, output_strides, output_shape,
                        inv_transforms.share(), interp_mode, border_mode, value, prefilter, stream);
        }
    }

    // Applies one or multiple 3D stretching/shrinking.
    template<typename T, typename S, typename C, typename = std::enable_if_t<details::is_valid_scale_v<3, T, S, C>>>
    void scale3D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                 const S& scaling_factors, const C& scaling_centers,
                 InterpMode interp_mode, BorderMode border_mode, T value, bool prefilter, Stream& stream) {

        constexpr bool SINGLE_SCALING = traits::is_float3_v<S>;
        constexpr bool SINGLE_CENTER = traits::is_float3_v<C>;

        auto getInvertTransform_ = [=](size_t index) -> float34_t {
            float3_t scaling_factor;
            if constexpr (SINGLE_SCALING)
                scaling_factor = scaling_factors;
            else
                scaling_factor = scaling_factors.get()[index];

            float3_t scaling_center;
            if constexpr (SINGLE_CENTER)
                scaling_center = scaling_centers;
            else
                scaling_center = scaling_centers.get()[index];

            return float34_t(noa::geometry::translate(scaling_center) *
                             float44_t(noa::geometry::scale(1.f / scaling_factor)) *
                             noa::geometry::translate(-scaling_center));
        };

        if ((SINGLE_SCALING && SINGLE_CENTER) || output_shape[0] == 1) {
            transform3D(input, input_strides, input_shape, output, output_strides, output_shape,
                        getInvertTransform_(0), interp_mode, border_mode, value, prefilter, stream);
        } else {
            memory::PtrHost<float34_t> inv_transforms(output_shape[0]);
            for (size_t i = 0; i < output_shape[0]; ++i)
                inv_transforms[i] = getInvertTransform_(i);
            transform3D(input, input_strides, input_shape, output, output_strides, output_shape,
                        inv_transforms.share(), interp_mode, border_mode, value, prefilter, stream);
        }
    }
}
