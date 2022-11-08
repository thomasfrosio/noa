#pragma once

#include "noa/common/geometry/Transform.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/geometry/Transform.h"

namespace noa::cuda::geometry::details {
    template<int NDIM, typename data_t, typename shift_t>
    constexpr bool is_valid_shift_v =
            traits::is_any_v<data_t, float, cfloat_t> &&
            ((NDIM == 2 && traits::is_any_v<shift_t, float2_t, shared_t<float2_t[]>>) ||
             (NDIM == 3 && traits::is_any_v<shift_t, float3_t, shared_t<float3_t[]>>));
}

namespace noa::cuda::geometry {
    // Applies one or multiple 2D translations.
    template<typename data_t, typename shift_t,
             typename = std::enable_if_t<details::is_valid_shift_v<2, data_t, shift_t>>>
    void shift2D(const shared_t<data_t[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<data_t[]>& output, dim4_t output_strides, dim4_t output_shape,
                 const shift_t& shifts, InterpMode interp_mode, BorderMode border_mode, bool prefilter,
                 Stream& stream) {

        constexpr bool SINGLE_SHIFT = traits::is_float2_v<shift_t>;
        auto shift2affine = [=](dim_t index) {
            if constexpr (SINGLE_SHIFT)
                return float23_t(noa::geometry::translate(-shifts));
            else
                return float23_t(noa::geometry::translate(-shifts.get()[index]));
        };

        if (SINGLE_SHIFT || output_shape[0] == 1) {
            transform2D(input, input_strides, input_shape, output, output_strides, output_shape,
                        shift2affine(0), interp_mode, border_mode, prefilter, stream);
        } else {
            memory::PtrPinned<float23_t> inv_matrices(output_shape[0]);
            for (dim_t i = 0; i < output_shape[0]; ++i)
                inv_matrices[i] = shift2affine(i);
            transform2D(input, input_strides, input_shape, output, output_strides, output_shape,
                        inv_matrices.share(), interp_mode, border_mode, prefilter, stream);
        }
    }

    // Applies one or multiple 3D translations.
    template<typename data_t, typename shift_t,
             typename = std::enable_if_t<details::is_valid_shift_v<3, data_t, shift_t>>>
    void shift3D(const shared_t<data_t[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<data_t[]>& output, dim4_t output_strides, dim4_t output_shape,
                 const shift_t& shifts, InterpMode interp_mode, BorderMode border_mode, bool prefilter,
                 Stream& stream) {
        constexpr bool SINGLE_SHIFT = traits::is_float3_v<shift_t>;
        auto shift2affine = [=](dim_t index) {
            if constexpr (SINGLE_SHIFT)
                return float34_t(noa::geometry::translate(-shifts));
            else
                return float34_t(noa::geometry::translate(-shifts.get()[index]));
        };

        if (SINGLE_SHIFT || output_shape[0] == 1) {
            transform3D(input, input_strides, input_shape, output, output_strides, output_shape,
                        shift2affine(0), interp_mode, border_mode, prefilter, stream);
        } else {
            memory::PtrPinned<float34_t> inv_matrices(output_shape[0]);
            for (dim_t i = 0; i < output_shape[0]; ++i)
                inv_matrices[i] = shift2affine(i);
            transform3D(input, input_strides, input_shape, output, output_strides, output_shape,
                        inv_matrices.share(), interp_mode, border_mode, prefilter, stream);
        }
    }
}
