/// \file noa/cpu/transform/Scale.h
/// \brief Scaling images and volumes using affine transforms.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/transform/Geometry.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/transform/Apply.h"

namespace noa::cpu::transform {
    /// Applies one or multiple 2D stretching/shrinking.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the prefiltered output which
    ///                             is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \tparam VECTOR              float2_t or float3_t.
    /// \param[in] input            On the \b host. Input array.
    /// \param[out] outputs         On the \b host. Output arrays. One per transformation.
    /// \param shape                Logical {fast, medium} shape of \p input and \p outputs.
    /// \param[in] scaling_factors  On the \b host. One per dimension. One per transformation.
    ///                             If float2_t: forward scaling factors along the {fast, medium} axes.
    ///                             If float3_t: first two values are the forward scaling factors. The third value is
    ///                             the in-plane magnification angle, in radians, defining the scaling axes.
    /// \param[in] scaling_centers  On the \b host. Scaling centers. One per transformation.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \note In-place computation is not allowed, i.e. \p input and \p outputs should not overlap.
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T, typename VECTOR>
    NOA_HOST void scale2D(const T* input, T* outputs, size2_t shape,
                          const VECTOR* scaling_factors, const float2_t* scaling_centers, uint nb_transforms,
                          InterpMode interp_mode, BorderMode border_mode, T value = T(0)) {

        auto getInvertTransform_ = [scaling_factors, scaling_centers](uint index) -> float23_t {
            if constexpr(traits::is_float2_v<VECTOR>) {
                return float23_t(noa::transform::translate(scaling_centers[index]) *
                                 float33_t(noa::transform::scale(1.f / scaling_factors[index])) *
                                 noa::transform::translate(-scaling_centers[index]));
            } else if constexpr(traits::is_float3_v<VECTOR>) {
                const float3_t& scaling = scaling_factors[index];
                float33_t matrix(noa::transform::rotate(-scaling.z) *
                                 noa::transform::scale(float2_t(1.f / scaling.x, 1.f / scaling.y)) *
                                 noa::transform::rotate(scaling.z)); // 2x2 linear -> 3x3 affine
                return float23_t(noa::transform::translate(scaling_centers[index]) *
                                 matrix *
                                 noa::transform::translate(-scaling_centers[index]));
            } else {
                static_assert(traits::always_false_v<VECTOR>);
            }
        };

        if (nb_transforms == 1U) { // allocate only if necessary
            apply2D<PREFILTER>(input, shape, outputs, shape, getInvertTransform_(0), interp_mode, border_mode, value);
        } else {
            memory::PtrHost<float23_t> inv_transforms(nb_transforms);
            for (uint i = 0; i < nb_transforms; ++i)
                inv_transforms[i] = getInvertTransform_(i);
            apply2D<PREFILTER>(input, shape, outputs, shape, inv_transforms.get(), 1U, interp_mode, border_mode, value);
        }
    }

    /// Applies a single 2D stretching/shrinking.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T, typename VECTOR>
    NOA_IH void scale2D(const T* input, T* output, size2_t shape,
                        VECTOR scaling_factor, float2_t scaling_center,
                        InterpMode interp_mode, BorderMode border_mode, T value = T(0)) {
        scale2D<PREFILTER>(input, output, shape, &scaling_factor, &scaling_center, 1, interp_mode, border_mode, value);
    }

    /// Applies one or multiple 3D stretching/shrinking.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the prefiltered output which
    ///                             is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \b host. Input array.
    /// \param[out] outputs         On the \b host. Output arrays. One per transformation.
    /// \param shape                Logical {fast, medium, slow} shape of \p input and \p outputs.
    /// \param[in] scaling_factors  On the \b host. One per dimension. One per transformation.
    /// \param[in] scaling_centers  On the \b host. Scaling centers. One per transformation.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \note In-place computation is not allowed, i.e. \p input and \p outputs should not overlap.
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void scale3D(const T* input, T* outputs, size3_t shape,
                          const float3_t* scaling_factors, const float3_t* scaling_centers, uint nb_transforms,
                          InterpMode interp_mode, BorderMode border_mode, T value = T(0)) {

        auto getInvertTransform_ = [scaling_factors, scaling_centers](uint index) -> float34_t {
            return float34_t(noa::transform::translate(scaling_centers[index]) *
                             float44_t(noa::transform::scale(1.f / scaling_factors[index])) *
                             noa::transform::translate(-scaling_centers[index]));
        };

        if (nb_transforms == 1U) { // allocate only if necessary
            apply3D<PREFILTER>(input, shape, outputs, shape, getInvertTransform_(0), interp_mode, border_mode, value);
        } else {
            memory::PtrHost<float34_t> inv_transforms(nb_transforms);
            for (uint i = 0; i < nb_transforms; ++i)
                inv_transforms[i] = getInvertTransform_(i);
            apply3D<PREFILTER>(input, shape, outputs, shape, inv_transforms.get(), 1U, interp_mode, border_mode, value);
        }
    }

    /// Applies a single 3D stretching/shrinking.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_IH void scale3D(const T* input, T* output, size3_t shape,
                        float3_t scaling_factor, float3_t scaling_center,
                        InterpMode interp_mode, BorderMode border_mode, T value = T(0)) {
        scale3D<PREFILTER>(input, output, shape, &scaling_factor, &scaling_center, 1, interp_mode, border_mode, value);
    }
}
