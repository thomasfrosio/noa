/// \file noa/cpu/transform/Scale.h
/// \brief Scales arrays.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/transform/Geometry.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/transform/Apply.h"

namespace noa::transform {
    /// Applies one or multiple 2D scaling/stretching.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the output of bspline::prefilter2D(),
    ///                             which is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \p host. Input array.
    /// \param[out] outputs         On the \p host. Output arrays. One per transformation. Shouldn't be equal to \p input.
    /// \param shape                Logical {fast, medium} shape of \p input and \p outputs.
    /// \param[in] scaling_factors  On the \p host. One per dimension. One per transformation.
    /// \param[in] scaling_centers  On the \p host. Scaling centers in \p input. One per transformation.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. All interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \note If the input and output window are meant to have different shapes and/or centers, use
    ///       transform::apply2D() instead.
    template<bool PREFILTER = true, typename T>
    NOA_IH void scale2D(const T* input, T* outputs, size2_t shape,
                        const float2_t* scaling_factors, const float2_t* scaling_centers, uint nb_transforms,
                        InterpMode interp_mode, BorderMode border_mode, T value) {
        if (nb_transforms == 1U) { // allocate only if necessary
            float23_t inv_transform(transform::translate(scaling_centers[0]) *
                                    float33_t(transform::scale(1.f / scaling_factors[0])) *
                                    transform::translate(-scaling_centers[0]));
            apply2D<PREFILTER>(input, shape, outputs, shape,
                               inv_transform, interp_mode, border_mode, value);
        } else {
            memory::PtrHost<float23_t> inv_transforms(nb_transforms);
            for (uint i = 0; i < nb_transforms; ++i)
                inv_transforms[i] = float23_t(transform::translate(scaling_centers[i]) *
                                              float33_t(transform::scale(1.f / scaling_factors[i])) *
                                              transform::translate(-scaling_centers[i]));
            apply2D<PREFILTER>(input, shape, outputs, shape,
                               inv_transforms.get(), 1U, interp_mode, border_mode, value);
        }
    }

    /// Applies a single 2D scaling/stretching.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void scale2D(const T* input, T* output, size2_t shape,
                          float2_t scaling_factor, float2_t scaling_center,
                          InterpMode interp_mode, BorderMode border_mode, T value) {
        float23_t inv_transform(transform::translate(scaling_center) *
                                float33_t(transform::scale(1.f / scaling_factor)) *
                                transform::translate(-scaling_center));
        apply2D<PREFILTER>(input, shape, output, shape, inv_transform, interp_mode, border_mode, value);
    }

    /// Applies one or multiple 3D scaling/stretching.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the output of bspline::prefilter3D(),
    ///                             which is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \p host. Input array.
    /// \param[out] outputs         On the \p host. Output arrays. One per transformation. Shouldn't be equal to \p input.
    /// \param shape                Logical {fast, medium, slow} shape of \p input and \p outputs.
    /// \param[in] scaling_factors  On the \p host. One per dimension. One per transformation.
    /// \param[in] scaling_centers  On the \p host. Scaling centers in \p input. One per transformation.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. All interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \note If the input and output window are meant to have different shapes and/or centers, use
    ///       transform::apply3D() instead.
    template<bool PREFILTER = true, typename T>
    NOA_IH void scale3D(const T* input, T* outputs, size3_t shape,
                        const float3_t* scaling_factors, const float3_t* scaling_centers, uint nb_transforms,
                        InterpMode interp_mode, BorderMode border_mode, T value) {
        if (nb_transforms == 1U) { // allocate only if necessary
            float34_t inv_transform(transform::translate(scaling_centers[0]) *
                                    float44_t(transform::scale(1.f / scaling_factors[0])) *
                                    transform::translate(-scaling_centers[0]));
            apply3D<PREFILTER>(input, shape, outputs, shape,
                               inv_transform, interp_mode, border_mode, value);
        } else {
            memory::PtrHost<float34_t> inv_transforms(nb_transforms);
            for (uint i = 0; i < nb_transforms; ++i)
                inv_transforms[i] = float34_t(transform::translate(scaling_centers[i]) *
                                              float44_t(transform::scale(1.f / scaling_factors[i])) *
                                              transform::translate(-scaling_centers[i]));
            apply3D<PREFILTER>(input, shape, outputs, shape,
                               inv_transforms.get(), 1U, interp_mode, border_mode, value);
        }
    }

    /// Applies a single 3D scaling/stretching.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void scale3D(const T* input, T* output, size3_t shape,
                          float3_t scaling_factor, float3_t scaling_center,
                          InterpMode interp_mode, BorderMode border_mode, T value) {
        float34_t inv_transform(transform::translate(scaling_center) *
                                float44_t(transform::scale(1.f / scaling_factor)) *
                                transform::translate(-scaling_center));
        apply3D<PREFILTER>(input, shape, output, shape, inv_transform, interp_mode, border_mode, value);
    }
}
