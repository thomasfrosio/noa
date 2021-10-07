/// \file noa/cpu/transform/Rotate.h
/// \brief Rotations of images and volumes using affine transforms.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/transform/Euler.h"
#include "noa/common/transform/Geometry.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/transform/Apply.h"

namespace noa::cpu::transform {
    /// Applies one or multiple 2D rotations.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the prefiltered output which
    ///                             is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \b host. Input array.
    /// \param[out] outputs         On the \b host. Output arrays. One per rotation.
    /// \param shape                Logical {fast, medium} shape of \p input and \p outputs.
    /// \param[in] rotations        On the \b host. Rotation angles, in radians. One per rotation.
    /// \param[in] rotation_centers On the \b host. Rotation centers. One per rotation.
    /// \param nb_rotations         Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \note In-place computation is not allowed, i.e. \p input and \p outputs should not overlap.
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T>
    NOA_IH void rotate2D(const T* input, T* outputs, size2_t shape,
                         const float* rotations, const float2_t* rotation_centers, uint nb_rotations,
                         InterpMode interp_mode, BorderMode border_mode, T value = T(0)) {

        auto getInvertTransform_ = [rotations, rotation_centers](uint index) {
            return float23_t(noa::transform::translate(rotation_centers[index]) *
                             float33_t(noa::transform::rotate(-rotations[index])) *
                             noa::transform::translate(-rotation_centers[index]));
        };

        if (nb_rotations == 1U) { // allocate only if necessary
            apply2D<PREFILTER>(input, shape, outputs, shape, getInvertTransform_(0), interp_mode, border_mode, value);
        } else {
            memory::PtrHost<float23_t> inv_transforms(nb_rotations);
            for (uint i = 0; i < nb_rotations; ++i)
                inv_transforms[i] = getInvertTransform_(i);
            apply2D<PREFILTER>(input, shape, outputs, shape, inv_transforms.get(), 1U, interp_mode, border_mode, value);
        }
    }

    /// Applies a single 2D rotation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_IH void rotate2D(const T* input, T* output, size2_t shape,
                         float rotation, float2_t rotation_center,
                         InterpMode interp_mode, BorderMode border_mode, T value = T(0)) {
        rotate2D<PREFILTER>(input, output, shape, &rotation, &rotation_center, 1, interp_mode, border_mode, value);
    }

    /// Applies one or multiple 3D rotations.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the prefiltered output which
    ///                             is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \b host. Input array.
    /// \param[out] outputs         On the \b host. Output arrays. One per rotation.
    /// \param output_shape         Logical {fast, medium, slow} shape of \p input and \p outputs.
    /// \param[in] rotations        On the \b host. 3x3 inverse rotation matrices. One per rotation.
    ///                             For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                             on the output array coordinates. This function assumes \p matrix is already
    ///                             inverted and pre-multiplies the coordinates with the matrix directly.
    /// \param[in] rotation_centers On the \b host. Rotation centers. One per rotation.
    /// \param nb_rotations         Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \note In-place computation is not allowed, i.e. \p input and \p outputs should not overlap.
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void rotate3D(const T* input, T* outputs, size3_t shape,
                           const float33_t* rotations, const float3_t* rotation_centers, uint nb_rotations,
                           InterpMode interp_mode, BorderMode border_mode, T value = T(0)) {

        auto getInvertTransform_ = [rotations, rotation_centers](uint index) {
            return float34_t(noa::transform::translate(rotation_centers[index]) *
                             float44_t(rotations[index]) *
                             noa::transform::translate(-rotation_centers[index]));
        };

        if (nb_rotations == 1U) { // allocate only if necessary
            apply3D<PREFILTER>(input, shape, outputs, shape, getInvertTransform_(0), interp_mode, border_mode, value);
        } else {
            memory::PtrHost<float34_t> inv_transforms(nb_rotations);
            for (uint i = 0; i < nb_rotations; ++i)
                inv_transforms[i] = getInvertTransform_(i);
            apply3D<PREFILTER>(input, shape, outputs, shape, inv_transforms.get(), 1U, interp_mode, border_mode, value);
        }
    }

    /// Applies a single 3D rotation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_IH void rotate3D(const T* input, T* output, size3_t shape,
                         float33_t rotation, float3_t rotation_center,
                         InterpMode interp_mode, BorderMode border_mode, T value = T(0)) {
        rotate3D<PREFILTER>(input, output, shape, &rotation, &rotation_center, 1, interp_mode, border_mode, value);
    }
}
