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
    ///                             shape as \p inputs is allocated and used to store the prefiltered output which
    ///                             is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs           On the \b host. Input arrays. One per rotation.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param[out] outputs         On the \b host. Output arrays. One per rotation.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param shape                Logical {fast, medium} shape of \p inputs and \p outputs.
    /// \param[in] rotations        On the \b host. Rotation angles, in radians. One per rotation.
    /// \param[in] rotation_centers On the \b host. Rotation centers. One per rotation.
    /// \param batches              Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p inputs and \p outputs should not overlap.
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T>
    NOA_IH void rotate2D(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch, size2_t shape,
                         const float* rotations, const float2_t* rotation_centers, size_t batches,
                         InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream) {

        auto getInvertTransform_ = [=](size_t index) {
            return float23_t(noa::transform::translate(rotation_centers[index]) *
                             float33_t(noa::transform::rotate(-rotations[index])) *
                             noa::transform::translate(-rotation_centers[index]));
        };

        if (batches == 1) {
            apply2D<PREFILTER>(inputs, input_pitch.x, shape, outputs, output_pitch.x, shape,
                               getInvertTransform_(0), interp_mode, border_mode, value, stream);
        } else {
            // Enqueue here to leave the allocation to the working thread so that calling thread doesn't have to wait.
            // Note that the stream can be captured and passed to itself.
            stream.enqueue([=, &stream]() {
                memory::PtrHost<float23_t> inv_transforms(batches);
                for (size_t i = 0; i < batches; ++i)
                    inv_transforms[i] = getInvertTransform_(i);
                apply2D<PREFILTER>(inputs, input_pitch, shape, outputs, output_pitch, shape,
                                   inv_transforms.get(), batches, interp_mode, border_mode, value, stream);
            });
        }
    }

    /// Applies a single 2D rotation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_IH void rotate2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                         float rotation, float2_t rotation_center,
                         InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream) {
        rotate2D<PREFILTER>(input, {input_pitch, 0}, output, {output_pitch, 0},
                            shape, &rotation, &rotation_center, 1, interp_mode, border_mode, value, stream);
    }

    /// Applies one or multiple 3D rotations.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p inputs is allocated and used to store the prefiltered output which
    ///                             is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs           On the \b host. Input arrays. One per rotation.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param[out] outputs         On the \b host. Output arrays. One per rotation.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param output_shape         Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param[in] rotations        On the \b host. 3x3 inverse rotation matrices. One per rotation.
    ///                             For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                             on the output array coordinates. This function assumes \p matrix is already
    ///                             inverted and pre-multiplies the coordinates with the matrix directly.
    /// \param[in] rotation_centers On the \b host. Rotation centers. One per rotation.
    /// \param batches              Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p inputs and \p outputs should not overlap.
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void rotate3D(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                           const float33_t* rotations, const float3_t* rotation_centers, size_t batches,
                           InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream) {

        auto getInvertTransform_ = [=](size_t index) {
            return float34_t(noa::transform::translate(rotation_centers[index]) *
                             float44_t(rotations[index]) *
                             noa::transform::translate(-rotation_centers[index]));
        };

        if (batches == 1) {
            apply3D<PREFILTER>(inputs, {input_pitch.x, input_pitch.y}, shape,
                               outputs, {output_pitch.x, output_pitch.y}, shape,
                               getInvertTransform_(0), interp_mode, border_mode, value, stream);
        } else {
            stream.enqueue([=, &stream]() {
                memory::PtrHost<float34_t> inv_transforms(batches);
                for (size_t i = 0; i < batches; ++i)
                    inv_transforms[i] = getInvertTransform_(i);
                apply3D<PREFILTER>(inputs, input_pitch, shape, outputs, output_pitch, shape,
                                   inv_transforms.get(), batches, interp_mode, border_mode, value, stream);
            });
        }
    }

    /// Applies a single 3D rotation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_IH void rotate3D(const T* input, size2_t input_pitch, T* output, size2_t output_pitch, size3_t shape,
                         float33_t rotation, float3_t rotation_center,
                         InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream) {
        rotate3D<PREFILTER>(input, {input_pitch, 0}, output, {output_pitch, 0},
                            shape, &rotation, &rotation_center, 1, interp_mode, border_mode, value, stream);
    }
}
