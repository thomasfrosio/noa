/// \file noa/cpu/geometry/Rotate.h
/// \brief Rotations of images and volumes using affine transforms.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Transform.h"

namespace noa::cpu::geometry {
    /// Applies one or multiple 2D rotations.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \b host. Input 2D array.
    /// \param input_stride         Rightmost stride, in elements, of \p input.
    /// \param input_shape          Rightmost shape of \p input.
    /// \param[out] output          On the \b host. Output 2D array.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape of \p output. The outermost dimension is the batch dimension.
    /// \param[in] rotations        On the \b host. Rotation angles, in radians. One per batch.
    /// \param[in] rotation_centers On the \b host. Rotation centers. One per batch.
    /// \param interp_mode          Interpolation/filter method. All interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p input and \p output should not overlap.
    /// \see "noa/cpu/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void rotate2D(const T* input, size4_t input_stride, size4_t input_shape,
                           T* output, size4_t output_stride, size4_t output_shape,
                           const float* rotations, const float2_t* rotation_centers,
                           InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream) {

        auto getInvertTransform_ = [=](size_t index) {
            return float23_t(noa::geometry::translate(rotation_centers[index]) *
                             float33_t(noa::geometry::rotate(-rotations[index])) *
                             noa::geometry::translate(-rotation_centers[index]));
        };

        if (output_shape[0] == 1) {
            transform2D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                   getInvertTransform_(0), interp_mode, border_mode, value, stream);
        } else {
            stream.enqueue([=, &stream]() {
                memory::PtrHost<float23_t> inv_transforms(output_shape[0]);
                for (size_t i = 0; i < output_shape[0]; ++i)
                    inv_transforms[i] = getInvertTransform_(i);
                transform2D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                       inv_transforms.get(), interp_mode, border_mode, value, stream);
            });
        }
    }

    /// Applies one 2D rotation to a (batched) array.
    /// See overload above for more details.
    template<bool PREFILTER = true, typename T>
    NOA_IH void rotate2D(const T* input, size4_t input_stride, size4_t input_shape,
                         T* output, size4_t output_stride, size4_t output_shape,
                         float rotation, float2_t rotation_center,
                         InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream) {
        const float23_t matrix{noa::geometry::translate(rotation_center) *
                               float33_t(noa::geometry::rotate(-rotation)) *
                               noa::geometry::translate(-rotation_center)};
        transform2D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                               matrix, interp_mode, border_mode, value, stream);
    }

    /// Applies one or multiple 3D rotations.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \b host. Input 3D array.
    /// \param input_stride         Rightmost stride, in elements, of \p input.
    /// \param input_shape          Rightmost shape of \p input.
    /// \param[out] output          On the \b host. Output 3D array.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape of \p output. The outermost dimension is the batch dimension.
    /// \param[in] rotations        On the \b host. 3x3 inverse rightmost rotation matrices. One per batch.
    /// \param[in] rotation_centers On the \b host. Rotation centers. One per batch.
    /// \param interp_mode          Interpolation/filter method. All interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p input and \p output should not overlap.
    /// \see "noa/cpu/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void rotate3D(const T* input, size4_t input_stride, size4_t input_shape,
                           T* output, size4_t output_stride, size4_t output_shape,
                           const float33_t* rotations, const float3_t* rotation_centers,
                           InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream) {

        auto getInvertTransform_ = [=](size_t index) {
            return float34_t(noa::geometry::translate(rotation_centers[index]) *
                             float44_t(rotations[index]) *
                             noa::geometry::translate(-rotation_centers[index]));
        };

        if (output_shape[0] == 1) {
            transform3D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                   getInvertTransform_(0), interp_mode, border_mode, value, stream);
        } else {
            stream.enqueue([=, &stream]() {
                memory::PtrHost<float34_t> inv_transforms(output_shape[0]);
                for (size_t i = 0; i < output_shape[0]; ++i)
                    inv_transforms[i] = getInvertTransform_(i);
                transform3D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                       inv_transforms.get(), interp_mode, border_mode, value, stream);
            });
        }
    }

    /// Applies one 3D rotation to a (batched) array.
    /// See overload above for more details.
    template<bool PREFILTER = true, typename T>
    NOA_IH void rotate3D(const T* input, size4_t input_stride, size4_t input_shape,
                           T* output, size4_t output_stride, size4_t output_shape,
                           float33_t rotation, float3_t rotation_center,
                           InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream) {
        const float34_t matrix{noa::geometry::translate(rotation_center) *
                               float44_t{rotation} *
                               noa::geometry::translate(-rotation_center)};
        transform3D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                               matrix, interp_mode, border_mode, value, stream);
    }
}