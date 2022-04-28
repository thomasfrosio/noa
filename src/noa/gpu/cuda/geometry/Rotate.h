/// \file noa/gpu/cuda/geometry/Rotate.h
/// \brief Rotations of images and volumes using affine transforms.
/// \author Thomas - ffyr2w
/// \date 22 Jul 2021

#pragma once

#include <memory>

#include "noa/common/Definitions.h"
#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/geometry/Transform.h"

namespace noa::cuda::geometry {
    /// Applies one or multiple 2D rotations.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, cfloat_t.
    /// \param[in] input            Input 2D array. If pre-filtering is required, should be on the \b device.
    ///                             Otherwise, can be on the \b host or \b device.
    /// \param input_stride         Rightmost stride, in elements, of \p input.
    ///                             The innermost dimension should be contiguous.
    /// \param input_shape          Rightmost shape, in elements, of \p input.
    /// \param[out] output          On the \b device. Output 2D array. Can be equal to \p input.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] rotations        On the \b host. Rotation angles, in radians. One per batch.
    /// \param[in] rotation_centers On the \b host. Rotation centers. One per batch.
    /// \param interp_mode          Filter method. Any of InterpMode.
    /// \param border_mode          Address mode. Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///                             The last two are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/cuda/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void rotate2D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                  const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                  const shared_t<float[]>& rotations, const shared_t<float2_t[]>& rotation_centers,
                  InterpMode interp_mode, BorderMode border_mode, Stream& stream) {

        auto getInvertTransform_ = [=](size_t index) {
            return float23_t{noa::geometry::translate(rotation_centers.get()[index]) *
                             float33_t{noa::geometry::rotate(-rotations.get()[index])} *
                             noa::geometry::translate(-rotation_centers.get()[index])};
        };

        if (output_shape[0] == 1) {
            transform2D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                   getInvertTransform_(0), interp_mode, border_mode, stream);
        } else {
            memory::PtrPinned<float23_t> inv_matrices{output_shape[0]};
            for (size_t i = 0; i < output_shape[0]; ++i)
                inv_matrices[i] = getInvertTransform_(i);
            transform2D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                   inv_matrices.share(), interp_mode, border_mode, stream);
        }
    }

    /// Applies one 2D rotation to a (batched) array.
    /// See overload above for more details.
    template<bool PREFILTER = true, typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    NOA_IH void rotate2D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                         const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                         float rotation, float2_t rotation_center,
                         InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        const float23_t matrix{noa::geometry::translate(rotation_center) *
                               float33_t{noa::geometry::rotate(-rotation)} *
                               noa::geometry::translate(-rotation_center)};
        transform2D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                               matrix, interp_mode, border_mode, stream);
    }

    /// Applies one or multiple 3D rotations.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, cfloat_t.
    /// \param[in] input            Input 3D array. If pre-filtering is required, should be on the \b device.
    ///                             Otherwise, can be on the \b host or \b device.
    /// \param input_stride         Rightmost stride, in elements, of \p input.
    ///                             The third-most and innermost dimensions should be contiguous.
    /// \param input_shape          Rightmost shape, in elements, of \p input.
    /// \param[out] output          On the \b device. Output 3D array. Can be equal to \p input.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] rotations        On the \b host. 3x3 inverse rightmost rotation matrices. One per batch.
    /// \param[in] rotation_centers On the \b host. Rotation centers. One per batch.
    /// \param interp_mode          Filter method. Any of InterpMode.
    /// \param border_mode          Address mode. Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///                             The last two are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/cuda/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void rotate3D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                  const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                  const shared_t<float33_t[]>& rotations,
                  const shared_t<float3_t[]>& rotation_centers,
                  InterpMode interp_mode, BorderMode border_mode, Stream& stream) {

        auto getInvertTransform_ = [=](size_t index) {
            return float34_t{noa::geometry::translate(rotation_centers.get()[index]) *
                             float44_t{rotations.get()[index]} *
                             noa::geometry::translate(-rotation_centers.get()[index])};
        };

        if (output_shape[0] == 1) {
            transform3D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                   getInvertTransform_(0), interp_mode, border_mode, stream);
        } else {
            memory::PtrPinned<float34_t> inv_matrices{output_shape[0]};
            for (size_t i = 0; i < output_shape[0]; ++i)
                inv_matrices[i] = getInvertTransform_(i);
            transform3D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                   inv_matrices.share(), interp_mode, border_mode, stream);
        }
    }

    /// Applies one 3D rotation to a (batched) array.
    /// See overload above for more details.
    template<bool PREFILTER = true, typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    NOA_IH void rotate3D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                         const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                         float33_t rotation, float3_t rotation_center,
                         InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        const float34_t matrix{noa::geometry::translate(rotation_center) *
                               float44_t{rotation} *
                               noa::geometry::translate(-rotation_center)};
        transform3D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                               matrix, interp_mode, border_mode, stream);
    }
}
