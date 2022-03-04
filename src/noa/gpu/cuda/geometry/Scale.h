/// \file noa/gpu/cuda/geometry/Scale.h
/// \brief Scaling images and volumes using affine transforms.
/// \author Thomas - ffyr2w
/// \date 22 Jul 2021

#pragma once

#include <memory>

#include "noa/common/Definitions.h"
#include "noa/common/geometry/Transform.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/geometry/Transform.h"

namespace noa::cuda::geometry {
    /// Applies one or multiple 2D scaling/stretching.
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
    /// \param[in] scaling_factors  On the \b host. Rightmost forward scaling factors. One per batch.
    /// \param[in] scaling_centers  On the \b host. Rightmost scaling centers. One per batch.
    /// \param interp_mode          Filter method. Any of InterpMode.
    /// \param border_mode          Address mode. Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///                             The last two are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/cuda/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void scale2D(const T* input, size4_t input_stride, size4_t input_shape,
                          T* output, size4_t output_stride, size4_t output_shape,
                          const float2_t* scaling_factors, const float2_t* scaling_centers,
                          InterpMode interp_mode, BorderMode border_mode, Stream& stream) {

        auto getInvertTransform_ = [=](size_t index) {
            return float23_t{noa::geometry::translate(scaling_centers[index]) *
                             float33_t{noa::geometry::scale(1.f / scaling_factors[index])} *
                             noa::geometry::translate(-scaling_centers[index])};
        };

        if (output_shape[0] == 1) {
            transform2D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                   getInvertTransform_(0), interp_mode, border_mode, stream);
        } else {
            std::unique_ptr<float23_t[]> inv_matrices = std::make_unique<float23_t[]>(output_shape[0]);
            for (size_t i = 0; i < output_shape[0]; ++i)
                inv_matrices[i] = getInvertTransform_(i);
            transform2D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                   inv_matrices.get(), interp_mode, border_mode, stream);
            // No need to synchronize since transform2D does it already.
        }
    }

    /// Applies one 2D scaling to a (batched) array.
    /// See overload above for more details.
    template<bool PREFILTER = true, typename T>
    NOA_IH void scale2D(const T* input, size4_t input_stride, size4_t input_shape,
                        T* output, size4_t output_stride, size4_t output_shape,
                        float2_t scaling_factor, float2_t scaling_center,
                        InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        const float23_t matrix{noa::geometry::translate(scaling_center) *
                               float33_t{noa::geometry::scale(1.f / scaling_factor)} *
                               noa::geometry::translate(-scaling_center)};
        transform2D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                               matrix, interp_mode, border_mode, stream);
    }

    /// Applies one or multiple 3D scaling/stretching.
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
    /// \param[in] scaling_factors  On the \b host. Rightmost forward scaling factors. One per batch.
    /// \param[in] scaling_centers  On the \b host. Rightmost scaling centers. One per batch.
    /// \param interp_mode          Filter method. Any of InterpMode.
    /// \param border_mode          Address mode. Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///                             The last two are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/cuda/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void scale3D(const T* input, size4_t input_stride, size4_t input_shape,
                          T* output, size4_t output_stride, size4_t output_shape,
                          const float3_t* scaling_factors, const float3_t* scaling_centers,
                          InterpMode interp_mode, BorderMode border_mode, Stream& stream) {

        auto getInvertTransform_ = [=](size_t index) -> float34_t {
            return float34_t{noa::geometry::translate(scaling_centers[index]) *
                             float44_t{noa::geometry::scale(1.f / scaling_factors[index])} *
                             noa::geometry::translate(-scaling_centers[index])};
        };

        if (output_shape[0] == 1) {
            transform3D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                   getInvertTransform_(0), interp_mode, border_mode, stream);
        } else {
            std::unique_ptr<float34_t[]> inv_matrices = std::make_unique<float34_t[]>(output_shape[0]);
            for (size_t i = 0; i < output_shape[0]; ++i)
                inv_matrices[i] = getInvertTransform_(i);
            transform3D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                                   inv_matrices.get(), interp_mode, border_mode, stream);
        }
    }

    /// Applies one 3D scaling to a (batched) array.
    /// See overload above for more details.
    template<bool PREFILTER = true, typename T>
    NOA_IH void scale3D(const T* input, size4_t input_stride, size4_t input_shape,
                        T* output, size4_t output_stride, size4_t output_shape,
                        float3_t scaling_factor, float3_t scaling_center,
                        InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        const float34_t matrix{noa::geometry::translate(scaling_center) *
                               float44_t{noa::geometry::scale(1.f / scaling_factor)} *
                               noa::geometry::translate(-scaling_center)};
        transform3D<PREFILTER>(input, input_stride, input_shape, output, output_stride, output_shape,
                               matrix, interp_mode, border_mode, stream);
    }
}
