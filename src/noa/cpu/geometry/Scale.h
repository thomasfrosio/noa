/// \file noa/cpu/geometry/Scale.h
/// \brief Scaling images and volumes using affine transforms.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/geometry/Transform.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Transform.h"

namespace noa::cpu::geometry {
    /// Applies one or multiple 2D stretching/shrinking.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \b host. Input 2D array.
    /// \param input_strides        BDHW strides, in elements, of \p input.
    /// \param input_shape          BDHW shape of \p input.
    /// \param[out] output          On the \b host. Output 2D array.
    /// \param output_strides       BDHW strides, in elements, of \p output.
    /// \param output_shape         BDHW shape of \p output.
    /// \param[in] scaling_factors  On the \b host. HW forward scaling factors. One per batch.
    /// \param[in] scaling_centers  On the \b host. HW scaling centers. One per batch.
    /// \param interp_mode          Interpolation/filter method. All interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param prefilter            Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p input and \p output should not overlap.
    /// \see "noa/cpu/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void scale2D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                 const shared_t<float2_t[]>& scaling_factors,
                 const shared_t<float2_t[]>& scaling_centers,
                 InterpMode interp_mode, BorderMode border_mode, T value, bool prefilter, Stream& stream) {

        auto getInvertTransform_ = [&](size_t index) {
            return float23_t(noa::geometry::translate(scaling_centers.get()[index]) *
                             float33_t(noa::geometry::scale(1.f / scaling_factors.get()[index])) *
                             noa::geometry::translate(-scaling_centers.get()[index]));
        };

        if (output_shape[0] == 1) {
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

    /// Applies one 2D scaling to a (batched) array.
    /// See overload above for more details.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    inline void scale2D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                        const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                        float2_t scaling_factor, float2_t scaling_center,
                        InterpMode interp_mode, BorderMode border_mode, T value, bool prefilter, Stream& stream) {
        const float23_t matrix(noa::geometry::translate(scaling_center) *
                               float33_t(noa::geometry::scale(1.f / scaling_factor)) *
                               noa::geometry::translate(-scaling_center));
        transform2D(input, input_strides, input_shape, output, output_strides, output_shape,
                    matrix, interp_mode, border_mode, value, prefilter, stream);
    }

    /// Applies one or multiple 3D stretching/shrinking.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \b host. Input 3D array.
    /// \param input_strides        BDHW strides, in elements, of \p input.
    /// \param input_shape          BDHW shape of \p input.
    /// \param[out] output          On the \b host. Output 3D array.
    /// \param output_strides       BDHW strides, in elements, of \p output.
    /// \param output_shape         BDHW shape of \p output.
    /// \param[in] scaling_factors  On the \b host. DHW forward scaling factors. One per batch.
    /// \param[in] scaling_centers  On the \b host. DHW scaling centers. One per batch.
    /// \param interp_mode          Interpolation/filter method. All interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param prefilter            Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p input and \p output should not overlap.
    /// \see "noa/cpu/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void scale3D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                 const shared_t<float3_t[]>& scaling_factors,
                 const shared_t<float3_t[]>& scaling_centers,
                 InterpMode interp_mode, BorderMode border_mode, T value, bool prefilter, Stream& stream) {

        auto getInvertTransform_ = [&](size_t index) -> float34_t {
            return float34_t(noa::geometry::translate(scaling_centers.get()[index]) *
                             float44_t(noa::geometry::scale(1.f / scaling_factors.get()[index])) *
                             noa::geometry::translate(-scaling_centers.get()[index]));
        };

        if (output_shape[0] == 1) {
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

    /// Applies one 3D scaling to a (batched) array.
    /// See overload above for more details.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    inline void scale3D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                        const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                        float3_t scaling_factor, float3_t scaling_center,
                        InterpMode interp_mode, BorderMode border_mode, T value, bool prefilter, Stream& stream) {
        const float34_t matrix(noa::geometry::translate(scaling_center) *
                               float44_t(noa::geometry::scale(1.f / scaling_factor)) *
                               noa::geometry::translate(-scaling_center));
        transform3D(input, input_strides, input_shape, output, output_strides, output_shape,
                    matrix, interp_mode, border_mode, value, prefilter, stream);
    }
}
