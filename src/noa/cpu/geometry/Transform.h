/// \file noa/cpu/geometry/Apply.h
/// \brief Apply linear and affine transforms to images and volumes.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/geometry/Symmetry.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry {
    /// Applies one or multiple 2D affine transforms.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p transforms, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix will
    ///          be applied for every batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation. However if the input is not batched, it is broadcast to all output batches,
    ///          effectively applying multiple transformations to the same 2D input array.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, double, cfloat_t or cdouble_t.
    /// \tparam MAT                 float23_t or float33_t.
    /// \param[in] input            On the \b host. Input 2D array.
    /// \param input_strides        BDHW strides, in elements, of \p input.
    /// \param input_shape          BDHW shape of \p input.
    /// \param[out] output          On the \b host. Output 2D array.
    /// \param output_strides       BDHW strides, in elements, of \p output.
    /// \param output_shape         BDHW shape of \p output.
    /// \param[in] matrices         One the \b host. 2x3 or 3x3 inverse HW affine matrices. One per batch.
    /// \param interp_mode          Interpolation/filter method. All interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p input and \p output should not overlap.
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                         (traits::is_float23_v<MAT> || traits::is_float33_v<MAT>)>>
    void transform2D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                     const shared_t<MAT[]>& matrices, InterpMode interp_mode, BorderMode border_mode,
                     T value, Stream& stream);

    /// Applies a single 2D affine (batched) transform.
    /// \see This function is has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                         (traits::is_float23_v<MAT> || traits::is_float33_v<MAT>)>>
    void transform2D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                     MAT matrix, InterpMode interp_mode, BorderMode border_mode,
                     T value, Stream& stream);

    /// Applies one or multiple 3D affine transforms.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p transforms, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix will
    ///          be applied for every batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation. However if the input is not batched, it is broadcast to all output batches,
    ///          effectively applying multiple transformations to the same 3D input array.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, double, cfloat_t or cdouble_t.
    /// \tparam MAT                 float34_t or float44_t.
    /// \param[in] input            On the \b host. Input 3D array.
    /// \param input_strides        BDHW strides, in elements, of \p input.
    /// \param input_shape          BDHW shape of \p input.
    /// \param[out] output          On the \b host. Output 3D array.
    /// \param output_strides       BDHW strides, in elements, of \p output.
    /// \param output_shape         BDHW shape of \p output.
    /// \param[in] matrices         One the \b host. 3x4 or 4x4 inverse DHW affine matrices. One per batch.
    /// \param interp_mode          Interpolation/filter method. All interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p input and \p output should not overlap.
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                         (traits::is_float34_v<MAT> || traits::is_float44_v<MAT>)>>
    void transform3D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                     const shared_t<MAT[]>& matrices, InterpMode interp_mode, BorderMode border_mode,
                     T value, Stream& stream);

    /// Applies one 3D affine transform to a (batched) array.
    /// \see This function is has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                         (traits::is_float34_v<MAT> || traits::is_float44_v<MAT>)>>
    void transform3D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                     MAT matrix, InterpMode interp_mode, BorderMode border_mode,
                     T value, Stream& stream);
}

// -- Apply symmetry -- //
namespace noa::cpu::geometry {
    using Symmetry = ::noa::geometry::Symmetry;

    /// Shifts, then rotates/scales and applies the symmetry on the 2D input array.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p transforms, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, the input can be batched
    ///          as well, resulting in a fully batched operation. However if the input is not batched, it is broadcast
    ///          to all output batches, effectively applying multiple transformations to the same 2D input array.
    ///
    /// \tparam PREFILTER       Whether or not the input should be prefiltered.
    ///                         Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Input array to transform.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param input_shape      BDHW shape, in elements, of \p input.
    /// \param[out] output      On the \b host. Transformed array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param output_shape     BDHW shape, in elements, of \p output.
    /// \param shift            HW forward shifts to apply before the other transformations.
    ///                         Positive shifts translate the object to the right.
    /// \param matrix           BDHW inverse rotation/scaling to apply after the shifts.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param center           HW index of the transformation center.
    ///                         Both \p matrix and \p symmetry operates around this center.
    /// \param interp_mode      Interpolation/filter mode. All interpolation modes are supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p input and \p output should not overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T,
             typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void transform2D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode, bool normalize, Stream& stream);

    /// Shifts, then rotates/scales and applies the symmetry on the 3D input array.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p transforms, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, the input can be batched
    ///          as well, resulting in a fully batched operation. However if the input is not batched, it is broadcast
    ///          to all output batches, effectively applying multiple transformations to the same 3D input array.
    ///
    /// \tparam PREFILTER       Whether or not the input should be prefiltered.
    ///                         Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Input array to transform.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param input_shape      BDHW shape, in elements, of \p input.
    /// \param[out] output      On the \b host. Transformed array.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param output_shape     BDHW shape, in elements, of \p output.
    /// \param shift            DHW forward shifts to apply before the other transformations.
    ///                         Positive shifts translate the object to the right.
    /// \param matrix           BDHW inverse rotation/scaling to apply after the shifts.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param center           DHW index of the transformation center.
    ///                         Both \p matrix and \p symmetry operates around this center.
    /// \param interp_mode      Interpolation/filter mode. All interpolation modes are supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p input and \p output should not overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T,
             typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void transform3D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode, bool normalize, Stream& stream);
}
