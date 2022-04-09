/// \file noa/cpu/geometry/Shift.h
/// \brief Pixel shifts for 2D and 3D (batched) arrays.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry {
    /// Applies one or multiple 2D shifts.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so
    ///          one can move the center of the output window relative to the input window with a simple shift,
    ///          effectively combining a shift and an extraction.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different shift will
    ///          be applied for every batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation. However if the input is not batched, it is broadcast to all output batches,
    ///          effectively applying multiple shifts to the same 2D input array.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \b host. Input 2D array.
    /// \param input_stride         Rightmost stride, in elements, of \p input.
    /// \param input_shape          Rightmost shape of \p input.
    /// \param[out] output          On the \b host. Output 2D array.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape of \p output. The outermost dimension is the batch dimension.
    /// \param[in] shifts           On the \b host. Rightmost forward shifts. One per batch.
    /// \param interp_mode          Interpolation/filter method. All interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p input and \p output should not overlap.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for shifts.
    template<bool PREFILTER = true, typename T>
    void shift2D(const shared_t<const T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 const shared_t<const float2_t[]>& shifts, InterpMode interp_mode, BorderMode border_mode,
                 T value, Stream& stream);

    /// Shifts a 2D (batched) array.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    void shift2D(const shared_t<const T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 float2_t shift, InterpMode interp_mode, BorderMode border_mode,
                 T value, Stream& stream);

    /// Applies one or multiple 3D shifts.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so
    ///          one can move the center of the output window relative to the input window with a simple shift,
    ///          effectively combining a shift and an extraction.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different shift will
    ///          be applied for every batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation. However if the input is not batched, it is broadcast to all output batches,
    ///          effectively applying multiple shifts to the same 3D input array.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \b host. Input 3D array.
    /// \param input_stride         Rightmost stride, in elements, of \p input.
    /// \param input_shape          Rightmost shape of \p input.
    /// \param[out] output          On the \b host. Output 3D array.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape of \p output. The outermost dimension is the batch dimension.
    /// \param[in] shifts           On the \b host. Rightmost forward shifts. One per batch.
    /// \param interp_mode          Interpolation/filter method. All interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p input and \p output should not overlap.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for shifts.
    template<bool PREFILTER = true, typename T>
    void shift3D(const shared_t<const T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 const shared_t<const float3_t[]>& shifts, InterpMode interp_mode, BorderMode border_mode,
                 T value, Stream& stream);

    /// Shifts a 3D (batched) array.
    /// See overload above for more details.
    template<bool PREFILTER = true, typename T>
    void shift3D(const shared_t<const T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 float3_t shift, InterpMode interp_mode, BorderMode border_mode,
                 T value, Stream& stream);
}
