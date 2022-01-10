/// \file noa/cpu/transform/Translate.h
/// \brief Translations for images and volumes.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::transform {
    /// Applies one or multiple 2D shifts.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so
    ///          one can move the center of the output window relative to the input window with a simple shift,
    ///          effectively combining a shift and an extraction.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p inputs is allocated and used to store the prefiltered output which
    ///                             is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs           On the \b host. Input arrays. One per shift.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param input_shape          Logical {fast, medium} shape of \p inputs.
    /// \param[out] outputs         On the \b host. Output arrays. One per shift.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param output_shape         Logical {fast, medium} shape of \p outputs.
    /// \param[in] shifts           On the \b host. One per dimension. One per shift.
    ///                             Positive values shift the "object" on the output window to the right.
    /// \param batches              Number of shifts to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p inputs and \p outputs should not overlap.
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for shifts.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void shift2D(const T* inputs, size2_t input_pitch, size2_t input_shape,
                          T* outputs, size2_t output_pitch, size2_t output_shape,
                          const float2_t* shifts, size_t batches,
                          InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream);

    /// Applies a single 2D shift.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_IH void shift2D(const T* input, size_t input_pitch, size2_t input_shape,
                        T* output, size_t output_pitch, size2_t output_shape,
                        float2_t shift, InterpMode interp_mode, BorderMode border_mode,
                        T value, Stream& stream) {
        shift2D<PREFILTER>(input, {input_pitch, 0}, input_shape, output, {output_pitch, 0}, output_shape,
                           &shift, 1, interp_mode, border_mode, value, stream);
    }

    /// Applies one or multiple 3D shifts.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so
    ///          one can move the center of the output window relative to the input window with a simple shift,
    ///          effectively combining a shift and an extraction.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p inputs is allocated and used to store the prefiltered output which
    ///                             is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs           On the \b host. Input arrays. One per shift.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param input_shape          Logical {fast, medium, slow} shape of \p inputs.
    /// \param[out] outputs         On the \b host. Output arrays. One per shift.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param output_shape         Logical {fast, medium, slow} shape of \p outputs.
    /// \param[in] shifts           On the \b host. One per dimension. One per shift.
    ///                             Positive values shift the "object" on the output window to the right.
    /// \param batches              Number of shifts to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note In-place computation is not allowed, i.e. \p inputs and \p outputs should not overlap.
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for shifts.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void shift3D(const T* inputs, size3_t input_pitch, size3_t input_shape,
                          T* outputs, size3_t output_pitch, size3_t output_shape,
                          const float3_t* shifts, size_t batches,
                          InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream);

    /// Applies a single 3D shift.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_IH void shift3D(const T* input, size2_t input_pitch, size3_t input_shape,
                        T* output, size2_t output_pitch, size3_t output_shape,
                        float3_t shift, InterpMode interp_mode, BorderMode border_mode,
                        T value, Stream& stream) {
        shift3D<PREFILTER>(input, {input_pitch, 0}, input_shape, output, {output_pitch, 0}, output_shape,
                           &shift, 1, interp_mode, border_mode, value, stream);
    }
}
