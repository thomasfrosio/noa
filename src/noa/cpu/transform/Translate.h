/// \file noa/cpu/transform/Translate.h
/// \brief Translations for images and volumes.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu::transform {
    /// Applies one or multiple 2D translations.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so
    ///          one can move the center of the output window relative to the input window with a simple translation,
    ///          effectively combining a translation and an extraction.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the prefiltered output which
    ///                             is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \b host. Input array.
    /// \param input_shape          Logical {fast, medium} shape of \p input.
    /// \param[out] outputs         On the \b host. Output arrays. One per rotation.
    /// \param output_shape         Logical {fast, medium} shape of \p outputs.
    /// \param[in] translations     On the \b host. One per dimension. One per translation.
    ///                             Positive values shift the "object" on the output window to the right.
    /// \param nb_translations      Number of translations to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \note In-place computation is not allowed, i.e. \p input and \p outputs should not overlap.
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for translations.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void translate2D(const T* input, size2_t input_shape, T* outputs, size2_t output_shape,
                              const float2_t* translations, size_t nb_translations,
                              InterpMode interp_mode, BorderMode border_mode, T value = T(0));

    /// Applies a single 2D translation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_IH void translate2D(const T* input, size2_t input_shape, T* output, size2_t output_shape,
                            float2_t translation, InterpMode interp_mode, BorderMode border_mode, T value = T(0)) {
        translate2D<PREFILTER>(input, input_shape, output, output_shape,
                               &translation, 1, interp_mode, border_mode, value);
    }

    /// Applies one or multiple 3D translations.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so
    ///          one can move the center of the output window relative to the input window with a simple translation,
    ///          effectively combining a translation and an extraction.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the prefiltered output which
    ///                             is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] input            On the \b host. Input array.
    /// \param input_shape          Logical {fast, medium, slow} shape of \p input.
    /// \param[out] outputs         On the \b host. Output arrays. One per rotation.
    /// \param output_shape         Logical {fast, medium, slow} shape of \p outputs.
    /// \param[in] translations     On the \b host. One per dimension. One per translation.
    ///                             Positive values shift the "object" on the output window to the right.
    /// \param nb_translations      Number of translations to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \note In-place computation is not allowed, i.e. \p input and \p outputs should not overlap.
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for translations.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void translate3D(const T* input, size3_t input_shape, T* outputs, size3_t output_shape,
                              const float3_t* translations, size_t nb_translations,
                              InterpMode interp_mode, BorderMode border_mode, T value = T(0));

    /// Applies a single 3D translation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_IH void translate3D(const T* input, size3_t input_shape, T* output, size3_t output_shape,
                            float3_t translation, InterpMode interp_mode, BorderMode border_mode, T value = T(0)) {
        translate3D<PREFILTER>(input, input_shape, output, output_shape,
                               &translation, 1, interp_mode, border_mode, value);
    }
}
