/// \file noa/gpu/cuda/transform/Translate.h
/// \brief Translate arrays or CUDA textures.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

// -- Using textures -- //
namespace noa::cuda::transform {
    /// Applies one or multiple 2D translations.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a additional translation in \p translations, one can move the center of the output window
    ///          relative to the input window, effectively combining a translation and an extraction.
    ///
    /// \tparam TEXTURE_OFFSET      Whether or not the 0.5 coordinate offset should be applied.
    ///                             CUDA texture uses a 0->N coordinate system, i.e. the output coordinates
    ///                             pointing at the center of a pixel is shifted by +0.5 compared to the index.
    ///                             On the input window, this is equivalent to a (left) shift of -0.5.
    ///                             If true, the function will add this offset to the output coordinates, otherwise,
    ///                             it assumes the offset is already included in \p translations.
    /// \tparam T                   float or cfloat_t.
    ///
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_shape        Logical {fast, medium} shape of \p texture.
    ///                             This is only used if \p texture_border_mode is BORDER_PERIODIC or BORDER_MIRROR.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. Any of InterpMode.
    /// \param texture_border_mode  Border/address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] outputs         On the \b device. Output arrays. One per translation.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param output_shape         Logical {fast, medium} shape of \p outputs.
    /// \param[in] translations     On the \b device. One per translation.
    ///                             Positive values shift the "object" on the output window to the right.
    /// \param nb_translations      Number of translations to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \see "noa/gpu/cuda/memory/PtrTexture.h" for more details on CUDA textures and how to use them.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST, and
    ///       require \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
    template<bool TEXTURE_OFFSET = true, typename T>
    NOA_HOST void translate2D(cudaTextureObject_t texture, size2_t texture_shape,
                              InterpMode texture_interp_mode, BorderMode texture_border_mode,
                              T* outputs, size_t output_pitch, size2_t output_shape,
                              const float2_t* translations, size_t nb_translations, Stream& stream);

    /// Translates a single 2D array.
    /// \see This function has the same features and limitations than the first overload above.
    template<bool TEXTURE_OFFSET = true, typename T>
    NOA_HOST void translate2D(cudaTextureObject_t texture, size2_t texture_shape,
                              InterpMode texture_interp_mode, BorderMode texture_border_mode,
                              T* output, size_t output_pitch, size2_t output_shape,
                              float2_t translation, Stream& stream);

    /// Applies one or multiple 3D translations.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a additional translation in \p translations, one can move the center of the output window
    ///          relative to the input window, effectively combining a translation and an extraction.
    ///
    /// \tparam TEXTURE_OFFSET      Whether or not the 0.5 coordinate offset should be applied.
    ///                             CUDA texture uses a 0->N coordinate system, i.e. the output coordinates
    ///                             pointing at the center of a pixel is shifted by +0.5 compared to the index.
    ///                             On the input window, this is equivalent to a (left) shift of -0.5.
    ///                             If true, the function will add this offset to the output coordinates, otherwise,
    ///                             it assumes the offset is already included in \p translations.
    /// \tparam T                   float or cfloat_t.
    ///
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_shape        Logical {fast, medium, slow} shape of \p texture.
    ///                             This is only used if \p texture_border_mode is BORDER_PERIODIC or BORDER_MIRROR.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. Any of InterpMode.
    /// \param texture_border_mode  Border/address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] outputs         On the \b device. Output arrays. One per translation.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param output_shape         Logical {fast, medium, slow} shape of \p outputs.
    /// \param[in] translations     On the \b device. One per translation.
    ///                             Positive values shift the "object" on the output window to the right.
    /// \param nb_translations      Number of translations to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \see "noa/gpu/cuda/memory/PtrTexture.h" for more details on CUDA textures and how to use them.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST, and
    ///       require \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
    template<bool TEXTURE_OFFSET = true, typename T>
    NOA_HOST void translate3D(cudaTextureObject_t texture, size3_t texture_shape,
                              InterpMode texture_interp_mode, BorderMode texture_border_mode,
                              T* outputs, size_t output_pitch, size3_t output_shape,
                              const float3_t* translations, size_t nb_translations, Stream& stream);

    /// Translates a single 3D array.
    /// \see This function has the same features and limitations than the first overload above.
    template<bool TEXTURE_OFFSET = true, typename T>
    NOA_HOST void translate3D(cudaTextureObject_t texture, size3_t texture_shape,
                              InterpMode texture_interp_mode, BorderMode texture_border_mode,
                              T* output, size_t output_pitch, size3_t output_shape,
                              float3_t translation, Stream& stream);
}

// -- Using arrays -- //
namespace noa::cuda::transform {
    /// Applies one or multiple 2D translations.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a additional translation in \p translations, one can move the center of the output window
    ///          relative to the input window, effectively combining a translation and an extraction.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true,
    ///                             a temporary array of the same shape as \p input is allocated and used to store the
    ///                             output of bspline::prefilter2D(), which is then used as input for the interpolation.
    /// \tparam TEXTURE_OFFSET      Whether or not the 0.5 coordinate offset should be applied.
    ///                             CUDA texture uses a 0->N coordinate system, i.e. the output coordinates
    ///                             pointing at the center of a pixel is shifted by +0.5 compared to the index.
    ///                             On the input window, this is equivalent to a (left) shift of -0.5.
    ///                             If true, the function will add this offset to the output coordinates, otherwise,
    ///                             it assumes the offset is already included in \p translations.
    /// \tparam T                   float or cfloat_t.
    ///
    /// \param[in] input            Input array. If pre-filtering is required (see \p PREFILTER), should be
    ///                             on the \b device. Otherwise, can be on the \b host or \b device.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param input_shape          Logical {fast, medium} shape of \p input.
    /// \param[out] outputs         On the \b device. Output arrays. One per translation. Can be equal to \p input.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param output_shape         Logical {fast, medium} shape of \p outputs.
    /// \param[in] translations     On the \b device. One per translation.
    ///                             Positive values shift the "object" on the output window to the right.
    /// \param nb_translations      Number of translations to compute.
    /// \param interp_mode          Interpolation/filter method. Any of InterpMode.
    /// \param border_mode          Border/address mode. Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or
    ///                             BORDER_MIRROR. BORDER_PERIODIC and BORDER_MIRROR are only supported with
    ///                             INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR.
    template<bool PREFILTER = true, bool TEXTURE_OFFSET = true, typename T>
    NOA_HOST void translate2D(const T* input, size_t input_pitch, size2_t input_shape,
                              T* outputs, size_t output_pitch, size2_t output_shape,
                              const float2_t* translations, size_t nb_translations,
                              InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies a single 2D translation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, bool TEXTURE_OFFSET = true, typename T>
    NOA_HOST void translate2D(const T* input, size_t input_pitch, size2_t input_shape,
                              T* output, size_t output_pitch, size2_t output_shape,
                              float2_t translation,
                              InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies one or multiple 3D translations.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a additional translation in \p translations, one can move the center of the output window
    ///          relative to the input window, effectively combining a translation and an extraction.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true,
    ///                             a temporary array of the same shape as \p input is allocated and used to store the
    ///                             output of bspline::prefilter3D(), which is then used as input for the interpolation.
    /// \tparam TEXTURE_OFFSET      Whether or not the 0.5 coordinate offset should be applied.
    ///                             CUDA texture uses a 0->N coordinate system, i.e. the output coordinates
    ///                             pointing at the center of a pixel is shifted by +0.5 compared to the index.
    ///                             On the input window, this is equivalent to a (left) shift of -0.5.
    ///                             If true, the function will add this offset to the output coordinates, otherwise,
    ///                             it assumes the offset is already included in \p translations.
    /// \tparam T                   float or cfloat_t.
    ///
    /// \param[in] input            Input array. If pre-filtering is required (see \p PREFILTER), should be
    ///                             on the \b device. Otherwise, can be on the \b host or \b device.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param input_shape          Logical {fast, medium, slow} shape of \p input.
    /// \param[out] outputs         On the \b device. Output arrays. One per translation. Can be equal to \p input.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param output_shape         Logical {fast, medium, slow} shape of \p outputs.
    /// \param[in] translations     On the \b device. One per translation.
    ///                             Positive values shift the "object" on the output window to the right.
    /// \param nb_translations      Number of translations to compute.
    /// \param interp_mode          Interpolation/filter method. Any of InterpMode.
    /// \param border_mode          Border/address mode. Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or
    ///                             BORDER_MIRROR. BORDER_PERIODIC and BORDER_MIRROR are only supported with
    ///                             INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
    template<bool PREFILTER = true, bool TEXTURE_OFFSET = true, typename T>
    NOA_HOST void translate3D(const T* input, size_t input_pitch, size3_t input_shape,
                              T* outputs, size_t output_pitch, size3_t output_shape,
                              const float3_t* translations, size_t nb_translations,
                              InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies a single 3D translation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, bool TEXTURE_OFFSET = true, typename T>
    NOA_HOST void translate3D(const T* input, size_t input_pitch, size3_t input_shape,
                              T* output, size_t output_pitch, size3_t output_shape,
                              float3_t translation,
                              InterpMode interp_mode, BorderMode border_mode, Stream& stream);
}
