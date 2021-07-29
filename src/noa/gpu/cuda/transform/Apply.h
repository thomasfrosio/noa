/// \file noa/gpu/cuda/transform/Apply.h
/// \brief Apply affine transforms to arrays or CUDA textures.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

// TODO Add batched versions. Atm, multiple rotations can be performed in one kernel launch but this is on the same input.
//      Batching everything could be quite expensive in term of memory, since each input needs a CUDA array.
//      In practice, it might be better to compute one batch at a time and reuse the same CUDA (or pitched 2D) array.

// -- Using textures -- //
namespace noa::cuda::transform {
    /// Applies one or multiple 2D affine transforms.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p transforms, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    ///
    /// \tparam TEXTURE_OFFSET      Whether or not the 0.5 coordinate offset should be applied.
    ///                             CUDA texture uses a 0->N coordinate system, i.e. the output coordinates
    ///                             pointing at the center of a pixel is shifted by +0.5 compared to the index.
    ///                             On the input window, this is equivalent to a (left) shift of -0.5.
    ///                             If true, the function will add this offset to the output coordinates, otherwise,
    ///                             it assumes the offset is already included in \p transforms.
    /// \tparam T                   float or cfloat_t.
    /// \tparam MATRIX              float23_t or float33_t.
    ///
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_shape        Logical {fast, medium} shape of \p texture.
    ///                             This is only used if \p texture_border_mode is BORDER_PERIODIC or BORDER_MIRROR.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture.
    ///                             Should be INTERP_NEAREST, INTERP_LINEAR, INTERP_COSINE or INTERP_CUBIC_BSPLINE.
    /// \param texture_border_mode  Border/address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] outputs         On the \b device. Output arrays. One per transformation.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param output_shape         Logical {fast, medium} shape of \p outputs.
    /// \param[in] transforms       One the \b device. 2x3 or 3x3 inverse affine matrices. One per transformation.
    ///                             For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                             on the output array coordinates. This functions assumes \p transforms are already
    ///                             inverted and pre-multiplies the coordinates with these matrices directly.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \see "noa/gpu/cuda/memory/PtrTexture.h" for more details on CUDA textures and how to use them.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR, and require
    ///       \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
    template<bool TEXTURE_OFFSET = true, typename T, typename MATRIX>
    NOA_HOST void apply2D(cudaTextureObject_t texture, size2_t texture_shape,
                          InterpMode texture_interp_mode, BorderMode texture_border_mode,
                          T* outputs, size_t output_pitch, size2_t output_shape,
                          const MATRIX* transforms, uint nb_transforms, Stream& stream);

    /// Applies a single 2D affine transform.
    /// \see This function is has the same features and limitations than the overload above.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool TEXTURE_OFFSET = true, typename T, typename MATRIX>
    NOA_HOST void apply2D(cudaTextureObject_t texture, size2_t texture_shape,
                          InterpMode texture_interp_mode, BorderMode texture_border_mode,
                          T* output, size_t output_pitch, size2_t output_shape,
                          MATRIX transform, Stream& stream);

    /// Applies one or multiple 3D affine transforms.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p transforms, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    ///
    /// \tparam TEXTURE_OFFSET      Whether or not the 0.5 coordinate offset should be applied.
    ///                             CUDA texture uses a 0->N coordinate system, i.e. the output coordinates
    ///                             pointing at the center of a pixel is shifted by +0.5 compared to the index.
    ///                             On the input window, this is equivalent to a (left) shift of -0.5.
    ///                             If true, the function will add this offset to the output coordinates, otherwise,
    ///                             it assumes the offset is already included in \p transforms.
    /// \tparam T                   float or cfloat_t.
    /// \tparam MATRIX              float34_t or float44_t.
    ///
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_shape        Logical {fast, medium, slow} shape of \p texture.
    ///                             This is only used if \p texture_border_mode is BORDER_PERIODIC or BORDER_MIRROR.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture.
    ///                             Should be INTERP_NEAREST, INTERP_LINEAR, INTERP_COSINE or INTERP_CUBIC_BSPLINE.
    /// \param texture_border_mode  Border/address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] outputs         On the \b device. Output arrays. One per transformation.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param output_shape         Logical {fast, medium, slow} shape of \p outputs.
    /// \param[in] transforms       One the \b device. 3x4 or 4x4 inverse affine matrices. One per transformation.
    ///                             For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                             on the output array coordinates. This functions assumes \p transforms are already
    ///                             inverted and pre-multiplies the coordinates with these matrices directly.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \see "noa/gpu/cuda/memory/PtrTexture.h" for more details on CUDA textures and how to use them.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR, and require
    ///       \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
    template<bool TEXTURE_OFFSET = true, typename T, typename MATRIX>
    NOA_HOST void apply3D(cudaTextureObject_t texture, size3_t texture_shape,
                          InterpMode texture_interp_mode, BorderMode texture_border_mode,
                          T* outputs, size_t output_pitch, size3_t output_shape,
                          const MATRIX* transforms, uint nb_transforms, Stream& stream);

    /// Applies a single 3D affine transform.
    /// \see This function is has the same features and limitations than the overload above.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool TEXTURE_OFFSET = true, typename T, typename MATRIX>
    NOA_HOST void apply3D(cudaTextureObject_t texture, size3_t texture_shape,
                          InterpMode texture_interp_mode, BorderMode texture_border_mode,
                          T* output, size_t output_pitch, size3_t output_shape,
                          MATRIX transform, Stream& stream);
}

// -- Using arrays -- //
namespace noa::cuda::transform {
    /// Applies one or multiple 2D affine transforms.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p transforms, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the output of bspline::prefilter2D(),
    ///                             which is then used as input for the interpolation.
    /// \tparam TEXTURE_OFFSET      Whether or not the 0.5 coordinate offset should be applied.
    ///                             CUDA texture uses a 0->N coordinate system, i.e. the output coordinates
    ///                             pointing at the center of a pixel is shifted by +0.5 compared to the index.
    ///                             On the input window, this is equivalent to a (left) shift of -0.5.
    ///                             If true, the function will add this offset to the output coordinates, otherwise,
    ///                             it assumes the offset is already included in \p transforms.
    /// \tparam T                   float or cfloat_t.
    /// \tparam MATRIX              float23_t or float33_t.
    ///
    /// \param input                On the \b device. Input array.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param input_shape          Logical {fast, medium} shape of \p input.
    /// \param[out] outputs         On the \b device. Output arrays. One per transformation. Can be equal to \p input.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param output_shape         Logical {fast, medium} shape of \p outputs.
    /// \param[in] transforms       One the \b device. 2x3 or 3x3 inverse affine matrices. One per transformation.
    ///                             For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                             on the output array coordinates. This functions assumes \p transforms are already
    ///                             inverted and pre-multiplies the coordinates with these matrices directly.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method.
    ///                             Should be INTERP_NEAREST, INTERP_LINEAR, INTERP_COSINE or INTERP_CUBIC_BSPLINE.
    /// \param border_mode          Border/address mode.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR.
    template<bool PREFILTER = true, bool TEXTURE_OFFSET = true, typename T, typename MATRIX>
    NOA_HOST void apply2D(const T* input, size_t input_pitch, size2_t input_shape,
                          T* outputs, size_t output_pitch, size2_t output_shape,
                          const MATRIX* transforms, uint nb_transforms,
                          InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies a single 2D affine transform.
    /// \see This function is has the same features and limitations than the overload above.
    template<bool PREFILTER = true, bool TEXTURE_OFFSET = true, typename T, typename MATRIX>
    NOA_HOST void apply2D(const T* input, size_t input_pitch, size2_t input_shape,
                          T* output, size_t output_pitch, size2_t output_shape,
                          MATRIX transform, InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies one or multiple 3D affine transforms.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p transforms, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the output of bspline::prefilter3D(),
    ///                             which is then used as input for the interpolation.
    /// \tparam TEXTURE_OFFSET      Whether or not the 0.5 coordinate offset should be applied.
    ///                             CUDA texture uses a 0->N coordinate system, i.e. the output coordinates
    ///                             pointing at the center of a pixel is shifted by +0.5 compared to the index.
    ///                             On the input window, this is equivalent to a (left) shift of -0.5.
    ///                             If true, the function will add this offset to the output coordinates, otherwise,
    ///                             it assumes the offset is already included in \p transforms.
    /// \tparam T                   float or cfloat_t.
    /// \tparam MATRIX              float34_t or float44_t.
    ///
    /// \param input                On the \b device. Input array.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param input_shape          Logical {fast, medium, slow} shape of \p input.
    /// \param[out] outputs         On the \b device. Output arrays. One per transformation. Can be equal to \p input.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param output_shape         Logical {fast, medium, slow} shape of \p outputs.
    /// \param[in] transforms       One the \b device. 3x4 or 4x4 inverse affine matrices. One per transformation.
    ///                             For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                             on the output array coordinates. This functions assumes \p transforms are already
    ///                             inverted and pre-multiplies the coordinates with these matrices directly.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method.
    ///                             Should be INTERP_NEAREST, INTERP_LINEAR, INTERP_COSINE or INTERP_CUBIC_BSPLINE.
    /// \param border_mode          Border/address mode.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR.
    template<bool PREFILTER = true, bool TEXTURE_OFFSET = true, typename T, typename MATRIX>
    NOA_HOST void apply3D(const T* input, size_t input_pitch, size3_t input_shape,
                          T* outputs, size_t output_pitch, size3_t output_shape,
                          const MATRIX* transforms, uint nb_transforms,
                          InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies a single 3D affine transform.
    /// \see This function is has the same features and limitations than the overload above.
    template<bool PREFILTER = true, bool TEXTURE_OFFSET = true, typename T, typename MATRIX>
    NOA_HOST void apply3D(const T* input, size_t input_pitch, size3_t input_shape,
                          T* output, size_t output_pitch, size3_t output_shape,
                          MATRIX transform, InterpMode interp_mode, BorderMode border_mode, Stream& stream);
}
