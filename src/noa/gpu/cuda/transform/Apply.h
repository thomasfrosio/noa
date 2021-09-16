/// \file noa/gpu/cuda/transform/Apply.h
/// \brief Apply affine transforms to arrays or CUDA textures.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include "noa/common/Definitions.h"
#include <noa/common/transform/Symmetry.h>
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
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. Any of InterpMode.
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
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST, and
    ///       require \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
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
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. Any of InterpMode.
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
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST, and
    ///       require \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
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
    ///                             is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true, a
    ///                             temporary array of the same shape as \p input is allocated and used to store the
    ///                             output of bspline::prefilter2D(), which is then used as input for the interpolation.
    /// \tparam TEXTURE_OFFSET      Whether or not the 0.5 coordinate offset should be applied.
    ///                             CUDA texture uses a 0->N coordinate system, i.e. the output coordinates
    ///                             pointing at the center of a pixel is shifted by +0.5 compared to the index.
    ///                             On the input window, this is equivalent to a (left) shift of -0.5.
    ///                             If true, the function will add this offset to the output coordinates, otherwise,
    ///                             it assumes the offset is already included in \p transforms.
    /// \tparam T                   float or cfloat_t.
    /// \tparam MATRIX              float23_t or float33_t.
    ///
    /// \param[in] input            Input array. If \p PREFILTER is true and \p interp_mode is INTERP_CUBIC_BSPLINE or
    ///                             INTERP_CUBIC_BSPLINE_FAST, should be on the \b device. Otherwise, can be on the
    ///                             \b host or \b device.
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
    /// \param interp_mode          Interpolation/filter method. Any of InterpMode.
    /// \param border_mode          Border/address mode.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
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
    ///                             is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true, a
    ///                             temporary array of the same shape as \p input is allocated and used to store the
    ///                             output of bspline::prefilter3D(), which is then used as input for the interpolation.
    /// \tparam TEXTURE_OFFSET      Whether or not the 0.5 coordinate offset should be applied.
    ///                             CUDA texture uses a 0->N coordinate system, i.e. the output coordinates
    ///                             pointing at the center of a pixel is shifted by +0.5 compared to the index.
    ///                             On the input window, this is equivalent to a (left) shift of -0.5.
    ///                             If true, the function will add this offset to the output coordinates, otherwise,
    ///                             it assumes the offset is already included in \p transforms.
    /// \tparam T                   float or cfloat_t.
    /// \tparam MATRIX              float34_t or float44_t.
    ///
    /// \param[in] input            Input array. If \p PREFILTER is true and \p interp_mode is INTERP_CUBIC_BSPLINE or
    ///                             INTERP_CUBIC_BSPLINE_FAST, should be on the \b device. Otherwise, can be on the
    ///                             \b host or \b device.
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
    /// \param interp_mode          Interpolation/filter method. Any of InterpMode.
    /// \param border_mode          Border/address mode.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
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

// -- Symmetry - using textures -- //
namespace noa::cuda::transform {
    using Symmetry = noa::transform::Symmetry;

    /// Shifts, rotate/scale and then apply the symmetry on the 2D input texture.
    /// \tparam T                       float or cfloat.
    /// \param texture                  Input texture bound to a CUDA array.
    /// \param texture_interp_mode      Interpolation/addressing mode of the texture. Any of InterpMode.
    /// \param[out] output              On the \b device. Symmetrized array.
    /// \param output_pitch             Pitch, in elements, of \p output.
    /// \param shape                    Physical {fast, medium} shape, in elements, of \p texture and \p output.
    /// \param center                   Transformation center. Both \p matrix and \p symmetry_matrices operates around this center.
    /// \param shifts                   Shifts to apply.
    /// \param matrix                   Rotation/scaling to apply after the shifts.
    ///                                 For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                                 on the output array coordinates. This functions assumes \p matrix is already
    ///                                 inverted and pre-multiplies the coordinates with the matrix directly.
    /// \param[in] symmetry_matrices    On the \b device. Matrices from the get() method of Symmetry. The identity
    ///                                 matrix is implicitly considered and should not be included here. They are
    ///                                 converted to 2x2 matrices, so really they should describe a C or D symmetry.
    /// \param symmetry_count           Number of matrices. If 0, \p symmetry_matrices is not read.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    /// \note The \p texture is expected to be set with BORDER_ZERO and unnormalized coordinates.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void apply2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* output, size_t output_pitch, size2_t shape,
                          float2_t center, float2_t shifts, float22_t matrix,
                          const float33_t* symmetry_matrices, uint symmetry_count, Stream& stream);

    /// Shifts, rotate/scale and then apply the symmetry on the 2D input texture.
    /// \tparam T                       float or cfloat.
    /// \param texture                  Input texture bound to a CUDA array.
    /// \param texture_interp_mode      Interpolation/addressing mode of the texture. Any of InterpMode.
    /// \param[out] output              On the \b device. Symmetrized array.
    /// \param output_pitch             Pitch, in elements, of \p output.
    /// \param shape                    Physical {fast, medium, slow} shape, in elements, of \p texture and \p output.
    /// \param center                   Transformation center. Both \p matrix and \p symmetry_matrices operates around this center.
    /// \param shifts                   Shifts to apply.
    /// \param matrix                   Rotation/scaling to apply after the shifts.
    ///                                 For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                                 on the output array coordinates. This functions assumes \p matrix is already
    ///                                 inverted and pre-multiplies the coordinates with the matrix directly.
    /// \param[in] symmetry_matrices    On the \b device. Matrices from the get() method of Symmetry. The identity
    ///                                 matrix is implicitly considered and should not be included here.
    /// \param symmetry_count           Number of matrices. If 0, \p symmetry_matrices is not read.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    /// \note The \p texture is expected to be set with BORDER_ZERO and unnormalized coordinates.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void apply3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* output, size_t output_pitch, size3_t shape,
                          float3_t center, float3_t shifts, float33_t matrix,
                          const float33_t* symmetry_matrices, uint symmetry_count, Stream& stream);
}

// -- Symmetry - using arrays -- //
namespace noa::cuda::transform {
    /// Shifts, rotate/scale and then apply the symmetry on the 2D input array.
    /// \tparam PREFILTER       Whether or not the input(s) should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true, the
    ///                         input(s) are pre-filtered using bspline::prefilter2D().
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param input            On the \b device. Input array to transform.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param output           On the \b device. Transformed output arrays.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Physical {fast, medium} shape of \p input and \p output, in elements.
    /// \param center           Transformation center. Both \p matrix and \p symmetry operates around this center.
    /// \param shifts           Shifts to apply.
    /// \param matrix           Rotation/scaling to apply after the shifts.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This functions assumes \p matrix is already
    ///                         inverted and pre-multiplies the coordinates with the matrix directly.
    /// \param symmetry         Symmetry operator to apply after the rotation/scaling. Should be a C or D symmetry.
    /// \param interp_mode      Interpolation/filter mode. Any of InterpMode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when this function returns.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    /// \note If there's no symmetry, equivalent results can be accomplished with the more flexible affine transforms.
    ///       Similarly, if the order of the transformations is not the desired one, a solution is to first transform
    ///       the array using the various transformation functions and then apply the symmetry on the transformed
    ///       array using the symmetrize functions in "noa/gpu/cuda/transform/Symmetry.h".
    template<bool PREFILTER = true, typename T>
    NOA_HOST void apply2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                          float2_t center, float2_t shifts, float22_t matrix, Symmetry symmetry,
                          InterpMode interp_mode, Stream& stream);

    /// Shifts, rotate/scale and then apply the symmetry on the 3D input array.
    /// \tparam PREFILTER       Whether or not the input(s) should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true, the
    ///                         input(s) are pre-filtered using bspline::prefilter3D().
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param input            On the \b device. Input array to transform.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param output           On the \b device. Transformed output arrays.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Physical {fast, medium, slow} shape of \p input and \p output, in elements.
    /// \param center           Transformation center. Both \p matrix and \p symmetry operates around this center.
    /// \param shifts           Shifts to apply.
    /// \param matrix           Rotation/scaling to apply after the shifts.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This functions assumes \p matrix is already
    ///                         inverted and pre-multiplies the coordinates with the matrix directly.
    /// \param symmetry         Symmetry operator to apply after the rotation/scaling.
    /// \param interp_mode      Interpolation/filter mode. Any of InterpMode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when this function returns.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    /// \note If there's no symmetry, equivalent results can be accomplished with the more flexible affine transforms.
    ///       Similarly, if the order of the transformations is not the desired one, a solution is to first transform
    ///       the array using the various transformation functions and then apply the symmetry on the transformed
    ///       array using the symmetrize functions in "noa/gpu/cuda/transform/Symmetry.h".
    template<bool PREFILTER = true, typename T>
    NOA_HOST void apply3D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape,
                          float3_t center, float3_t shifts, float33_t matrix, Symmetry symmetry,
                          InterpMode interp_mode, Stream& stream);
}
