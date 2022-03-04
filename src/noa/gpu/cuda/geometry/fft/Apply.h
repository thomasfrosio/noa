#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/transform/Symmetry.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

// -- Using textures -- //
namespace noa::cuda::transform::fft {
    using Remap = noa::fft::Remap;

    /// Rotates/scales a non-redundant FFT.
    /// \tparam REMAP               Remap operation. Should be HC2HC or HC2H.
    /// \tparam T                   float, cfloat_t.
    /// \param texture              Non-redundant FFT to transform. Should use unnormalized coordinates.
    /// \param texture_interp_mode  Interpolation/filtering mode of \p texture. Cubic modes are currently not supported.
    /// \param[out] outputs         On the \b device. Non-redundant transformed FFT. One per transformation.
    /// \param output_pitch         Pitch, in \p T elements, of \p outputs.
    /// \param shape                Logical {fast, medium} shape, in \p T elements, of \p texture and \p outputs.
    /// \param[in] transforms       On the \b device. 2x2 inverse rotation/scaling matrix.
    /// \param[in] shifts           On the \b device. One per transformation. If nullptr or if \p T is real, it is ignored.
    ///                             2D real-space shift to apply (as phase shift) after the transformation.
    /// \param nb_transforms        Number of transformations.
    /// \param max_frequency        Maximum output frequency to consider, in cycle/pix.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    NOA_HOST void apply2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* outputs, size_t output_pitch, size2_t shape,
                          const float22_t* transforms, const float2_t* shifts, size_t nb_transforms,
                          float max_frequency, Stream& stream);

    /// Rotates/scales a non-redundant FFT.
    /// Overload applying the same shift to all transforms.
    template<Remap REMAP, typename T>
    NOA_HOST void apply2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* outputs, size_t output_pitch, size2_t shape,
                          const float22_t* transforms, float2_t shift, size_t nb_transforms,
                          float max_frequency, Stream& stream);

    /// Rotates/scales a non-redundant FFT.
    /// Overload for a single transform.
    template<Remap REMAP, typename T>
    NOA_HOST void apply2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* output, size_t output_pitch, size2_t shape,
                          float22_t transform, float2_t shift,
                          float max_frequency, Stream& stream);

    /// Rotates/scales a non-redundant FFT.
    /// \tparam REMAP               Remap operation. Should be HC2HC or HC2H.
    /// \tparam T                   float, cfloat_t.
    /// \param texture              Non-redundant FFT to transform. Should use unnormalized coordinates.
    /// \param texture_interp_mode  Interpolation/filtering mode of \p texture. Cubic modes are currently not supported.
    /// \param[out] outputs         On the \b device. Non-redundant transformed FFT. One per transformation.
    /// \param output_pitch         Pitch, in \p T elements, of \p outputs.
    /// \param shape                Logical {fast, medium, slow} shape, in \p T elements, of \p texture and \p outputs.
    /// \param[in] transforms       On the \b device. 3x3 inverse rotation/scaling matrix.
    /// \param[in] shifts           On the \b device. One per transformation. If nullptr or if \p T is real, it is ignored.
    ///                             3D real-space shift to apply (as phase shift) after the transformation.
    /// \param nb_transforms        Number of transformations.
    /// \param max_frequency        Maximum output frequency to consider, in cycle/pix.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    NOA_HOST void apply3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* outputs, size_t output_pitch, size3_t shape,
                          const float33_t* transforms, const float3_t* shifts, size_t nb_transforms,
                          float max_frequency, Stream& stream);

    /// Rotates/scales a non-redundant FFT.
    /// Overload applying the same shift to all transforms.
    template<Remap REMAP, typename T>
    NOA_HOST void apply3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* outputs, size_t output_pitch, size3_t shape,
                          const float33_t* transforms, float3_t shift, size_t nb_transforms,
                          float max_frequency, Stream& stream);

    /// Rotates/scales a non-redundant FFT.
    /// Overload for a single transform.
    template<Remap REMAP, typename T>
    NOA_HOST void apply3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* output, size_t output_pitch, size3_t shape,
                          float33_t transform, float3_t shift,
                          float max_frequency, Stream& stream);
}

// -- Using arrays -- //
namespace noa::cuda::transform::fft {
    /// Rotates/scales a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        On the \b host or \b device. Non-redundant FFT to transform.
    /// \param input_pitch      Pitch, in \p T elements, of \p input.
    /// \param[out] outputs     On the \b device. Non-redundant transformed FFT. One per transformation.
    /// \param output_pitch     Pitch, in \p T elements, of \p outputs.
    /// \param shape            Logical {fast, medium} shape, in \p T elements, of \p input and \p outputs.
    /// \param[in] transforms   On the \b device. 2x2 inverse rotation/scaling matrix.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transforms are already
    ///                         inverted and pre-multiplies the coordinates with these matrices directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       On the \b device. One per transformation. If nullptr or if \p T is real, it is ignored.
    ///                         2D real-space shift to apply (as phase shift) after the transformation.
    /// \param nb_transforms    Number of transformations.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note \p input can be equal to \p outputs.
    template<Remap REMAP, typename T>
    NOA_HOST void apply2D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size2_t shape,
                          const float22_t* transforms, const float2_t* shifts, size_t nb_transforms,
                          float max_frequency, InterpMode interp_mode, Stream& stream);

    /// Rotates/scales a non-redundant FFT.
    /// Overload applying the same shift to all transforms.
    template<Remap REMAP, typename T>
    NOA_HOST void apply2D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size2_t shape,
                          const float22_t* transforms, float2_t shift, size_t nb_transforms,
                          float max_frequency, InterpMode interp_mode, Stream& stream);

    /// Rotates/scales a non-redundant FFT.
    /// Overload for a single transform.
    template<Remap REMAP, typename T>
    NOA_HOST void apply2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                          float22_t transform, float2_t shift,
                          float max_frequency, InterpMode interp_mode, Stream& stream);

    /// Rotates/scales a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        On the \b host or \b device. Non-redundant FFT to transform.
    /// \param input_pitch      Pitch, in \p T elements, of \p input.
    /// \param[out] outputs     On the \b device. Non-redundant transformed FFT. One per transformation.
    /// \param output_pitch     Pitch, in \p T elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape, in \p T elements, of \p input and \p outputs.
    /// \param[in] transforms   On the \b device. 3x3 inverse rotation/scaling matrix.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transforms are already
    ///                         inverted and pre-multiplies the coordinates with these matrices directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       On the \b device. One per transformation. If nullptr or if \p T is real, it is ignored.
    ///                         3D real-space shift to apply (as phase shift) after the transformation.
    /// \param nb_transforms    Number of transformations.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note \p input can be equal to \p outputs.
    template<Remap REMAP, typename T>
    NOA_HOST void apply3D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size3_t shape,
                          const float33_t* transforms, const float3_t* shifts, size_t nb_transforms,
                          float max_frequency, InterpMode interp_mode, Stream& stream);

    /// Rotates/scales a non-redundant FFT.
    /// Overload applying the same shift to all transforms.
    template<Remap REMAP, typename T>
    NOA_HOST void apply3D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size3_t shape,
                          const float33_t* transforms, float3_t shift, size_t nb_transforms,
                          float max_frequency, InterpMode interp_mode, Stream& stream);

    /// Rotates/scales a non-redundant FFT.
    /// Overload for a single transform.
    template<Remap REMAP, typename T>
    NOA_HOST void apply3D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape,
                          float33_t transform, float3_t shift,
                          float max_frequency, InterpMode interp_mode, Stream& stream);
}

// --- With symmetry, using textures ---
namespace noa::cuda::transform::fft {
    using Symmetry = ::noa::transform::Symmetry;

    /// Rotates/scales and then symmetrizes a non-redundant FFT.
    /// \tparam REMAP                   Remap operation. Should be HC2HC or HC2H.
    /// \tparam T                       float, cfloat_t.
    /// \param texture                  Non-redundant FFT to transform. Should use unnormalized coordinates.
    /// \param texture_interp_mode      Interpolation/filtering mode of \p texture. Cubic modes are currently not supported.
    /// \param[out] output              On the \b device. Non-redundant transformed FFT.
    /// \param output_pitch             Pitch, in \p T elements, of \p output.
    /// \param shape                    Logical {fast, medium} shape, in \p T elements, of \p texture and \p output.
    /// \param[in] transform            On the \b device. 2x2 inverse rotation/scaling matrix.
    /// \param[in] symmetry_matrices    On the \b device. Symmetry matrices.
    /// \param symmetry_count           Number of symmetry matrices.
    /// \param shift                    2D real-space shift to apply (as phase shift) after the transformation.
    /// \param max_frequency            Maximum frequency to consider, in cycle/pix.
    /// \param normalize                Whether \p output should be normalized to have the same range as the input data.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    NOA_HOST void apply2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* output, size_t output_pitch, size2_t shape,
                          float22_t transform, const float33_t* symmetry_matrices, size_t symmetry_count,
                          float2_t shift, float max_frequency, bool normalize, Stream& stream);

    /// Rotates/scales and then symmetrizes a non-redundant FFT.
    /// \tparam REMAP                   Remap operation. Should be HC2HC or HC2H.
    /// \tparam T                       float, cfloat_t.
    /// \param texture                  Non-redundant FFT to transform. Should use unnormalized coordinates.
    /// \param texture_interp_mode      Interpolation/filtering mode of \p texture. Cubic modes are currently not supported.
    /// \param[out] output              On the \b device. Non-redundant transformed FFT.
    /// \param output_pitch             Pitch, in \p T elements, of \p outputs.
    /// \param shape                    Logical {fast, medium, slow} shape, in \p T elements, of \p texture and \p output.
    /// \param[in] transform            On the \b device. 3x3 inverse rotation/scaling matrix.
    /// \param[in] symmetry_matrices    On the \b device. Symmetry matrices.
    /// \param symmetry_count           Number of symmetry matrices.
    /// \param shift                    3D real-space shift to apply (as phase shift) after the transformation.
    /// \param max_frequency            Maximum frequency to consider, in cycle/pix.
    /// \param normalize                Whether \p output should be normalized to have the same range as the input data.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    NOA_HOST void apply3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                          T* output, size_t output_pitch, size3_t shape,
                          float33_t transform, const float33_t* symmetry_matrices, size_t symmetry_count,
                          float3_t shift, float max_frequency, bool normalize, Stream& stream);
}

// --- With symmetry, using arrays ---
namespace noa::cuda::transform::fft {
    /// Rotates/scales and then symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        On the \b host or \b device. Non-redundant FFT to transform.
    /// \param input_pitch      Pitch, in \p T elements, of \p input.
    /// \param[out] output      On the \b device. Non-redundant transformed FFT.
    /// \param output_pitch     Pitch, in \p T elements, of \p output.
    /// \param shape            Logical {fast, medium} shape, in \p T elements, of \p texture and \p output.
    /// \param[in] transform    On the \b device. 2x2 inverse rotation/scaling matrix.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transform is already
    ///                         inverted and pre-multiplies the coordinates with this matrix directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in
    ///                         real space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param shift            2D real-space shift to apply (as phase shift) after the transformation.
    ///                         If \p T is real, it is ignored.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note \p input can be equal to \p outputs.
    template<Remap REMAP, typename T>
    NOA_HOST void apply2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                          float22_t transform, const Symmetry& symmetry, float2_t shift,
                          float max_frequency, InterpMode interp_mode, bool normalize, Stream& stream);

    /// Rotates/scales and then symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        On the \b host or \b device. Non-redundant FFT to transform.
    /// \param input_pitch      Pitch, in \p T elements, of \p input.
    /// \param[out] outputs     On the \b device. Non-redundant transformed FFT. One per transformation.
    /// \param output_pitch     Pitch, in \p T elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape, in \p T elements, of \p texture and \p outputs.
    /// \param[in] transform    On the \b device. 3x3 inverse rotation/scaling matrix.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transform is already
    ///                         inverted and pre-multiplies the coordinates with this matrix directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in
    ///                         real space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] shift        On the \b device. If \p T is real, it is ignored.
    ///                         3D real-space shift to apply (as phase shift) after the transformation.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note \p input can be equal to \p outputs.
    template<Remap REMAP, typename T>
    NOA_HOST void apply3D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape,
                          float33_t transform, const Symmetry& symmetry, float3_t shift,
                          float max_frequency, InterpMode interp_mode, bool normalize, Stream& stream);
}
