#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/transform/Symmetry.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

// -- Using textures -- //
namespace noa::cuda::transform::fft {
    using Remap = ::noa::fft::Remap;
    using Symmetry = ::noa::transform::Symmetry;

    /// Symmetrizes a non-redundant FFT.
    /// \tparam REMAP                   Remap operation. Should be HC2HC or HC2H.
    /// \tparam T                       float, cfloat_t.
    /// \param texture                  Non-redundant FFT to symmetrize. Should use unnormalized coordinates.
    /// \param texture_interp_mode      Interpolation/filtering mode of \p texture. Cubic modes are currently not supported.
    /// \param[out] output              On the \b device. Non-redundant symmetrized FFT.
    /// \param output_pitch             Pitch, in \p T elements, of \p output.
    /// \param shape                    Logical {fast, medium} shape, in \p T elements, of \p texture and \p output.
    /// \param[in] symmetry_matrices    On the \b device. Symmetry matrices.
    /// \param symmetry_count           Number of symmetry matrices.
    /// \param shift                    2D real-space shift to apply (as phase shift) after the transformation.
    /// \param max_frequency            Maximum frequency to consider, in cycle/pix.
    /// \param normalize                Whether \p output should be normalized to have the same range as the input data.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    NOA_HOST void symmetrize2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                               T* output, size_t output_pitch, size2_t shape,
                               const float33_t* symmetry_matrices, size_t symmetry_count, float2_t shift,
                               float max_frequency, bool normalize, Stream& stream);

    /// Symmetrizes a non-redundant FFT.
    /// \tparam REMAP                   Remap operation. Should be HC2HC or HC2H.
    /// \tparam T                       float, cfloat_t.
    /// \param texture                  Non-redundant FFT to symmetrize. Should use unnormalized coordinates.
    /// \param texture_interp_mode      Interpolation/filtering mode of \p texture. Cubic modes are currently not supported.
    /// \param[out] output              On the \b device. Non-redundant symmetrized FFT.
    /// \param output_pitch             Pitch, in \p T elements, of \p output.
    /// \param shape                    Logical {fast, medium, slow} shape, in \p T elements, of \p texture and \p output.
    /// \param[in] symmetry_matrices    On the \b device. Symmetry matrices.
    /// \param symmetry_count           Number of symmetry matrices.
    /// \param shift                    3D real-space shift to apply (as phase shift) after the transformation.
    /// \param max_frequency            Maximum frequency to consider, in cycle/pix.
    /// \param normalize                Whether \p output should be normalized to have the same range as the input data.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    NOA_HOST void symmetrize3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                               T* output, size_t output_pitch, size3_t shape,
                               const float33_t* symmetry_matrices, size_t symmetry_count, float3_t shift,
                               float max_frequency, bool normalize, Stream& stream);
}

// -- Using arrays -- //
namespace noa::cuda::transform::fft {
    /// Symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to symmetrize.
    /// \param input_pitch      Pitch, in \p T elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant symmetrized FFT.
    /// \param output_pitch     Pitch, in \p T elements, of \p output.
    /// \param shape            Logical {fast, medium} shape, in \p T elements, of \p input and \p output.
    /// \param[in] symmetry     Symmetry operator to apply.
    /// \param shift            2D real-space shift to apply (as phase shift) after the transformation.
    ///                         If \p T is real, it is ignored.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are left unchanged.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note \p input can be equal to \p output.
    template<Remap REMAP, typename T>
    NOA_HOST void symmetrize2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                               const Symmetry& symmetry, float2_t shift,
                               float max_frequency, InterpMode interp_mode, bool normalize, Stream& stream);

    /// Symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to symmetrize.
    /// \param input_pitch      Pitch, in \p T elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant symmetrized FFT.
    /// \param output_pitch     Pitch, in \p T elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape, in \p T elements, of \p input and \p output.
    /// \param[in] symmetry     Symmetry operator to apply.
    /// \param shift            3D real-space shift to apply (as phase shift) after the transformation.
    ///                         If \p T is real, it is ignored.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are left unchanged.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note \p input can be equal to \p output.
    template<Remap REMAP, typename T>
    NOA_HOST void symmetrize3D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape,
                               const Symmetry& symmetry, float3_t shift,
                               float max_frequency, InterpMode interp_mode, bool normalize, Stream& stream);
}
