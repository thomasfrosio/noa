#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/transform/Symmetry.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::transform::fft::details {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename T, typename U, typename V, typename W>
    void applyND(const T* inputs, U input_pitch, T* outputs, U output_pitch, U shape,
                 const V* transforms, const W* shifts, size_t batches,
                 float cutoff, InterpMode interp_mode, Stream& stream);
}

namespace noa::cpu::transform::fft {
    using Remap = noa::fft::Remap;

    /// Rotates/scales a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Non-redundant FFT to transform. One per transformation.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Non-redundant transformed FFT. One per transformation.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium} shape, in \p T elements, of \p inputs and \p outputs.
    /// \param[in] transforms   On the \b host. 2x2 inverse rotation/scaling matrix. One per transformation.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transforms are already
    ///                         inverted and pre-multiplies the coordinates with these matrices directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       On the \b host. One per transformation. If nullptr or if \p T is real, it is ignored.
    ///                         2D real-space shift to apply (as phase shift) after the transformation.
    /// \param batches          Number of transformations.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p inputs and \p outputs should not overlap.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    NOA_IH void apply2D(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch, size2_t shape,
                        const float22_t* transforms, const float2_t* shifts, size_t batches,
                        float cutoff, InterpMode interp_mode, Stream& stream) {
        details::applyND<REMAP>(inputs, input_pitch, outputs, output_pitch, shape, transforms, shifts, batches,
                                cutoff, interp_mode, stream);
    }

    /// Rotates/scales a non-redundant FFT.
    /// Overload for one single transformation.
    template<Remap REMAP, typename T>
    NOA_IH void apply2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                        float22_t transform, float2_t shift,
                        float cutoff, InterpMode interp_mode, Stream& stream) {
        apply2D<REMAP>(input, {input_pitch, 0}, output, {output_pitch, 0}, shape,
                       &transform, &shift, 1, cutoff, interp_mode, stream);
    }

    /// Rotates/scales a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Non-redundant FFT to transform. One per transformation.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Non-redundant transformed FFT. One per transformation.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape, in \p T elements, of \p inputs and \p outputs.
    /// \param[in] transforms   On the \b host. 3x3 inverse rotation/scaling matrix. One per transformation.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transforms are already
    ///                         inverted and pre-multiplies the coordinates with these matrices directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       On the \b host. One per transformation. If nullptr or if \p T is real, it is ignored.
    ///                         3D real-space shift to apply (as phase shift) after the transformation.
    /// \param batches          Number of transformations.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p inputs and \p outputs should not overlap.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    NOA_IH void apply3D(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                        const float33_t* transforms, const float3_t* shifts, size_t batches,
                        float cutoff, InterpMode interp_mode, Stream& stream) {
        details::applyND<REMAP>(inputs, input_pitch, outputs, output_pitch, shape, transforms, shifts, batches,
                                cutoff, interp_mode, stream);
    }

    /// Rotates/scales a non-redundant FFT.
    /// Overload for one single transformation.
    template<Remap REMAP, typename T>
    NOA_IH void apply3D(const T* input, size2_t input_pitch, T* output, size2_t output_pitch, size3_t shape,
                        float33_t transform, float3_t shift,
                        float cutoff, InterpMode interp_mode, Stream& stream) {
        apply3D<REMAP>(input, {input_pitch, 0}, output, {output_pitch, 0}, shape,
                       &transform, &shift, 1, cutoff, interp_mode, stream);
    }
}

namespace noa::cpu::transform::fft {
    using Symmetry = ::noa::transform::Symmetry;

    /// Rotates/scales and then symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to transform and symmetrize.
    /// \param[out] output      On the \b host. Non-redundant transformed and symmetrized FFT.
    /// \param shape            Logical {fast, medium} shape, in \p T elements, of \p input and \p output.
    /// \param transform        On the \b host. 2x2 inverse rotation/scaling matrix.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transform is already
    ///                         inverted and pre-multiplies the coordinates with this matrix directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param shift            2D real-space shift to apply (as phase shift) after the transformation and symmetry.
    ///                         If \p T is real, it is ignored.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    /// \todo ADD TESTS!
    template<Remap REMAP, typename T>
    NOA_HOST void apply2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                          float22_t transform, const Symmetry& symmetry, float2_t shift,
                          float cutoff, InterpMode interp_mode, bool normalize, Stream& stream);

    /// Rotates/scales and then symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to transform and symmetrize.
    /// \param[out] output      On the \b host. Non-redundant transformed and symmetrized FFT.
    /// \param shape            Logical {fast, medium, slow} shape, in \p T elements, of \p input and \p output.
    /// \param transform        On the \b host. 3x3 inverse rotation/scaling matrix.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This function assumes \p transform is already
    ///                         inverted and pre-multiplies the coordinates with this matrix directly.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param shift            3D real-space shift to apply (as phase shift) after the transformation and symmetry.
    ///                         If \p T is real, it is ignored.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    /// \todo ADD TESTS!
    template<Remap REMAP, typename T>
    NOA_HOST void apply3D(const T* input, size2_t input_pitch, T* output, size2_t output_pitch, size3_t shape,
                          float33_t transform, const Symmetry& symmetry, float3_t shift,
                          float cutoff, InterpMode interp_mode, bool normalize, Stream& stream);
}
