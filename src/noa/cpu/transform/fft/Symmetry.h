#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/transform/fft/Apply.h"

namespace noa::cpu::transform::fft {
    /// Symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to symmetrize.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant symmetrized FFT.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium} shape, in \p T elements, of \p input and \p output.
    /// \param[in] symmetry     Symmetry operator to apply.
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
    NOA_IH void symmetrize2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                             const Symmetry& symmetry, float2_t shift, float cutoff, InterpMode interp_mode,
                             bool normalize, Stream& stream) {
        if (!symmetry.count())
            return memory::copy(input, input_pitch, output, output_pitch, shapeFFT(shape), 1, stream);
        apply2D<REMAP>(input, input_pitch, output, output_pitch, shape, float22_t(), symmetry,
                       shift, cutoff, interp_mode, normalize, stream);
    }

    /// Symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to symmetrize.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant symmetrized FFT.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape, in \p T elements, of \p input and \p output.
    /// \param[in] symmetry     Symmetry operator to apply.
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
    NOA_IH void symmetrize3D(const T* input, size2_t input_pitch, T* output, size2_t output_pitch, size3_t shape,
                             const Symmetry& symmetry, float3_t shift, float cutoff, InterpMode interp_mode,
                             bool normalize, Stream& stream) {
        if (!symmetry.count())
            return memory::copy(input, input_pitch, output, output_pitch, shapeFFT(shape), 1, stream);
        apply2D<REMAP>(input, input_pitch, output, output_pitch, shape, float33_t(), symmetry,
                       shift, cutoff, interp_mode, normalize, stream);
    }
}
