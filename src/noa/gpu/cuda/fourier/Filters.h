#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Fourier {
    /**
     * Applies a lowpass filter to the input array(s).
     * @tparam T            float, double, cfloat_t, cdouble_t.
     * @param inputs        Input non-redundant, non-centered transforms. One per batch.
     * @param outputs       Output non-redundant, non-centered transforms. One per batch. Can be the same as @a inputs.
     * @param shape         Logical {fast, medium, slow} shape.
     * @param freq_cutoff   Frequency cutoff, from 0 to 0.5. At this frequency, the pass starts to roll-off.
     * @param freq_width    Width of the Hann window, in frequencies, from 0 to 0.5.
     * @param batches       Number of batches.
     *
     * @warning This function is asynchronous relative to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void lowpass(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs, size3_t shape,
                          float freq_cutoff, float freq_width, uint batches, Stream& stream);

    /// @see This is version for contiguous layouts. See version with pitch for more details.
    /// @warning This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void lowpass(T* inputs, T* outputs, size3_t shape, float freq_cutoff, float freq_width,
                        uint batches, Stream& stream) {
        size_t pitch = shape.x / 2U + 1U;
        lowpass(inputs, pitch, outputs, pitch, shape, freq_cutoff, freq_width, batches, stream);
    }

    /**
     * Computes a lowpass filter.
     * @tparam T                float, double.
     * @param output_lowpass    Output lowpass filter. At least getElementsFFT(shape) * sizeof(T) bytes.
     * @param shape             Logical {fast, medium, slow} shape.
     * @param freq_cutoff       Frequency cutoff, from 0 to 0.5. At this frequency, the pass starts to roll-off.
     * @param freq_width        Width of the Hann window, in frequencies, from 0 to 0.5.
     *
     * @warning This function is asynchronous relative to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void lowpass(T* output_lowpass, size_t pitch_output_lowpass, size3_t shape,
                          float freq_cutoff, float freq_width, Stream& stream);

    /// @see This is version for contiguous layouts. See version with pitch for more details.
    /// @warning This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void lowpass(T* output_lowpass, size3_t shape, float freq_cutoff, float freq_width, Stream& stream) {
        lowpass(output_lowpass, shape.x / 2U + 1U, shape, freq_cutoff, freq_width, stream);
    }

    /**
     * Applies a highpass filter to the input array(s).
     * @tparam T            float, double, cfloat_t, cdouble_t.
     * @param inputs        Input non-redundant, non-centered transforms. One per batch.
     * @param outputs       Output non-redundant, non-centered transforms. One per batch. Can be the same as @a inputs.
     * @param shape         Logical {fast, medium, slow} shape.
     * @param freq_cutoff   Frequency cutoff, from 0 to 0.5. At this frequency, the pass is fully recovered.
     * @param freq_width    Width of the Hann window, in frequencies, from 0 to 0.5.
     * @param batches       Number of batches.
     *
     * @warning This function is asynchronous relative to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void highpass(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs, size3_t shape,
                           float freq_cutoff, float freq_width, uint batches, Stream& stream);

    /// @see This is version for contiguous layouts. See version with pitch for more details.
    /// @warning This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void highpass(T* inputs, T* outputs, size3_t shape, float freq_cutoff, float freq_width,
                         uint batches, Stream& stream) {
        size_t pitch = shape.x / 2U + 1U;
        highpass(inputs, pitch, outputs, pitch, shape, freq_cutoff, freq_width, batches, stream);
    }

    /**
     * Computes a highpass filter.
     * @tparam T                float, double.
     * @param output_highpass   Output lowpass filter. At least getElementsFFT(shape) * sizeof(T) bytes.
     * @param shape             Logical {fast, medium, slow} shape.
     * @param freq_cutoff       Frequency cutoff, from 0 to 0.5. At this frequency, the pass is fully recovered.
     * @param freq_width        Width of the Hann window, in frequencies, from 0 to 0.5.
     *
     * @warning This function is asynchronous relative to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void highpass(T* output_highpass, size_t pitch_output_highpass, size3_t shape,
                           float freq_cutoff, float freq_width, Stream& stream);

    /// @see This is version for contiguous layouts. See version with pitch for more details.
    /// @warning This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void highpass(T* output_highpass, size3_t shape, float freq_cutoff, float freq_width, Stream& stream) {
        highpass(output_highpass, shape.x / 2U + 1U, shape, freq_cutoff, freq_width, stream);
    }

    /**
     * Applies a bandpass filter to the input array(s).
     * @tparam T            float, double, cfloat_t, cdouble_t.
     * @param inputs        Input non-redundant, non-centered transforms. One per batch.
     * @param outputs       Output non-redundant, non-centered transforms. One per batch. Can be the same as @a inputs.
     * @param shape         Logical {fast, medium, slow} shape.
     * @param freq_cutoff_1 First frequency cutoff, from 0 to 0.5. At this frequency, the pass is fully recovered.
     * @param freq_cutoff_2 Second frequency cutoff, from 0 to 0.5. At this frequency, the pass starts to roll-off.
     * @param freq_width_1  Width of the Hann window between 0 and @a freq_cutoff_1, in frequencies, from 0 to 0.5.
     * @param freq_width_2  Width of the Hann window between @a freq_cutoff_2 and 0.5, in frequencies, from 0 to 0.5.
     * @param batches       Number of batches.
     *
     * @warning This function is asynchronous relative to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void bandpass(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs, size3_t shape,
                           float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2,
                           uint batches, Stream& stream);

    /// @see This is version for contiguous layouts. See version with pitch for more details.
    /// @warning This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void bandpass(T* inputs, T* outputs, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                         float freq_width_1, float freq_width_2, uint batches, Stream& stream) {
        size_t pitch = shape.x / 2U + 1U;
        bandpass(inputs, pitch, outputs, pitch, shape,
                 freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2, batches, stream);
    }

    /**
     * Computes a bandpass filter.
     * @tparam T                float, double.
     * @param output_bandpass   Output bandpass filter. At least getElementsFFT(shape) * sizeof(T) bytes.
     * @param shape             Logical {fast, medium, slow} shape.
     * @param freq_cutoff_1     First frequency cutoff, from 0 to 0.5. At this frequency, the pass is fully recovered.
     * @param freq_cutoff_2     Second frequency cutoff, from 0 to 0.5. At this frequency, the pass starts to roll-off.
     * @param freq_width_1      Width of the Hann window between 0 and @a freq_cutoff_1, in frequencies, from 0 to 0.5.
     * @param freq_width_2      Width of the Hann window between @a freq_cutoff_2 and 0.5, in frequencies, from 0 to 0.5.
     *
     * @warning This function is asynchronous relative to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void bandpass(T* output_bandpass, size_t pitch_output_bandpass, size3_t shape,
                           float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2,
                           Stream& stream);

    /// @see This is version for contiguous layouts. See version with pitch for more details.
    /// @warning This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void bandpass(T* output_bandpass, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                         float freq_width_1, float freq_width_2, Stream& stream) {
        bandpass(output_bandpass, shape.x / 2U + 1U, shape,
                 freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2, stream);
    }
}
