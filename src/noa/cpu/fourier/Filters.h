#include "noa/Definitions.h"
#include "noa/Types.h"

/*
 * These filters are all using a raised-cosine (Hann) window. The cutoffs and window width are specified in fractional
 * reciprocal lattice units from 0 to 0.5. Anything outside of this range is still valid).
 *
 * For instance, given an 64x64 image with a pixel size of 1.4 A/pixel. To lowpass filter this image at a resolution
 * of 8 A, the frequency cutoff should be 1.4 / 8 = 0.175. Note that multiplying this normalized value by the
 * dimension of the image gives us the number of oscillations in the real-space image at this frequency (or the
 * resolution shell in Fourier space), i.e. 0.175 * 64 = 22.4. Naturally, the Nyquist frequency is at 0.5 in fractional
 * reciprocal lattice units and, for this example, at the 64th shell.
 */

namespace Noa::Fourier {
    /**
     * Applies a lowpass filter to the input array(s).
     * @tparam T            float, double, cfloat_t, cdouble_t.
     * @param inputs        Input non-redundant, non-centered transforms. One per batch.
     * @param outputs       Output non-redundant, non-centered transforms. One per batch. Can be the same as @a inputs.
     * @param shape         Logical {fast, medium, slow} shape.
     * @param freq_cutoff   Frequency cutoff, from 0 to 0.5. At this frequency, the pass starts to roll-off.
     * @param freq_width    Width of the Hann window, in frequencies, from 0 to 0.5.
     * @param batches       Number of batches.
     */
    template<typename T>
    NOA_HOST void lowpass(T* inputs, T* outputs, size3_t shape, float freq_cutoff, float freq_width, uint batches);

    /**
     * Computes a lowpass filter.
     * @tparam T                float, double.
     * @param output_lowpass    Output lowpass filter. At least getElementsFFT(shape) * sizeof(T) bytes.
     * @param shape             Logical {fast, medium, slow} shape.
     * @param freq_cutoff       Frequency cutoff, from 0 to 0.5. At this frequency, the pass starts to roll-off.
     * @param freq_width        Width of the Hann window, in frequencies, from 0 to 0.5.
     */
    template<typename T>
    NOA_HOST void lowpass(T* output_lowpass, size3_t shape, float freq_cutoff, float freq_width);

    /**
     * Applies a highpass filter to the input array(s).
     * @tparam T            float, double, cfloat_t, cdouble_t.
     * @param inputs        Input non-redundant, non-centered transforms. One per batch.
     * @param outputs       Output non-redundant, non-centered transforms. One per batch. Can be the same as @a inputs.
     * @param shape         Logical {fast, medium, slow} shape.
     * @param freq_cutoff   Frequency cutoff, from 0 to 0.5. At this frequency, the pass is fully recovered.
     * @param freq_width    Width of the Hann window, in frequencies, from 0 to 0.5.
     * @param batches       Number of batches.
     */
    template<typename T>
    NOA_HOST void highpass(T* inputs, T* outputs, size3_t shape, float freq_cutoff, float freq_width, uint batches);

    /**
     * Computes a highpass filter.
     * @tparam T                float, double.
     * @param output_highpass   Output lowpass filter. At least getElementsFFT(shape) * sizeof(T) bytes.
     * @param shape             Logical {fast, medium, slow} shape.
     * @param freq_cutoff       Frequency cutoff, from 0 to 0.5. At this frequency, the pass is fully recovered.
     * @param freq_width        Width of the Hann window, in frequencies, from 0 to 0.5.
     */
    template<typename T>
    NOA_HOST void highpass(T* output_highpass, size3_t shape, float freq_cutoff, float freq_width);

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
     */
    template<typename T>
    NOA_HOST void bandpass(T* inputs, T* outputs, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                           float freq_width_1, float freq_width_2, uint batches);

    /**
     * Computes a bandpass filter.
     * @tparam T                float, double.
     * @param output_bandpass   Output bandpass filter. At least getElementsFFT(shape) * sizeof(T) bytes.
     * @param shape             Logical {fast, medium, slow} shape.
     * @param freq_cutoff_1     First frequency cutoff, from 0 to 0.5. At this frequency, the pass is fully recovered.
     * @param freq_cutoff_2     Second frequency cutoff, from 0 to 0.5. At this frequency, the pass starts to roll-off.
     * @param freq_width_1      Width of the Hann window between 0 and @a freq_cutoff_1, in frequencies, from 0 to 0.5.
     * @param freq_width_2      Width of the Hann window between @a freq_cutoff_2 and 0.5, in frequencies, from 0 to 0.5.
     */
    template<typename T>
    NOA_HOST void bandpass(T* output_bandpass, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                           float freq_width_1, float freq_width_2);
}
