#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

// TODO(TF) Add all remaining layouts

namespace noa::cpu::transform::fft {
    using Remap = noa::fft::Remap;

    /// Phase-shifts the non-redundant FFT transform.
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T               cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Non-redundant FFT to phase-shift. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Non-redundant phase-shifted FFT. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium} shape of \p inputs and \p outputs.
    /// \param[in] shifts       On the \b host. 2D real-space shift to apply (as phase) to the transform(s).
    /// \param batches          Number of contiguous batches to shift.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p inputs and \p outputs can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename T>
    NOA_HOST void shift2D(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch, size2_t shape,
                          const float2_t* shifts, size_t batches, Stream& stream);

    /// Phase-shifts the non-redundant FFT transform.
    /// Overload taking the same shift for all batches.
    template<Remap REMAP, typename T>
    NOA_HOST void shift2D(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch, size2_t shape,
                          float2_t shift, size_t batches, Stream& stream);

    /// Phase-shifts the non-redundant FFT transform.
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T               cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Non-redundant FFT to phase-shift. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Non-redundant phase-shifted FFT. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param[in] shifts       On the \b host. 3D real-space shift to apply (as phase) to the transform(s).
    /// \param batches          Number of contiguous batches to shift.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p inputs and \p outputs can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename T>
    NOA_HOST void shift3D(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                          const float3_t* shifts, size_t batches, Stream& stream);

    /// Phase-shifts the non-redundant FFT transform.
    /// Overload taking the same shift for all batches.
    template<Remap REMAP, typename T>
    NOA_HOST void shift3D(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                          float3_t shift, size_t batches, Stream& stream);
}
