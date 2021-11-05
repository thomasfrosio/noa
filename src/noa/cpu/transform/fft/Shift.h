#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu::transform::fft {
    using Remap = noa::fft::Remap;

    /// Phase-shifts the non-redundant FFT transform.
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T               cfloat_t, cdouble_t.
    /// \param inputs           On the \b host. Non-redundant FFT to phase-shift. One per batch.
    /// \param outputs          On the \b host. Non-redundant phase-shifted FFT. One per batch.
    /// \param shape            Logical {fast, medium} shape of \p inputs and \p outputs.
    /// \param shifts           2D real-space shift to apply (as phase) to the transform(s).
    /// \param batches          Number of contiguous batches to shift.
    /// \note \p inputs and \p outputs can overlap if no remapping is done, i.e. H2H or H2HC.
    template<Remap REMAP = Remap::H2H, typename T>
    NOA_HOST void shift2D(const T* inputs, T* outputs, size2_t shape, const float2_t* shifts, size_t batches);

    /// Phase-shifts the non-redundant FFT transform.
    /// Overload taking the same shift for all batches.
    template<Remap REMAP = Remap::H2H, typename T>
    NOA_HOST void shift2D(const T* inputs, T* outputs, size2_t shape, float2_t shift, size_t batches);

    /// Phase-shifts the non-redundant FFT transform.
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T               cfloat_t, cdouble_t.
    /// \param inputs           On the \b host. Non-redundant FFT to phase-shift. One per batch.
    /// \param outputs          On the \b host. Non-redundant phase-shifted FFT. One per batch.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param shifts           3D real-space shift to apply (as phase) to the transform(s).
    /// \param batches          Number of contiguous batches to shift.
    /// \note \p inputs and \p outputs can overlap if no remapping is done, i.e. H2H or H2HC.
    template<Remap REMAP = Remap::H2H, typename T>
    NOA_HOST void shift3D(const T* inputs, T* outputs, size3_t shape, const float3_t* shifts, size_t batches);

    /// Phase-shifts the non-redundant FFT transform.
    /// Overload taking the same shift for all batches.
    template<Remap REMAP = Remap::H2H, typename T>
    NOA_HOST void shift3D(const T* inputs, T* outputs, size3_t shape, float3_t shift, size_t batches);
}
