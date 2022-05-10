#pragma once

#include "noa/unified/Array.h"

namespace noa::signal::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_shift_v =
            traits::is_any_v<T, cfloat_t, cdouble_t> &&
            (REMAP == H2H || REMAP == H2HC || REMAP == HC2H || REMAP == HC2HC);
}

namespace noa::signal::fft {
    using Remap = noa::fft::Remap;

    /// Phase-shifts a non-redundant 2D (batched) FFT.
    /// \tparam REMAP       Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T           cfloat_t or cdouble_t.
    /// \param[in] input    2D FFT to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output  Non-redundant phase-shifted 2D FFT.
    /// \param shape        Rightmost logical shape.
    /// \param[in] shifts   Rightmost 2D phase-shift to apply.
    /// \param cutoff       Maximum output frequency to consider, in cycle/pix.
    ///                     Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                     Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    /// \note \p shifts should be a 1D contiguous row vector, with one shift per output batch.
    ///       If \p output is on the GPU, \p shifts can be on any device, including the CPU.
    ///       If \p output is on the CPU, \p shifts should be dereferencable by the CPU.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift2D(const Array<T>& input, const Array<T>& output, size4_t shape,
                 const Array<float2_t>& shifts, float cutoff = 0.5f);

    ///  Phase-shifts a non-redundant 2D (batched) FFT.
    /// \tparam REMAP       Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T           cfloat_t or cdouble_t.
    /// \param[in] input    2D FFT to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output  Non-redundant phase-shifted 2D FFT.
    /// \param shape        Rightmost logical shape.
    /// \param shift        Rightmost 2D phase-shift to apply.
    /// \param cutoff       Maximum output frequency to consider, in cycle/pix.
    ///                     Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                     Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift2D(const Array<T>& input, const Array<T>& output, size4_t shape,
                 float2_t shift, float cutoff = 0.5f);

    /// Phase-shifts a non-redundant 3D (batched) FFT transform.
    /// \tparam REMAP       Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T           cfloat_t or cdouble_t.
    /// \param[in] input    3D FFT to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output  Non-redundant phase-shifted 3D FFT.
    /// \param shape        Rightmost logical shape.
    /// \param[in] shifts   Rightmost 3D phase-shift to apply.
    /// \param cutoff       Maximum output frequency to consider, in cycle/pix.
    ///                     Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                     Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    /// \note \p shifts should be a 1D contiguous row vector, with one shift per output batch.
    ///       If \p output is on the GPU, \p shifts can be on any device, including the CPU.
    ///       If \p output is on the CPU, \p shifts should be dereferencable by the CPU.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift3D(const Array<T>& input, const Array<T>& output, size4_t shape,
                 const Array<float3_t>& shifts, float cutoff = 0.5f);

    ///  Phase-shifts a non-redundant 3D (batched) FFT.
    /// \tparam REMAP       Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T           cfloat_t or cdouble_t.
    /// \param[in] input    3D FFT to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output  Non-redundant phase-shifted 3D FFT.
    /// \param shape        Rightmost logical shape.
    /// \param shift        Rightmost 3D phase-shift to apply.
    /// \param cutoff       Maximum output frequency to consider, in cycle/pix.
    ///                     Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                     Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift3D(const Array<T>& input, const Array<T>& output, size4_t shape,
                 float3_t shift, float cutoff = 0.5f);
}

#define NOA_UNIFIED_SHIFT_
#include "noa/unified/signal/fft/Shift.inl"
#undef NOA_UNIFIED_SHIFT_
