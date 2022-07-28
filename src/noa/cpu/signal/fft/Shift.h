#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

// TODO(TF) Add all remaining layouts

namespace noa::cpu::signal::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_shift_v =
            traits::is_any_v<T, cfloat_t, cdouble_t> &&
            (REMAP == H2H || REMAP == H2HC || REMAP == HC2H || REMAP == HC2HC);
}

namespace noa::cpu::signal::fft {
    using Remap = noa::fft::Remap;

    /// Phase-shifts a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T               cfloat_t or cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant 2D FFT to phase-shift.
    ///                         If nullptr, it is ignored and the phase-shifts are saved in \p output.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant phase-shifted 2D FFT.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape, in elements, of \p input and \p output.
    /// \param[in] shifts       On the \b host. HW 2D phase-shift to apply.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are not phase-shifted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift2D(const shared_t<T[]>& input, size4_t input_strides,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                 const shared_t<float2_t[]>& shifts, float cutoff, Stream& stream);

    ///  Phase-shifts a non-redundant 2D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift2D(const shared_t<T[]>& input, size4_t input_strides,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                 float2_t shift, float cutoff, Stream& stream);

    /// Phase-shifts a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T               cfloat_t or cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant 3D FFT to phase-shift.
    ///                         If nullptr, it is ignored and the shifts are saved in \p output.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant phase-shifted 3D FFT.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape, in elements, of \p input and \p output.
    /// \param[in] shifts       On the \b host. DHW 3D phase-shift to apply.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are not phase-shifted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift3D(const shared_t<T[]>& input, size4_t input_strides,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                 const shared_t<float3_t[]>& shifts, float cutoff, Stream& stream);

    ///  Phase-shifts a non-redundant 3D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift3D(const shared_t<T[]>& input, size4_t input_strides,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                 float3_t shift, float cutoff, Stream& stream);
}
