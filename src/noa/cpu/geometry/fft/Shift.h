#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

// TODO(TF) Add all remaining layouts

namespace noa::cpu::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Phase-shifts a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T               cfloat_t or cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant 2D FFT to phase-shift.
    ///                         If nullptr, it is ignored and the phase-shifts are saved in \p output.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant phase-shifted 2D FFT.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape, in elements, of \p input and \p output.
    ///                         The outermost dimension is the batch.
    /// \param[in] shifts       On the \b host. Rightmost 2D real-space forward shift to apply (as phase shift).
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are not phase-shifted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename T>
    void shift2D(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                 const shared_t<float2_t[]>& shifts, float cutoff, Stream& stream);

    ///  Phase-shifts a non-redundant 2D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void shift2D(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                 float2_t shift, float cutoff, Stream& stream);

    /// Phase-shifts a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T               cfloat_t or cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant 3D FFT to phase-shift.
    ///                         If nullptr, it is ignored and the shifts are saved in \p output.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant phase-shifted 3D FFT.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape, in elements, of \p input and \p output.
    ///                         The outermost dimension is the batch.
    /// \param[in] shifts       On the \b host. Rightmost 3D real-space forward shift to apply (as phase shift).
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are not phase-shifted.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename T>
    void shift3D(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                 const shared_t<float3_t[]>& shifts, float cutoff, Stream& stream);

    ///  Phase-shifts a non-redundant 3D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void shift3D(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                 float3_t shift, float cutoff, Stream& stream);
}
