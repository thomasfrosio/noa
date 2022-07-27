#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/geometry/Symmetry.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_xform_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> && (REMAP == HC2HC || REMAP == HC2H);
}

namespace noa::cpu::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant 2D FFT to transform.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant transformed 2D FFT.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape, in elements, of \p input and \p output.
    /// \param[in] matrices     On the \b host. 2x2 inverse HW rotation/scaling matrix. One per batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       On the \b host. One per batch. If nullptr or if \p T is real, it is ignored.
    ///                         HW 2D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xform_v<REMAP, T>>>
    void transform2D(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                     const shared_t<float22_t[]>& matrices,
                     const shared_t<float2_t[]>& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream);

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xform_v<REMAP, T>>>
    void transform2D(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                     float22_t matrix, float2_t shift,
                     float cutoff, InterpMode interp_mode, Stream& stream);

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant 3D FFT to transform.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant transformed 3D FFT.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape, in elements, of \p input and \p output.
    /// \param[in] matrices     On the \b host. 3x3 inverse DHW rotation/scaling matrix. One per batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       On the \b host. One per batch. If nullptr or if \p T is real, it is ignored.
    ///                         DHW 3D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note \p input and \p output should not overlap.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xform_v<REMAP, T>>>
    void transform3D(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                     const shared_t<float33_t[]>& matrices,
                     const shared_t<float3_t[]>&  shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream);

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xform_v<REMAP, T>>>
    void transform3D(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                     float33_t matrix, float3_t shift,
                     float cutoff, InterpMode interp_mode, Stream& stream);
}

namespace noa::cpu::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    /// Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant 2D FFT to transform.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant transformed 2D FFT.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape, in elements, of \p input and \p output.
    /// \param[in] matrix       2x2 inverse HW rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] shift        HW 2D real-space forward shift to apply (as phase shift) after the transformation.
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
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xform_v<REMAP, T>>>
    void transform2D(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream);

    /// Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant 3D FFT to transform.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant transformed 3D FFT.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape, in elements, of \p input and \p output.
    /// \param[in] matrix       3x3 inverse DHW rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] shift        DHW 3D real-space forward shift to apply (as phase shift) after the transformation.
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
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xform_v<REMAP, T>>>
    void transform3D(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream);
}
