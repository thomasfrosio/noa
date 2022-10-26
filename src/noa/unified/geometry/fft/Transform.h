#pragma once

#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"

#include "noa/unified/Array.h"

namespace noa::geometry::fft::details {
    using namespace ::noa::fft;

    template<int NDIM, Remap REMAP, typename T, typename M, typename S>
    constexpr bool is_valid_transform_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> && (REMAP == HC2HC || REMAP == HC2H) &&
            ((NDIM == 2 && traits::is_any_v<M, float22_t, Array<float22_t>> && traits::is_any_v<S, float2_t, shared_t<float2_t[]>>) ||
             (NDIM == 3 && traits::is_any_v<M, float33_t, Array<float33_t>> && traits::is_any_v<S, float3_t, shared_t<float3_t[]>>));

    template<Remap REMAP, typename T>
    constexpr bool is_valid_transform_sym_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> && (REMAP == HC2HC || REMAP == HC2H);
}

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    /// \tparam REMAP       Remap operation. Should be HC2HC or HC2H.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \tparam M           float22_t or Array<float22_t>.
    /// \tparam S           float2_t or Array<float2_t>.
    /// \param[in] input    Non-redundant 2D FFT to transform.
    /// \param[out] output  Non-redundant transformed 2D FFT.
    /// \param shape        BDHW logical shape of \p input and \p output.
    /// \param[in] matrices 2x2 inverse HW rotation/scaling matrix. One, or if an array is provided, one per output batch.
    ///                     If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                     space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts   2D real-space HW forward shift to apply (as phase shift) after the transformation.
    ///                     One, or if an array is provided, one per output batch.
    ///                     If an empty array is entered or if \p T is real, it is ignored.
    /// \param cutoff       Maximum output frequency to consider, in cycle/pix.
    ///                     Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                     Frequencies higher than this value are set to 0.
    /// \param interp_mode  Interpolation/filtering mode. Cubic modes are currently not supported.
    ///
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p matrices and \p shifts should be accessible by the CPU.\n
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p input should be in the rightmost order and the width dimension should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input, \p matrices and \p shifts can be on any device, including the CPU.\n
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T, typename M, typename S,
             typename = std::enable_if_t<details::is_valid_transform_v<2, REMAP, T, M, S>>>
    void transform2D(const Array<T>& input, const Array<T>& output, dim4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR);

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, the border mode should be BORDER_ZERO and un-normalized
    ///          coordinates should be used.
    template<Remap REMAP, typename T, typename M, typename S,
             typename = std::enable_if_t<details::is_valid_transform_v<2, REMAP, T, M, S>>>
    void transform2D(const Texture<T>& input, const Array<T>& output, dim4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff = 0.5f);

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    /// \tparam REMAP       Remap operation. Should be HC2HC or HC2H.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] input    Non-redundant 3D FFT to transform.
    /// \param[out] output  Non-redundant transformed 3D FFT.
    /// \param shape        BDHW logical shape of \p input and \p output.
    /// \param[in] matrices 3x3 inverse DHW rotation/scaling matrix. One, or if an array is provided, one per output batch.
    ///                     If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                     space, a scaling of 1/S should be used in Fourier space. One per output batch.
    /// \param[in] shifts   DHW 3D real-space forward shift to apply (as phase shift) after the transformation.
    ///                     One, or if an array is provided, one per output batch.
    ///                     If an empty array is entered or if \p T is real, it is ignored.
    /// \param cutoff       Maximum output frequency to consider, in cycle/pix.
    ///                     Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                     Frequencies higher than this value are set to 0.
    /// \param interp_mode  Interpolation/filtering mode. Cubic modes are currently not supported.
    ///
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p matrices and \p shifts should be accessible by the CPU.\n
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p input should be in the rightmost order and the depth and width dimension should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input, \p matrices and \p shifts can be on any device, including the CPU.\n
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T, typename M, typename S,
             typename = std::enable_if_t<details::is_valid_transform_v<3, REMAP, T, M, S>>>
    void transform3D(const Array<T>& input, const Array<T>& output, dim4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR);

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, the border mode should be BORDER_ZERO and un-normalized
    ///          coordinates should be used.
    template<Remap REMAP, typename T, typename M, typename S,
                          typename = std::enable_if_t<details::is_valid_transform_v<3, REMAP, T, M, S>>>
    void transform3D(const Texture<T>& input, const Array<T>& output, dim4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff = 0.5f);
}

namespace noa::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    /// Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    /// \tparam REMAP       Remap operation. Should be HC2HC or HC2H.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] input    Non-redundant 2D FFT to transform.
    /// \param[out] output  Non-redundant transformed 2D FFT.
    /// \param shape        BDHW logical shape of \p input and \p output.
    /// \param[in] matrix   2x2 inverse HW rotation/scaling matrix.
    ///                     If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                     space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry Symmetry operator to apply after the rotation/scaling.
    /// \param[in] shift    HW 2D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff       Maximum output frequency to consider, in cycle/pix.
    ///                     Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                     Frequencies higher than this value are set to 0.
    /// \param interp_mode  Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p input should be in the rightmost order and the width dimension should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input can be on any device, including the CPU.\n
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void transform2D(const Array<T>& input, const Array<T>& output, dim4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);

    /// Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, the border mode should be BORDER_ZERO and un-normalized
    ///          coordinates should be used.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void transform2D(const Texture<T>& input, const Array<T>& output, dim4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff = 0.5f, bool normalize = true);

    /// Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    /// \tparam REMAP       Remap operation. Should be HC2HC or HC2H.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] input    Non-redundant 3D FFT to transform.
    /// \param[out] output  Non-redundant transformed 3D FFT.
    /// \param shape        BDHW logical shape of \p input and \p output.
    /// \param[in] matrix   3x3 inverse DHW rotation/scaling matrix.
    ///                     If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                     space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry Symmetry operator to apply after the rotation/scaling.
    /// \param[in] shift    DHW 3D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff       Maximum output frequency to consider, in cycle/pix.
    ///                     Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                     Frequencies higher than this value are set to 0.
    /// \param interp_mode  Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p input should be in the rightmost order and the depth and width dimension should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input can be on any device, including the CPU.\n
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void transform3D(const Array<T>& input, const Array<T>& output, dim4_t shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);

    /// Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, the border mode should be BORDER_ZERO and un-normalized
    ///          coordinates should be used.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void transform3D(const Texture<T>& input, const Array<T>& output, dim4_t shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff = 0.5f, bool normalize = true);
}

#define NOA_UNIFIED_FFT_TRANSFORM_
#include "noa/unified/geometry/fft/Transform.inl"
#undef NOA_UNIFIED_FFT_TRANSFORM_
