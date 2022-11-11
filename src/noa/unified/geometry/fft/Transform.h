#pragma once

#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"
#include "noa/common/geometry/Symmetry.h"
#include "noa/unified/Array.h"
#include "noa/unified/Texture.h"

namespace noa::geometry::fft::details {
    using namespace ::noa::fft;

    template<int NDIM, Remap REMAP, typename Value, typename Matrix, typename Shift>
    constexpr bool is_valid_transform_v =
            traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> && (REMAP == HC2HC || REMAP == HC2H) &&
            ((NDIM == 2 &&
              traits::is_any_v<Matrix, float22_t, Array<float22_t>> &&
              traits::is_any_v<Shift, float2_t, Array<float2_t>>) ||
             (NDIM == 3 &&
              traits::is_any_v<Matrix, float33_t, Array<float33_t>> &&
              traits::is_any_v<Shift, float3_t, Array<float3_t>>));

    template<Remap REMAP, typename Value>
    constexpr bool is_valid_transform_sym_v =
            traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> && (REMAP == HC2HC || REMAP == HC2H);
}

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam Value           float, double, cfloat_t, cdouble_t.
    /// \tparam Matrix          float22_t or Array<float22_t>.
    /// \tparam Shift           float2_t or Array<float2_t>.
    /// \param[in] input        Non-redundant 2D FFT to transform.
    /// \param[out] output      Non-redundant transformed 2D FFT.
    /// \param shape            BDHW logical shape of \p input and \p output.
    /// \param[in] inv_matrices 2x2 inverse HW rotation/scaling matrix.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S
    ///                         in real space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] post_shifts  2D real-space HW forward shift to apply (as phase shift) after the transformation.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If an empty array is entered or if \p T is real, it is ignored.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
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
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_v<2, REMAP, Value, Matrix, Shift>>>
    void transform2D(const Array<Value>& input, const Array<Value>& output, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& post_shifts,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR);

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, the border mode should be BORDER_ZERO and un-normalized
    ///          coordinates should be used.
    template<Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_v<2, REMAP, Value, Matrix, Shift>>>
    void transform2D(const Texture<Value>& input, const Array<Value>& output, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& post_shifts,
                     float cutoff = 0.5f);

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam Value           float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 3D FFT to transform.
    /// \param[out] output      Non-redundant transformed 3D FFT.
    /// \param shape            BDHW logical shape of \p input and \p output.
    /// \param[in] inv_matrices 3x3 inverse DHW rotation/scaling matrix.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space. One per output batch.
    /// \param[in] post_shifts  DHW 3D real-space forward shift to apply (as phase shift) after the transformation.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If an empty array is entered or if \p T is real, it is ignored.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
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
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename Value, typename Matrix, typename Shift,
             typename = std::enable_if_t<details::is_valid_transform_v<3, REMAP, Value, Matrix, Shift>>>
    void transform3D(const Array<Value>& input, const Array<Value>& output, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& post_shifts,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR);

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, the border mode should be BORDER_ZERO and un-normalized
    ///          coordinates should be used.
    template<Remap REMAP, typename Value, typename Matrix, typename Shift,
            typename = std::enable_if_t<details::is_valid_transform_v<3, REMAP, Value, Matrix, Shift>>>
    void transform3D(const Texture<Value>& input, const Array<Value>& output, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& post_shifts,
                     float cutoff = 0.5f);
}

namespace noa::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    /// Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 2D FFT to transform.
    /// \param[out] output      Non-redundant transformed 2D FFT.
    /// \param shape            BDHW logical shape of \p input and \p output.
    /// \param[in] inv_matrix   2x2 inverse HW rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] post_shift   HW 2D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
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
    template<Remap REMAP, typename Value, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform2D(const Array<Value>& input, const Array<Value>& output, dim4_t shape,
                     float22_t inv_matrix, const Symmetry& symmetry, float2_t post_shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);

    /// Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, the border mode should be BORDER_ZERO and un-normalized
    ///          coordinates should be used.
    template<Remap REMAP, typename Value, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform2D(const Texture<Value>& input, const Array<Value>& output, dim4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff = 0.5f, bool normalize = true);

    /// Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 3D FFT to transform.
    /// \param[out] output      Non-redundant transformed 3D FFT.
    /// \param shape            BDHW logical shape of \p input and \p output.
    /// \param[in] inv_matrix   3x3 inverse DHW rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] post_shift   DHW 3D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
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
    template<Remap REMAP, typename Value, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform3D(const Array<Value>& input, const Array<Value>& output, dim4_t shape,
                     float33_t inv_matrix, const Symmetry& symmetry, float3_t post_shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);

    /// Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, the border mode should be BORDER_ZERO and un-normalized
    ///          coordinates should be used.
    template<Remap REMAP, typename Value, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform3D(const Texture<Value>& input, const Array<Value>& output, dim4_t shape,
                     float33_t inv_matrix, const Symmetry& symmetry, float3_t post_shift,
                     float cutoff = 0.5f, bool normalize = true);

    /// Symmetrizes a non-redundant 2D (batched) FFT.
    /// \details This function has the same features and limitations as transform2D.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam Value           float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 2D FFT to transform.
    /// \param[out] output      Non-redundant transformed 2D FFT.
    /// \param shape            Rightmost logical shape of \p input and \p output.
    /// \param[in] symmetry     Symmetry operator to apply.
    /// \param[in] post_shift   Rightmost 2D real-space forward shift to apply (as phase shift) after the symmetry.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    template<Remap REMAP, typename Value, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void symmetrize2D(const Array<Value>& input, const Array<Value>& output, dim4_t shape,
                      const Symmetry& symmetry, float2_t post_shift,
                      float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true) {
        transform2D<REMAP>(input, output, shape, float22_t{}, symmetry, post_shift, cutoff, interp_mode, normalize);
    }

    /// Symmetrizes a non-redundant 3D (batched) FFT.
    /// \details This function has the same features and limitations as transform2D.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 3D FFT to transform.
    /// \param[out] output      Non-redundant transformed 3D FFT.
    /// \param shape            Rightmost logical shape of \p input and \p output.
    /// \param[in] symmetry     Symmetry operator to apply.
    /// \param[in] post_shift   Rightmost 3D real-space forward shift to apply (as phase shift) after the symmetry.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    template<Remap REMAP, typename Value, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, Value>>>
    void symmetrize3D(const Array<Value>& input, const Array<Value>& output, dim4_t shape,
                      const Symmetry& symmetry, float3_t post_shift,
                      float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true) {
        transform3D<REMAP>(input, output, shape, float33_t{}, symmetry, post_shift, cutoff, interp_mode, normalize);
    }
}
