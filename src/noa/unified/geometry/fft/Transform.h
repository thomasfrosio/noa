#pragma once

#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"

#include "noa/cpu/geometry/fft/Transform.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Transform.h"
#endif

#include "noa/unified/Array.h"

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 2D FFT to transform.
    /// \param[out] output      Non-redundant transformed 2D FFT.
    /// \param[in] matrices     2x2 inverse rightmost rotation/scaling matrix. One per output batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       Rightmost 2D real-space forward shift to apply (as phase shift) after the transformation.
    ///                         One per output batch. If empty or if \p T is real, it is ignored.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    ///
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input, \p matrices and \p shifts can be on any device, including the CPU.\n
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     const Array<float22_t>& matrices, const Array<float2_t>& shifts,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR);

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     float22_t matrix, float2_t shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR);

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 3D FFT to transform.
    /// \param[out] output      Non-redundant transformed 3D FFT.
    /// \param[in] matrices     3x3 inverse rightmost rotation/scaling matrix. One per output batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       Rightmost 3D real-space forward shift to apply (as phase shift) after the transformation.
    ///                         One per output batch. If empty or if \p T is real, it is ignored.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    ///
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input, \p matrices and \p shifts can be on any device, including the CPU.\n
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     const Array<float33_t>& matrices, const Array<float3_t>& shifts,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR);

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     float33_t matrix, float3_t shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR);
}

namespace noa::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    /// Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 2D FFT to transform.
    /// \param[out] output      Non-redundant transformed 2D FFT.
    /// \param[in] matrix       2x2 inverse rightmost rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] shift        Rightmost 2D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    ///
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input can be on any device, including the CPU.\n
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);

    /// Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 3D FFT to transform.
    /// \param[out] output      Non-redundant transformed 3D FFT.
    /// \param[in] matrix       3x3 inverse rightmost rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] shift        Rightmost 3D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    ///
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input can be on any device, including the CPU.\n
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);
}

#define NOA_UNIFIED_FFT_TRANSFORM_
#include "noa/unified/geometry/fft/Transform.inl"
#undef NOA_UNIFIED_FFT_TRANSFORM_
