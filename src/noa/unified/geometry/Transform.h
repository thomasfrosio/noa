#pragma once

#include "noa/common/geometry/Symmetry.h"
#include "noa/unified/Array.h"

namespace noa::geometry::details {
    template<int NDIM, typename T, typename M>
    constexpr bool is_valid_transform_v =
            traits::is_any_v<T, float, cfloat_t, double, cdouble_t> &&
            ((NDIM == 2 && traits::is_any_v<M, float23_t, float33_t>) ||
             (NDIM == 3 && traits::is_any_v<M, float34_t, float44_t>));
}

// -- Affine transformations -- //
namespace noa::geometry {
    /// Applies one or multiple 2D affine transforms.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    ///
    /// \tparam PREFILTER   Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \tparam M           float23_t or float33_t.
    /// \param[in] input    Input 2D array.
    /// \param[out] output  Output 2D array.
    /// \param[in] matrices 2x3 or 3x3 inverse rightmost affine matrices. One per output batch.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param border_mode  Address mode. See BorderMode.
    /// \param value        Constant value to use for out-of-bounds coordinates.
    ///                     Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p input and \p output should be on the same device.\n
    ///         - \p matrices can be on any device as long as they are dereferencable by the CPU.\n
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p matrices can be on any device, including the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<bool PREFILTER = true, typename T, typename M,
             typename = std::enable_if_t<details::is_valid_transform_v<2, T, M>>>
    void transform2D(const Array<T>& input, const Array<T>& output, const Array<M>& matrices,
                     InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0});

    /// Applies one or multiple 2D affine transforms.
    /// \see This function is has the same features and limitations than the overload above,
    ///      but is using the same matrix for all transformations.
    template<bool PREFILTER = true, typename T, typename M,
             typename = std::enable_if_t<details::is_valid_transform_v<2, T, M>>>
    void transform2D(const Array<T>& input, const Array<T>& output, M matrix,
                     InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0});

    /// Applies one or multiple 3D affine transforms.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    ///
    /// \tparam PREFILTER   Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \tparam M           float34_t or float44_t.
    /// \param[in] input    Input 3D array.
    /// \param[out] output  Output 3D array.
    /// \param[in] matrices 3x4 or 4x4 inverse rightmost affine matrices. One per output batch.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param border_mode  Address mode. See BorderMode.
    /// \param value        Constant value to use for out-of-bounds coordinates.
    ///                     Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p input and \p output should be on the same device.\n
    ///         - \p matrices can be on any device as long as they are dereferencable by the CPU.\n
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p matrices can be on any device, including the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<bool PREFILTER = true, typename T, typename M,
             typename = std::enable_if_t<details::is_valid_transform_v<3, T, M>>>
    void transform3D(const Array<T>& input, const Array<T>& output, const Array<M>& matrices,
                     InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0});

    /// Applies one or multiple 3D affine transforms.
    /// \see This function is has the same features and limitations than the overload above,
    ///      but is using the same matrix for all transformations.
    template<bool PREFILTER = true, typename T, typename M,
             typename = std::enable_if_t<details::is_valid_transform_v<3, T, M>>>
    void transform3D(const Array<T>& input, const Array<T>& output, M matrix,
                     InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0});
}

// -- With symmetry -- //
namespace noa::geometry {
    /// Shifts, then rotates/scales and applies the symmetry on the 2D input array.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    ///
    /// \tparam PREFILTER   Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 2D array.
    /// \param[out] output  Output 2D array.
    /// \param shift        Rightmost forward shift to apply before the other transformations.
    ///                     Positive shifts translate the object to the right.
    /// \param matrix       Rightmost inverse rotation/scaling to apply after the shift.
    /// \param[in] symmetry Symmetry operator to apply after the rotation/scaling.
    /// \param center       Rightmost index of the transformation center.
    ///                     Both \p matrix and \p symmetry operates around this center.
    /// \param interp_mode  Interpolation/filter method. All interpolation modes are supported.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p input and \p output should be on the same device.\n
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    template<bool PREFILTER = true, typename T,
             typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t, double, cdouble_t>>>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);

    /// Shifts, then rotates/scales and applies the symmetry on the 3D input array.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    ///
    /// \tparam PREFILTER   Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 3D array.
    /// \param[out] output  Output 3D array.
    /// \param shift        Rightmost forward shift to apply before the other transformations.
    ///                     Positive shifts translate the object to the right.
    /// \param matrix       Rightmost inverse rotation/scaling to apply after the shift.
    /// \param[in] symmetry Symmetry operator to apply after the rotation/scaling.
    /// \param center       Rightmost index of the transformation center.
    ///                     Both \p matrix and \p symmetry operates around this center.
    /// \param interp_mode  Interpolation/filter mode. All interpolation modes are supported.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p input and \p output should be on the same device.\n
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    template<bool PREFILTER = true, typename T,
             typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t, double, cdouble_t>>>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);
}

#define NOA_UNIFIED_TRANSFORM_
#include "noa/unified/geometry/Transform.inl"
#undef NOA_UNIFIED_TRANSFORM_
