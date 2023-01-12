#pragma once

#include "noa/common/geometry/Symmetry.h"
#include "noa/unified/Array.h"
#include "noa/unified/Texture.h"

namespace noa::geometry::details {
    template<int NDIM, typename Value, typename Matrix>
    constexpr bool is_valid_transform_v =
            traits::is_any_v<Value, float, cfloat_t, double, cdouble_t> &&
            ((NDIM == 2 && traits::is_any_v<Matrix, float23_t, float33_t, Array<float23_t>, Array<float33_t>>) ||
             (NDIM == 3 && traits::is_any_v<Matrix, float34_t, float44_t, Array<float34_t>, Array<float44_t>>));
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
    ///
    /// \tparam Value           float, double, cfloat_t or cdouble_t.
    /// \tparam Matrix          float23_t, float33_t or an array of these types.
    /// \param[in,out] input    Input 2D array. It can be overwritten depending on \p prefilter.
    /// \param[out] output      Output 2D array.
    /// \param[in] inv_matrices 2x3 or 3x3 inverse HW affine matrices.
    ///                         One, or if an array is entered, one per output batch.
    /// \param interp_mode      Filter mode. See InterpMode.
    /// \param border_mode      Address mode. See BorderMode.
    /// \param value            Constant value to use for out-of-bounds coordinates.
    ///                         Only used if \p border_mode is BORDER_VALUE.
    /// \param prefilter        Whether or not the input should be prefiltered in-place.
    ///                         Only used if \p interp_mode is INTERP_CUBIC_BSPLINE(_FAST).
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_transform_v<2, Value, Matrix>>>
    void transform2D(const Array<Value>& input, const Array<Value>& output, const Matrix& inv_matrices,
                     InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO,
                     Value value = Value{0}, bool prefilter = true);

    /// Applies one or multiple 2D affine transforms.
    /// \details This overload has the same features and limitations as the overload taking Arrays.
    ///          This is mostly for the GPU, since "CPU textures" are simple Arrays.
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_transform_v<2, Value, Matrix>>>
    void transform2D(const Texture<Value>& input, const Array<Value>& output, const Matrix& inv_matrices);

    /// Applies one or multiple 3D affine transforms.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    ///
    /// \tparam Value           float, double, cfloat_t or cdouble_t.
    /// \tparam Matrix          float34_t, float44_t or an array of these types.
    /// \param[in,out] input    Input 3D array. It can be overwritten depending on \p prefilter.
    /// \param[out] output      Output 3D array.
    /// \param[in] inv_matrices 3x4 or 4x4 inverse DHW affine matrix/matrices.
    ///                         One, or if an array is entered, one per output batch.
    /// \param interp_mode      Filter mode. See InterpMode.
    /// \param border_mode      Address mode. See BorderMode.
    /// \param value            Constant value to use for out-of-bounds coordinates.
    ///                         Only used if \p border_mode is BORDER_VALUE.
    /// \param prefilter        Whether or not the input should be prefiltered in-place.
    ///                         Only used if \p interp_mode is INTERP_CUBIC_BSPLINE(_FAST).
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_transform_v<3, Value, Matrix>>>
    void transform3D(const Array<Value>& input, const Array<Value>& output, const Matrix& inv_matrices,
                     InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO,
                     Value value = Value{0}, bool prefilter = true);

    /// Applies one or multiple 3D affine transforms.
    /// \details This overload has the same features and limitations as the overload taking Arrays.
    ///          This is mostly for the GPU, since "CPU textures" are simple Arrays.
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_transform_v<3, Value, Matrix>>>
    void transform3D(const Texture<Value>& input, const Array<Value>& output, const Matrix& inv_matrices);
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
    /// \tparam Value       float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 2D array. It can be overwritten depending on \p prefilter.
    /// \param[out] output  Output 2D array.
    /// \param shift        HW forward shift to apply before the other transformations.
    ///                     Positive shifts translate the object to the right.
    /// \param inv_matrices HW inverse rotation/scaling to apply after the shift.
    /// \param[in] symmetry Symmetry operator to apply after the rotation/scaling.
    /// \param center       HW index of the transformation center.
    ///                     Both \p matrix and \p symmetry operates around this center.
    /// \param interp_mode  Interpolation/filter method. All interpolation modes are supported.
    /// \param prefilter    Whether or not the input should be prefiltered in-place.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE(_FAST).
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, cfloat_t, double, cdouble_t>>>
    void transform2D(const Array<Value>& input, const Array<Value>& output,
                     float2_t shift, float22_t inv_matrices, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode = INTERP_LINEAR, bool prefilter = true, bool normalize = true);

    /// Shifts, then rotates/scales and applies the symmetry on the 2D input array.
    /// \details This overload has the same features and limitations as the overload taking Arrays.
    ///          This is mostly for the GPU, since "CPU textures" are simple Arrays.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, cfloat_t, double, cdouble_t>>>
    void transform2D(const Texture<Value>& input, const Array<Value>& output,
                     float2_t shift, float22_t inv_matrices, const Symmetry& symmetry, float2_t center,
                     bool normalize = true);

    /// Shifts, then rotates/scales and applies the symmetry on the 3D input array.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    ///
    /// \tparam Value       float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 3D array. It can be overwritten depending on \p prefilter.
    /// \param[out] output  Output 3D array.
    /// \param shift        DHW forward shift to apply before the other transformations.
    ///                     Positive shifts translate the object to the right.
    /// \param inv_matrices DHW inverse rotation/scaling to apply after the shift.
    /// \param[in] symmetry Symmetry operator to apply after the rotation/scaling.
    /// \param center       DHW index of the transformation center.
    ///                     Both \p matrix and \p symmetry operates around this center.
    /// \param interp_mode  Interpolation/filter mode. All interpolation modes are supported.
    /// \param prefilter    Whether or not the input should be prefiltered in-place.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE(_FAST).
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, cfloat_t, double, cdouble_t>>>
    void transform3D(const Array<Value>& input, const Array<Value>& output,
                     float3_t shift, float33_t inv_matrices, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode = INTERP_LINEAR, bool prefilter = true, bool normalize = true);

    /// Shifts, then rotates/scales and applies the symmetry on the 3D input array.
    /// \details This overload has the same features and limitations as the overload taking Arrays.
    ///          This is mostly for the GPU, since "CPU textures" are simple Arrays.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, cfloat_t, double, cdouble_t>>>
    void transform3D(const Texture<Value>& input, const Array<Value>& output,
                     float3_t shift, float33_t inv_matrices, const Symmetry& symmetry, float3_t center,
                     bool normalize = true);

    /// Symmetrizes the 2D (batched) input array.
    /// \tparam Value           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 2D array. It can be overwritten depending on \p prefilter.
    /// \param[out] output  Output 2D array.
    /// \param[in] symmetry Symmetry operator.
    /// \param center       HW center of the symmetry.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param prefilter    Whether the input should be prefiltered in-place.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE(_FAST).
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, cfloat_t, double, cdouble_t>>>
    void symmetrize2D(const Array<Value>& input, const Array<Value>& output,
                      const Symmetry& symmetry, float2_t center,
                      InterpMode interp_mode = INTERP_LINEAR, bool prefilter = true, bool normalize = true);

    /// Symmetrizes the 2D (batched) input array.
    /// \details This overload has the same features and limitations as the overload taking Arrays.
    ///          This is mostly for the GPU, since "CPU textures" are simple Arrays.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, cfloat_t, double, cdouble_t>>>
    void symmetrize2D(const Texture<Value>& input, const Array<Value>& output,
                      const Symmetry& symmetry, float2_t center,
                      bool normalize = true);

    /// Symmetrizes the 3D (batched) input array.
    /// \tparam Value       float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 3D array. It can be overwritten depending on \p prefilter.
    /// \param[out] output  Output 3D array.
    /// \param[in] symmetry Symmetry operator.
    /// \param center       DHW center of the symmetry.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param prefilter    Whether or not the input should be prefiltered in-place.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, cfloat_t, double, cdouble_t>>>
    void symmetrize3D(const Array<Value>& input, const Array<Value>& output,
                      const Symmetry& symmetry, float3_t center,
                      InterpMode interp_mode = INTERP_LINEAR, bool prefilter = true, bool normalize = true);

    /// Symmetrizes the 2D (batched) input array.
    /// \details This overload has the same features and limitations as the overload taking Arrays.
    ///          This is mostly for the GPU, since "CPU textures" are simple Arrays.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, cfloat_t, double, cdouble_t>>>
    void symmetrize3D(const Texture<Value>& input, const Array<Value>& output,
                      const Symmetry& symmetry, float3_t center,
                      bool normalize = true);
}
