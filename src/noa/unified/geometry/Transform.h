#pragma once

#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"

#include "noa/cpu/geometry/Transform.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Transform.h"
#endif

#include "noa/unified/Array.h"

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
    /// \tparam PREFILTER   Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \tparam MAT         float23_t or float33_t.
    /// \param[in] input    Input 2D array.
    /// \param[out] output  Output 2D array.
    /// \param[in] matrices 2x3 or 3x3 inverse rightmost affine matrices. One per output batch.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param border_mode  Address mode. See BorderMode.
    /// \param value        Constant value to use for out-of-bounds coordinates.
    ///                     Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p matrices can be on any device, including the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float23_v<MAT> || traits::is_float33_v<MAT>>>
    void transform2D(const Array<T>& input, const Array<T>& output, const Array<MAT>& matrices,
                     InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0}) {
        NOA_CHECK(matrices.shape()[3] == output.shape()[0],
                  "The number of matrices, specified as a row vector, should be equal to the number of batches "
                  "in the output, bot got {} matrices and {} output batches", matrices.shape()[3], output.shape()[0]);
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(matrices.dereferencable(), "The matrices should be accessible to the host");
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            if (matrices.device().gpu())
                Stream::current(matrices.device()).synchronize();
            cpu::geometry::transform2D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    matrices.share(), interp_mode, border_mode, value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(input.contiguous()[3],
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          input.shape(), input.stride());

                if (input.device().cpu() || matrices.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::transform2D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        matrices.share(), interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Applies one or multiple 2D affine transforms.
    /// \see This function is has the same features and limitations than the overload above,
    ///      but is using the same matrix for all transformations.
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float23_v<MAT> || traits::is_float33_v<MAT>>>
    void transform2D(const Array<T>& input, const Array<T>& output, MAT matrix,
                     InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0}) {
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::transform2D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    matrix, interp_mode, border_mode, value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(input.contiguous()[3],
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          input.shape(), input.stride());

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::transform2D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        matrix, interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Applies one or multiple 3D affine transforms.
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
    /// \tparam MAT         float34_t or float44_t.
    /// \param[in] input    Input 3D array.
    /// \param[out] output  Output 3D array.
    /// \param[in] matrices 3x4 or 4x4 inverse rightmost affine matrices. One per output batch.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param border_mode  Address mode. See BorderMode.
    /// \param value        Constant value to use for out-of-bounds coordinates.
    ///                     Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p matrices can be on any device, including the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float34_v<MAT> || traits::is_float44_v<MAT>>>
    void transform3D(const Array<T>& input, const Array<T>& output, const Array<MAT>& matrices,
                     InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0}) {
        NOA_CHECK(matrices.shape()[3] == output.shape()[0],
                  "The number of matrices, specified as a row vector, should be equal to the number of batches "
                  "in the output, bot got {} matrices and {} output batches", matrices.shape()[3], output.shape()[0]);
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(matrices.dereferencable(), "The matrices should be accessible to the host");
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            if (matrices.device().gpu())
                Stream::current(matrices.device()).synchronize();
            cpu::geometry::transform3D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    matrices.share(), interp_mode, border_mode, value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(input.contiguous()[1] && input.contiguous()[3],
                          "The third-most and innermost dimension of the input should be contiguous, "
                          "but got shape {} and stride {}", input.shape(), input.stride());

                if (input.device().cpu() || matrices.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::transform3D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        matrices.share(), interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Applies one or multiple 3D affine transforms.
    /// \see This function is has the same features and limitations than the overload above,
    ///      but is using the same matrix for all transformations.
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float34_v<MAT> || traits::is_float44_v<MAT>>>
    void transform3D(const Array<T>& input, const Array<T>& output, MAT matrix,
                     InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0}) {
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::transform3D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    matrix, interp_mode, border_mode, value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(input.contiguous()[1] && input.contiguous()[3],
                          "The third-most and innermost dimension of the input should be contiguous, "
                          "but got shape {} and stride {}", input.shape(), input.stride());

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::transform3D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        matrix, interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
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
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    template<bool PREFILTER = true, typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode = INTERP_LINEAR, bool normalize = true) {
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::transform2D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    shift, matrix, symmetry, center, interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(input.contiguous()[3],
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          input.shape(), input.stride());

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::transform2D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        shift, matrix, symmetry, center, interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

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
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    template<bool PREFILTER = true, typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode = INTERP_LINEAR, bool normalize = true) {
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::transform3D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    shift, matrix, symmetry, center, interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(input.contiguous()[1] && input.contiguous()[3],
                          "The third-most and innermost dimension of the input should be contiguous, "
                          "but got shape {} and stride {}", input.shape(), input.stride());

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::transform3D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        shift, matrix, symmetry, center, interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
