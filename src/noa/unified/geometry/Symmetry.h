#pragma once

#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"

#include "noa/cpu/geometry/Symmetry.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Symmetry.h"
#endif

#include "noa/unified/Array.h"

namespace noa::geometry {
    /// Symmetrizes the 2D (batched) input array.
    /// \tparam PREFILTER   Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 2D array.
    /// \param[out] output  Output 2D array.
    /// \param[in] symmetry Symmetry operator.
    /// \param center       Rightmost center of the symmetry.
    /// \param interp_mode  Filter mode. See InterpMode.
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
    void symmetrize2D(const Array<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float2_t center,
                      InterpMode interp_mode = INTERP_LINEAR, bool normalize = true) {
        NOA_CHECK(input.shape()[2] == input.shape()[2] &&
                  input.shape()[1] == input.shape()[1],
                  "The input {} and output {} shapes don't match", input.shape(), output.shape());
        size4_t input_stride = input.stride();
        if (input.shape()[0] == 1) {
            input_stride[0] = 0;
        } else if (input.shape()[0] == output.shape()[0]) {
            NOA_THROW("The number of batches in the input ({}) is not compatible with the number of "
                      "batches in the output ({})", input.shape()[0], output.shape()[0]);
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::symmetrize2D<PREFILTER>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    symmetry, center, interp_mode, normalize, stream.cpu());
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
                NOA_CHECK(input_stride[3] == 1,
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          input.shape(), input_stride);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::symmetrize2D<PREFILTER>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        symmetry, center, interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Symmetrizes the 3D (batched) input array.
    /// \tparam PREFILTER   Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 3D array.
    /// \param[out] output  Output 3D array.
    /// \param[in] symmetry Symmetry operator.
    /// \param center       Rightmost center of the symmetry.
    /// \param interp_mode  Filter mode. See InterpMode.
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
    void symmetrize3D(const Array<T[]>& input, const Array<T[]>& output,
                      const Symmetry& symmetry, float3_t center,
                      InterpMode interp_mode = INTERP_LINEAR, bool normalize = true) {
        NOA_CHECK(input.shape()[3] == input.shape()[3] &&
                  input.shape()[2] == input.shape()[2] &&
                  input.shape()[1] == input.shape()[1],
                  "The input {} and output {} shapes don't match", input.shape(), output.shape());
        size4_t input_stride = input.stride();
        if (input.shape()[0] == 1) {
            input_stride[0] = 0;
        } else if (input.shape()[0] == output.shape()[0]) {
            NOA_THROW("The number of batches in the input ({}) is not compatible with the number of "
                      "batches in the output ({})", input.shape()[0], output.shape()[0]);
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::symmetrize3D<PREFILTER>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    symmetry, center, interp_mode, normalize, stream.cpu());
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
                NOA_CHECK(indexing::isContiguous(input_stride, input.shape())[1] && input_stride[3] == 1,
                          "The third-most and innermost dimension of the input should be contiguous, "
                          "but got shape {} and stride {}", input.shape(), input_stride);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::symmetrize3D<PREFILTER>(
                        input.share(), input_stride,
                        output.share(), output.stride(), output.shape(),
                        symmetry, center, interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
