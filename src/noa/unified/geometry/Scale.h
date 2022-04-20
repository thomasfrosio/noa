#pragma once

#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"

#include "noa/cpu/geometry/Scale.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Scale.h"
#endif

#include "noa/unified/Array.h"

namespace noa::geometry {
    /// Applies one or multiple 2D shrinking/stretching.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, double, cfloat_t or cdouble_t.
    /// \param[in] input            Input 2D array.
    /// \param[out] output          Output 2D array.
    /// \param[in] scaling_factors  Rightmost forward scaling factors. One per output batch.
    /// \param[in] scaling_centers  Rightmost scaling centers. One per output batch.
    /// \param interp_mode          Filter mode. See InterpMode.
    /// \param border_mode          Address mode. See BorderMode.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \see "noa/unified/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p scaling_factors and \p scaling_centers should be accessible by the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<bool PREFILTER = true, typename T>
    void scale2D(const Array<T>& input, const Array<T>& output,
                 const Array<float2_t>& scaling_factors, const Array<float2_t>& scaling_centers,
                 InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0}) {
        NOA_CHECK(scaling_factors.shape()[3] == output.shape()[0] &&
                  scaling_centers.shape()[3] == output.shape()[0],
                  "The number of scaling factors and scaling centers, specified as a row vector, should be equal to "
                  "the number of batches in the output, bot got {} scaling factors, {} scaling centers and {} "
                  "output batches", scaling_factors.shape()[3], scaling_centers.shape()[3], output.shape()[0]);
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);
        NOA_CHECK(scaling_factors.dereferencable() && scaling_centers.dereferencable(),
                  "The rotation parameters should be accessible to the host");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            if (scaling_factors.device().gpu())
                Stream::current(scaling_factors.device()).synchronize();
            if (scaling_centers.device().gpu() && scaling_factors.device() != scaling_centers.device())
                Stream::current(scaling_centers.device()).synchronize();

            cpu::geometry::scale2D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    scaling_factors.share(), scaling_centers.share(),
                    interp_mode, border_mode, value, stream.cpu());
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

                if (input.device().cpu() || scaling_factors.device().cpu() || scaling_centers.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::scale2D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        scaling_factors.share(), scaling_centers.share(),
                        interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Applies one or multiple 2D shrinking/stretching.
    /// \see This function is has the same features and limitations than the overload above,
    ///      but is using the same rotation for all batches.
    template<bool PREFILTER = true, typename T>
    void scale2D(const Array<T>& input, const Array<T>& output,
                  float2_t scaling_factors, float2_t scaling_centers,
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

            cpu::geometry::scale2D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    scaling_factors, scaling_centers,
                    interp_mode, border_mode, value, stream.cpu());
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
                cuda::geometry::scale2D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        scaling_factors, scaling_centers,
                        interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Applies one or multiple 3D shrinking/stretching.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, double, cfloat_t or cdouble_t.
    /// \param[in] input            Input 3D array.
    /// \param[out] output          Output 3D array.
    /// \param[in] scaling_factors  Rightmost forward scaling factors. One per output batch.
    /// \param[in] scaling_centers  Rightmost scaling centers. One per output batch.
    /// \param interp_mode          Filter mode. See InterpMode.
    /// \param border_mode          Address mode. See BorderMode.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \see "noa/unified/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p scaling_factors and \p scaling_centers should be accessible by the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<bool PREFILTER = true, typename T>
    void scale3D(const Array<T>& input, const Array<T>& output,
                  const Array<float3_t>& scaling_factors, const Array<float3_t>& scaling_centers,
                  InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0}) {
        NOA_CHECK(scaling_factors.shape()[3] == output.shape()[0] &&
                  scaling_centers.shape()[3] == output.shape()[0],
                  "The number of scaling factors and scaling centers, specified as a row vector, should be equal to "
                  "the number of batches in the output, bot got {} scaling factors, {} scaling centers and {} "
                  "output batches", scaling_factors.shape()[3], scaling_centers.shape()[3], output.shape()[0]);
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);
        NOA_CHECK(scaling_factors.dereferencable() && scaling_centers.dereferencable(),
                  "The rotation parameters should be accessible to the host");

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            if (scaling_factors.device().gpu())
                Stream::current(scaling_factors.device()).synchronize();
            if (scaling_centers.device().gpu() && scaling_factors.device() != scaling_centers.device())
                Stream::current(scaling_centers.device()).synchronize();

            cpu::geometry::scale3D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    scaling_factors.share(), scaling_centers.share(),
                    interp_mode, border_mode, value, stream.cpu());
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
                          "The third-most innermost dimension of the input should be contiguous, "
                          "but got shape {} and stride {}", input.shape(), input.stride());

                if (input.device().cpu() || scaling_factors.device().cpu() || scaling_centers.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::scale3D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        scaling_factors.share(), scaling_centers.share(),
                        interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Applies one 3D rotation to a (batched) array.
    /// \see This function is has the same features and limitations than the overload above,
    ///      but is using the same rotation for all batches.
    template<bool PREFILTER = true, typename T>
    void scale3D(const Array<T>& input, const Array<T>& output,
                  float3_t scaling_factor, float3_t scaling_center,
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

            cpu::geometry::scale3D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    scaling_factor, scaling_center,
                    interp_mode, border_mode, value, stream.cpu());
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
                          "The third-most innermost dimension of the input should be contiguous, "
                          "but got shape {} and stride {}", input.shape(), input.stride());

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::scale3D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        scaling_factor, scaling_center,
                        interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
