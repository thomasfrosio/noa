#pragma once

#ifndef NOA_UNIFIED_SCALE_
#error "This is a private header"
#endif

#include "noa/cpu/geometry/Scale.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Scale.h"
#endif

namespace noa::geometry {
    template<bool PREFILTER, typename T>
    void scale2D(const Array<T>& input, const Array<T>& output,
                 const Array<float2_t>& scaling_factors, const Array<float2_t>& scaling_centers,
                 InterpMode interp_mode, BorderMode border_mode, T value) {
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

    template<bool PREFILTER, typename T>
    void scale2D(const Array<T>& input, const Array<T>& output,
                 float2_t scaling_factors, float2_t scaling_centers,
                 InterpMode interp_mode, BorderMode border_mode, T value) {
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

    template<bool PREFILTER, typename T>
    void scale3D(const Array<T>& input, const Array<T>& output,
                 const Array<float3_t>& scaling_factors, const Array<float3_t>& scaling_centers,
                 InterpMode interp_mode, BorderMode border_mode, T value) {
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

    template<bool PREFILTER, typename T>
    void scale3D(const Array<T>& input, const Array<T>& output,
                 float3_t scaling_factor, float3_t scaling_center,
                 InterpMode interp_mode, BorderMode border_mode, T value) {
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
