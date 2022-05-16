#pragma once

#ifndef NOA_UNIFIED_ROTATE_
#error "This is a private header"
#endif

#include "noa/cpu/geometry/Rotate.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Rotate.h"
#endif

namespace noa::geometry {
    template<bool PREFILTER, typename T, typename>
    void rotate2D(const Array<T>& input, const Array<T>& output,
                  const Array<float>& rotations, const Array<float2_t>& rotation_centers,
                  InterpMode interp_mode, BorderMode border_mode, T value) {
        NOA_CHECK(rotations.shape().ndim() == 1 && rotations.shape()[3] == output.shape()[0] &&
                  rotations.contiguous()[3],
                  "The number of rotations, specified as a contiguous row vector, should be equal to the number "
                  "of batches in the output, got {} rotations and {} output batches",
                  rotations.shape()[3], output.shape()[0]);
        NOA_CHECK(rotation_centers.shape().ndim() == 1 && rotation_centers.shape()[3] == output.shape()[0] &&
                  rotation_centers.contiguous()[3],
                  "The number of rotations, specified as a contiguous row vector, should be equal to the number "
                  "of batches in the output, got {} rotations and {} output batches",
                  rotation_centers.shape()[3], output.shape()[0]);
        NOA_CHECK(rotations.dereferencable() && rotation_centers.dereferencable(),
                  "The rotation parameters should be accessible to the CPU");

        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get() != output.get(), "In-place transformations are not supported");

            if (rotations.device().gpu())
                Stream::current(rotations.device()).synchronize();
            if (rotation_centers.device().gpu() && rotations.device() != rotation_centers.device())
                Stream::current(rotation_centers.device()).synchronize();

            cpu::geometry::rotate2D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    rotations.share(), rotation_centers.share(),
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

                if (input.device().cpu() || rotations.device().cpu() || rotation_centers.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::rotate2D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        rotations.share(), rotation_centers.share(),
                        interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<bool PREFILTER, typename T, typename>
    void rotate2D(const Array<T>& input, const Array<T>& output,
                  float rotation, float2_t rotation_center,
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
            NOA_CHECK(input.get() != output.get(), "In-place transformations are not supported");

            cpu::geometry::rotate2D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    rotation, rotation_center,
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
                cuda::geometry::rotate2D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        rotation, rotation_center,
                        interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<bool PREFILTER, typename T, typename>
    void rotate3D(const Array<T>& input, const Array<T>& output,
                  const Array<float33_t>& rotations, const Array<float3_t>& rotation_centers,
                  InterpMode interp_mode, BorderMode border_mode, T value) {
        NOA_CHECK(rotations.shape().ndim() == 1 && rotations.shape()[3] == output.shape()[0] &&
                  rotations.contiguous()[3],
                  "The number of rotations, specified as a contiguous row vector, should be equal to the number "
                  "of batches in the output, got {} rotations and {} output batches",
                  rotations.shape()[3], output.shape()[0]);
        NOA_CHECK(rotation_centers.shape().ndim() == 1 && rotation_centers.shape()[3] == output.shape()[0] &&
                  rotation_centers.contiguous()[3],
                  "The number of rotations, specified as a contiguous row vector, should be equal to the number "
                  "of batches in the output, got {} rotations and {} output batches",
                  rotation_centers.shape()[3], output.shape()[0]);
        NOA_CHECK(rotations.dereferencable() && rotation_centers.dereferencable(),
                  "The rotation parameters should be accessible to the CPU");

        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get() != output.get(), "In-place transformations are not supported");

            if (rotations.device().gpu())
                Stream::current(rotations.device()).synchronize();
            if (rotation_centers.device().gpu() && rotations.device() != rotation_centers.device())
                Stream::current(rotation_centers.device()).synchronize();

            cpu::geometry::rotate3D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    rotations.share(), rotation_centers.share(),
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

                if (input.device().cpu() || rotations.device().cpu() || rotation_centers.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::rotate3D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        rotations.share(), rotation_centers.share(),
                        interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<bool PREFILTER, typename T, typename>
    void rotate3D(const Array<T>& input, const Array<T>& output,
                  float33_t rotation, float3_t rotation_center,
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
            NOA_CHECK(input.get() != output.get(), "In-place transformations are not supported");

            cpu::geometry::rotate3D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    rotation, rotation_center,
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
                cuda::geometry::rotate3D<PREFILTER>(
                        input.share(), input.stride(), input.shape(),
                        output.share(), output.stride(), output.shape(),
                        rotation, rotation_center,
                        interp_mode, border_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
