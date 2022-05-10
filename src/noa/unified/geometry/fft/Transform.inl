#pragma once

#ifndef NOA_UNIFIED_FFT_TRANSFORM_
#error "This is a private header"
#endif

#include "noa/cpu/geometry/fft/Transform.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Transform.h"
#endif

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     const Array<float22_t>& matrices, const Array<float2_t>& shifts,
                     float cutoff, InterpMode interp_mode) {
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
            NOA_CHECK(matrices.dereferencable() && shifts.dereferencable(),
                      "The rotation parameters should be accessible to the host");

            if (matrices.device().gpu())
                Stream::current(matrices.device()).synchronize();
            if (shifts.device().gpu() && matrices.device() != shifts.device())
                Stream::current(shifts.device()).synchronize();

            cpu::geometry::fft::transform2D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrices.share(), shifts.share(), cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(input_stride[3] == 1,
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          input.shape(), input_stride);

                if (input.device().cpu() || matrices.device().cpu() || shifts.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform2D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrices.share(), shifts.share(), cutoff, interp_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     float22_t matrix, float2_t shift,
                     float cutoff, InterpMode interp_mode) {
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

            cpu::geometry::fft::transform2D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrix, shift, cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(input_stride[3] == 1,
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          input.shape(), input_stride);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform2D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrix, shift, cutoff, interp_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     const Array<float33_t>& matrices, const Array<float3_t>& shifts,
                     float cutoff, InterpMode interp_mode) {
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
            NOA_CHECK(matrices.dereferencable() && shifts.dereferencable(),
                      "The rotation parameters should be accessible to the host");

            if (matrices.device().gpu())
                Stream::current(matrices.device()).synchronize();
            if (shifts.device().gpu() && matrices.device() != shifts.device())
                Stream::current(shifts.device()).synchronize();

            cpu::geometry::fft::transform3D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrices.share(), shifts.share(), cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isContiguous(input_stride, input.shape())[1] && input_stride[3] == 1,
                          "The third-most and innermost dimension of the input should be contiguous, but got shape {} "
                          "and stride {}", input.shape(), input_stride);

                if (input.device().cpu() || matrices.device().cpu() || shifts.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform3D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrices.share(), shifts.share(), cutoff, interp_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     float33_t matrix, float3_t shift,
                     float cutoff, InterpMode interp_mode) {
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

            cpu::geometry::fft::transform3D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrix, shift, cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isContiguous(input_stride, input.shape())[1] && input_stride[3] == 1,
                          "The third-most and innermost dimension of the input should be contiguous, but got shape {} "
                          "and stride {}", input.shape(), input_stride);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform3D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrix, shift, cutoff, interp_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

namespace noa::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    template<Remap REMAP, typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize) {
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

            cpu::geometry::fft::transform2D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(input_stride[3] == 1,
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          input.shape(), input_stride);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform2D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize) {
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

            cpu::geometry::fft::transform3D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isContiguous(input_stride, input.shape())[1] && input_stride[3] == 1,
                          "The third-most and innermost dimension of the input should be contiguous, but got shape {} "
                          "and stride {}", input.shape(), input_stride);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform3D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
