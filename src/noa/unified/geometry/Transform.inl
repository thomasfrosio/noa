#pragma once

#ifndef NOA_UNIFIED_TRANSFORM_
#error "This is a private header"
#endif

#include "noa/cpu/geometry/Transform.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Transform.h"
#endif

// -- Affine transformations -- //
namespace noa::geometry {
    template<bool PREFILTER, typename T, typename MAT, typename>
    void transform2D(const Array<T>& input, const Array<T>& output, const Array<MAT>& matrices,
                     InterpMode interp_mode, BorderMode border_mode, T value) {
        NOA_CHECK(matrices.shape()[3] == output.shape()[0] && matrices.shape().ndim() == 1 && matrices.contiguous(),
                  "The number of matrices, specified as a contiguous row vector, should be equal to the number "
                  "of batches in the output, got {} matrices and {} output batches", matrices.shape()[3], output.shape()[0]);
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
            NOA_CHECK(input.get() != output.get(), "In-place transformations are not supported");

            if (matrices.device().gpu())
                Stream::current(matrices.device()).synchronize();
            cpu::geometry::transform2D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    matrices.share(), interp_mode, border_mode, value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
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

    template<bool PREFILTER, typename T, typename MAT, typename>
    void transform2D(const Array<T>& input, const Array<T>& output, MAT matrix,
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

            cpu::geometry::transform2D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    matrix, interp_mode, border_mode, value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
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

    template<bool PREFILTER, typename T, typename MAT, typename>
    void transform3D(const Array<T>& input, const Array<T>& output, const Array<MAT>& matrices,
                     InterpMode interp_mode, BorderMode border_mode, T value) {
        NOA_CHECK(matrices.shape()[3] == output.shape()[0] && matrices.shape().ndim() == 1 && matrices.contiguous(),
                  "The number of matrices, specified as a contiguous row vector, should be equal to the number "
                  "of batches in the output, got {} matrices and {} output batches", matrices.shape()[3], output.shape()[0]);
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
            NOA_CHECK(input.get() != output.get(), "In-place transformations are not supported");

            if (matrices.device().gpu())
                Stream::current(matrices.device()).synchronize();
            cpu::geometry::transform3D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    matrices.share(), interp_mode, border_mode, value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
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

    template<bool PREFILTER, typename T, typename MAT, typename>
    void transform3D(const Array<T>& input, const Array<T>& output, MAT matrix,
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

            cpu::geometry::transform3D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    matrix, interp_mode, border_mode, value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
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

namespace noa::geometry {
    template<bool PREFILTER, typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode, bool normalize) {
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

            cpu::geometry::transform2D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    shift, matrix, symmetry, center, interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
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

    template<bool PREFILTER, typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode, bool normalize) {
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

            cpu::geometry::transform3D<PREFILTER>(
                    input.share(), input.stride(), input.shape(),
                    output.share(), output.stride(), output.shape(),
                    shift, matrix, symmetry, center, interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
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
