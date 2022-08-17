#pragma once

#ifndef NOA_UNIFIED_TRANSFORM_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/geometry/Transform.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Transform.h"
#endif

// -- Affine transformations -- //
namespace noa::geometry {
    template<typename T, typename M, typename>
    void transform2D(const Array<T>& input, const Array<T>& output, const M& matrices,
                     InterpMode interp_mode, BorderMode border_mode, T value, bool prefilter) {
        constexpr bool SINGLE_MATRIX = traits::is_floatXX_v<M>;
        using matrix_t = std::conditional_t<SINGLE_MATRIX, M, shared_t<traits::value_type_t<M>>>;
        const matrix_t* matrices_;

        if constexpr (!traits::is_floatXX_v<M>) {
            NOA_CHECK(indexing::isVector(matrices.shape()) &&
                      matrices.shape().elements() == output.shape()[0] &&
                      matrices.contiguous(),
                      "The number of matrices, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} matrices and {} output batches",
                      matrices.shape().elements(), output.shape()[0]);
            matrices_ = &matrices.share();
        } else {
            matrices_ = &matrices;
        }

        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);
        NOA_CHECK(input.shape()[1] == 1 && output.shape()[1] == 1,
                  "The input and output arrays should be 2D, but got shape input:{}, output:{}",
                  input.shape(), output.shape());

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get() != output.get(), "In-place transformations are not supported");

            if constexpr (!SINGLE_MATRIX) {
                NOA_CHECK(matrices.dereferenceable(), "The matrices should be accessible to the host");
                if (matrices.device().gpu())
                    Stream::current(matrices.device()).synchronize();
            }
            cpu::geometry::transform2D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    *matrices_, interp_mode, border_mode, value, prefilter, stream.cpu());

        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(indexing::isRightmost(input.strides()) && input.strides()[3] == 1,
                          "The input should be in the rightmost order and the width dimension should be contiguous, "
                          "but got shape {} and stride {}", input.shape(), input.strides());

                bool sync_cpu = input.device().cpu();
                if constexpr (!SINGLE_MATRIX)
                    sync_cpu = matrices.device().cpu() ? true : sync_cpu;
                if (sync_cpu)
                    Stream::current(Device{}).synchronize();

                cuda::geometry::transform2D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        *matrices_, interp_mode, border_mode, prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename M, typename>
    void transform2D(const Texture<T>& input, const Array<T>& output, const M& matrices) {
        NOA_CHECK(input.device() == output.device(),
                  "The input texture and output array must be on the same device, "
                  "but got input:{} and output:{}", input.device(), output.device());

        if (output.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            transform2D(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()), output,
                        matrices, input.interp(), input.border(), texture.cvalue, false);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
        } else {
            constexpr bool SINGLE_MATRIX = traits::is_floatXX_v<M>;
            using matrix_t = std::conditional_t<SINGLE_MATRIX, M, shared_t<traits::value_type_t<M>>>;
            matrix_t* matrices_;

            if constexpr (!SINGLE_MATRIX) {
                NOA_CHECK(indexing::isVector(matrices.shape()) &&
                          matrices.shape().elements() == output.shape()[0] &&
                          matrices.contiguous(),
                          "The number of matrices, specified as a contiguous vector, should be equal to the number "
                          "of batches in the output, got {} matrices and {} output batches",
                          matrices.shape().elements(), output.shape()[0]);

                if (matrices.device().cpu())
                    Stream::current(Device{}).synchronize();

                matrices_ = &matrices.share();
            } else {
                matrices_ = &matrices;
            }

            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);
            NOA_CHECK(input.shape()[1] == 1 && output.shape()[1] == 1,
                      "The input texture and output array should be 2D, but got shape input:{}, output:{}",
                      input.shape(), output.shape());

            const Device device = output.device();
            NOA_CHECK(device == input.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);

            const cuda::Texture<T>& texture = input.cuda();
            Stream& stream = Stream::current(device);
            cuda::geometry::transform2D(
                    texture.array, texture.texture, size2_t(input.shape().get(2)),
                    input.interp(), input.border(),
                    output.share(), output.strides(), output.shape(),
                    *matrices_, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename T, typename M, typename>
    void transform3D(const Array<T>& input, const Array<T>& output, const M& matrices,
                     InterpMode interp_mode, BorderMode border_mode, T value, bool prefilter) {
        constexpr bool SINGLE_MATRIX = traits::is_floatXX_v<M>;
        using matrix_t = std::conditional_t<SINGLE_MATRIX, M, shared_t<traits::value_type_t<M>>>;
        matrix_t* matrices_;

        if constexpr (!traits::is_floatXX_v<M>) {
            NOA_CHECK(indexing::isVector(matrices.shape()) &&
                      matrices.shape().elements() == output.shape()[0] &&
                      matrices.contiguous(),
                      "The number of matrices, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} matrices and {} output batches",
                      matrices.shape().elements(), output.shape()[0]);
            matrices_ = &matrices.share();
        } else {
            matrices_ = &matrices;
        }

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

            if constexpr (!SINGLE_MATRIX) {
                NOA_CHECK(matrices.dereferenceable(), "The matrices should be accessible to the host");
                if (matrices.device().gpu())
                    Stream::current(matrices.device()).synchronize();
            }
            cpu::geometry::transform3D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    *matrices_, interp_mode, border_mode, value, prefilter, stream.cpu());

        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(indexing::isRightmost(input.strides()) &&
                          indexing::isContiguous(input.strides(), input.shape())[1] &&
                          input.strides()[3] == 1,
                          "The input should be in the rightmost order and the height and width dimension should be "
                          "contiguous, but got shape {} and stride {}", input.shape(), input.strides());

                bool sync_cpu = input.device().cpu();
                if constexpr (!SINGLE_MATRIX)
                    sync_cpu = matrices.device().cpu() ? true : sync_cpu;
                if (sync_cpu)
                    Stream::current(Device{}).synchronize();

                cuda::geometry::transform3D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        *matrices_, interp_mode, border_mode, prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename M, typename>
    void transform3D(const Texture<T>& input, const Array<T>& output, const M& matrices) {
        NOA_CHECK(input.device() == output.device(),
                  "The input texture and output array must be on the same device, "
                  "but got input:{} and output:{}", input.device(), output.device());

        if (output.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            transform3D(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()), output,
                        matrices, input.interp(), input.border(), texture.cvalue, false);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
        } else {
            constexpr bool SINGLE_MATRIX = traits::is_floatXX_v<M>;
            using matrix_t = std::conditional_t<SINGLE_MATRIX, M, shared_t<traits::value_type_t<M>>>;
            matrix_t* matrices_;

            if constexpr (!SINGLE_MATRIX) {
                NOA_CHECK(indexing::isVector(matrices.shape()) &&
                          matrices.shape().elements() == output.shape()[0] &&
                          matrices.contiguous(),
                          "The number of matrices, specified as a contiguous vector, should be equal to the number "
                          "of batches in the output, got {} matrices and {} output batches",
                          matrices.shape().elements(), output.shape()[0]);

                if (matrices.device().cpu())
                    Stream::current(Device{}).synchronize();

                matrices_ = &matrices.share();
            } else {
                matrices_ = &matrices;
            }

            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);

            const Device device = output.device();
            NOA_CHECK(device == input.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);

            const cuda::Texture<T>& texture = input.cuda();
            Stream& stream = Stream::current(device);
            cuda::geometry::transform3D(
                    texture.array, texture.texture, size3_t(input.shape().get(1)),
                    input.interp(), input.border(),
                    output.share(), output.strides(), output.shape(),
                    *matrices_, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }
}

namespace noa::geometry {
    template<typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize) {
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

            cpu::geometry::transform2D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    shift, matrix, symmetry, center, interp_mode, prefilter, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(indexing::isRightmost(input.strides()) && input.strides()[3] == 1,
                          "The input should be in the rightmost order and the width dimension should be contiguous, "
                          "but got shape {} and stride {}", input.shape(), input.strides());

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::transform2D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        shift, matrix, symmetry, center, interp_mode, prefilter, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void transform2D(const Texture<T>& input, const Array<T>& output,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     bool normalize) {
        if (input.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            transform2D(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()), output,
                        shift, matrix, symmetry, center, input.interp(), false, normalize);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
        } else {
            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);

            const Device device = output.device();
            NOA_CHECK(device == input.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = input.cuda();
            cuda::geometry::transform2D(
                    texture.array, texture.texture, input.interp(),
                    output.share(), output.strides(), output.shape(),
                    shift, matrix, symmetry, center, normalize, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize) {
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

            cpu::geometry::transform3D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    shift, matrix, symmetry, center, interp_mode, prefilter, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(indexing::isRightmost(input.strides()) &&
                          indexing::isContiguous(input.strides(), input.shape())[1] &&
                          input.strides()[3] == 1,
                          "The input should be in the rightmost order and the height and width dimension should be "
                          "contiguous, but got shape {} and stride {}", input.shape(), input.strides());

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::transform3D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        shift, matrix, symmetry, center, interp_mode, prefilter, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void transform3D(const Texture<T>& input, const Array<T>& output,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     bool normalize) {
        if (input.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            transform3D(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()), output,
                        shift, matrix, symmetry, center, input.interp(), false, normalize);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
        } else {
            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);

            const Device device = output.device();
            NOA_CHECK(device == input.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = input.cuda();
            cuda::geometry::transform3D(
                    texture.array, texture.texture, input.interp(),
                    output.share(), output.strides(), output.shape(),
                    shift, matrix, symmetry, center, normalize, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }
}
