#pragma once

#ifndef NOA_UNIFIED_FFT_TRANSFORM_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/geometry/fft/Transform.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Transform.h"
#endif

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename T, typename M, typename S, typename>
    void transform2D(const Array<T>& input, const Array<T>& output, size4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff, InterpMode interp_mode) {
        NOA_CHECK(shape[3] / 2 + 1 == input.shape()[3] && input.shape()[3] == output.shape()[3] &&
                  shape[2] == input.shape()[2] && input.shape()[2] == output.shape()[2],
                  "The non-redundant input {} and/or output {} shapes don't match the logical shape {}",
                  input.shape(), output.shape(), shape);

        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);
        size4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        constexpr bool SINGLE_MATRIX = traits::is_float22_v<M>;
        using matrix_t = std::conditional_t<SINGLE_MATRIX, M, shared_t<traits::value_type_t<M>>>;
        const matrix_t* matrices_;
        if constexpr (!SINGLE_MATRIX) {
            NOA_CHECK(matrices.shape().elements() == output.shape()[0] &&
                      indexing::isVector(matrices.shape()) && matrices.contiguous(),
                      "The number of matrices, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} matrices and {} output batches",
                      matrices.shape().elements(), output.shape()[0]);
            matrices_ = &matrices.share();
        } else {
            matrices_ = &matrices;
        }

        constexpr bool SINGLE_SHIFT = traits::is_float2_v<M>;
        using shift_t = std::conditional_t<SINGLE_SHIFT, S, shared_t<traits::value_type_t<S>>>;
        const shift_t* shifts_;
        if constexpr (!SINGLE_SHIFT) {
            NOA_CHECK(shifts.empty() ||
                      (shifts.shape().elements() == output.shape()[0] &&
                       indexing::isVector(shifts.shape()) && shifts.contiguous()),
                      "The number of shifts, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} shifts and {} output batches",
                      shifts.shape().elements(), output.shape()[0]);
            shifts_ = &shifts.share();
        } else {
            shifts_ = &shifts;
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            if constexpr (!SINGLE_MATRIX) {
                NOA_CHECK(matrices.dereferenceable(), "The matrices should be accessible to the CPU");
                if (matrices.device().gpu())
                    Stream::current(matrices.device()).synchronize();
                if constexpr (!SINGLE_SHIFT) {
                    NOA_CHECK(shifts.empty() || shifts.dereferenceable(), "The shifts should be accessible to the CPU");
                    if (!shifts.empty() && shifts.device().gpu() && matrices.device() != shifts.device())
                        Stream::current(shifts.device()).synchronize();
                }
            } else if constexpr (!SINGLE_SHIFT) {
                if (!shifts.empty() && shifts.device().gpu())
                    Stream::current(shifts.device()).synchronize();
                NOA_CHECK(shifts.empty() || shifts.dereferenceable(), "The shifts should be accessible to the CPU");
            }

            cpu::geometry::fft::transform2D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    *matrices_, *shifts_, cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isRightmost(input_strides) && input_strides[3] == 1,
                          "The input should be in the rightmost order and the width dimension should be "
                          "contiguous, but got shape {} and strides {}", input.shape(), input_strides);

                bool sync_cpu = false;
                if (input.device().cpu())
                    sync_cpu = true;
                if constexpr (!SINGLE_MATRIX)
                    if (!sync_cpu && matrices.device().cpu())
                        sync_cpu = true;
                if constexpr (!SINGLE_SHIFT)
                    if (!sync_cpu && !shifts.empty() && shifts.device().cpu())
                        sync_cpu = true;
                if (sync_cpu)
                    Stream::current(Device{}).synchronize();

                cuda::geometry::fft::transform2D<REMAP>(
                        input.share(), input_strides,
                        output.share(), output.strides(), shape,
                        *matrices_, *shifts_, cutoff, interp_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename M, typename S, typename>
    void transform2D(const Texture<T>& input, const Array<T>& output, size4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff) {
        if (input.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            transform2D<REMAP>(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()),
                               output, shape, matrices, shifts, cutoff, input.interp());
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
            NOA_THROW("Double-precision floating-points are not supported");
        } else {
            NOA_CHECK(shape[3] / 2 + 1 == input.shape()[3] && input.shape()[3] == output.shape()[3] &&
                      shape[2] == input.shape()[2] && input.shape()[2] == output.shape()[2],
                      "The non-redundant input {} and/or output {} shapes don't match the logical shape {}",
                      input.shape(), output.shape(), shape);
            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);

            constexpr bool SINGLE_MATRIX = traits::is_float22_v<M>;
            using matrix_t = std::conditional_t<SINGLE_MATRIX, M, shared_t<traits::value_type_t<M>>>;
            const matrix_t* matrices_;
            if constexpr (!SINGLE_MATRIX) {
                NOA_CHECK(matrices.shape().elements() == output.shape()[0] &&
                          indexing::isVector(matrices.shape()) && matrices.contiguous(),
                          "The number of matrices, specified as a contiguous vector, should be equal to the number "
                          "of batches in the output, got {} matrices and {} output batches",
                          matrices.shape().elements(), output.shape()[0]);
                matrices_ = &matrices.share();
            } else {
                matrices_ = &matrices;
            }

            constexpr bool SINGLE_SHIFT = traits::is_float2_v<M>;
            using shift_t = std::conditional_t<SINGLE_SHIFT, S, shared_t<traits::value_type_t<S>>>;
            const shift_t* shifts_;
            if constexpr (!SINGLE_SHIFT) {
                NOA_CHECK(shifts.empty() ||
                          (shifts.shape().elements() == output.shape()[0] &&
                           indexing::isVector(shifts.shape()) && shifts.contiguous()),
                          "The number of shifts, specified as a contiguous vector, should be equal to the number "
                          "of batches in the output, got {} shifts and {} output batches",
                          shifts.shape().elements(), output.shape()[0]);
                shifts_ = &shifts.share();
            } else {
                shifts_ = &shifts;
            }

            const Device device = output.device();
            NOA_CHECK(input.device() == device,
                      "The input texture and output array should be on the same device, but got input:{} and output:{}",
                      input.device(), device);

            bool sync_cpu = false;
            if constexpr (!SINGLE_MATRIX)
                if (matrices.device().cpu())
                    sync_cpu = true;
            if constexpr (!SINGLE_SHIFT)
                if (!shifts.empty() && shifts.device().cpu())
                    sync_cpu = true;
            if (sync_cpu)
                Stream::current(Device{}).synchronize();

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = input.cuda();
            cuda::geometry::fft::transform2D<REMAP>(
                    texture.array, texture.texture, input.interp(),
                    output.share(), output.strides(), output.shape(),
                    *matrices_, *shifts_, cutoff, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<Remap REMAP, typename T, typename M, typename S, typename>
    void transform3D(const Array<T>& input, const Array<T>& output, size4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff, InterpMode interp_mode) {
        NOA_CHECK(shape[3] / 2 + 1 == input.shape()[3] && input.shape()[3] == output.shape()[3] &&
                  shape[2] == input.shape()[2] && input.shape()[2] == output.shape()[2],
                  shape[1] == input.shape()[1] && input.shape()[1] == output.shape()[1],
                  "The non-redundant input {} and/or output {} shapes don't match the logical shape {}",
                  input.shape(), output.shape(), shape);

        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);
        size4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;


        constexpr bool SINGLE_MATRIX = traits::is_float33_v<M>;
        using matrix_t = std::conditional_t<SINGLE_MATRIX, M, shared_t<traits::value_type_t<M>>>;
        const matrix_t* matrices_;
        if constexpr (!SINGLE_MATRIX) {
            NOA_CHECK(matrices.shape().elements() == output.shape()[0] &&
                      indexing::isVector(matrices.shape()) && matrices.contiguous(),
                      "The number of matrices, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} matrices and {} output batches",
                      matrices.shape().elements(), output.shape()[0]);
            matrices_ = &matrices.share();
        } else {
            matrices_ = &matrices;
        }

        constexpr bool SINGLE_SHIFT = traits::is_float3_v<M>;
        using shift_t = std::conditional_t<SINGLE_SHIFT, S, shared_t<traits::value_type_t<S>>>;
        const shift_t* shifts_;
        if constexpr (!SINGLE_SHIFT) {
            NOA_CHECK(shifts.empty() ||
                      (shifts.shape().elements() == output.shape()[0] &&
                       indexing::isVector(shifts.shape()) && shifts.contiguous()),
                      "The number of shifts, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} shifts and {} output batches",
                      shifts.shape().elements(), output.shape()[0]);
            shifts_ = &shifts.share();
        } else {
            shifts_ = &shifts;
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            if constexpr (!SINGLE_MATRIX) {
                NOA_CHECK(matrices.dereferenceable(), "The matrices should be accessible to the CPU");
                if (matrices.device().gpu())
                    Stream::current(matrices.device()).synchronize();
                if constexpr (!SINGLE_SHIFT) {
                    NOA_CHECK(shifts.empty() || shifts.dereferenceable(), "The shifts should be accessible to the CPU");
                    if (!shifts.empty() && shifts.device().gpu() && matrices.device() != shifts.device())
                        Stream::current(shifts.device()).synchronize();
                }
            } else if constexpr (!SINGLE_SHIFT) {
                if (!shifts.empty() && shifts.device().gpu())
                    Stream::current(shifts.device()).synchronize();
                NOA_CHECK(shifts.empty() || shifts.dereferenceable(), "The shifts should be accessible to the CPU");
            }

            cpu::geometry::fft::transform3D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    *matrices_, *shifts_, cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isRightmost(input_strides) &&
                          indexing::isContiguous(input_strides, input.shape())[1] && input_strides[3] == 1,
                          "The input should be in the rightmost order and the depth and width dimension should be "
                          "contiguous, but got shape {} and strides {}", input.shape(), input_strides);

                bool sync_cpu = false;
                if (input.device().cpu())
                    sync_cpu = true;
                if constexpr (!SINGLE_MATRIX)
                    if (!sync_cpu && matrices.device().cpu())
                        sync_cpu = true;
                if constexpr (!SINGLE_SHIFT)
                    if (!sync_cpu && !shifts.empty() && shifts.device().cpu())
                        sync_cpu = true;
                if (sync_cpu)
                    Stream::current(Device{}).synchronize();

                cuda::geometry::fft::transform3D<REMAP>(
                        input.share(), input_strides,
                        output.share(), output.strides(), shape,
                        *matrices_, *shifts_, cutoff, interp_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename M, typename S, typename>
    void transform3D(const Texture<T>& input, const Array<T>& output, size4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff) {
        if (input.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            transform3D<REMAP>(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()),
                               output, shape, matrices, shifts, cutoff, input.interp());
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
            NOA_THROW("Double-precision floating-points are not supported");
        } else {
            NOA_CHECK(shape[3] / 2 + 1 == input.shape()[3] && input.shape()[3] == output.shape()[3] &&
                      shape[2] == input.shape()[2] && input.shape()[2] == output.shape()[2],
                      shape[1] == input.shape()[1] && input.shape()[1] == output.shape()[1],
                      "The non-redundant input {} and/or output {} shapes don't match the logical shape {}",
                      input.shape(), output.shape(), shape);
            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);

            constexpr bool SINGLE_MATRIX = traits::is_float33_v<M>;
            using matrix_t = std::conditional_t<SINGLE_MATRIX, M, shared_t<traits::value_type_t<M>>>;
            const matrix_t* matrices_;
            if constexpr (!SINGLE_MATRIX) {
                NOA_CHECK(matrices.shape().elements() == output.shape()[0] &&
                          indexing::isVector(matrices.shape()) && matrices.contiguous(),
                          "The number of matrices, specified as a contiguous vector, should be equal to the number "
                          "of batches in the output, got {} matrices and {} output batches",
                          matrices.shape().elements(), output.shape()[0]);
                matrices_ = &matrices.share();
            } else {
                matrices_ = &matrices;
            }

            constexpr bool SINGLE_SHIFT = traits::is_float3_v<M>;
            using shift_t = std::conditional_t<SINGLE_SHIFT, S, shared_t<traits::value_type_t<S>>>;
            const shift_t* shifts_;
            if constexpr (!SINGLE_SHIFT) {
                NOA_CHECK(shifts.empty() ||
                          (shifts.shape().elements() == output.shape()[0] &&
                           indexing::isVector(shifts.shape()) && shifts.contiguous()),
                          "The number of shifts, specified as a contiguous vector, should be equal to the number "
                          "of batches in the output, got {} shifts and {} output batches",
                          shifts.shape().elements(), output.shape()[0]);
                shifts_ = &shifts.share();
            } else {
                shifts_ = &shifts;
            }

            const Device device = output.device();
            NOA_CHECK(input.device() == device,
                      "The input texture and output array should be on the same device, but got input:{} and output:{}",
                      input.device(), device);

            bool sync_cpu = false;
            if constexpr (!SINGLE_MATRIX)
                if (matrices.device().cpu())
                    sync_cpu = true;
            if constexpr (!SINGLE_SHIFT)
                if (!shifts.empty() && shifts.device().cpu())
                    sync_cpu = true;
            if (sync_cpu)
                Stream::current(Device{}).synchronize();

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = input.cuda();
            cuda::geometry::fft::transform3D<REMAP>(
                    texture.array, texture.texture, input.interp(),
                    output.share(), output.strides(), output.shape(),
                    *matrices_, *shifts_, cutoff, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }
}

namespace noa::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    template<Remap REMAP, typename T, typename>
    void transform2D(const Array<T>& input, const Array<T>& output, size4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize) {
        NOA_CHECK(shape[3] / 2 + 1 == input.shape()[3] && input.shape()[3] == output.shape()[3] &&
                  shape[2] == input.shape()[2] && input.shape()[2] == output.shape()[2],
                  "The non-redundant input {} and/or output {} shapes don't match the logical shape {}",
                  input.shape(), output.shape(), shape);

        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);
        size4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::fft::transform2D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isRightmost(input_strides) && input_strides[3] == 1,
                          "The input should be in the rightmost order and the width dimension should be "
                          "contiguous, but got shape {} and strides {}", input.shape(), input_strides);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform2D<REMAP>(
                        input.share(), input_strides,
                        output.share(), output.strides(), shape,
                        matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void transform2D(const Texture<T>& input, const Array<T>& output, size4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, bool normalize) {
        if (input.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            transform3D<REMAP>(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()),
                               output, shape, matrix, symmetry, shift, cutoff, input.interp(), normalize);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
            NOA_THROW("Double-precision floating-points are not supported");
        } else {
            NOA_CHECK(shape[3] / 2 + 1 == input.shape()[3] && input.shape()[3] == output.shape()[3] &&
                      shape[2] == input.shape()[2] && input.shape()[2] == output.shape()[2],
                      "The non-redundant input {} and/or output {} shapes don't match the logical shape {}",
                      input.shape(), output.shape(), shape);
            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);

            const Device device = output.device();
            NOA_CHECK(device == input.device(),
                      "The input texture and output array must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = input.cuda();
            cuda::geometry::fft::transform2D<REMAP>(
                    texture.array, texture.texture, input.interp(),
                    output.share(), output.strides(), output.shape(),
                    matrix, symmetry, shift, cutoff, normalize, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<Remap REMAP, typename T, typename>
    void transform3D(const Array<T>& input, const Array<T>& output, size4_t shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize) {
        NOA_CHECK(shape[3] / 2 + 1 == input.shape()[3] && input.shape()[3] == output.shape()[3] &&
                  shape[2] == input.shape()[2] && input.shape()[2] == output.shape()[2],
                  shape[1] == input.shape()[1] && input.shape()[1] == output.shape()[1],
                  "The non-redundant input {} and/or output {} shapes don't match the logical shape {}",
                  input.shape(), output.shape(), shape);

        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);
        size4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::fft::transform3D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isRightmost(input_strides) &&
                          indexing::isContiguous(input_strides, input.shape())[1] && input_strides[3] == 1,
                          "The input should be in the rightmost order and the depth and width dimension should be "
                          "contiguous, but got shape {} and strides {}", input.shape(), input_strides);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform3D<REMAP>(
                        input.share(), input_strides,
                        output.share(), output.strides(), shape,
                        matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void transform3D(const Texture<T>& input, const Array<T>& output, size4_t shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, bool normalize) {
        if (input.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            transform3D<REMAP>(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()),
                               output, shape, matrix, symmetry, shift, cutoff, input.interp(), normalize);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
            NOA_THROW("Double-precision floating-points are not supported");
        } else {
            NOA_CHECK(shape[3] / 2 + 1 == input.shape()[3] && input.shape()[3] == output.shape()[3] &&
                      shape[2] == input.shape()[2] && input.shape()[2] == output.shape()[2],
                      shape[1] == input.shape()[1] && input.shape()[1] == output.shape()[1],
                      "The non-redundant input {} and/or output {} shapes don't match the logical shape {}",
                      input.shape(), output.shape(), shape);
            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);

            const Device device = output.device();
            NOA_CHECK(device == input.device(),
                      "The input texture and output array must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = input.cuda();
            cuda::geometry::fft::transform2D<REMAP>(
                    texture.array, texture.texture, input.interp(),
                    output.share(), output.strides(), output.shape(),
                    matrix, symmetry, shift, cutoff, normalize, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }
}
