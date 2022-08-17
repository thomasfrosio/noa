#pragma once

#ifndef NOA_UNIFIED_ROTATE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/geometry/Rotate.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Rotate.h"
#endif

namespace noa::geometry {
    template<typename T, typename R, typename C, typename>
    void rotate2D(const Array<T>& input, const Array<T>& output,
                  const R& rotations, const C& rotation_centers,
                  InterpMode interp_mode, BorderMode border_mode,
                  T value, bool prefilter) {
        constexpr bool SINGLE_ROTATION = traits::is_float_v<R>;
        constexpr bool SINGLE_CENTER = traits::is_float2_v<C>;
        using rotation_t = std::conditional_t<SINGLE_ROTATION, R, shared_t<traits::value_type_t<R>>>;
        using center_t = std::conditional_t<SINGLE_CENTER, C, shared_t<traits::value_type_t<C>>>;
        const rotation_t* rotations_;
        const center_t* centers_;

        if constexpr (!SINGLE_ROTATION) {
            NOA_CHECK(indexing::isVector(rotations.shape()) &&
                      rotations.shape().elements() == output.shape()[0] &&
                      rotations.contiguous(),
                      "The number of rotations, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} rotations and {} output batches",
                      rotations.shape().elements(), output.shape()[0]);
            NOA_CHECK(rotations.dereferenceable(), "The rotations should be accessible to the CPU");
            rotations_ = &rotations.share();
        } else {
            rotations_ = &rotations;
        }

        if constexpr (!SINGLE_CENTER) {
            NOA_CHECK(indexing::isVector(rotation_centers.shape()) &&
                      rotation_centers.shape().elements() == output.shape()[0] &&
                      rotation_centers.contiguous(),
                      "The number of rotation centers, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} rotation centers and {} output batches",
                      rotation_centers.shape().elements(), output.shape()[0]);
            NOA_CHECK(rotation_centers.dereferenceable(), "The rotations centers should be accessible to the CPU");
            centers_ = &rotation_centers.share();
        } else {
            centers_ = &rotation_centers;
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

            if constexpr (!SINGLE_ROTATION) {
                if (rotations.device().gpu())
                    Stream::current(rotations.device()).synchronize();
                if constexpr (!SINGLE_CENTER) {
                    if (rotation_centers.device().gpu() && rotations.device() != rotation_centers.device())
                        Stream::current(rotation_centers.device()).synchronize();
                }
            } else if constexpr (!SINGLE_CENTER) {
                if (rotation_centers.device().gpu())
                    Stream::current(rotation_centers.device()).synchronize();
            }

            cpu::geometry::rotate2D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    *rotations_, *centers_,
                    interp_mode, border_mode, value, prefilter, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
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
                if constexpr (!SINGLE_ROTATION)
                    sync_cpu = rotations.device().cpu() ? true : sync_cpu;
                if constexpr (!SINGLE_CENTER)
                    sync_cpu = rotation_centers.device().cpu() ? true : sync_cpu;
                if (sync_cpu)
                    Stream::current(Device{}).synchronize();

                cuda::geometry::rotate2D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        *rotations_, *centers_,
                        interp_mode, border_mode, prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename R, typename C, typename>
    void rotate3D(const Array<T>& input, const Array<T>& output,
                  const R& rotations, const C& rotation_centers,
                  InterpMode interp_mode, BorderMode border_mode,
                  T value, bool prefilter) {
        constexpr bool SINGLE_ROTATION = traits::is_float33_v<R>;
        constexpr bool SINGLE_CENTER = traits::is_float3_v<C>;
        using rotation_t = std::conditional_t<SINGLE_ROTATION, R, shared_t<traits::value_type_t<R>>>;
        using center_t = std::conditional_t<SINGLE_CENTER, C, shared_t<traits::value_type_t<C>>>;
        const rotation_t* rotations_;
        const center_t* centers_;

        if constexpr (!SINGLE_ROTATION) {
            NOA_CHECK(indexing::isVector(rotations.shape()) &&
                      rotations.shape().elements() == output.shape()[0] &&
                      rotations.contiguous(),
                      "The number of rotations, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} rotations and {} output batches",
                      rotations.shape().elements(), output.shape()[0]);
            NOA_CHECK(rotations.dereferenceable(), "The rotations should be accessible to the CPU");
            rotations_ = &rotations.share();
        } else {
            rotations_ = &rotations;
        }

        if constexpr (!SINGLE_CENTER) {
            NOA_CHECK(indexing::isVector(rotation_centers.shape()) &&
                      rotation_centers.shape().elements() == output.shape()[0] &&
                      rotation_centers.contiguous(),
                      "The number of rotation centers, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} rotation centers and {} output batches",
                      rotation_centers.shape().elements(), output.shape()[0]);
            NOA_CHECK(rotation_centers.dereferenceable(), "The rotations centers should be accessible to the CPU");
            centers_ = &rotation_centers.share();
        } else {
            centers_ = &rotation_centers;
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

            if constexpr (!SINGLE_ROTATION) {
                if (rotations.device().gpu())
                    Stream::current(rotations.device()).synchronize();
                if constexpr (!SINGLE_CENTER) {
                    if (rotation_centers.device().gpu() && rotations.device() != rotation_centers.device())
                        Stream::current(rotation_centers.device()).synchronize();
                }
            } else if constexpr (!SINGLE_CENTER) {
                if (rotation_centers.device().gpu())
                    Stream::current(rotation_centers.device()).synchronize();
            }

            cpu::geometry::rotate3D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    *rotations_, *centers_,
                    interp_mode, border_mode, value, prefilter, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
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
                if constexpr (!SINGLE_ROTATION)
                    sync_cpu = rotations.device().cpu() ? true : sync_cpu;
                if constexpr (!SINGLE_CENTER)
                    sync_cpu = rotation_centers.device().cpu() ? true : sync_cpu;
                if (sync_cpu)
                    Stream::current(Device{}).synchronize();

                cuda::geometry::rotate3D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        *rotations_, *centers_,
                        interp_mode, border_mode, prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
