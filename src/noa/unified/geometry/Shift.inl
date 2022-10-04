#pragma once

#ifndef NOA_UNIFIED_SHIFT_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/geometry/Shift.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Shift.h"
#endif

namespace noa::geometry {
    template<typename T, typename S, typename>
    void shift2D(const Array<T>& input, const Array<T>& output, const S& shifts,
                 InterpMode interp_mode, BorderMode border_mode, T value, bool prefilter) {
        constexpr bool SINGLE_SHIFT = traits::is_float2_v<S>;
        using shift_t = std::conditional_t<SINGLE_SHIFT, S, shared_t<traits::value_type_t<S>>>;
        const shift_t* shifts_;

        if constexpr (!SINGLE_SHIFT) {
            NOA_CHECK(indexing::isVector(shifts.shape()) &&
                      shifts.elements() == output.shape()[0] &&
                      shifts.contiguous(),
                      "The number of shifts, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} shifts and {} output batches",
                      shifts.elements(), output.shape()[0]);
            shifts_ = &shifts.share();
        } else {
            shifts_ = &shifts;
        }

        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");
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
            NOA_CHECK(!indexing::isOverlap(input, output), "Input and output arrays should not overlap");

            if constexpr (!SINGLE_SHIFT) {
                NOA_CHECK(shifts.dereferenceable(), "The rotation parameters should be accessible to the CPU");
                if (shifts.device().gpu())
                    Stream::current(shifts.device()).synchronize();
            }

            cpu::geometry::shift2D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    *shifts_, interp_mode, border_mode, value, prefilter, stream.cpu());
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
                if constexpr (!SINGLE_SHIFT)
                    sync_cpu = shifts.device().cpu() ? true : sync_cpu;
                if (sync_cpu)
                    Stream::current(Device{}).synchronize();

                cuda::geometry::shift2D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        *shifts_, interp_mode, border_mode, prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename S, typename>
    void shift2D(const Texture<T>& input, const Array<T>& output, const S& shifts) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");

        if (input.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            shift2D(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()), output, shifts,
                    input.interp(), input.border(), texture.cvalue, false);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
        } else {
            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);
            NOA_CHECK(input.shape()[1] == 1 && output.shape()[1] == 1,
                      "The input and output arrays should be 2D, but got shape input:{}, output:{}",
                      input.shape(), output.shape());

            constexpr bool SINGLE_SHIFT = traits::is_float2_v<S>;
            using shift_t = std::conditional_t<SINGLE_SHIFT, S, shared_t<traits::value_type_t<S>>>;
            const shift_t* shifts_;

            if constexpr (!SINGLE_SHIFT) {
                NOA_CHECK(indexing::isVector(shifts.shape()) &&
                          shifts.elements() == output.shape()[0] &&
                          shifts.contiguous(),
                          "The number of shifts, specified as a contiguous vector, should be equal to the number "
                          "of batches in the output, got {} shifts and {} output batches",
                          shifts.elements(), output.shape()[0]);

                if (shifts.device().cpu())
                    Stream::current(Device{}).synchronize();

                shifts_ = &shifts.share();
            } else {
                shifts_ = &shifts;
            }

            const Device device = output.device();
            NOA_CHECK(device == input.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = input.cuda();
            cuda::geometry::shift2D(
                    texture.array, texture.texture, size2_t(input.shape().get(2)),
                    input.interp(), input.border(), output.share(), output.strides(), output.shape(),
                    *shifts_, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename T, typename S, typename>
    void shift3D(const Array<T>& input, const Array<T>& output, const S& shifts,
                 InterpMode interp_mode, BorderMode border_mode, T value, bool prefilter) {
        constexpr bool SINGLE_SHIFT = traits::is_float3_v<S>;
        using shift_t = std::conditional_t<SINGLE_SHIFT, S, shared_t<traits::value_type_t<S>>>;
        const shift_t* shifts_;

        if constexpr (!SINGLE_SHIFT) {
            NOA_CHECK(indexing::isVector(shifts.shape()) &&
                      shifts.elements() == output.shape()[0] &&
                      shifts.contiguous(),
                      "The number of shifts, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} shifts and {} output batches",
                      shifts.elements(), output.shape()[0]);
            shifts_ = &shifts.share();
        } else {
            shifts_ = &shifts;
        }

        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(!indexing::isOverlap(input, output), "Input and output arrays should not overlap");

            if constexpr (!SINGLE_SHIFT) {
                NOA_CHECK(shifts.dereferenceable(), "The rotation parameters should be accessible to the CPU");
                if (shifts.device().gpu())
                    Stream::current(shifts.device()).synchronize();
            }

            cpu::geometry::shift3D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    *shifts_, interp_mode, border_mode, value, prefilter, stream.cpu());
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
                if constexpr (!SINGLE_SHIFT)
                    sync_cpu = shifts.device().cpu() ? true : sync_cpu;
                if (sync_cpu)
                    Stream::current(Device{}).synchronize();

                cuda::geometry::shift2D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        *shifts_, interp_mode, border_mode, prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename S, typename>
    void shift3D(const Texture<T>& input, const Array<T>& output, const S& shifts) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");

        if (input.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            shift3D(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()), output, shifts,
                    input.interp(), input.border(), texture.cvalue, false);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
        } else {
            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);

            constexpr bool SINGLE_SHIFT = traits::is_float3_v<S>;
            using shift_t = std::conditional_t<SINGLE_SHIFT, S, shared_t<traits::value_type_t<S>>>;
            const shift_t* shifts_;

            if constexpr (!SINGLE_SHIFT) {
                NOA_CHECK(indexing::isVector(shifts.shape()) &&
                          shifts.elements() == output.shape()[0] &&
                          shifts.contiguous(),
                          "The number of shifts, specified as a contiguous vector, should be equal to the number "
                          "of batches in the output, got {} shifts and {} output batches",
                          shifts.elements(), output.shape()[0]);

                if (shifts.device().cpu())
                    Stream::current(Device{}).synchronize();

                shifts_ = &shifts.share();
            } else {
                shifts_ = &shifts;
            }

            const Device device = output.device();
            NOA_CHECK(device == input.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = input.cuda();
            cuda::geometry::shift3D(
                    texture.array, texture.texture, size2_t(input.shape().get(2)),
                    input.interp(), input.border(), output.share(), output.strides(), output.shape(),
                    *shifts_, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }
}
