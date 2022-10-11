#pragma once

#ifndef NOA_UNIFIED_SCALE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/geometry/Scale.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Scale.h"
#endif

namespace noa::geometry {
    template<typename T, typename S, typename C, typename>
    void scale2D(const Array<T>& input, const Array<T>& output,
                 const S& scaling_factors, const C& scaling_centers,
                 InterpMode interp_mode, BorderMode border_mode,
                 T value, bool prefilter) {
        constexpr bool SINGLE_SCALE = traits::is_float2_v<S>;
        constexpr bool SINGLE_CENTER = traits::is_float2_v<C>;
        using scale_t = traits::shared_type_t<S>;
        using center_t = traits::shared_type_t<C>;
        const scale_t* scaling_factors_;
        const center_t* centers_;

        if constexpr (!SINGLE_SCALE) {
            NOA_CHECK(indexing::isVector(scaling_factors.shape()) &&
                      scaling_factors.elements() == output.shape()[0] &&
                      scaling_factors.contiguous(),
                      "The number of scaling factors, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} scaling factors and {} output batches",
                      scaling_factors.elements(), output.shape()[0]);
            NOA_CHECK(scaling_factors.dereferenceable(), "The scaling factors should be accessible to the CPU");
            scaling_factors_ = &scaling_factors.share();
        } else {
            scaling_factors_ = &scaling_factors;
        }

        if constexpr (!SINGLE_CENTER) {
            NOA_CHECK(indexing::isVector(scaling_centers.shape()) &&
                      scaling_centers.elements() == output.shape()[0] &&
                      scaling_centers.contiguous(),
                      "The number of scaling centers, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} scaling centers and {} output batches",
                      scaling_centers.elements(), output.shape()[0]);
            NOA_CHECK(scaling_centers.dereferenceable(), "The scaling centers should be accessible to the CPU");
            centers_ = &scaling_centers.share();
        } else {
            centers_ = &scaling_centers;
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

            if constexpr (!SINGLE_SCALE) {
                if (scaling_factors.device().gpu())
                    Stream::current(scaling_factors.device()).synchronize();
                if constexpr (!SINGLE_CENTER) {
                    if (scaling_centers.device().gpu() && scaling_factors.device() != scaling_centers.device())
                        Stream::current(scaling_centers.device()).synchronize();
                }
            } else if constexpr (!SINGLE_CENTER) {
                if (scaling_centers.device().gpu())
                    Stream::current(scaling_centers.device()).synchronize();
            }

            cpu::geometry::rotate2D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    *scaling_factors_, *centers_,
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
                if constexpr (!SINGLE_SCALE)
                    sync_cpu = scaling_factors.device().cpu() ? true : sync_cpu;
                if constexpr (!SINGLE_CENTER)
                    sync_cpu = scaling_centers.device().cpu() ? true : sync_cpu;
                if (sync_cpu)
                    Stream::current(Device{}).synchronize();

                cuda::geometry::rotate2D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        *scaling_factors_, *centers_,
                        interp_mode, border_mode, prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename S, typename C, typename>
    void scale3D(const Array<T>& input, const Array<T>& output,
                 const S& scaling_factors, const C& scaling_centers,
                 InterpMode interp_mode, BorderMode border_mode,
                 T value, bool prefilter) {
        constexpr bool SINGLE_SCALE = traits::is_float3_v<S>;
        constexpr bool SINGLE_CENTER = traits::is_float3_v<C>;
        using scale_t = traits::shared_type_t<S>;
        using center_t = traits::shared_type_t<C>;
        const scale_t* scaling_factors_;
        const center_t* centers_;

        if constexpr (!SINGLE_SCALE) {
            NOA_CHECK(indexing::isVector(scaling_factors.shape()) &&
                      scaling_factors.elements() == output.shape()[0] &&
                      scaling_factors.contiguous(),
                      "The number of scaling factors, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} scaling factors and {} output batches",
                      scaling_factors.elements(), output.shape()[0]);
            NOA_CHECK(scaling_factors.dereferenceable(), "The scaling factors should be accessible to the CPU");
            scaling_factors_ = &scaling_factors.share();
        } else {
            scaling_factors_ = &scaling_factors;
        }

        if constexpr (!SINGLE_CENTER) {
            NOA_CHECK(indexing::isVector(scaling_centers.shape()) &&
                      scaling_centers.elements() == output.shape()[0] &&
                      scaling_centers.contiguous(),
                      "The number of scaling centers, specified as a contiguous vector, should be equal to the number "
                      "of batches in the output, got {} scaling centers and {} output batches",
                      scaling_centers.elements(), output.shape()[0]);
            NOA_CHECK(scaling_centers.dereferenceable(), "The scaling centers should be accessible to the CPU");
            centers_ = &scaling_centers.share();
        } else {
            centers_ = &scaling_centers;
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

            if constexpr (!SINGLE_SCALE) {
                if (scaling_factors.device().gpu())
                    Stream::current(scaling_factors.device()).synchronize();
                if constexpr (!SINGLE_CENTER) {
                    if (scaling_centers.device().gpu() && scaling_factors.device() != scaling_centers.device())
                        Stream::current(scaling_centers.device()).synchronize();
                }
            } else if constexpr (!SINGLE_CENTER) {
                if (scaling_centers.device().gpu())
                    Stream::current(scaling_centers.device()).synchronize();
            }

            cpu::geometry::rotate3D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    *scaling_factors_, *centers_,
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
                if constexpr (!SINGLE_SCALE)
                    sync_cpu = scaling_factors.device().cpu() ? true : sync_cpu;
                if constexpr (!SINGLE_CENTER)
                    sync_cpu = scaling_centers.device().cpu() ? true : sync_cpu;
                if (sync_cpu)
                    Stream::current(Device{}).synchronize();

                cuda::geometry::rotate3D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        *scaling_factors_, *centers_,
                        interp_mode, border_mode, prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
