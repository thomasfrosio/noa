#pragma once

#ifndef NOA_UNIFIED_POLAR_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/geometry/Polar.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Polar.h"
#endif

namespace noa::geometry {
    template<typename T, typename>
    void cartesian2polar(const Array<T>& cartesian, const Array<T>& polar,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter) {
        NOA_CHECK(!cartesian.empty() && !polar.empty(), "Empty array detected");
        NOA_CHECK(cartesian.shape()[0] == 1 || cartesian.shape()[0] == polar.shape()[0],
                  "The number of batches in the cartesian array ({}) is not compatible with the number of "
                  "batches in the polar array ({})", cartesian.shape()[0], polar.shape()[0]);
        NOA_CHECK(cartesian.shape()[1] == 1 && polar.shape()[1] == 1, "3D arrays are not supported");

        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == cartesian.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", cartesian.device(), device);
            NOA_CHECK(cartesian.get() != polar.get(), "In-place transformations are not supported");
            NOA_CHECK(!indexing::isOverlap(cartesian, polar), "Input and output arrays should not overlap");

            cpu::geometry::cartesian2polar(
                    cartesian.share(), cartesian.strides(), cartesian.shape(),
                    polar.share(), polar.strides(), polar.shape(),
                    cartesian_center, radius_range, angle_range, log, interp, prefilter, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        prefilter && (interp == INTERP_CUBIC_BSPLINE || interp == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == cartesian.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", cartesian.device(), device);
                NOA_CHECK(indexing::isRightmost(cartesian.strides()) && cartesian.strides()[3] == 1,
                          "The input should be in the rightmost order and its width dimension should be contiguous, "
                          "but got shape {} and strides {}", cartesian.shape(), cartesian.strides());

                cuda::geometry::cartesian2polar(
                        cartesian.share(), cartesian.strides(), cartesian.shape(),
                        polar.share(), polar.strides(), polar.shape(),
                        cartesian_center, radius_range, angle_range, log, interp, prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void cartesian2polar(const Texture<T>& cartesian, const Array<T>& polar,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log) {
        NOA_CHECK(cartesian.border() == BORDER_ZERO,
                  "The texture should use the {} mode, but got {}",
                  BORDER_ZERO, cartesian.border());

        if (cartesian.device().cpu()) {
            const cpu::Texture<T>& texture = cartesian.cpu();
            cartesian2polar(Array<T>(texture.ptr, cartesian.shape(), texture.strides, cartesian.options()), polar,
                            cartesian_center, radius_range, angle_range, log, cartesian.interp(), false);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
        } else {
            NOA_CHECK(cartesian.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", cartesian.shape()[0]);
            NOA_CHECK(cartesian.shape()[1] == 1 && polar.shape()[1] == 1,
                      "The input texture and output array should be 2D, but got shape input:{}, output:{}",
                      cartesian.shape(), polar.shape());

            const Device device = cartesian.device();
            NOA_CHECK(device == polar.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", device, polar.device());

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = cartesian.cuda();
            cuda::geometry::cartesian2polar(
                    texture.array, texture.texture, cartesian.interp(),
                    polar.share(), polar.strides(), polar.shape(),
                    cartesian_center, radius_range, angle_range, log, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename T, typename>
    void polar2cartesian(const Array<T>& polar, const Array<T>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter) {
        NOA_CHECK(!cartesian.empty() && !polar.empty(), "Empty array detected");
        NOA_CHECK(polar.shape()[0] == 1 || polar.shape()[0] == cartesian.shape()[0],
                  "The number of batches in the polar array ({}) is not compatible with the number of "
                  "batches in the cartesian array ({})", polar.shape()[0], cartesian.shape()[0]);
        NOA_CHECK(cartesian.shape()[1] == 1 && polar.shape()[1] == 1, "3D arrays are not supported");

        const Device device = cartesian.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == polar.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", polar.device(), device);
            NOA_CHECK(!indexing::isOverlap(cartesian, polar), "Input and output arrays should not overlap");

            cpu::geometry::cartesian2polar(
                    polar.share(), polar.strides(), polar.shape(),
                    cartesian.share(), cartesian.strides(), cartesian.shape(),
                    cartesian_center, radius_range, angle_range, log, interp, prefilter, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        prefilter && (interp == INTERP_CUBIC_BSPLINE || interp == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == polar.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", polar.device(), device);
                NOA_CHECK(indexing::isRightmost(polar.strides()) && polar.contiguous()[3],
                          "The input should be in the rightmost order and its width dimension should be contiguous, "
                          "but got shape {} and strides {}", polar.shape(), polar.strides());

                cuda::geometry::cartesian2polar(
                        polar.share(), polar.strides(), polar.shape(),
                        cartesian.share(), cartesian.strides(), cartesian.shape(),
                        cartesian_center, radius_range, angle_range, log, interp, prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void polar2cartesian(const Texture<T>& polar, const Array<T>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log) {
        NOA_CHECK(polar.border() == BORDER_ZERO,
                  "The texture should use the {} mode, but got {}",
                  BORDER_ZERO, polar.border());

        if (polar.device().cpu()) {
            const cpu::Texture<T>& texture = polar.cpu();
            polar2cartesian(Array<T>(texture.ptr, polar.shape(), texture.strides, polar.options()), cartesian,
                            cartesian_center, radius_range, angle_range, log, polar.interp(), false);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
        } else {
            NOA_CHECK(polar.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", polar.shape()[0]);
            NOA_CHECK(polar.shape()[1] == 1 && polar.shape()[1] == 1,
                      "The input texture and output array should be 2D, but got shape input:{}, output:{}",
                      polar.shape(), cartesian.shape());

            const Device device = polar.device();
            NOA_CHECK(device == cartesian.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", device, cartesian.device());

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = polar.cuda();
            cuda::geometry::polar2cartesian(
                    texture.array, texture.texture, polar.interp(), float2_t(polar.shape().get(2)),
                    cartesian.share(), cartesian.strides(), cartesian.shape(),
                    cartesian_center, radius_range, angle_range, log, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }
}
