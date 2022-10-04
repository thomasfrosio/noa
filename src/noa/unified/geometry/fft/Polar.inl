#pragma once

#ifndef NOA_UNIFIED_FFT_POLAR_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/geometry/fft/Polar.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Polar.h"
#endif

namespace noa::geometry::fft {
    template<Remap REMAP, typename T, typename>
    void cartesian2polar(const Array<T>& cartesian, size4_t cartesian_shape, const Array<T>& polar,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, InterpMode interp) {
        NOA_CHECK(!cartesian.empty() && !polar.empty(), "Empty array detected");
        NOA_CHECK(cartesian.shape()[0] == 1 || cartesian.shape()[0] == polar.shape()[0],
                  "The number of batches in the cartesian array ({}) is not compatible with the number of "
                  "batches in the polar array ({})", cartesian.shape()[0], polar.shape()[0]);
        NOA_CHECK(all(cartesian.shape() == cartesian_shape.fft()),
                  "The non-redundant FFT with shape {} doesn't match the logical shape {}",
                  cartesian.shape(), cartesian_shape);
        NOA_CHECK(cartesian.shape()[1] == 1 && polar.shape()[1] == 1, "3D arrays are not supported");

        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == cartesian.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", cartesian.device(), device);
            NOA_CHECK(!indexing::isOverlap(cartesian, polar), "Input and output arrays should not overlap");

            cpu::geometry::fft::cartesian2polar<REMAP>(
                    cartesian.share(), cartesian.strides(), cartesian_shape,
                    polar.share(), polar.strides(), polar.shape(),
                    frequency_range, angle_range, log, interp, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isRightmost(cartesian.strides()) && cartesian.strides()[3] == 1,
                          "The input should be in the rightmost order and the width dimension should be contiguous, "
                          "but got shape {} and strides {}", cartesian.shape(), cartesian.strides());

                if (cartesian.device().cpu())
                    Stream::current(Device{}).synchronize();

                cuda::geometry::fft::cartesian2polar<REMAP>(
                        cartesian.share(), cartesian.strides(), cartesian_shape,
                        polar.share(), polar.strides(), polar.shape(),
                        frequency_range, angle_range, log, interp, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void cartesian2polar(const Texture<T>& cartesian, size4_t cartesian_shape, const Array<T>& polar,
                         float2_t frequency_range, float2_t angle_range,
                         bool log) {
        NOA_CHECK(!cartesian.empty() && !polar.empty(), "Empty array detected");

        if (cartesian.device().cpu()) {
            const cpu::Texture<T>& texture = cartesian.cpu();
            cartesian2polar<REMAP>(Array<T>(texture.ptr, cartesian.shape(), texture.strides, cartesian.options()),
                                   cartesian_shape, polar, frequency_range, angle_range, log, cartesian.interp());
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported by this function");
        } else {
            NOA_CHECK(cartesian.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", cartesian.shape()[0]);

            const Device device = polar.device();
            NOA_CHECK(device == cartesian.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", cartesian.device(), device);

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = cartesian.cuda();
            cuda::geometry::fft::cartesian2polar<REMAP>(
                    texture.array, texture.texture, cartesian.interp(),
                    polar.share(), polar.strides(), polar.shape(),
                    frequency_range, angle_range, log, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }
}
