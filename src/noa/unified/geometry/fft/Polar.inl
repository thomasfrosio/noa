#pragma once

#ifndef NOA_UNIFIED_FFT_POLAR_
#error "This is a private header"
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
        NOA_CHECK(cartesian.shape()[0] == 1 || cartesian.shape()[0] == polar.shape()[0],
                  "The number of batches in the cartesian array ({}) is not compatible with the number of "
                  "batches in the polar array ({})", cartesian.shape()[0], polar.shape()[0]);
        NOA_CHECK(all(cartesian.shape() == cartesian_shape.fft()),
                  "The non-redundant FFT with shape {} doesn't match the logical shape {}",
                  cartesian.shape(), cartesian_shape);

        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == cartesian.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", cartesian.device(), device);
            NOA_CHECK(cartesian.get() != polar.get(), "In-place transformations are not supported");

            cpu::geometry::fft::cartesian2polar<REMAP>(
                    cartesian.share(), cartesian.strides(), cartesian_shape,
                    polar.share(), polar.strides(), polar.shape(),
                    frequency_range, angle_range, log, interp, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                NOA_CHECK(cartesian.contiguous()[3],
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          cartesian.shape(), cartesian.strides());

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
}
