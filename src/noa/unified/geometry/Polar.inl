#pragma once

#ifndef NOA_UNIFIED_POLAR_
#error "This is a private header"
#endif

#include "noa/cpu/geometry/Polar.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Polar.h"
#endif


namespace noa::geometry {
    template<bool PREFILTER, typename T, typename>
    void cartesian2polar(const Array<T>& cartesian, const Array<T>& polar,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp) {
        NOA_CHECK(cartesian.shape()[0] == 1 || cartesian.shape()[0] == polar.shape()[0],
                  "The number of batches in the cartesian array ({}) is not compatible with the number of "
                  "batches in the polar array ({})", cartesian.shape()[0], polar.shape()[0]);

        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == cartesian.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", cartesian.device(), device);
            NOA_CHECK(cartesian.get() != polar.get(), "In-place transformations are not supported");

            cpu::geometry::cartesian2polar<PREFILTER>(
                    cartesian.share(), cartesian.stride(), cartesian.shape(),
                    polar.share(), polar.stride(), polar.shape(),
                    cartesian_center, radius_range, angle_range, log, interp, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        PREFILTER && (interp == INTERP_CUBIC_BSPLINE || interp == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == cartesian.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", cartesian.device(), device);
                NOA_CHECK(cartesian.contiguous()[3],
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          cartesian.shape(), cartesian.stride());

                cuda::geometry::cartesian2polar<PREFILTER>(
                        cartesian.share(), cartesian.stride(), cartesian.shape(),
                        polar.share(), polar.stride(), polar.shape(),
                        cartesian_center, radius_range, angle_range, log, interp, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<bool PREFILTER, typename T, typename>
    void polar2cartesian(const Array<T>& polar, const Array<T>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp) {
        NOA_CHECK(polar.shape()[0] == 1 || polar.shape()[0] == cartesian.shape()[0],
                  "The number of batches in the polar array ({}) is not compatible with the number of "
                  "batches in the cartesian array ({})", polar.shape()[0], cartesian.shape()[0]);

        const Device device = cartesian.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == polar.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", polar.device(), device);
            NOA_CHECK(polar.get() != cartesian.get(), "In-place transformations are not supported");

            cpu::geometry::cartesian2polar<PREFILTER>(
                    polar.share(), polar.stride(), polar.shape(),
                    cartesian.share(), cartesian.stride(), cartesian.shape(),
                    cartesian_center, radius_range, angle_range, log, interp, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        PREFILTER && (interp == INTERP_CUBIC_BSPLINE || interp == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == polar.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", polar.device(), device);
                NOA_CHECK(polar.contiguous()[3],
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          polar.shape(), polar.stride());

                cuda::geometry::cartesian2polar<PREFILTER>(
                        polar.share(), polar.stride(), polar.shape(),
                        cartesian.share(), cartesian.stride(), cartesian.shape(),
                        cartesian_center, radius_range, angle_range, log, interp, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
