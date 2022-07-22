#pragma once

#ifndef NOA_UNIFIED_PREFILTER_
#error "This is a private header"
#endif

#include "noa/cpu/geometry/Prefilter.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Prefilter.h"
#endif

namespace noa::geometry::bspline {
    template<typename T, typename>
    void prefilter(const Array<T>& input, const Array<T>& output) {
        size4_t input_stride = input.strides();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::bspline::prefilter(
                    input.share(), input_stride,
                    output.share(), output.strides(),
                    output.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::bspline::prefilter(
                    input.share(), input_stride,
                    output.share(), output.strides(),
                    output.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
