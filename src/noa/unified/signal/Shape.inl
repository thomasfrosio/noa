#pragma once

#ifndef NOA_UNIFIED_SHAPE_
#error "This is a private header"
#endif

#include "noa/cpu/signal/Shape.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Shape.h"
#endif

namespace noa::filter {
    template<bool INVERT, typename T, typename>
    void sphere(const Array<T>& input, const Array<T>& output,
                float3_t center, float radius, float taper_size) {
        size4_t input_stride = input.stride();
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
            cpu::signal::sphere(input.share(), input_stride,
                                output.share(), output.stride(), output.shape(),
                                center, radius, taper_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::sphere(input.share(), input_stride,
                                 output.share(), output.stride(), output.shape(),
                                 center, radius, taper_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<bool INVERT, typename T, typename>
    void sphere(const Array<T>& input, const Array<T>& output,
                float2_t center, float radius, float taper_size) {
        return sphere<INVERT>(input, output, float3_t{0, center[0], center[1]}, radius, taper_size);
    }

    template<bool INVERT, typename T, typename>
    void rectangle(const Array<T>& input, const Array<T>& output,
                   float3_t center, float3_t radius, float taper_size) {
        size4_t input_stride = input.stride();
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
            cpu::signal::rectangle(input.share(), input_stride,
                                   output.share(), output.stride(), output.shape(),
                                   center, radius, taper_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::rectangle(input.share(), input_stride,
                                    output.share(), output.stride(), output.shape(),
                                    center, radius, taper_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<bool INVERT, typename T, typename>
    void rectangle(const Array<T>& input, const Array<T>& output,
                   float2_t center, float2_t radius, float taper_size) {
        return rectangle<INVERT>(input, output, float3_t{0, center[0], center[1]},
                                 float3_t{1, radius[0], radius[1]}, taper_size);
    }

    template<bool INVERT, typename T, typename>
    void cylinder(const Array<T>& input, const Array<T>& output,
                  float3_t center, float radius, float length, float taper_size) {
        size4_t input_stride = input.stride();
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
            cpu::signal::cylinder(input.share(), input_stride,
                                  output.share(), output.stride(), output.shape(),
                                  center, radius, length, taper_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::cylinder(input.share(), input_stride,
                                   output.share(), output.stride(), output.shape(),
                                   center, radius, length, taper_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
