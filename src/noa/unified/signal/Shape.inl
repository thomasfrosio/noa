#pragma once

#ifndef NOA_UNIFIED_SHAPE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/signal/Shape.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Shape.h"
#endif

namespace noa::signal {
    template<typename T, typename>
    void ellipse(const Array<T>& input, const Array<T>& output,
                 float3_t center, float3_t radius, float taper_size, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        const bool is_empty = input.empty();
        dim4_t input_strides = input.strides();
        if (!is_empty && !indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(is_empty || device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            if (invert) {
                cpu::signal::ellipse<true>(input.share(), input_strides,
                                           output.share(), output.strides(), output.shape(),
                                           center, radius, taper_size, stream.cpu());
            } else {
                cpu::signal::ellipse<false>(input.share(), input_strides,
                                            output.share(), output.strides(), output.shape(),
                                            center, radius, taper_size, stream.cpu());
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            if (invert) {
                cuda::signal::ellipse<true>(input.share(), input_strides,
                                            output.share(), output.strides(), output.shape(),
                                            center, radius, taper_size, stream.cuda());
            } else {
                cuda::signal::ellipse<false>(input.share(), input_strides,
                                             output.share(), output.strides(), output.shape(),
                                             center, radius, taper_size, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void ellipse(const Array<T>& input, const Array<T>& output,
                 float2_t center, float2_t radius, float taper_size, bool invert) {
        return sphere(input, output, float3_t{0, center[0], center[1]}, float3_t{0, radius[0], radius[1]},
                      taper_size, invert);
    }

    template<typename T, typename>
    void sphere(const Array<T>& input, const Array<T>& output,
                float3_t center, float radius, float taper_size, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        const bool is_empty = input.empty();
        dim4_t input_strides = input.strides();
        if (!is_empty && !indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(is_empty || device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            if (invert) {
                cpu::signal::sphere<true>(input.share(), input_strides,
                                          output.share(), output.strides(), output.shape(),
                                          center, radius, taper_size, stream.cpu());
            } else {
                cpu::signal::sphere<false>(input.share(), input_strides,
                                           output.share(), output.strides(), output.shape(),
                                           center, radius, taper_size, stream.cpu());
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            if (invert) {
                cuda::signal::sphere<true>(input.share(), input_strides,
                                           output.share(), output.strides(), output.shape(),
                                           center, radius, taper_size, stream.cuda());
            } else {
                cuda::signal::sphere<false>(input.share(), input_strides,
                                            output.share(), output.strides(), output.shape(),
                                            center, radius, taper_size, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void sphere(const Array<T>& input, const Array<T>& output,
                float2_t center, float radius, float taper_size, bool invert) {
        return sphere(input, output, float3_t{0, center[0], center[1]}, radius, taper_size, invert);
    }

    template<typename T, typename>
    void rectangle(const Array<T>& input, const Array<T>& output,
                   float3_t center, float3_t radius, float taper_size, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        const bool is_empty = input.empty();
        dim4_t input_strides = input.strides();
        if (!is_empty && !indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(is_empty || device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            if (invert) {
                cpu::signal::rectangle<true>(input.share(), input_strides,
                                             output.share(), output.strides(), output.shape(),
                                             center, radius, taper_size, stream.cpu());
            } else {
                cpu::signal::rectangle<false>(input.share(), input_strides,
                                              output.share(), output.strides(), output.shape(),
                                              center, radius, taper_size, stream.cpu());
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            if (invert) {
                cuda::signal::rectangle<true>(input.share(), input_strides,
                                              output.share(), output.strides(), output.shape(),
                                              center, radius, taper_size, stream.cuda());
            } else {
                cuda::signal::rectangle<false>(input.share(), input_strides,
                                               output.share(), output.strides(), output.shape(),
                                               center, radius, taper_size, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void rectangle(const Array<T>& input, const Array<T>& output,
                   float2_t center, float2_t radius, float taper_size, bool invert) {
        return rectangle(input, output, float3_t{0, center[0], center[1]},
                         float3_t{1, radius[0], radius[1]}, taper_size, invert);
    }

    template<typename T, typename>
    void cylinder(const Array<T>& input, const Array<T>& output,
                  float3_t center, float radius, float length, float taper_size, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        const bool is_empty = input.empty();
        dim4_t input_strides = input.strides();
        if (!is_empty && !indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(is_empty || device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            if (invert) {
                cpu::signal::cylinder<true>(input.share(), input_strides,
                                            output.share(), output.strides(), output.shape(),
                                            center, radius, length, taper_size, stream.cpu());
            } else {
                cpu::signal::cylinder<false>(input.share(), input_strides,
                                             output.share(), output.strides(), output.shape(),
                                             center, radius, length, taper_size, stream.cpu());
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            if (invert) {
                cuda::signal::cylinder<true>(input.share(), input_strides,
                                             output.share(), output.strides(), output.shape(),
                                             center, radius, length, taper_size, stream.cuda());
            } else {
                cuda::signal::cylinder<false>(input.share(), input_strides,
                                              output.share(), output.strides(), output.shape(),
                                              center, radius, length, taper_size, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
