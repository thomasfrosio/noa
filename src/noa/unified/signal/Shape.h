#pragma once

#include "noa/cpu/signal/Shape.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Shape.h"
#endif

#include "noa/unified/Array.h"

namespace noa::filter {
    /// Returns or applies a spherical mask.
    /// \tparam INVERT      Whether the mask should be inverted. If true, everything within the sphere is removed.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    Array to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array. Can be equal to \p input.
    /// \param center       Rightmost center of the sphere.
    /// \param radius       Radius, in elements, of the sphere.
    /// \param taper_size   Width, in elements, of the raised-cosine, including the first zero.
    template<bool INVERT = false, typename T>
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

    /// Returns or applies a rectangular mask.
    /// \tparam INVERT      Whether the mask should be inverted. If true, everything within the rectangle is removed.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    Array to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array. Can be equal to \p input.
    /// \param center       Rightmost center of the rectangle.
    /// \param radius       Rightmost radius, in elements, of the rectangle.
    /// \param taper_size   Width, in elements, of the raised-cosine, including the first zero.
    template<bool INVERT = false, typename T>
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

    /// Returns or applies a cylindrical mask.
    /// \tparam INVERT      Whether the mask should be inverted. If true, everything within the cylinder is removed.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    Array to mask. If empty, write the mask in \p output.
    /// \param[out] output  Masked array. Can be equal to \p input.
    /// \param center       Rightmost center of the cylinder, in \p T elements.
    /// \param radius       Radius of the cylinder.
    /// \param length       Length of the cylinder along the third-most dimension.
    /// \param taper_size   Width, in elements, of the raised-cosine, including the first zero.
    template<bool INVERT = false, typename T>
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
