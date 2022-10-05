#pragma once

#ifndef NOA_UNIFIED_SHIFT_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/signal/fft/Shift.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Shift.h"
#endif

namespace noa::signal::fft {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename T, typename>
    void shift2D(const Array<T>& input, const Array<T>& output, dim4_t shape,
                 const Array<float2_t>& shifts, float cutoff) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        dim4_t input_strides = input.strides();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
            NOA_CHECK(REMAP == Remap::H2H || REMAP == Remap::HC2HC || input.get() != output.get(),
                      "In-place remapping is not allowed");
        }

        NOA_CHECK(shifts.elements() == output.shape()[0] &&
                  indexing::isVector(shifts.shape()) && shifts.contiguous(),
                  "The input shift(s) should be entered as a 1D contiguous vector, with one shift per output batch, "
                  "but got shift {} and output {}", shifts.shape(), output.shape());

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(shifts.dereferenceable(), "The shifts should be accessible by the CPU");
            cpu::signal::fft::shift2D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    shifts.share(), cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift2D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    shifts.share(), cutoff, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void shift2D(const Array<T>& input, const Array<T>& output, dim4_t shape,
                 float2_t shift, float cutoff) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        dim4_t input_strides = input.strides();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
            NOA_CHECK(REMAP == Remap::H2H || REMAP == Remap::HC2HC || input.get() != output.get(),
                      "In-place remapping is not allowed");
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::shift2D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    shift, cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift2D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    shift, cutoff, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void shift3D(const Array<T>& input, const Array<T>& output, dim4_t shape,
                 const Array<float3_t>& shifts, float cutoff) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        dim4_t input_strides = input.strides();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
            NOA_CHECK(REMAP == Remap::H2H || REMAP == Remap::HC2HC || input.get() != output.get(),
                      "In-place remapping is not allowed");
        }

        NOA_CHECK(shifts.elements() == output.shape()[0] &&
                  indexing::isVector(shifts.shape()) && shifts.contiguous(),
                  "The input shift(s) should be entered as a 1D contiguous vector, with one shift per output batch, "
                  "but got shift {} and output {}", shifts.shape(), output.shape());

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(shifts.dereferenceable(), "The matrices should be accessible to the host");
            cpu::signal::fft::shift3D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    shifts.share(), cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift3D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    shifts.share(), cutoff, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void shift3D(const Array<T>& input, const Array<T>& output, dim4_t shape,
                 float3_t shift, float cutoff) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        dim4_t input_strides = input.strides();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
            NOA_CHECK(REMAP == Remap::H2H || REMAP == Remap::HC2HC || input.get() != output.get(),
                      "In-place remapping is not allowed");
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::shift3D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    shift, cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift3D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    shift, cutoff, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
