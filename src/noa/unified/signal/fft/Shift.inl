#pragma once

#ifndef NOA_UNIFIED_SHIFT_
#error "This is a private header"
#endif

#include "noa/cpu/signal/fft/Shift.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Shift.h"
#endif

namespace noa::signal::fft {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename T>
    void shift2D(const Array<T>& input, const Array<T>& output, size4_t shape,
                 const Array<float2_t>& shifts, float cutoff) {
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "The non-redundant FFT with a shape of [logical:{}, pitch:{}] is expected, but got pitch of {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        size4_t input_stride = input.stride();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
            NOA_CHECK(REMAP == Remap::H2H || REMAP == Remap::HC2HC || input.get() != output.get(),
                      "In-place remapping is not allowed");
        }

        NOA_CHECK(shifts.shape()[3] == output.shape()[3] && shifts.shape().ndim() == 1 && shifts.contiguous(),
                  "The input shift(s) should be entered as a 1D contiguous row vector, with one shift per output batch, "
                  "but got shift {} and output {}", shifts.shape(), output.shape());

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(shifts.dereferencable(), "The shifts should be accessible by the CPU");
            cpu::signal::fft::shift2D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), shape,
                    shifts.share(), cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift2D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), shape,
                    shifts.share(), cutoff, stream.cpu());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T>
    void shift2D(const Array<T>& input, const Array<T>& output, size4_t shape,
                 float2_t shift, float cutoff) {
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "The non-redundant FFT with a shape of [logical:{}, pitch:{}] is expected, but got pitch of {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        size4_t input_stride = input.stride();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
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
                    input.share(), input.stride(),
                    output.share(), output.stride(), shape,
                    shift, cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift2D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), shape,
                    shift, cutoff, stream.cpu());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T>
    void shift3D(const Array<T>& input, const Array<T>& output, size4_t shape,
                 const Array<float3_t>& shifts, float cutoff) {
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "The non-redundant FFT with a shape of [logical:{}, pitch:{}] is expected, but got pitch of {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        size4_t input_stride = input.stride();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
            NOA_CHECK(REMAP == Remap::H2H || REMAP == Remap::HC2HC || input.get() != output.get(),
                      "In-place remapping is not allowed");
        }

        NOA_CHECK(shifts.shape()[3] == output.shape()[3] && shifts.shape().ndim() == 1 && shifts.contiguous(),
                  "The input shift(s) should be entered as a 1D contiguous row vector, with one shift per output batch, "
                  "but got shift {} and output {}", shifts.shape(), output.shape());

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(shifts.dereferencable(), "The matrices should be accessible to the host");
            cpu::signal::fft::shift3D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), shape,
                    shifts.share(), cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift3D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), shape,
                    shifts.share(), cutoff, stream.cpu());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T>
    void shift3D(const Array<T>& input, const Array<T>& output, size4_t shape,
                 float3_t shift, float cutoff) {
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "The non-redundant FFT with a shape of [logical:{}, pitch:{}] is expected, but got pitch of {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        size4_t input_stride = input.stride();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
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
                    input.share(), input.stride(),
                    output.share(), output.stride(), shape,
                    shift, cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift3D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), shape,
                    shift, cutoff, stream.cpu());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
