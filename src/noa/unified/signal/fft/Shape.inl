#pragma once

#ifndef NOA_UNIFIED_SHAPE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/signal/fft/Shape.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Shape.h"
#endif

namespace noa::signal::fft {
    template<fft::Remap REMAP, typename Value, typename>
    void ellipse(const Array<Value>& input, const Array<Value>& output,
                 float3_t center, float3_t radius, float edge_size,
                 float33_t inv_matrix, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !indexing::isOverlap(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);
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
            cpu::signal::fft::ellipse<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::ellipse<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<fft::Remap REMAP, typename Value, typename>
    void ellipse(const Array<Value>& input, const Array<Value>& output,
                 float2_t center, float2_t radius, float edge_size,
                 float22_t inv_matrix, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK(output.shape()[1] == 1,
                  "This overload doesn't supports 3D arrays. Use the overload for 2D and 3D arrays");
        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !indexing::isOverlap(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);
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
            cpu::signal::fft::ellipse<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::ellipse<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<fft::Remap REMAP, typename Value, typename>
    void sphere(const Array<Value>& input, const Array<Value>& output,
                float3_t center, float radius, float edge_size,
                float33_t inv_matrix, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !indexing::isOverlap(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);
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
            cpu::signal::fft::sphere<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::sphere<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<fft::Remap REMAP, typename Value, typename>
    void sphere(const Array<Value>& input, const Array<Value>& output,
                float2_t center, float radius, float edge_size,
                float22_t inv_matrix, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK(output.shape()[1] == 1,
                  "This overload doesn't supports 3D arrays. Use the overload for 2D and 3D arrays");
        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !indexing::isOverlap(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);
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
            cpu::signal::fft::sphere<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::sphere<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<fft::Remap REMAP, typename Value, typename>
    void rectangle(const Array<Value>& input, const Array<Value>& output,
                   float3_t center, float3_t radius, float edge_size,
                   float33_t inv_matrix, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !indexing::isOverlap(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);
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
            cpu::signal::fft::rectangle<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::rectangle<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<fft::Remap REMAP, typename Value, typename>
    void rectangle(const Array<Value>& input, const Array<Value>& output,
                   float2_t center, float2_t radius, float edge_size,
                   float22_t inv_matrix, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK(output.shape()[1] == 1,
                  "This overload doesn't supports 3D arrays. Use the overload for 2D and 3D arrays");
        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !indexing::isOverlap(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);
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
            cpu::signal::fft::rectangle<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::rectangle<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size, inv_matrix, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<fft::Remap REMAP, typename Value, typename>
    void cylinder(const Array<Value>& input, const Array<Value>& output,
                  float3_t center, float radius, float length, float edge_size,
                  float33_t inv_matrix, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || !indexing::isOverlap(input, output),
                  "In-place computation is not supported with remapping {}", REMAP);
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
            cpu::signal::fft::cylinder<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, length, edge_size, inv_matrix, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::cylinder<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, length, edge_size, inv_matrix, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
