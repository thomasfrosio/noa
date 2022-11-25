#pragma once

#ifndef NOA_UNIFIED_SHAPE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/signal/Shape.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Shape.h"
#endif

namespace noa::signal::details {
    // The backends only accept the truncated affine matrices.
    template<int32_t NDIM, typename Matrix>
    constexpr auto squareOrTruncatedMatrix(const Matrix& matrix) noexcept {
        if constexpr (NDIM == 2 && traits::is_float33_v<Matrix>)
            return float23_t(matrix);
        else if constexpr (NDIM == 3 && traits::is_float44_v<Matrix>)
            return float34_t(matrix);
        else
            return matrix;
    }
}

namespace noa::signal {
    template<typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void ellipse(const Array<Value>& input, const Array<Value>& output,
                 float3_t center, float3_t radius, float edge_size,
                 Matrix inv_matrix, Functor functor, CValue cvalue, bool invert) {
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
            cpu::signal::ellipse(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<3>(inv_matrix),
                    functor, cvalue, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::ellipse(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<3>(inv_matrix),
                    functor, cvalue, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void ellipse(const Array<Value>& input, const Array<Value>& output,
                 float2_t center, float2_t radius, float edge_size,
                 Matrix inv_matrix, Functor functor, CValue cvalue, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK(output.shape()[1] == 1,
                  "This overload doesn't supports 3D arrays. Use the overload for 2D and 3D arrays");
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
            cpu::signal::ellipse(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<2>(inv_matrix),
                    functor, cvalue, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::ellipse(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<2>(inv_matrix),
                    functor, cvalue, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void sphere(const Array<Value>& input, const Array<Value>& output,
                float3_t center, float radius, float edge_size,
                Matrix inv_matrix, Functor functor, CValue cvalue, bool invert) {
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
            cpu::signal::sphere(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<3>(inv_matrix),
                    functor, cvalue, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::sphere(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<3>(inv_matrix),
                    functor, cvalue, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void sphere(const Array<Value>& input, const Array<Value>& output,
                float2_t center, float radius, float edge_size,
                Matrix inv_matrix, Functor functor, CValue cvalue, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK(output.shape()[1] == 1,
                  "This overload doesn't supports 3D arrays. Use the overload for 2D and 3D arrays");
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
            cpu::signal::sphere(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<2>(inv_matrix),
                    functor, cvalue, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::sphere(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<2>(inv_matrix),
                    functor, cvalue, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void rectangle(const Array<Value>& input, const Array<Value>& output,
                   float3_t center, float3_t radius, float edge_size,
                   Matrix inv_matrix, Functor functor, CValue cvalue, bool invert) {
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
            cpu::signal::rectangle(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<3>(inv_matrix),
                    functor, cvalue, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::rectangle(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<3>(inv_matrix),
                    functor, cvalue, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void rectangle(const Array<Value>& input, const Array<Value>& output,
                   float2_t center, float2_t radius, float edge_size,
                   Matrix inv_matrix, Functor functor, CValue cvalue, bool invert) {
        NOA_CHECK(!output.empty(), "Empty array detected");
        NOA_CHECK(output.shape()[1] == 1,
                  "This overload doesn't supports 3D arrays. Use the overload for 2D and 3D arrays");
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
            cpu::signal::rectangle(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<2>(inv_matrix),
                    functor, cvalue, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::rectangle(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, edge_size,
                    details::squareOrTruncatedMatrix<2>(inv_matrix),
                    functor, cvalue, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void cylinder(const Array<Value>& input, const Array<Value>& output,
                  float3_t center, float radius, float length, float edge_size,
                  Matrix inv_matrix, Functor functor, CValue cvalue, bool invert) {
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
            cpu::signal::cylinder(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, length, edge_size,
                    details::squareOrTruncatedMatrix<3>(inv_matrix),
                    functor, cvalue, invert, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::cylinder(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    center, radius, length, edge_size,
                    details::squareOrTruncatedMatrix<3>(inv_matrix),
                    functor, cvalue, invert, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
