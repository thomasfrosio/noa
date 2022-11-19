#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/cpu/signal/fft/Shape.h"

namespace noa::cpu::signal::details {
    using namespace ::noa::fft;
    template<int32_t NDIM, typename Value, typename Matrix>
    constexpr bool is_valid_shape_v =
            traits::is_any_v<Value, float, cfloat_t, double, cdouble_t> &&
            (NDIM == 2 && traits::is_any_v<Matrix, float22_t, float23_t> ||
             NDIM == 3 && traits::is_any_v<Matrix, float33_t, float34_t>);
}

namespace noa::cpu::signal {
    // Returns or applies an elliptical mask.
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix>>>
    void ellipse(const shared_t<Value[]>& input, dim4_t input_strides,
                 const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t center, float3_t radius, float edge_size,
                 Matrix inv_matrix, bool invert, Stream& stream) {
        cpu::signal::fft::ellipse<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, invert, stream);
    }

    // Returns or applies a 2D elliptical mask.
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_shape_v<2, Value, Matrix>>>
    void ellipse(const shared_t<Value[]>& input, dim4_t input_strides,
                 const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                 float2_t center, float2_t radius, float edge_size,
                 Matrix inv_matrix, bool invert, Stream& stream) {
        cpu::signal::fft::ellipse<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, invert, stream);
    }

    // Returns or applies a spherical mask.
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix>>>
    void sphere(const shared_t<Value[]>& input, dim4_t input_strides,
                const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                float3_t center, float radius, float edge_size,
                Matrix inv_matrix, bool invert, Stream& stream) {
        cpu::signal::fft::sphere<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, invert, stream);
    }

    // Returns or applies a 2D spherical mask.
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_shape_v<2, Value, Matrix>>>
    void sphere(const shared_t<Value[]>& input, dim4_t input_strides,
                const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                float2_t center, float radius, float edge_size,
                Matrix inv_matrix, bool invert, Stream& stream) {
        cpu::signal::fft::sphere<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, invert, stream);
    }

    // Returns or applies a rectangular mask.
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix>>>
    void rectangle(const shared_t<Value[]>& input, dim4_t input_strides,
                   const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                   float3_t center, float3_t radius, float edge_size,
                   Matrix inv_matrix, bool invert, Stream& stream) {
        cpu::signal::fft::rectangle<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, invert, stream);
    }

    // Returns or applies a 2D rectangular mask.
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_shape_v<2, Value, Matrix>>>
    void rectangle(const shared_t<Value[]>& input, dim4_t input_strides,
                   const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                   float2_t center, float2_t radius, float edge_size,
                   Matrix inv_matrix, bool invert, Stream& stream) {
        cpu::signal::fft::rectangle<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, invert, stream);
    }

    // Returns or applies a cylindrical mask.
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix>>>
    void cylinder(const shared_t<Value[]>& input, dim4_t input_strides,
                  const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                  float3_t center, float radius, float length, float edge_size,
                  Matrix inv_matrix, bool invert, Stream& stream) {
        cpu::signal::fft::cylinder<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, length, edge_size, inv_matrix, invert, stream);
    }
}
