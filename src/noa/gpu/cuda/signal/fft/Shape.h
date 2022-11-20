#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Functors.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::signal::fft::details {
    using namespace ::noa::fft;
    template<int32_t NDIM, Remap REMAP, typename Value, typename Matrix, typename Functor>
    constexpr bool is_valid_shape_v =
            traits::is_any_v<Value, float, cfloat_t, double, cdouble_t> &&
            (REMAP == F2F || REMAP == FC2FC || REMAP == F2FC || REMAP == FC2F) &&
            traits::is_any_v<Functor, noa::math::multiply_t, noa::math::plus_t> &&
            (NDIM == 2 && traits::is_any_v<Matrix, float22_t, float23_t> ||
             NDIM == 3 && traits::is_any_v<Matrix, float33_t, float34_t>);
}

namespace noa::cuda::signal::fft {
    using namespace ::noa::fft;

    // Returns or applies an elliptical mask.
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename = std::enable_if_t<details::is_valid_shape_v<3, REMAP, Value, Matrix, Functor>>>
    void ellipse(const shared_t<Value[]>& input, dim4_t input_strides,
                 const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t center, float3_t radius, float edge_size,
                 Matrix inv_matrix, Functor functor, bool invert, Stream& stream);

    // Returns or applies a 2D elliptical mask.
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename = std::enable_if_t<details::is_valid_shape_v<2, REMAP, Value, Matrix, Functor>>>
    void ellipse(const shared_t<Value[]>& input, dim4_t input_strides,
                 const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                 float2_t center, float2_t radius, float edge_size,
                 Matrix inv_matrix, Functor functor, bool invert, Stream& stream);

    // Returns or applies a spherical mask.
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename = std::enable_if_t<details::is_valid_shape_v<3, REMAP, Value, Matrix, Functor>>>
    void sphere(const shared_t<Value[]>& input, dim4_t input_strides,
                const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                float3_t center, float radius, float edge_size,
                Matrix inv_matrix, Functor functor, bool invert, Stream& stream);

    // Returns or applies a 2D spherical mask.
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename = std::enable_if_t<details::is_valid_shape_v<2, REMAP, Value, Matrix, Functor>>>
    void sphere(const shared_t<Value[]>& input, dim4_t input_strides,
                const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                float2_t center, float radius, float edge_size,
                Matrix inv_matrix, Functor functor, bool invert, Stream& stream);

    // Returns or applies a rectangular mask.
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename = std::enable_if_t<details::is_valid_shape_v<3, REMAP, Value, Matrix, Functor>>>
    void rectangle(const shared_t<Value[]>& input, dim4_t input_strides,
                   const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                   float3_t center, float3_t radius, float edge_size,
                   Matrix inv_matrix, Functor functor, bool invert, Stream& stream);

    // Returns or applies a 2D rectangular mask.
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename = std::enable_if_t<details::is_valid_shape_v<2, REMAP, Value, Matrix, Functor>>>
    void rectangle(const shared_t<Value[]>& input, dim4_t input_strides,
                   const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                   float2_t center, float2_t radius, float edge_size,
                   Matrix inv_matrix, Functor functor, bool invert, Stream& stream);

    // Returns or applies a cylindrical mask.
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename = std::enable_if_t<details::is_valid_shape_v<3, REMAP, Value, Matrix, Functor>>>
    void cylinder(const shared_t<Value[]>& input, dim4_t input_strides,
                  const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                  float3_t center, float radius, float length, float edge_size,
                  Matrix inv_matrix, Functor functor, bool invert, Stream& stream);
}
