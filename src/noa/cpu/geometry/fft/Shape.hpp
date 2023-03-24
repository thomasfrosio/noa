#pragma once

#include "noa/core/Types.hpp"

namespace noa::cpu::geometry::fft::details {
    using namespace ::noa::fft;
    template<size_t NDIM, Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue>
    constexpr bool is_valid_shape_v =
            noa::traits::is_any_v<Value, f32, f64, c32, c64> &&
            std::is_same_v<CValue, noa::traits::value_type_t<Value>> &&
            (REMAP == F2F || REMAP == FC2FC || REMAP == F2FC || REMAP == FC2F) &&
            noa::traits::is_any_v<Functor, noa::multiply_t, noa::plus_t> &&
            (NDIM == 2 && noa::traits::is_any_v<Matrix, Float22, Float23> ||
             NDIM == 3 && noa::traits::is_any_v<Matrix, Float33, Float34>);
}

namespace noa::cpu::geometry::fft {
    using namespace ::noa::fft;

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<3, REMAP, Value, Matrix, Functor, CValue>>>
    void ellipse(const Value* input, Strides4<i64> input_strides,
                 Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                 Vec3<f32> center, Vec3<f32> radius, f32 edge_size,
                 Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads);

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<2, REMAP, Value, Matrix, Functor, CValue>>>
    void ellipse(const Value* input, Strides4<i64> input_strides,
                 Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                 Vec2<f32> center, Vec2<f32> radius, f32 edge_size,
                 Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads);

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<3, REMAP, Value, Matrix, Functor, CValue>>>
    void sphere(const Value* input, Strides4<i64> input_strides,
                Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                Vec3<f32> center, f32 radius, f32 edge_size,
                Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads);

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<2, REMAP, Value, Matrix, Functor, CValue>>>
    void sphere(const Value* input, Strides4<i64> input_strides,
                Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                Vec2<f32> center, f32 radius, f32 edge_size,
                Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads);

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<3, REMAP, Value, Matrix, Functor, CValue>>>
    void rectangle(const Value* input, Strides4<i64> input_strides,
                   Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                   Vec3<f32> center, Vec3<f32> radius, f32 edge_size,
                   Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads);

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<2, REMAP, Value, Matrix, Functor, CValue>>>
    void rectangle(const Value* input, Strides4<i64> input_strides,
                   Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                   Vec2<f32> center, Vec2<f32> radius, f32 edge_size,
                   Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads);

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<3, REMAP, Value, Matrix, Functor, CValue>>>
    void cylinder(const Value* input, Strides4<i64> input_strides,
                  Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                  Vec3<f32> center, f32 radius, f32 length, f32 edge_size,
                  Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads);
}
