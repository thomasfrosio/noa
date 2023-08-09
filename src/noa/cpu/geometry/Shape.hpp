#pragma once

#include "noa/core/Types.hpp"
#include "noa/cpu/geometry/fft/Shape.hpp"

namespace noa::cpu::geometry::details {
    using namespace ::noa::fft;
    template<size_t NDIM, typename Value, typename Matrix, typename Functor, typename CValue>
    constexpr bool is_valid_shape_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            std::is_same_v<CValue, nt::value_type_t<Value>> &&
            nt::is_any_v<Functor, noa::multiply_t, noa::plus_t> &&
            (NDIM == 2 && nt::is_any_v<Matrix, Float22, Float23, const Float22*, const Float23*> ||
             NDIM == 3 && nt::is_any_v<Matrix, Float33, Float34, const Float33*, const Float34*>);
}

namespace noa::cpu::geometry {
    template<typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix, Functor, CValue>>>
    void ellipse(const Value* input, const Strides4<i64>& input_strides,
                 Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                 const Vec3<f32>& center, const Vec3<f32>& radius, f32 edge_size,
                 const Matrix& inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        cpu::geometry::fft::ellipse<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, functor, cvalue, invert, threads);
    }

    template<typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<2, Value, Matrix, Functor, CValue>>>
    void ellipse(const Value* input, const Strides4<i64>& input_strides,
                 Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                 const Vec2<f32>& center, const Vec2<f32>& radius, f32 edge_size,
                 const Matrix& inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        cpu::geometry::fft::ellipse<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, functor, cvalue, invert, threads);
    }

    template<typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix, Functor, CValue>>>
    void sphere(const Value* input, const Strides4<i64>& input_strides,
                Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                const Vec3<f32>& center, f32 radius, f32 edge_size,
                const Matrix& inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        cpu::geometry::fft::sphere<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, functor, cvalue, invert, threads);
    }

    template<typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<2, Value, Matrix, Functor, CValue>>>
    void sphere(const Value* input, const Strides4<i64>& input_strides,
                Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                const Vec2<f32>& center, f32 radius, f32 edge_size,
                const Matrix& inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        cpu::geometry::fft::sphere<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, functor, cvalue, invert, threads);
    }

    template<typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix, Functor, CValue>>>
    void rectangle(const Value* input, const Strides4<i64>& input_strides,
                   Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                   const Vec3<f32>& center, const Vec3<f32>& radius, f32 edge_size,
                   const Matrix& inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        cpu::geometry::fft::rectangle<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, functor, cvalue, invert, threads);
    }

    template<typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<2, Value, Matrix, Functor, CValue>>>
    void rectangle(const Value* input, const Strides4<i64>& input_strides,
                   Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                   const Vec2<f32>& center, const Vec2<f32>& radius, f32 edge_size,
                   const Matrix& inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        cpu::geometry::fft::rectangle<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, edge_size, inv_matrix, functor, cvalue, invert, threads);
    }

    template<typename Value, typename Matrix, typename Functor,
             typename CValue = traits::value_type_t<Value>,
             typename = std::enable_if_t<details::is_valid_shape_v<3, Value, Matrix, Functor, CValue>>>
    void cylinder(const Value* input, const Strides4<i64>& input_strides,
                  Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                  const Vec3<f32>& center, f32 radius, f32 length, f32 edge_size,
                  const Matrix& inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        cpu::geometry::fft::cylinder<fft::FC2FC>(
                input, input_strides, output, output_strides, shape,
                center, radius, length, edge_size, inv_matrix, functor, cvalue, invert, threads);
    }
}
