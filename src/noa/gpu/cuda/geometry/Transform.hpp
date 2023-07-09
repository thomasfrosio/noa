#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/geometry/Euler.hpp"
#include "noa/core/geometry/Symmetry.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::geometry::details {
    template<int NDIM, typename T, typename M>
    constexpr bool is_valid_transform_v =
            noa::traits::is_any_v<T, f32, f64, c32, c64> &&
            ((NDIM == 2 && noa::traits::is_any_v<M, Float23, Float33, const Float23*, const Float33*>) ||
             (NDIM == 3 && noa::traits::is_any_v<M, Float34, Float44, const Float34*, const Float44*>));

    template<int NDIM, typename T, typename M>
    constexpr bool is_valid_transform_texture_v =
            noa::traits::is_any_v<T, f32, c32> &&
            ((NDIM == 2 && noa::traits::is_any_v<M, Float23, Float33, const Float23*, const Float33*>) ||
             (NDIM == 3 && noa::traits::is_any_v<M, Float34, Float44, const Float34*, const Float44*>));
}

namespace noa::cuda::geometry {
    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_transform_v<2, Value, Matrix>>>
    void transform_2d(
            const Value* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Matrix& inv_matrices, InterpMode interp_mode, BorderMode border_mode,
            Value cvalue, Stream& stream);

    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_transform_v<3, Value, Matrix>>>
    void transform_3d(
            const Value* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Matrix& inv_matrices, InterpMode interp_mode, BorderMode border_mode,
            Value cvalue, Stream& stream);

    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_transform_texture_v<2, Value, Matrix>>>
    void transform_2d(
            cudaArray* array, cudaTextureObject_t texture, const Shape4<i64>& texture_shape,
            InterpMode texture_interp_mode, BorderMode texture_border_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Matrix& inv_matrices, Stream& stream);

    template<typename Value, typename Matrix,
             typename = std::enable_if_t<details::is_valid_transform_texture_v<3, Value, Matrix>>>
    void transform_3d(
            cudaArray* array, cudaTextureObject_t texture, const Shape4<i64>& texture_shape,
            InterpMode texture_interp_mode, BorderMode texture_border_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Matrix& inv_matrices, Stream& stream);
}

namespace noa::cuda::geometry {
    using Symmetry = noa::geometry::Symmetry;

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, f64, c32, c64>>>
    void transform_and_symmetrize_2d(
            const Value* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec2<f32>& shift, const Float22& inv_matrix,
            const Symmetry& symmetry, const Vec2<f32>& center,
            InterpMode interp_mode, bool normalize, Stream& stream);

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, f64, c32, c64>>>
    void transform_and_symmetrize_3d(
            const Value* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec3<f32>& shift, const Float33& inv_matrix,
            const Symmetry& symmetry, const Vec3<f32>& center,
            InterpMode interp_mode, bool normalize, Stream& stream);

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, f64, c32, c64>>>
    void symmetrize_2d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, const Symmetry& symmetry, const Vec2<f32>& center,
            InterpMode interp_mode, bool normalize, Stream& stream);

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, f64, c32, c64>>>
    void symmetrize_3d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, const Symmetry& symmetry, const Vec3<f32>& center,
            InterpMode interp_mode, bool normalize, Stream& stream);

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, c32>>>
    void transform_and_symmetrize_2d(
            cudaArray* array, cudaTextureObject_t texture,
            InterpMode texture_interp_mode, const Shape4<i64>& texture_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec2<f32>& shift, const Float22& inv_matrix,
            const Symmetry& symmetry, const Vec2<f32>& center,
            bool normalize, Stream& stream);

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, c32>>>
    void transform_and_symmetrize_3d(
            cudaArray* array,cudaTextureObject_t texture,
            InterpMode texture_interp_mode, const Shape4<i64>& texture_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec3<f32>& shift, const Float33& inv_matrix,
            const Symmetry& symmetry, const Vec3<f32>& center,
            bool normalize, Stream& stream);

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, c32>>>
    void symmetrize_2d(
            cudaArray* array,
            cudaTextureObject_t texture,
            InterpMode texture_interp_mode, const Shape4<i64>& texture_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Symmetry& symmetry, const Vec2<f32>& center, bool normalize, Stream& stream);

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, c32>>>
    void symmetrize_3d(
            cudaArray* array,
            cudaTextureObject_t texture,
            InterpMode texture_interp_mode, const Shape4<i64>& texture_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Symmetry& symmetry, const Vec3<f32>& center, bool normalize, Stream& stream);
}
