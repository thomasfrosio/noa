#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"

namespace noa::cpu::math::details {
    template<typename T>
    constexpr bool is_valid_dot_t = noa::traits::is_any_v<T, i32, i64, u32, u64, f32, f64, c32, c64>;

    template<typename T>
    constexpr bool is_valid_matmul_t = noa::traits::is_any_v<T, f32, f64, c32, c64>;
}

namespace noa::cpu::math {
    // Returns the vector-vector dot product.
    template<typename T, typename = std::enable_if_t<details::is_valid_dot_t<T>>>
    T dot(const Shared<T[]>& lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
          const Shared<T[]>& rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
          Stream& stream);

    // Computes the (batched) vector-vector dot product.
    template<typename T, typename = std::enable_if_t<details::is_valid_dot_t<T>>>
    void dot(const Shared<T[]>& lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
             const Shared<T[]>& rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
             const Shared<T[]>& output, Stream& stream);

    // Computes a scalar-matrix-matrix product and add the result to a scalar-matrix product, with general matrices.
    template<typename T, typename = std::enable_if_t<details::is_valid_matmul_t<T>>>
    void matmul(const Shared<T[]>& lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
                const Shared<T[]>& rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
                T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
                const Shared<T[]>& output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                Stream& stream);

    // Computes a matrix-matrix product with general matrices.
    template<typename T, typename = std::enable_if_t<details::is_valid_matmul_t<T>>>
    void matmul(const Shared<T[]>& lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
                const Shared<T[]>& rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
                const Shared<T[]>& output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                Stream& stream) {
        matmul(lhs, lhs_strides, lhs_shape, rhs, rhs_strides, rhs_shape,
               T{1}, T{0}, false, false,
               output, output_strides, output_shape, stream);
    }
}
