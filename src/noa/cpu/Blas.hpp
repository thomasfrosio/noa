#pragma once

#include "noa/core/types/Shape.hpp"

namespace noa::cpu {
    // Computes a scalar-matrix-matrix product and add the result to a scalar-matrix product, with general matrices.
    template<typename T>// requires nt::is_any_v<T, f32, f64, c32, c64>
    void matmul(const T* lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
                const T* rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
                T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
                T* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                i64 n_threads);
}
