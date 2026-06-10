#pragma once

#include "noa/runtime/core/Shape.hpp"

namespace noa::cpu {
    // Computes a scalar-matrix-matrix product and add the result to a scalar-matrix product, with general matrices.
    template<typename T>// requires nt::is_any_v<T, f32, f64, c32, c64>
    void matmul(const T* lhs, const Strides3& lhs_strides, const Shape3& lhs_shape,
                const T* rhs, const Strides3& rhs_strides, const Shape3& rhs_shape,
                T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
                T* output, const Strides3& output_strides, const Shape3& output_shape,
                isize n_threads);
}
