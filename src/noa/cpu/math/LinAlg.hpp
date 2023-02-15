#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"

namespace noa::cpu::math::details {
    template<typename T, typename U>
    constexpr bool is_valid_lstsq_t =
            noa::traits::is_any_v<T, f32, f64, c32, c64> &&
            noa::traits::is_almost_same_v<noa::traits::value_type_t<T>, U>;
}

namespace noa::cpu::math {
    // TODO Add solve(...), like https://github.com/scipy/linalg/_basic.py#L40 using e.g. genv
    // TODO Add svd

    // Computes least-squares solution to equation Ax = b.
    // a: Dense MxN matrix. Can be batched. It is overwritten.
    // b: Dense MxK matrix, where K can be 1. Can be batched. It is overwritten with the NxK solution matrix.
    // In the case of M < N, \p b should be extended to fit the output and its shape should reflect that,
    // i.e. its second-most dimension should have max(M,N) elements.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_lstsq_t<T, U>>>
    void lstsq(T* a, const Strides4<i64>& a_strides, const Shape4<i64>& a_shape,
               T* b, const Strides4<i64>& b_strides, const Shape4<i64>& b_shape,
               f32 cond, U* svd);

    // Fits a polynomial 2D surface onto a regular grid represented by a 2D image.
    // order 1: p[0] + p[1]*x + p[2]*y = 0
    // order 2: p[0] + p[1]*x + p[2]*y + p[3]*x*y + p[4]*x*x + p[5]*y*y = 0
    // order 3: p[0] + p[1]*x + p[2]*y + p[3]*x*y + p[4]*x*x + p[5]*y*y + p[6]*x^2*y + p[7]*x*y^2 + p[8]*x^3 + p[9]*y^3 = 0
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    void surface(T* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                 T* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                 bool subtract, i32 order, T* parameters);
}
