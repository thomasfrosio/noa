#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/common/Functors.h"

namespace noa::cpu::math::details {
    template<typename T, typename U>
    constexpr bool is_valid_lstsq_t = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                      traits::is_almost_same_v<traits::value_type_t<T>, U>;
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
    void lstsq(const shared_t<T[]>& a, dim4_t a_strides, dim4_t a_shape,
               const shared_t<T[]>& b, dim4_t b_strides, dim4_t b_shape,
               float cond, const shared_t<U[]>& svd,
               Stream& stream);

    // Fits a polynomial 2D surface onto a regular grid represented by a 2D image.
    // order 1: p[0] + p[1]*x + p[2]*y = 0
    // order 2: p[0] + p[1]*x + p[2]*y + p[3]*x*y + p[4]*x*x + p[5]*y*y = 0
    // order 3: p[0] + p[1]*x + p[2]*y + p[3]*x*y + p[4]*x*x + p[5]*y*y + p[6]*x^2*y + p[7]*x*y^2 + p[8]*x^3 + p[9]*y^3 = 0
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void surface(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, bool subtract,
                 int order, const shared_t<T[]>& parameters, Stream& stream);
}
