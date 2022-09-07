#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Functors.h"

#include "noa/cpu/Stream.h"

namespace noa::cpu::math {
    // Element-wise transformation using an unary operator()(\p T) -> \p U
    template<typename T, typename U, typename UnaryOp>
    void ewise(const shared_t<T[]>& input, size4_t input_strides,
               const shared_t<U[]>& output, size4_t output_strides,
               size4_t shape, UnaryOp unary_op, Stream& stream);

    // Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    template<typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U>>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides, U rhs,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, BinaryOp binary_op, Stream& stream);

    // Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    template<typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<T>>>
    void ewise(T lhs, const shared_t<U[]>& rhs, size4_t rhs_strides,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, BinaryOp binary_op, Stream& stream);

    // Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               const shared_t<U[]>& rhs, size4_t rhs_strides,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, BinaryOp binary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p T, \p U, \p U) -> \p V
    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U> && noa::traits::is_data_v<V>>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides, U mhs, V rhs,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<V>>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               const shared_t<U[]>& mhs, size4_t mhs_strides, V rhs,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U>>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides, U mhs,
               const shared_t<V[]>& rhs, size4_t rhs_strides,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               const shared_t<U[]>& mhs, size4_t mhs_strides,
               const shared_t<V[]>& rhs, size4_t rhs_strides,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);
}

#define NOA_EWISE_INL_
#include "noa/cpu/math/Ewise.inl"
#undef NOA_EWISE_INL_
