#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Functors.h"

#include "noa/cpu/Stream.h"

namespace noa::cpu::math {
    // Element-wise transformation using an unary operator()(In) -> Out
    template<typename In, typename Out, typename UnaryOp>
    void ewise(const shared_t<In[]>& input, dim4_t input_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, UnaryOp&& unary_op, Stream& stream);

    // Element-wise transformation using a binary operator()(Lhs, Rhs) -> Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<Rhs>>>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp&& binary_op, Stream& stream);

    // Element-wise transformation using a binary operator()(Lhs, Rhs) -> Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<Lhs>>>
    void ewise(Lhs lhs,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp&& binary_op, Stream& stream);

    // Element-wise transformation using a binary operator()(Lhs, Rhs) -> Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp&& binary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(Lhs, Mhs, Rhs) -> Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<Mhs> && noa::traits::is_data_v<Rhs>>>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               Mhs mhs,
               Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp&& trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(Lhs, Mhs, Rhs) -> Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<Rhs>>>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               const shared_t<Mhs[]>& mhs, dim4_t mhs_strides,
               Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp&& trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(Lhs, Mhs, Rhs) -> Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<Mhs>>>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               Mhs mhs,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp&& trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(Lhs, Mhs, Rhs) -> Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               const shared_t<Mhs[]>& mhs, dim4_t mhs_strides,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp&& trinary_op, Stream& stream);
}

#define NOA_EWISE_INL_
#include "noa/cpu/math/Ewise.inl"
#undef NOA_EWISE_INL_
