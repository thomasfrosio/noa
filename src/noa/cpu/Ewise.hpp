#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"
#include "noa/cpu/utils/EwiseUnary.hpp"
#include "noa/cpu/utils/EwiseBinary.hpp"
#include "noa/cpu/utils/EwiseTrinary.hpp"

namespace noa::cpu {
    // Element-wise transformation using a unary operator()(In) -> Out
    template<typename In, typename Out, typename UnaryOp>
    void ewise_unary(const In* input, const Strides4<i64>& input_strides,
                     Out* output, const Strides4<i64>& output_strides,
                     const Shape4<i64>& shape, UnaryOp&& unary_op, i64 threads) {
        noa::cpu::utils::ewise_unary(
                input, input_strides,
                output, output_strides,
                shape, std::forward<UnaryOp>(unary_op), threads);
    }

    // Element-wise transformation using a binary operator()(Lhs, Rhs) -> Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp>
    void ewise_binary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                      const Rhs* rhs, const Strides4<i64>& rhs_strides,
                      Out* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp&& binary_op, i64 threads) {
        noa::cpu::utils::ewise_binary(
                lhs, lhs_strides,
                rhs, rhs_strides,
                output, output_strides,
                shape, std::forward<BinaryOp>(binary_op), threads);
    }

    // Element-wise transformation using a binary operator()(Lhs, Rhs) -> Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp>
    void ewise_binary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                      Rhs rhs,
                      Out* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp&& binary_op, i64 threads) {
        noa::cpu::utils::ewise_binary(
                lhs, lhs_strides,
                rhs,
                output, output_strides,
                shape, std::forward<BinaryOp>(binary_op), threads);
    }

    // Element-wise transformation using a binary operator()(Lhs, Rhs) -> Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp>
    void ewise_binary(Lhs lhs,
                      const Rhs* rhs, const Strides4<i64>& rhs_strides,
                      Out* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp&& binary_op, i64 threads) {
        noa::cpu::utils::ewise_binary(
                lhs,
                rhs, rhs_strides,
                output, output_strides,
                shape, std::forward<BinaryOp>(binary_op), threads);
    }

    // Element-wise transformation using a trinary operator()(Lhs, Mhs, Rhs) -> Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, i64 threads) {
        noa::cpu::utils::ewise_trinary(
                lhs, lhs_strides,
                mhs, mhs_strides,
                rhs, rhs_strides,
                output, output_strides,
                shape, std::forward<TrinaryOp>(trinary_op), threads);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(Lhs lhs,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, i64 threads) {
        noa::cpu::utils::ewise_trinary(
                lhs,
                mhs, mhs_strides,
                rhs, rhs_strides,
                output, output_strides,
                shape, std::forward<TrinaryOp>(trinary_op), threads);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       Mhs mhs,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, i64 threads) {
        noa::cpu::utils::ewise_trinary(
                lhs, lhs_strides,
                mhs,
                rhs, rhs_strides,
                output, output_strides,
                shape, std::forward<TrinaryOp>(trinary_op), threads);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       Rhs rhs,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, i64 threads) {
        noa::cpu::utils::ewise_trinary(
                lhs, lhs_strides,
                mhs, mhs_strides,
                rhs,
                output, output_strides,
                shape, std::forward<TrinaryOp>(trinary_op), threads);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       Mhs mhs,
                       Rhs rhs,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, i64 threads) {
        noa::cpu::utils::ewise_trinary(
                lhs, lhs_strides,
                mhs,
                rhs,
                output, output_strides,
                shape, std::forward<TrinaryOp>(trinary_op), threads);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(Lhs lhs,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       Rhs rhs,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, i64 threads) {
        noa::cpu::utils::ewise_trinary(
                lhs,
                mhs, mhs_strides,
                rhs,
                output, output_strides,
                shape, std::forward<TrinaryOp>(trinary_op), threads);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(Lhs lhs,
                       Mhs mhs,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, i64 threads) {
        noa::cpu::utils::ewise_trinary(
                lhs,
                mhs,
                rhs, rhs_strides,
                output, output_strides,
                shape, std::forward<TrinaryOp>(trinary_op), threads);
    }
}
