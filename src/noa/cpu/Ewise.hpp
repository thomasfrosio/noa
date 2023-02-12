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
    void ewise_unary(const Shared<In[]>& input, const Strides4<i64>& input_strides,
                     const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                     const Shape4<i64>& shape, UnaryOp&& unary_op, Stream& stream) {
        const auto threads = stream.threads();
        stream.enqueue([=, op = std::forward<UnaryOp>(unary_op)]() {
            noa::cpu::utils::ewise_unary(
                    input.get(), input_strides,
                    output.get(), output_strides,
                    shape, op, threads);
        });
    }

    // Element-wise transformation using a binary operator()(Lhs, Rhs) -> Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp>
    void ewise_binary(const Shared<Lhs[]>& lhs, const Strides4<i64>& lhs_strides,
                      const Shared<Rhs[]>& rhs, const Strides4<i64>& rhs_strides,
                      const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp&& binary_op, Stream& stream) {
        const auto threads = stream.threads();
        stream.enqueue([=, op = std::forward<BinaryOp>(binary_op)]() {
            noa::cpu::utils::ewise_binary(
                    lhs.get(), lhs_strides,
                    rhs.get(), rhs_strides,
                    output.get(), output_strides,
                    shape, op, threads);
        });
    }

    // Element-wise transformation using a binary operator()(Lhs, Rhs) -> Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp>
    void ewise_binary(const Shared<Lhs[]>& lhs, const Strides4<i64>& lhs_strides,
                      Rhs rhs,
                      const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp&& binary_op, Stream& stream) {
        const auto threads = stream.threads();
        stream.enqueue([=, op = std::forward<BinaryOp>(binary_op)]() {
            noa::cpu::utils::ewise_binary(
                    lhs.get(), lhs_strides,
                    rhs,
                    output.get(), output_strides,
                    shape, op, threads);
        });
    }

    // Element-wise transformation using a binary operator()(Lhs, Rhs) -> Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp>
    void ewise_binary(Lhs lhs,
                      const Shared<Rhs[]>& rhs, const Strides4<i64>& rhs_strides,
                      const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp&& binary_op, Stream& stream) {
        const auto threads = stream.threads();
        stream.enqueue([=, op = std::forward<BinaryOp>(binary_op)]() {
            noa::cpu::utils::ewise_binary(
                    lhs,
                    rhs.get(), rhs_strides,
                    output.get(), output_strides,
                    shape, op, threads);
        });
    }

    // Element-wise transformation using a trinary operator()(Lhs, Mhs, Rhs) -> Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(const Shared<Lhs[]>& lhs, const Strides4<i64>& lhs_strides,
                       const Shared<Mhs[]>& mhs, const Strides4<i64>& mhs_strides,
                       const Shared<Rhs[]>& rhs, const Strides4<i64>& rhs_strides,
                       const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, Stream& stream) {
        const auto threads = stream.threads();
        stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
            noa::cpu::utils::ewise_trinary(
                    lhs.get(), lhs_strides,
                    mhs.get(), mhs_strides,
                    rhs.get(), rhs_strides,
                    output.get(), output_strides,
                    shape, op, threads);
        });
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(Lhs lhs,
                       const Shared<Mhs[]>& mhs, const Strides4<i64>& mhs_strides,
                       const Shared<Rhs[]>& rhs, const Strides4<i64>& rhs_strides,
                       const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, Stream& stream) {
        const auto threads = stream.threads();
        stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
            noa::cpu::utils::ewise_trinary(
                    lhs,
                    mhs.get(), mhs_strides,
                    rhs.get(), rhs_strides,
                    output.get(), output_strides,
                    shape, op, threads);
        });
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(const Shared<Lhs[]>& lhs, const Strides4<i64>& lhs_strides,
                       Mhs mhs,
                       const Shared<Rhs[]>& rhs, const Strides4<i64>& rhs_strides,
                       const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, Stream& stream) {
        const auto threads = stream.threads();
        stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
            noa::cpu::utils::ewise_trinary(
                    lhs.get(), lhs_strides,
                    mhs,
                    rhs.get(), rhs_strides,
                    output.get(), output_strides,
                    shape, op, threads);
        });
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(const Shared<Lhs[]>& lhs, const Strides4<i64>& lhs_strides,
                       const Shared<Mhs[]>& mhs, const Strides4<i64>& mhs_strides,
                       Rhs rhs,
                       const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, Stream& stream) {
        const auto threads = stream.threads();
        stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
            noa::cpu::utils::ewise_trinary(
                    lhs.get(), lhs_strides,
                    mhs.get(), mhs_strides,
                    rhs,
                    output.get(), output_strides,
                    shape, op, threads);
        });
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(const Shared<Lhs[]>& lhs, const Strides4<i64>& lhs_strides,
                       Mhs mhs,
                       Rhs rhs,
                       const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, Stream& stream) {
        const auto threads = stream.threads();
        stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
            noa::cpu::utils::ewise_trinary(
                    lhs.get(), lhs_strides,
                    mhs,
                    rhs,
                    output.get(), output_strides,
                    shape, op, threads);
        });
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(Lhs lhs,
                       const Shared<Mhs[]>& mhs, const Strides4<i64>& mhs_strides,
                       Rhs rhs,
                       const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, Stream& stream) {
        const auto threads = stream.threads();
        stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
            noa::cpu::utils::ewise_trinary(
                    lhs,
                    mhs.get(), mhs_strides,
                    rhs,
                    output.get(), output_strides,
                    shape, op, threads);
        });
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise_trinary(Lhs lhs,
                       Mhs mhs,
                       const Shared<Rhs[]>& rhs, const Strides4<i64>& rhs_strides,
                       const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp&& trinary_op, Stream& stream) {
        const auto threads = stream.threads();
        stream.enqueue([=, op = std::forward<TrinaryOp>(trinary_op)]() {
            noa::cpu::utils::ewise_trinary(
                    lhs,
                    mhs,
                    rhs.get(), rhs_strides,
                    output.get(), output_strides,
                    shape, op, threads);
        });
    }
}
