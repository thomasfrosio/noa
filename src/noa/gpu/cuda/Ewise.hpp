#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::details {
    template<typename In, typename Out, typename Op>
    constexpr bool is_valid_ewise_unary_v =
            (traits::is_any_v<Out, i8, i16, i32, i64> && std::is_same_v<In, Out> && traits::is_any_v<Op, copy_t, square_t, abs_t, negate_t, one_minus_t, nonzero_t, logical_not_t>) ||
            (traits::is_any_v<Out, u8, u16, u32, u64> && std::is_same_v<In, Out> && traits::is_any_v<Op, copy_t, square_t, nonzero_t, logical_not_t>) ||
            (traits::is_restricted_int_v<In> && std::is_same_v<Out, bool> && traits::is_any_v<Op, nonzero_t, logical_not_t>) ||
            (traits::is_real_v<Out> && std::is_same_v<In, Out> && traits::is_any_v<Op, copy_t, square_t, abs_t, negate_t, one_minus_t, inverse_t, sqrt_t, rsqrt_t, exp_t, log_t, cos_t, sin_t, one_log_t, abs_one_log_t>) ||
            (traits::is_real_v<Out> && std::is_same_v<In, Out> && traits::is_any_v<Op, round_t, rint_t, ceil_t, floor_t, trunc_t>) ||
            (traits::is_complex_v<Out> && std::is_same_v<In, Out> && traits::is_any_v<Op, one_minus_t, square_t, inverse_t, normalize_t, conj_t>) ||
            (traits::is_complex_v<In> && std::is_same_v<Out, traits::value_type_t<In>> && traits::is_any_v<Op, abs_t, real_t, imag_t, abs_squared_t, abs_one_log_t>);

    template<typename Lhs, typename Rhs, typename Out, typename Op>
    constexpr bool is_valid_ewise_binary_v =
            (traits::is_restricted_int_v<Out> && traits::are_all_same_v<Lhs, Rhs, Out> &&
             traits::is_any_v<Op, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t, min_t, max_t,
                      equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, modulo_t, logical_and_t, logical_or_t>) ||
            (std::is_same_v<Out, bool> && traits::is_any_v<Lhs, i16, i32, i64, u16, u32, u64> && std::is_same_v<Lhs, Rhs> &&
             traits::is_any_v<Op, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, modulo_t, logical_and_t, logical_or_t>) ||
            (traits::is_real_v<Out> && traits::are_all_same_v<Lhs, Rhs, Out> &&
             traits::is_any_v<Op, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t, min_t, max_t, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, pow_t>) ||
            (traits::is_real_v<Lhs> && std::is_same_v<Lhs, Rhs> && std::is_same_v<Out, bool> &&
             traits::is_any_v<Op, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, pow_t>) ||
            (traits::is_complex_v<Out> && traits::are_all_same_v<Lhs, Rhs, Out> && traits::is_any_v<Op, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t, multiply_conj_t>) ||
            (traits::is_complex_v<Out> && std::is_same_v<Lhs, traits::value_type_t<Out>> && std::is_same_v<Rhs, Out> && traits::is_any_v<Op, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t>) ||
            (traits::is_complex_v<Out> && std::is_same_v<Rhs, traits::value_type_t<Out>> && std::is_same_v<Lhs, Out> && traits::is_any_v<Op, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t>);

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename Op>
    constexpr bool is_valid_ewise_trinary_bool_v =
            traits::is_restricted_scalar_v<Lhs> &&
            (traits::are_all_same_v<Lhs, Mhs, Rhs, Out> || traits::are_all_same_v<Lhs, Mhs, Rhs> && std::is_same_v<Out, bool>)
            && traits::is_any_v<Op, within_t, within_equal_t>;

    template<typename Op>
    constexpr bool is_valid_ewise_trinary_arithmetic_op_v =
            traits::is_any_v<Op,
                    plus_t, plus_minus_t, plus_multiply_t, plus_divide_t,
                    minus_t, minus_plus_t, minus_multiply_t, minus_divide_t,
                    multiply_t, multiply_plus_t, multiply_minus_t, multiply_divide_t,
                    divide_t, divide_plus_t, divide_minus_t, divide_multiply_t, divide_epsilon_t>;

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename Op>
    constexpr bool is_valid_ewise_trinary_arithmetic_v =
            is_valid_ewise_trinary_arithmetic_op_v<Op> &&
            ((traits::is_restricted_scalar_v<Lhs> && traits::are_all_same_v<Lhs, Mhs, Rhs, Out>) ||
             (traits::are_all_same_v<traits::value_type_t<Lhs>, traits::value_type_t<Mhs>, traits::value_type_t<Rhs>, traits::value_type_t<Out>> &&
              (traits::is_complex_v<Lhs> || traits::is_complex_v<Mhs> || traits::is_complex_v<Rhs>) && traits::is_complex_v<Out>));

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename Op>
    constexpr bool is_valid_ewise_trinary_v =
            is_valid_ewise_trinary_bool_v<Lhs, Mhs, Rhs, Out, Op> ||
            is_valid_ewise_trinary_arithmetic_v<Lhs, Mhs, Rhs, Out, Op> ||
            (traits::is_restricted_scalar_v<Lhs> &&
             traits::are_all_same_v<Lhs, Mhs, Rhs, Out> &&
             std::is_same_v<Op, clamp_t>);
}

namespace noa::cuda {
    // Element-wise transformation using a unary operator()(In) -> Out
    template<typename In, typename Out, typename UnaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_unary_v<In, Out, UnaryOp>>>
    void ewise_unary(const In* input, const Strides4<i64>& input_strides,
                     Out* output, const Strides4<i64>& output_strides,
                     const Shape4<i64>& shape, UnaryOp unary_op, Stream& stream);

    // Element-wise transformation using a binary operator()(Lhs, Rhs) -> Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_binary_v<Lhs, Rhs, Out, BinaryOp>>>
    void ewise_binary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                      const Rhs* rhs, const Strides4<i64>& rhs_strides,
                      Out* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp binary_op, Stream& stream);

    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_binary_v<Lhs, Rhs, Out, BinaryOp>>>
    void ewise_binary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                      Rhs rhs,
                      Out* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp binary_op, Stream& stream);

    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_binary_v<Lhs, Rhs, Out, BinaryOp>>>
    void ewise_binary(Lhs lhs,
                      const Rhs* rhs, const Strides4<i64>& rhs_strides,
                      Out* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp binary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(Lhs, Mhs, Rhs) -> Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream);

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise_trinary(Lhs lhs,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream);

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       Mhs mhs,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream);

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       Rhs rhs,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream);

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       Mhs mhs,
                       Rhs rhs,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream);

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise_trinary(Lhs lhs,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       Rhs rhs,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream);

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise_trinary(Lhs lhs,
                       Mhs mhs,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream);
}
