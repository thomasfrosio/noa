#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Functors.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::math::details {
    using namespace noa::math;
    using namespace noa::traits;

    template<typename In, typename Out, typename Op>
    constexpr bool is_valid_ewise_unary_v =
            (is_any_v<Out, int16_t, int32_t, int64_t> && std::is_same_v<In, Out> && is_any_v<Op, copy_t, square_t, abs_t, negate_t, one_minus_t, nonzero_t, logical_not_t>) ||
            (is_any_v<Out, uint16_t, uint32_t, uint64_t> && std::is_same_v<In, Out> && is_any_v<Op, copy_t, square_t, nonzero_t, logical_not_t>) ||
            (is_any_v<In, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t> && std::is_same_v<Out, bool> && is_any_v<Op, nonzero_t, logical_not_t>) ||
            (is_float_v<Out> && std::is_same_v<In, Out> && is_any_v<Op, copy_t, square_t, abs_t, negate_t, one_minus_t, inverse_t, sqrt_t, rsqrt_t, exp_t, log_t, cos_t, sin_t, one_log_t, abs_one_log_t>) ||
            (is_float_v<Out> && std::is_same_v<In, Out> && is_any_v<Op, round_t, rint_t, ceil_t, floor_t, trunc_t>) ||
            (is_complex_v<Out> && std::is_same_v<In, Out> && is_any_v<Op, one_minus_t, square_t, inverse_t, normalize_t, conj_t>) ||
            (is_complex_v<In> && std::is_same_v<Out, value_type_t<In>> && is_any_v<Op, abs_t, real_t, imag_t, abs_squared_t, abs_one_log_t>);

    template<typename Lhs, typename Rhs, typename Out, typename Op>
    constexpr bool is_valid_ewise_binary_v =
            (is_any_v<Out, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t> && are_all_same_v<Lhs, Rhs, Out> &&
             is_any_v<Op, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t, min_t, max_t,
                      equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, modulo_t, logical_and_t, logical_or_t>) ||
            (std::is_same_v<Out, bool> && is_any_v<Lhs, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t> && std::is_same_v<Lhs, Rhs> &&
             is_any_v<Op, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, modulo_t, logical_and_t, logical_or_t>) ||
            (is_float_v<Out> && are_all_same_v<Lhs, Rhs, Out> &&
             is_any_v<Op, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t, min_t, max_t, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, pow_t>) ||
            (is_float_v<Lhs> && std::is_same_v<Lhs, Rhs> && std::is_same_v<Out, bool> &&
             is_any_v<Op, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, pow_t>) ||
            (is_complex_v<Out> && are_all_same_v<Lhs, Rhs, Out> && is_any_v<Op, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t, multiply_conj_t>) ||
            (is_complex_v<Out> && std::is_same_v<Lhs, value_type_t<Out>> && std::is_same_v<Rhs, Out> && is_any_v<Op, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t>) ||
            (is_complex_v<Out> && std::is_same_v<Rhs, value_type_t<Out>> && std::is_same_v<Lhs, Out> && is_any_v<Op, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t>);

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename Op>
    constexpr bool is_valid_ewise_trinary_bool_v =
            is_any_v<Lhs, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, half_t, float, double> &&
            (are_all_same_v<Lhs, Mhs, Rhs, Out> || are_all_same_v<Lhs, Mhs, Rhs> && std::is_same_v<Out, bool>)
            && is_any_v<Op, within_t, within_equal_t, clamp_t>;

    template<typename Op>
    constexpr bool is_valid_ewise_trinary_arithmetic_op_v =
            is_any_v<Op,
                     plus_t, plus_minus_t, plus_multiply_t, plus_divide_t,
                     minus_t, minus_plus_t, minus_multiply_t, minus_divide_t,
                     multiply_t, multiply_plus_t, multiply_minus_t, multiply_divide_t,
                     divide_t, divide_plus_t, divide_minus_t, divide_multiply_t, divide_epsilon_t>;

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename Op>
    constexpr bool is_valid_ewise_trinary_arithmetic_v =
            is_valid_ewise_trinary_arithmetic_op_v<Op> &&
            ((is_any_v<Lhs, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, half_t, float, double> && are_all_same_v<Lhs, Mhs, Rhs, Out>) ||
             (are_all_same_v<traits::value_type_t<Lhs>, traits::value_type_t<Mhs>, traits::value_type_t<Rhs>, traits::value_type_t<Out>> &&
              (is_complex_v<Lhs> || is_complex_v<Mhs> || is_complex_v<Rhs>) && is_complex_v<Out>));

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename Op>
    constexpr bool is_valid_ewise_trinary_v = is_valid_ewise_trinary_bool_v<Lhs, Mhs, Rhs, Out, Op> || is_valid_ewise_trinary_arithmetic_v<Lhs, Mhs, Rhs, Out, Op>;
}

namespace noa::cuda::math {
    // Element-wise transformation using an unary operator()(\p In) -> \p Out
    template<typename In, typename Out, typename UnaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_unary_v<In, Out, UnaryOp>>>
    void ewise(const shared_t<In[]>& input, dim4_t input_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, UnaryOp unary_op, Stream& stream);
}

namespace noa::cuda::math {
    // Element-wise transformation using a binary operator()(\p Lhs, \p Rhs) -> \p Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp,
             std::enable_if_t<details::is_valid_ewise_binary_v<Lhs, Rhs, Out, BinaryOp>, bool> = true>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp binary_op, Stream& stream);

    // Element-wise transformation using a binary operator()(\p Lhs, \p Rhs) -> \p Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp,
             std::enable_if_t<details::is_valid_ewise_binary_v<Lhs, Rhs, Out, BinaryOp>, bool> = true>
    void ewise(Lhs lhs,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp binary_op, Stream& stream);

    // Element-wise transformation using a binary operator()(\p Lhs, \p Rhs) -> \p Out
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_binary_v<Lhs, Rhs, Out, BinaryOp>>>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp binary_op, Stream& stream);
}

namespace noa::cuda::math {
    // Element-wise transformation using a trinary operator()(\p Lhs, \p Mhs, \p Rhs) -> \p Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise(Lhs lhs,
               const shared_t<Mhs[]>& mhs, dim4_t mhs_strides,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p Lhs, \p Mhs, \p Rhs) -> \p Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               Mhs mhs,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p Lhs, \p Mhs, \p Rhs) -> \p Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               const shared_t<Mhs[]>& mhs, dim4_t mhs_strides,
               Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p Lhs, \p Mhs, \p Rhs) -> \p Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise(Lhs lhs,
               Mhs mhs,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p Lhs, \p Mhs, \p Rhs) -> \p Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise(Lhs lhs,
               const shared_t<Mhs[]>& mhs, dim4_t mhs_strides,
               Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p Lhs, \p Mhs, \p Rhs) -> \p Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               Mhs mhs,
               Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p Lhs, \p Mhs, \p Rhs) -> \p Out
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<Lhs, Mhs, Rhs, Out, TrinaryOp>>>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               const shared_t<Mhs[]>& mhs, dim4_t mhs_strides,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp trinary_op, Stream& stream);
}
