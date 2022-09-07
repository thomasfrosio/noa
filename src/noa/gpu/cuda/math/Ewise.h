#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Functors.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::math::details {
    using namespace noa::math;
    using namespace noa::traits;

    template<typename in_t, typename out_t, typename op_t>
    constexpr bool is_valid_ewise_unary_v =
            (is_any_v<out_t, int16_t, int32_t, int64_t> && std::is_same_v<in_t, out_t> && is_any_v<op_t, copy_t, square_t, abs_t, negate_t, one_minus_t, nonzero_t, logical_not_t>) ||
            (is_any_v<out_t, uint16_t, uint32_t, uint64_t> && std::is_same_v<in_t, out_t> && is_any_v<op_t, copy_t, square_t, nonzero_t, logical_not_t>) ||
            (is_any_v<in_t, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t> && std::is_same_v<out_t, bool> && is_any_v<op_t, nonzero_t, logical_not_t>) ||
            (is_float_v<out_t> && std::is_same_v<in_t, out_t> && is_any_v<op_t, copy_t, square_t, abs_t, negate_t, one_minus_t, inverse_t, sqrt_t, rsqrt_t, exp_t, log_t, cos_t, sin_t>) ||
            (is_float_v<out_t> && std::is_same_v<in_t, out_t> && is_any_v<op_t, round_t, rint_t, ceil_t, floor_t, trunc_t>) ||
            (is_complex_v<out_t> && std::is_same_v<in_t, out_t> && is_any_v<op_t, one_minus_t, square_t, inverse_t, normalize_t, conj_t>) ||
            (is_complex_v<in_t> && std::is_same_v<out_t, value_type_t<in_t>> && is_any_v<op_t, abs_t, real_t, imag_t, abs_squared_t>);

    template<typename lhs_t, typename rhs_t, typename out_t, typename op_t>
    constexpr bool is_valid_ewise_binary_v =
            (is_any_v<out_t, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t> && are_all_same_v<lhs_t, rhs_t, out_t> &&
             is_any_v<op_t, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t, min_t, max_t,
                      equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, modulo_t, logical_and_t, logical_or_t>) ||
            (std::is_same_v<out_t, bool> && is_any_v<lhs_t, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t> && std::is_same_v<lhs_t, rhs_t> &&
             is_any_v<op_t, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, modulo_t, logical_and_t, logical_or_t>) ||
            (is_float_v<out_t> && are_all_same_v<lhs_t, rhs_t, out_t> &&
             is_any_v<op_t, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t, min_t, max_t, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, pow_t>) ||
            (is_float_v<lhs_t> && std::is_same_v<lhs_t, rhs_t> && std::is_same_v<out_t, bool> &&
             is_any_v<op_t, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t, pow_t>) ||
            (is_complex_v<out_t> && are_all_same_v<lhs_t, rhs_t, out_t> && is_any_v<op_t, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t, multiply_conj_t>) ||
            (is_complex_v<out_t> && std::is_same_v<lhs_t, value_type_t<out_t>> && std::is_same_v<rhs_t, out_t> && is_any_v<op_t, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t>) ||
            (is_complex_v<out_t> && std::is_same_v<rhs_t, value_type_t<out_t>> && std::is_same_v<lhs_t, out_t> && is_any_v<op_t, plus_t, minus_t, multiply_t, divide_t, divide_safe_t, dist2_t>);

    template<typename lhs_t, typename mhs_t, typename rhs_t, typename out_t, typename op_t>
    constexpr bool is_valid_ewise_trinary_v =
            (is_any_v<lhs_t, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, half_t, float, double> && are_all_same_v<lhs_t, mhs_t, rhs_t, out_t> && is_any_v<op_t, within_t, within_equal_t, clamp_t, fma_t, plus_divide_t, divide_epsilon_t>) ||
            (is_any_v<lhs_t, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, half_t, float, double> && are_all_same_v<lhs_t, mhs_t, rhs_t> && std::is_same_v<out_t, bool> && is_any_v<op_t, within_t, within_equal_t, clamp_t>) ||
            ((is_float_v<lhs_t> || is_complex_v<lhs_t>) && are_all_same_v<lhs_t, mhs_t, rhs_t, out_t> && is_any_v<op_t, fma_t, plus_divide_t, divide_epsilon_t>) ||
            (is_complex_v<lhs_t> && std::is_same_v<lhs_t, out_t> && are_all_same_v<mhs_t, rhs_t, value_type_t<out_t>> && is_any_v<op_t, fma_t, plus_divide_t, divide_epsilon_t>);
}

namespace noa::cuda::math {
    // Element-wise transformation using an unary operator()(\p T) -> \p U
    template<typename T, typename U, typename UnaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_unary_v<T, U, UnaryOp>>>
    void ewise(const shared_t<T[]>& input, size4_t input_strides,
               const shared_t<U[]>& output, size4_t output_strides,
               size4_t shape, UnaryOp unary_op, Stream& stream);
}

namespace noa::cuda::math {
    // Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    template<typename T, typename U, typename V, typename BinaryOp,
             std::enable_if_t<details::is_valid_ewise_binary_v<T, U, V, BinaryOp>, bool> = true>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides, U rhs,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, BinaryOp binary_op, Stream& stream);

    // Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    template<typename T, typename U, typename V, typename BinaryOp,
             std::enable_if_t<details::is_valid_ewise_binary_v<T, U, V, BinaryOp>, bool> = true>
    void ewise(T lhs, const shared_t<U[]>& rhs, size4_t rhs_strides,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, BinaryOp binary_op, Stream& stream);

    // Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    template<typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_binary_v<T, U, V, BinaryOp>>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               const shared_t<U[]>& rhs, size4_t rhs_strides,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, BinaryOp binary_op, Stream& stream);
}

namespace noa::cuda::math {
    // Element-wise transformation using a trinary operator()(\p T, \p U, \p U) -> \p V
    template<typename T, typename U, typename V, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<T, U, U, V, TrinaryOp>>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides, U mhs, U rhs,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<T, U, V, W, TrinaryOp>>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               const shared_t<U[]>& mhs, size4_t mhs_strides, V rhs,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<T, U, V, W, TrinaryOp>>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides, V mhs,
               const shared_t<U[]>& rhs, size4_t rhs_strides,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);

    // Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             typename = std::enable_if_t<details::is_valid_ewise_trinary_v<T, U, V, W, TrinaryOp>>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               const shared_t<U[]>& mhs, size4_t mhs_strides,
               const shared_t<V[]>& rhs, size4_t rhs_strides,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);
}
