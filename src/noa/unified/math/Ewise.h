#pragma once

#include "noa/unified/Array.h"

// -- Unary operators -- //
namespace noa::math {
    /// Element-wise transformation using an unary operator()(\p In) -> \p Out.
    /// \param[in] input    Input to transform.
    /// \param[out] output  Transformed array.
    /// \param unary_op     Unary operator. The output is explicitly cast to \p Out.
    /// \note On the GPU, supported operators and types are limited to the following list:
    ///     Integers:
    ///       - (copy|square|abs|negate|one_minus|nonzero|logical_not)_t(A) -> A
    ///       - (copy|square|nonzero|logical_not)_t(B) -> B
    ///       - (nonzero|logical_not)_t(A|B) -> bool
    ///     Floating-points:
    ///       - (copy|square|abs|negate|one_minus|inverse)_t(C) -> C
    ///       - (sqrt|rsqrt|exp|log|cos|sin|one_log|abs_one_log)_t(C) -> C
    ///       - (round|rint|ceil|floor|trunc)_t(C) -> C
    ///     Complex:
    ///       - (square|one_minus|inverse|normalize|conj)_t(D) -> D
    ///       - (abs|abs_squared|abs_one_log|real|imag)_t(D) -> C
    ///     Where:
    ///         A = int16_t, int32_t, or int64_t
    ///         B = uint16_t, uint32_t, or uint64_t
    ///         C = half_t, float, or double
    ///         D = chalf_t, cfloat_t, or cdouble_t
    template<typename In, typename Out, typename UnaryOp>
    void ewise(const Array<In>& input, const Array<Out>& output, UnaryOp&& unary_op);
}

// -- Binary operators -- //
namespace noa::math {
    /// Element-wise transformation using a binary operator()(\p Lhs, \p Rhs) -> \p Out.
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param binary_op    Binary operator. The output is explicitly cast to \p Out.
    /// \note On the GPU, supported operators and types are limited to the following list:
    ///     Integers:
    ///       - (plus|minus|multiply|divide|divide_safe|dist2|min|max)_t(A,A) -> A
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal|modulo|logical_and|logical_or)_t(A,A) -> A
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal|logical_and|logical_not)_t(A,A) -> bool
    ///     Floating-points:
    ///       - (plus|minus|multiply|divide|divide_safe|dist2|min|max)_t(B,B) -> B
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal|pow)_t(B,B) -> B
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal)_t(B,B) -> bool
    ///     Complex:
    ///       - (plus|minus|multiply|divide|divide_safe|dist2|multiply_conj)_t(C,C) -> C
    ///       - (plus|minus|multiply|divide|divide_safe|dist2)_t(C,B) -> C
    ///       - (plus|minus|multiply|divide|divide_safe|dist2)_t(B,C) -> C
    ///     Where:
    ///         A = int16_t, int32_t, int64_t, uint16_t, uint32_t, or uint64_t
    ///         B = half_t, float, or double
    ///         C = chalf_t, cfloat_t, or cdouble_t
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<Rhs>>>
    void ewise(const Array<Lhs>& lhs, Rhs rhs, const Array<Out>& output, BinaryOp&& binary_op);

    /// Element-wise transformation using a binary operator()(\p Lhs, \p Rhs) -> \p Out.
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param binary_op    Binary operator. The output is explicitly cast to \p Out.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<Lhs>>>
    void ewise(Lhs lhs, const Array<Rhs>& rhs, const Array<Out>& output, BinaryOp&& binary_op);

    /// Element-wise transformation using a binary operator()(\p Lhs, \p Rhs) -> \p Out.
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param binary_op    Binary operator. The output is explicitly cast to \p Out.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp>
    void ewise(const Array<Lhs>& lhs, const Array<Rhs>& rhs, const Array<Out>& output, BinaryOp&& binary_op);
}

// -- Trinary operators -- //
namespace noa::math {
    /// Element-wise transformation using a trinary operator()(\p Lhs, \p Mhs, \p Rhs) -> \p Out.
    /// \param[in] lhs      On the \b device. Left-hand side argument.
    /// \param mhs          Middle-hand side argument.
    /// \param rhs          Right-hand side argument.
    /// \param[out] output  On the \b device. Transformed array.
    /// \param trinary_op   Trinary operator. The output is explicitly cast to \p Out.
    /// \note On the GPU, supported operators and types are limited to the following list:
    ///     Integers and Floating-points:
    ///       - (within|within_equal|clamp)_t(A,A,A) -> A or bool
    ///       - (plus|plus_minus|plus_multiply|plus_divide)_t(A,A,A) -> A
    ///       - (minus|minus_plus|minus_multiply|minus_divide)_t(A,A,A) -> A
    ///       - (multiply|multiply_plus|multiply_minus|multiply_divide)_t(A,A,A) -> A
    ///       - (divide|divide_plus|divide_minus|divide_multiply|divide_epsilon)_t(A,A,A) -> A
    ///     Complex:
    ///       - (plus|plus_minus|plus_multiply|plus_divide)_t(B,B,B) -> C
    ///       - (minus|minus_plus|minus_multiply|minus_divide)_t(B,B,B) -> C
    ///       - (multiply|multiply_plus|multiply_minus|multiply_divide)_t(B,B,B) -> C
    ///       - (divide|divide_plus|divide_minus|divide_multiply|divide_epsilon)_t(B,B,B) -> C
    ///     Where:
    ///         A = (u)int16_t, (u)int32_t, (u)int64_t, half_t, float or double
    ///         B = half_t, float, double, chalf_t, cfloat_t or cdouble_t
    ///         C = chalf_t, cfloat_t or cdouble_t
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<Mhs> && noa::traits::is_data_v<Rhs>>>
    void ewise(const Array<Lhs>& lhs, Mhs mhs, Rhs rhs, const Array<Out>& output, TrinaryOp&& trinary_op);

    /// Element-wise transformation using a trinary operator()(\p Lhs, \p Mhs, \p Rhs) -> \p Out.
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] mhs      Middle-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param trinary_op   Trinary operator. The output is explicitly cast to \p Out.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<Rhs>>>
    void ewise(const Array<Lhs>& lhs, const Array<Mhs>& mhs, Rhs rhs,
               const Array<Out>& output, TrinaryOp&& trinary_op);

    /// Element-wise transformation using a trinary operator()(\p Lhs, \p Mhs, \p Rhs) -> \p Out.
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] mhs      Middle-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param trinary_op   Trinary operator. The output is explicitly cast to \p Out.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<Mhs>>>
    void ewise(const Array<Lhs>& lhs, Mhs mhs, const Array<Rhs>& rhs,
               const Array<Out>& output, TrinaryOp&& trinary_op);

    /// Element-wise transformation using a trinary operator()(\p Lhs, \p Mhs, \p Rhs) -> \p Out.
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] mhs      Middle-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param trinary_op   Trinary operator. The output is explicitly cast to \p Out.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp>
    void ewise(const Array<Lhs>& lhs, const Array<Mhs>& mhs, const Array<Rhs>& rhs,
               const Array<Out>& output, TrinaryOp&& trinary_op);
}

#define NOA_UNIFIED_EWISE_
#include "noa/unified/math/Ewise.inl"
#undef NOA_UNIFIED_EWISE_
