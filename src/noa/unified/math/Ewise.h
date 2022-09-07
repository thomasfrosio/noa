#pragma once

#include "noa/unified/Array.h"

// -- Unary operators -- //
namespace noa::math {
    /// Element-wise transformation using an unary operator()(\p T) -> \p U
    /// \param[in] input    Input to transform.
    /// \param[out] output  Transformed array.
    /// \note On the GPU, supported operators and types are limited to the following list:
    ///     Integers:
    ///       - (copy|square|abs|negate|one_minus|nonzero|logical_not)_t(A) -> A
    ///       - (copy|square|nonzero|logical_not)_t(B) -> B
    ///       - (nonzero|logical_not)_t(A|B) -> bool
    ///     Floating-points:
    ///       - (copy|square|abs|negate|one_minus|inverse|sqrt|rsqrt|exp|log|cos|sin)_t(C) -> C
    ///       - (round|rint|ceil|floor|trunc)_t(C) -> C
    ///     Complex:
    ///       - (square|one_minus|inverse|normalize|conj)_t(D) -> D
    ///       - (abs|abs_squared|real|imag)_t(D) -> C
    ///     Where:
    ///         A = int16_t, int32_t, or int64_t
    ///         B = uint16_t, uint32_t, or uint64_t
    ///         C = half_t, float, or double
    ///         D = chalf_t, cfloat_t, or cdouble_t
    template<typename T, typename U, typename UnaryOp>
    void ewise(const Array<T>& input, const Array<U>& output, UnaryOp unary_op);
}

// -- Binary operators -- //
namespace noa::math {
    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param binary_op    Binary operator. The output is explicitly casted to \p V.
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
    template<typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U>>>
    void ewise(const Array<T>& lhs, U rhs, const Array<V>& output, BinaryOp binary_op);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param binary_op    Binary operator. The output is explicitly casted to \p V.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<T>>>
    void ewise(T lhs, const Array<U>& rhs, const Array<V>& output, BinaryOp binary_op);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param binary_op    Binary operator. The output is explicitly casted to \p V.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const Array<T>& lhs, const Array<U>& rhs, const Array<V>& output, BinaryOp binary_op);
}

// -- Trinary operators -- //
namespace noa::math {
    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p U) -> \p V
    /// \param[in] lhs      On the \b device. Left-hand side argument.
    /// \param mhs          Middle-hand side argument.
    /// \param rhs          Right-hand side argument.
    /// \param[out] output  On the \b device. Transformed array.
    /// \param trinary_op   Trinary operation function object that will be applied.
    /// \note On the GPU, supported operators and types are limited to the following list:
    ///     Integers:
    ///       - (within|within_equal|clamp)_t(A,A,A) -> A or bool
    ///       - (fma|plus_divide|divide_threshold)_t(A,A,A) -> A
    ///     Floating-points:
    ///       - (within|within_equal|clamp)_t(B,B,B) -> B or bool
    ///       - (fma|plus_divide|divide_threshold)_t(B,B,B) -> B
    ///     Complex:
    ///       - (fma|plus_divide|divide_threshold)_t(C,C,C) -> C
    ///       - (fma|plus_divide|divide_threshold)_t(C,B,B) -> C
    ///     Where:
    ///         A = (u)int16_t, (u)int32_t, (u)int64_t
    ///         B = half_t, float, double
    ///         C = chalf_t, cfloat_t, cdouble_t
    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U> && noa::traits::is_data_v<V>>>
    void ewise(const Array<T>& lhs, U mhs, V rhs, const Array<W>& output, TrinaryOp trinary_op);

    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] mhs      Middle-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param trinary_op   Trinary operation function object that will be applied.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<V>>>
    void ewise(const Array<T>& lhs, const Array<U>& mhs, V rhs,
               const Array<W>& output, TrinaryOp trinary_op);

    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] mhs      Middle-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param trinary_op   Trinary operation function object that will be applied.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename T, typename U, typename V, typename W, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U>>>
    void ewise(const Array<T>& lhs, U mhs, const Array<V>& rhs,
               const Array<W>& output, TrinaryOp trinary_op);

    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    /// \param[in] lhs      Left-hand side argument.
    /// \param[in] mhs      Middle-hand side argument.
    /// \param[in] rhs      Right-hand side argument.
    /// \param[out] output  Transformed array.
    /// \param trinary_op   Trinary operation function object that will be applied.
    /// \note On the GPU, the same operators and types are supported as in the overload above.
    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    void ewise(const Array<T>& lhs, const Array<U>& mhs, const Array<V>& rhs,
               const Array<W>& output, TrinaryOp trinary_op);
}

#define NOA_UNIFIED_EWISE_
#include "noa/unified/math/Ewise.inl"
#undef NOA_UNIFIED_EWISE_
