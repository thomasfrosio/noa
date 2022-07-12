#pragma once

#include "noa/unified/Array.h"

namespace noa::cpu::math::details {
    template<typename T, typename U>
    constexpr bool is_valid_lstsq_t = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
    traits::is_almost_same_v<traits::value_type_t<T>, U>;
}

namespace noa::cpu::math {
    /// Computes least-squares solution to equation Ax = b.
    /// \details Computes a vector x such that the 2-norm |b - A x| is minimized. Several right hand side vectors b
    ///          and solution vectors x can be handled in a single call, in which case they are stored as the columns
    ///          of the matrix b.
    ///
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in,out] a        On the \b host. Dense MxN matrix. Can be batched. It is overwritten.
    /// \param[in,out] b        On the \b host. Dense MxK matrix, where K can be 1. Can be batched.
    ///                         It is overwritten with the NxK solution matrix.
    /// \param cond             Used to determine effective rank of \p a. Cutoff for "small" singular values.
    ///                         Singular values smaller than \p cond * largest_singular_value are considered zero.
    ///                         If <= 0, machine precision is used instead.
    /// \param[out] svd         The min(M,N) singular values of \p a in decreasing order.
    ///                         The condition number of \p a in the 2-norm is s[0]/s[min(M,N)-1].
    ///                         If empty, a complete orthogonal factorization of \p a is used to solve the problem,
    ///                         which can be slightly faster on many problems but the SVD method is more stable.
    ///
    /// \note Some LAPACKE interfaces (e.g. OpenBLAS) do not natively support row-major matrices and transposes
    ///       them internally, requiring more memory and running slower. If the matrices \p a and \p b (if K > 1)
    ///       are column major, this function will correctly identify it and will call the column-major implementation.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_lstsq_t<T, U>>>
    void lstsq(const Array<T>& a, const Array<T>& b, float cond = 0, const Array<U>& svd = {});
}

#define NOA_UNIFIED_LINALG_
#include "noa/unified/math/LinAlg.inl"
#undef NOA_UNIFIED_LINALG_
