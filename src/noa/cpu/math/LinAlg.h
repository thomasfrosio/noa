#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/common/Functors.h"

namespace noa::cpu::math::details {
    template<typename T, typename U>
    constexpr bool is_valid_lstsq_t = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                      traits::is_almost_same_v<traits::value_type_t<T>, U>;
}

namespace noa::cpu::math {
    // TODO Add solve(...), like https://github.com/scipy/linalg/_basic.py#L40 using e.g. genv
    // TODO Add svd, inverse and determinant

    /// Computes least-squares solution to equation Ax = b.
    /// \details Computes a vector x such that the 2-norm |b - A x| is minimized. Several right hand side vectors b
    ///          and solution vectors x can be handled in a single call, in which case they are stored as the columns
    ///          of the matrix b.
    ///
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in,out] a        On the \b host. Dense MxN matrix. Can be batched. It is overwritten.
    /// \param a_stride         Rightmost stride of \p a. The two innermost dimension should not be broadcast.
    ///                         If row-major: the rows (i.e. the second-most dimension) can be padded.
    ///                         If column-major: the columns (i.e. the innermost dimension) can be padded.
    /// \param a_shape          Rightmost shape of \p a.
    /// \param[in,out] b        On the \b host. Dense MxK matrix, where K can be 1. Can be batched.
    ///                         It is overwritten with the NxK solution matrix.
    /// \param b_stride         Rightmost stride of \p b.
    ///                         If row-major: the rows (i.e. the second-most dimension) can be padded.
    ///                         If column-major: the columns (i.e. the innermost dimension) can be padded.
    /// \param b_shape          Rightmost shape of \p b. If K is 1, \p b can be a row vector.
    ///                         In the case of M < N, \p b should be extended to fit the output and its shape should
    ///                         reflect that, i.e. its second-most dimension should have max(M,N) elements.
    /// \param cond             Used to determine effective rank of \p a. Cutoff for "small" singular values.
    ///                         Singular values smaller than \p cond * largest_singular_value are considered zero.
    ///                         If <= 0, machine precision is used instead.
    /// \param[out] svd         The min(M,N) singular values of \p a in decreasing order.
    ///                         The condition number of \p a in the 2-norm is s[0]/s[min(M,N)-1].
    ///                         If nullptr, a complete orthogonal factorization of \p a is used to solve the problem,
    ///                         which can be slightly faster on many problems but the SVD method is more stable.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note Some LAPACKE interfaces (e.g. OpenBLAS) do not natively support row-major matrices and transposes
    ///       them internally, requiring more memory and running slower. If the matrices \p a and \p b (if K > 1)
    ///       are column major, this function will correctly identify it and will call the column-major implementation.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_lstsq_t<T, U>>>
    void lstsq(const shared_t<T[]>& a, size4_t a_stride, size4_t a_shape,
               const shared_t<T[]>& b, size4_t b_stride, size4_t b_shape,
               float cond, const shared_t<U[]>& svd,
               Stream& stream);
}
