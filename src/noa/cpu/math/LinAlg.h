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
    // TODO Add svd

    /// Computes least-squares solution to equation Ax = b.
    /// \details Computes a vector x such that the 2-norm |b - A x| is minimized. Several right hand side vectors b
    ///          and solution vectors x can be handled in a single call, in which case they are stored as the columns
    ///          of the matrix b.
    ///
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in,out] a        On the \b host. Dense MxN matrix. Can be batched. It is overwritten.
    /// \param a_strides        BDHW strides of \p a. The two innermost dimension should not be broadcast.
    /// \param a_shape          BDHW shape of \p a.
    /// \param[in,out] b        On the \b host. Dense MxK matrix, where K can be 1. Can be batched.
    ///                         It is overwritten with the NxK solution matrix.
    /// \param b_strides        BDHW strides of \p b.
    /// \param b_shape          BDHW shape of \p b.
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
    /// \note The memory layout is restricted: \p a and \p b should not overlap and should either be row-major or
    ///       column-major. The innermost dimension of the matrices should be contiguous and the second-most dimension
    ///       cannot be broadcast but can be padded.
    /// \note Some LAPACKE interfaces (e.g. OpenBLAS) do not natively support row-major matrices and transposes
    ///       them internally, requiring more memory and running slower. As such, it might be more efficient to for
    ///       the matrices \p a and \p b (if K > 1) to be column major.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_lstsq_t<T, U>>>
    void lstsq(const shared_t<T[]>& a, size4_t a_strides, size4_t a_shape,
               const shared_t<T[]>& b, size4_t b_strides, size4_t b_shape,
               float cond, const shared_t<U[]>& svd,
               Stream& stream);

    /// Fits a polynomial 2D surface onto a regular grid represented by a 2D image.
    /// \tparam T               float or double.
    /// \param[in] input        On the \b host. Input 2D array(s).
    /// \param input_strides    BDHW strides of \p input.
    /// \param input_shape      BDHW shape of \p input.
    /// \param[out] output      On the \b host. Output surface or image subtracted with the surface.
    ///                         Can be equal to input. If nullptr, it is ignored.
    /// \param output_strides   BDHW shape of \p output.
    /// \param output_shape     BDHW shape of \p output.
    /// \param subtract         Whether \p input should be subtracted by the surface and saved in \p output.
    ///                         If true, the input and output should have the same shape.
    /// \param order            Order of the polynomial, plane: 1, quadratic: 2 or cubic: 3.
    /// \param[out] parameters  On the \p host. Surface parameters. One set per batch.
    ///                         If \p order is 1: p[0] + p[1]*x + p[2]*y = 0
    ///                         If \p order is 2: p[0] + p[1]*x + p[2]*y + p[3]*x*y + p[4]*x*x + p[5]*y*y = 0
    ///                         If \p order is 3: p[0] + p[1]*x + p[2]*y + p[3]*x*y + p[4]*x*x + p[5]*y*y + ...
    ///                                           p[6]*x^2*y + p[7]*x*y^2 + p[8]*x^3 + p[9]*y^3 = 0
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note This function can allocate a lot of memory, at least (\p order + 1) times the input size.
    /// \note The current implementation is faster if \p input is in the rightmost order.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void surface(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape, bool subtract,
                 int order, const shared_t<T[]>& parameters, Stream& stream);
}
