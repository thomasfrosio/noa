#pragma once

#include "noa/Array.h"

namespace noa::math::details {
    template<typename T, typename U>
    constexpr bool is_valid_lstsq_t = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                      traits::is_almost_same_v<traits::value_type_t<T>, U>;
}

namespace noa::math {
    /// Computes least-squares solution to equation Ax = b.
    /// \details Computes a vector x such that the 2-norm |b - A x| is minimized. Several right hand side vectors b
    ///          and solution vectors x can be handled in a single call, in which case they are stored as the columns
    ///          of the matrix b. Furthermore, the entire problem can be batched.
    ///
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \tparam U               float or double.
    /// \param[in,out] a        Dense M-by-N matrix. Can be batched. It is overwritten.
    /// \param[in,out] b        Dense M-by-K matrix, where K can be 1.
    /// \param[out] x           Dense N-by-K solution matrix.
    /// \param cond             Used to determine effective rank of \p a. Cutoff for "small" singular values.
    ///                         Singular values smaller than \p cond * largest_singular_value are considered zero.
    ///                         If <= 0, machine precision is used instead.
    /// \param[out] svd         The min(M,N) singular values of \p a in decreasing order.
    ///                         If empty, a complete orthogonal factorization of \p a is used to solve the problem,
    ///                         which can be slightly faster on many problems but the SVD method is more stable.
    ///
    /// \note For optimization purposes, \p x can point to the same memory as \p b. In this case, \p b will be
    ///       overwritten to fit the N-by-K solution matrix, thus, \p b should have a shape of max(M,N)-by-K.
    ///       If \p x and \p b do not point to the same memory, the implementation may have to copy \p b into a
    ///       temporary array.
    /// \note In most cases, it is more efficient if \p a, \p b and \p x (if K > 1) are column major and contiguous.
    ///       In any case, the innermost dimension should be contiguous and the second-most dimension can either be
    ///       contiguous or padded.
    /// \note This function is currently not supported on the GPU.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_lstsq_t<T, U>>>
    void lstsq(const Array<T>& a, const Array<T>& b, const Array<T>& x,
               float cond = 0, const Array<U>& svd = {});

    /// Fits a polynomial 2D surface onto a regular grid represented by a 2D image.
    /// \tparam T               float or double.
    /// \param[in] input        Input 2D array(s).
    /// \param order            Order of the polynomial, plane: 1, quadratic: 2 or cubic: 3.
    /// \param[out] output      Output surface or image subtracted with the surface.
    ///                         Can be equal to input. If empty, it is ignored.
    /// \param subtract         Whether \p input should be subtracted by the surface and saved in \p output.
    ///                         If true, the input and output should have the same shape.
    /// \param[out] parameters  Surface parameters. One set per batch.
    ///                         If \p order is 1: p[0] + p[1]*x + p[2]*y = 0
    ///                         If \p order is 2: p[0] + p[1]*x + p[2]*y + p[3]*x*y + p[4]*x*x + p[5]*y*y = 0
    ///                         If \p order is 3: p[0] + p[1]*x + p[2]*y + p[3]*x*y + p[4]*x*x + p[5]*y*y + ...
    ///                                           p[6]*x^2*y + p[7]*x*y^2 + p[8]*x^3 + p[9]*y^3 = 0
    /// \note This function can allocate a lot of memory, at least (\p order + 1) times the input size.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void surface(const Array<T>& input, int order,
                 const Array<T>& output = {}, bool subtract = false,
                 const Array<T>& parameters = {});
}

#define NOA_UNIFIED_LINALG_
#include "noa/math/details/LinAlg.inl"
#undef NOA_UNIFIED_LINALG_
