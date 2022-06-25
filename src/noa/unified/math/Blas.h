#pragma once

#include "noa/unified/Array.h"

namespace noa::math::details {
    template<typename T>
    constexpr bool is_valid_dot_t = traits::is_any_v<T, uint32_t, uint64_t, int32_t, int64_t,
                                                     float, double, cfloat_t, cdouble_t>;

    template<typename T>
    constexpr bool is_valid_matmul_t = traits::is_any_v<T, float, double, cfloat_t, cdouble_t>;
}

namespace noa::math {
    /// Returns the vector-vector dot product.
    /// \tparam T       (u)int32_t, (u)int64_t, float, double, cfloat_t and cdouble_t.
    /// \param[in] lhs  Unbatched row or column vector.
    /// \param[in] rhs  Unbatched row or column vector.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<typename T, typename = std::enable_if_t<details::is_valid_dot_t<T>>>
    T dot(const Array<T>& lhs, const Array<T>& rhs);

    /// Computes the (batched) vector-vector dot product.
    /// \tparam T           (u)int32_t, (u)int64_t, float, double, cfloat_t and cdouble_t.
    /// \param[in] lhs      (Batched) row or column vector.
    /// \param[in] rhs      (Batched) row or column vector.
    /// \param[out] output  Output contiguous vector with the dot products. One element per batch.
    ///                     If \p lhs and \p rhs are on the GPU, \p output can be on any device, including the CPU.
    ///                     Otherwise, they must be dereferencable by the CPU.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<typename T, typename = std::enable_if_t<details::is_valid_dot_t<T>>>
    void dot(const Array<T>& lhs, const Array<T>& rhs, const Array<T>& output);

    /// Computes a matrix-matrix product, with general matrices.
    /// \details This function computes a matrix-matrix product, but it also accepts vectors.
    ///          As such, it can computes a matrix-vector product, a vector-matrix product and
    ///          the vector-vector outer-product or dot product.
    /// \tparam T           float, double, cfloat, cdouble.
    /// \param[in] lhs      Dense {B,1,M,K} matrix.
    /// \param[in] rhs      Dense {B,1,K,N} matrix.
    /// \param[out] output  Dense {B,1,M,N} matrix.
    /// \note The innermost dimension of the matrices should be contiguous and
    ///       the second-most dimension cannot be broadcast.
    template<typename T, typename = std::enable_if_t<details::is_valid_matmul_t<T>>>
    void matmul(const Array<T>& lhs, const Array<T>& rhs, const Array<T>& output);

    /// Computes a scalar-matrix-matrix product, with general matrices and adds the result to a scalar-matrix product.
    /// \details This function computes a matrix-matrix product, but it also accepts vectors.
    ///          As such, it can computes a matrix-vector product, a vector-matrix product and
    ///          the vector-vector outer-product or dot product. The operation is defined as:
    ///          \p output  = \p alpha * \p lhs * \p rhs + \p beta * \p output.
    /// \tparam T           float, double, cfloat, cdouble.
    /// \param[in] lhs      Dense {B,1,M,K} matrix.
    /// \param[in] rhs      Dense {B,1,K,N} matrix.
    /// \param[out] output  Dense {B,1,M,N} matrix.
    /// \param alpha        Scalar for the scalar-matrix-matrix product.
    /// \param beta         Scalar for the scalar-matrix product.
    /// \param lhs_op       Whether \p lhs should be transposed (and possibly conjugated) before the operation.
    /// \param rhs_op       Whether \p rhs should be transposed (and possibly conjugated) before the operation.
    /// \note The innermost dimension of the matrices (before transposition) should be contiguous and
    ///       the second-most dimension cannot be broadcast.
    template<typename T, typename = std::enable_if_t<details::is_valid_matmul_t<T>>>
    void matmul(const Array<T>& lhs, const Array<T>& rhs, const Array<T>& output,
                T alpha, T beta,
                BlasTranspose lhs_transpose = BLAS_TRANSPOSE_NONE,
                BlasTranspose rhs_transpose = BLAS_TRANSPOSE_NONE);
}

#define NOA_UNIFIED_BLAS_
#include "noa/unified/math/Blas.inl"
#undef NOA_UNIFIED_BLAS_
