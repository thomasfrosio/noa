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
    [[nodiscard]] T dot(const Array<T>& lhs, const Array<T>& rhs);

    /// Computes the (batched) vector-vector dot product.
    /// \tparam T           (u)int32_t, (u)int64_t, float, double, cfloat_t and cdouble_t.
    /// \param[in] lhs      (Batched) row or column vector.
    /// \param[in] rhs      (Batched) row or column vector.
    /// \param[out] output  Output contiguous vector with the dot products. One element per batch.
    ///                     If \p lhs and \p rhs are on the GPU, \p output can be on any device, including the CPU.
    ///                     Otherwise, it must be dereferenceable by the CPU.
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
    /// \note The memory layout is restricted: \p lhs, \p rhs and \p output should not overlap. All matrices should
    ///       either be row-major or column-major. The innermost dimension of the matrices should be contiguous and
    ///       the second-most dimension cannot be broadcast.
    template<typename T, typename = std::enable_if_t<details::is_valid_matmul_t<T>>>
    void matmul(const Array<T>& lhs, const Array<T>& rhs, const Array<T>& output);

    /// Computes a scalar-matrix-matrix product, with general matrices and adds the result to a scalar-matrix product.
    /// \details This function computes a matrix-matrix product, but it also accepts vectors.
    ///          As such, it can computes a matrix-vector product, a vector-matrix product and
    ///          the vector-vector outer-product or dot product. The operation is defined as:
    ///          \p output = \p alpha * \p lhs * \p rhs + \p beta * \p output.
    /// \tparam T           float, double, cfloat, cdouble.
    /// \param[in] lhs      Dense {B,1,M,K} matrix.
    /// \param[in] rhs      Dense {B,1,K,N} matrix.
    /// \param[out] output  Dense {B,1,M,N} matrix.
    /// \param alpha        Scalar for the scalar-matrix-matrix product.
    /// \param beta         Scalar for the scalar-matrix product. If T{0}, \p output doesn't need to be set.
    /// \param lhs_op       Whether \p lhs should be transposed before the operation.
    ///                     In this case, the matrix {B,1,K,M} is expected.
    /// \param rhs_op       Whether \p rhs should be transposed before the operation.
    ///                     In this case, the matrix {B,1,N,K} is expected.
    /// \note The memory layout is restricted: \p lhs and \p rhs should not overlap with \p output. All matrices should
    ///       either be row-major or column-major (before transposition). The innermost dimension of the matrices
    ///       (before transposition) should be contiguous and the second-most dimension cannot be broadcast.
    template<typename T, typename = std::enable_if_t<details::is_valid_matmul_t<T>>>
    void matmul(const Array<T>& lhs, const Array<T>& rhs, const Array<T>& output,
                T alpha, T beta, bool lhs_transpose = false, bool rhs_transpose = false);
}

#define NOA_UNIFIED_BLAS_
#include "noa/unified/math/Blas.inl"
#undef NOA_UNIFIED_BLAS_
