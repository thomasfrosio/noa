#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::math::details {
    template<typename T>
    constexpr bool is_valid_dot_t = traits::is_any_v<uint32_t, uint64_t, int32_t, int64_t,
                                                     float, double, cfloat_t, cdouble_t>;

    template<typename T>
    constexpr bool is_valid_matmul_t = traits::is_any_v<float, double, cfloat_t, cdouble_t>;
}

namespace noa::cpu::math {
    using noa::math::BlasTranspose;

    /// Returns the vector-vector dot product.
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t and cdouble_t.
    /// \param[in] lhs          On the \b host. Unbatched row or column vector.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param lhs_shape        Rightmost shape of \p lhs.
    /// \param[in] rhs          On the \b host. Unbatched row or column vector.
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param rhs_shape        Rightmost shape of \p rhs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<typename T, typename = std::enable_if_t<details::is_valid_dot_t<T>>>
    T dot(const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
          const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
          Stream& stream);

    /// Computes the (batched) vector-vector dot product.
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t and cdouble_t.
    /// \param[in] lhs          On the \b host. (Batched) row or column vector.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param lhs_shape        Rightmost shape of \p lhs.
    /// \param[in] rhs          On the \b host. (Batched) row or column vector.
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param rhs_shape        Rightmost shape of \p rhs.
    /// \param[out] output      On the \b host. Output dot products. One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<typename T, typename = std::enable_if_t<details::is_valid_dot_t<T>>>
    void dot(const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
             const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
             const std::shared_ptr<T[]>& output, Stream& stream);

    /// Computes a scalar-matrix-matrix product and add the result to a scalar-matrix product, with general matrices.
    /// \tparam T               float, double, cfloat, cdouble.
    /// \param lhs_transpose    Transpose operation of \p lhs before the matrix-matrix product.
    /// \param rhs_transpose    Transpose operation of \p rhs before the matrix-matrix product.
    /// \param alpha            A scalar to multiply the matrix-matrix product with.
    /// \param[in] lhs          On the \b host. Dense MxK matrix. Can be batched.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param lhs_shape        Rightmost shape of \p lhs.
    /// \param[in] rhs          On the \b host. Dense KxN matrix. Can be batched.
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param rhs_shape        Rightmost shape of \p rhs.
    /// \param beta             A scalar to multiply \p output with. If T(0), \p output doesn't need to be set.
    /// \param[in,out] output   On the \b host. Dense MxN matrix. Can be batched.
    /// \param output_stride    Rightmost stride of \p output.
    /// \param output_shape     Rightmost shape of \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note The innermost dimension of the matrices (before transposition) should be contiguous and
    ///       the second-most dimension cannot be broadcast.
    template<typename T, typename = std::enable_if_t<details::is_valid_matmul_t<T>>>
    void matmul(BlasTranspose lhs_transpose, BlasTranspose rhs_transpose, T alpha,
                const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
                const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
                T beta, const std::shared_ptr<T[]>& output, size4_t output_stride, size4_t output_shape,
                Stream& stream);

    /// Computes a matrix-matrix product with general matrices.
    /// \tparam T               float, double, cfloat, cdouble.
    /// \param[in] lhs          On the \b host. Dense MxK matrix. Can be batched.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param lhs_shape        Rightmost shape of \p lhs.
    /// \param[in] rhs          On the \b host. Dense KxN matrix. Can be batched.
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param rhs_shape        Rightmost shape of \p rhs.
    /// \param[out] output      On the \b host. Dense MxN matrix. Can be batched.
    /// \param output_stride    Rightmost stride of \p output.
    /// \param output_shape     Rightmost shape of \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note The innermost dimension of the matrices (before transposition) should be contiguous and
    ///       the second-most dimension cannot be broadcast.
    template<typename T, typename = std::enable_if_t<details::is_valid_matmul_t<T>>>
    void matmul(const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
                const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
                const std::shared_ptr<T[]>& output, size4_t output_stride, size4_t output_shape,
                Stream& stream) {
        matmul(BlasTranspose::BLAS_TRANSPOSE_NONE, BlasTranspose::BLAS_TRANSPOSE_NONE, T{1},
               lhs, lhs_stride, lhs_shape, rhs, rhs_stride, rhs_shape,
               T{0}, output, output_stride, output_shape, stream);
    }
}
