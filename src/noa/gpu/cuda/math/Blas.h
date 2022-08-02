#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::math::details {
    void cublasClearCache(int device);

    template<typename T>
    constexpr bool is_valid_dot_t = traits::is_any_v<T, uint32_t, uint64_t, int32_t, int64_t,
                                                     float, double, cfloat_t, cdouble_t>;

    template<typename T>
    constexpr bool is_valid_matmul_t = traits::is_any_v<T, float, double, cfloat_t, cdouble_t>;
}

namespace noa::cuda::math {
    /// Returns the vector-vector dot product.
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t and cdouble_t.
    /// \param[in] lhs          On the \b device. Unbatched row or column vector.
    /// \param lhs_strides      BDHW strides of \p lhs.
    /// \param lhs_shape        BDHW shape of \p lhs.
    /// \param[in] rhs          On the \b device. Unbatched row or column vector.
    /// \param rhs_strides      BDHW strides of \p rhs.
    /// \param rhs_shape        BDHW shape of \p rhs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<typename T, typename = std::enable_if_t<details::is_valid_dot_t<T>>>
    T dot(const std::shared_ptr<T[]>& lhs, size4_t lhs_strides, size4_t lhs_shape,
          const std::shared_ptr<T[]>& rhs, size4_t rhs_strides, size4_t rhs_shape,
          Stream& stream);

    /// Computes the (batched) vector-vector dot product.
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t and cdouble_t.
    /// \param[in] lhs          On the \b device. (Batched) row or column vector.
    /// \param lhs_strides      BDHW strides of \p lhs.
    /// \param lhs_shape        BDHW shape of \p lhs.
    /// \param[in] rhs          On the \b device. (Batched) row or column vector.
    /// \param rhs_strides      BDHW strides of \p rhs.
    /// \param rhs_shape        BDHW shape of \p rhs.
    /// \param[out] output      On the \b host or \b device. Output dot products. One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<typename T, typename = std::enable_if_t<details::is_valid_dot_t<T>>>
    void dot(const std::shared_ptr<T[]>& lhs, size4_t lhs_strides, size4_t lhs_shape,
             const std::shared_ptr<T[]>& rhs, size4_t rhs_strides, size4_t rhs_shape,
             const std::shared_ptr<T[]>& output, Stream& stream);

    /// Computes a scalar-matrix-matrix product and add the result to a scalar-matrix product, with general matrices.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] lhs          On the \b device. Dense MxK matrix. Can be batched.
    /// \param lhs_strides      BDHW strides of \p lhs.
    /// \param lhs_shape        BDHW shape of \p lhs.
    /// \param[in] rhs          On the \b device. Dense KxN matrix. Can be batched.
    /// \param rhs_strides      BDHW strides of \p rhs.
    /// \param rhs_shape        BDHW shape of \p rhs.
    /// \param alpha            A scalar to multiply the matrix-matrix product with.
    /// \param beta             A scalar to multiply \p output with. If T(0), \p output doesn't need to be set.
    /// \param lhs_transpose    Whether \p lhs should be transposed before the matrix-matrix product.
    ///                         In this case, the matrix KxM is expected.
    /// \param rhs_transpose    Whether \p rhs should be transposed before the matrix-matrix product.
    ///                         In this case, the matrix NxK is expected.
    /// \param[in,out] output   On the \b device. Dense MxN matrix. Can be batched.
    /// \param output_strides   BDHW strides of \p output.
    /// \param output_shape     BDHW shape of \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note The memory layout is restricted: \p lhs, \p rhs and \p output should not overlap. All matrices should
    ///       either be row-major or column-major (before transposition). The innermost dimension of the matrices
    ///       (before transposition) should be contiguous and the second-most dimension cannot be broadcast.
    template<typename T, typename = std::enable_if_t<details::is_valid_matmul_t<T>>>
    void matmul(const std::shared_ptr<T[]>& lhs, size4_t lhs_strides, size4_t lhs_shape,
                const std::shared_ptr<T[]>& rhs, size4_t rhs_strides, size4_t rhs_shape,
                T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
                const std::shared_ptr<T[]>& output, size4_t output_strides, size4_t output_shape,
                Stream& stream);

    /// Computes a matrix-matrix product with general matrices.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] lhs          On the \b device. Dense MxK matrix. Can be batched.
    /// \param lhs_strides      BDHW strides of \p lhs.
    /// \param lhs_shape        BDHW shape of \p lhs.
    /// \param[in] rhs          On the \b device. Dense KxN matrix. Can be batched.
    /// \param rhs_strides      BDHW strides of \p rhs.
    /// \param rhs_shape        BDHW shape of \p rhs.
    /// \param[out] output      On the \b device. Dense MxN matrix. Can be batched.
    /// \param output_strides   BDHW strides of \p output.
    /// \param output_shape     BDHW shape of \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note The memory layout is restricted: \p lhs, \p rhs and \p output should not overlap. All matrices should
    ///       either be row-major or column-major. The innermost dimension of the matrices should be contiguous and
    ///       the second-most dimension cannot be broadcast.
    template<typename T, typename = std::enable_if_t<details::is_valid_matmul_t<T>>>
    void matmul(const std::shared_ptr<T[]>& lhs, size4_t lhs_strides, size4_t lhs_shape,
                const std::shared_ptr<T[]>& rhs, size4_t rhs_strides, size4_t rhs_shape,
                const std::shared_ptr<T[]>& output, size4_t output_strides, size4_t output_shape,
                Stream& stream) {
        matmul(lhs, lhs_strides, lhs_shape, rhs, rhs_strides, rhs_shape,
               T{1}, T{0}, false, false,
               output, output_strides, output_shape, stream);
    }
}
