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
    // Returns the vector-vector dot product.
    template<typename T, typename = std::enable_if_t<details::is_valid_dot_t<T>>>
    T dot(const std::shared_ptr<T[]>& lhs, size4_t lhs_strides, size4_t lhs_shape,
          const std::shared_ptr<T[]>& rhs, size4_t rhs_strides, size4_t rhs_shape,
          Stream& stream);

    // Computes the (batched) vector-vector dot product.
    template<typename T, typename = std::enable_if_t<details::is_valid_dot_t<T>>>
    void dot(const std::shared_ptr<T[]>& lhs, size4_t lhs_strides, size4_t lhs_shape,
             const std::shared_ptr<T[]>& rhs, size4_t rhs_strides, size4_t rhs_shape,
             const std::shared_ptr<T[]>& output, Stream& stream);

    // Computes a scalar-matrix-matrix product and add the result to a scalar-matrix product, with general matrices.
    template<typename T, typename = std::enable_if_t<details::is_valid_matmul_t<T>>>
    void matmul(const std::shared_ptr<T[]>& lhs, size4_t lhs_strides, size4_t lhs_shape,
                const std::shared_ptr<T[]>& rhs, size4_t rhs_strides, size4_t rhs_shape,
                T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
                const std::shared_ptr<T[]>& output, size4_t output_strides, size4_t output_shape,
                Stream& stream);

    // Computes a matrix-matrix product with general matrices.
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
