#pragma once

#include "noa/cpu/math/Blas.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Blas.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::math::details {
    template<typename T>
    constexpr bool is_valid_dot_t = noa::traits::is_any_v<T, i32, i64, u32, u64, f32, f64, c32, c64>;

    template<typename T>
    constexpr bool is_valid_matmul_t = noa::traits::is_any_v<T, f32, f64, c32, c64>;
}

namespace noa::math {
    /// Returns the vector-vector dot product.
    /// \tparam Lhs, Rhs    Array or View of i32, i64, u32, u64, f32, f64, c32, or c64.
    /// \param[in] lhs      Unbatched row or column vector.
    /// \param[in] rhs      Unbatched row or column vector.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<typename Lhs, typename Rhs, typename = std::enable_if_t<
             noa::traits::are_array_or_view_v<Lhs, Rhs> &&
             noa::traits::are_almost_same_value_type_v<Lhs, Rhs> &&
             details::is_valid_dot_t<noa::traits::mutable_value_type_t<Lhs>>>>
    [[nodiscard]] auto dot(const Lhs& lhs, const Rhs& rhs) {
        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty(), "Empty array detected");
        NOA_CHECK(!lhs.shape().is_batched() && lhs.shape().ndim() <= 2 &&
                  !rhs.shape().is_batched() && rhs.shape().ndim() <= 2,
                  "The inputs should be unbatched row or column vectors, but got shape lhs:{} and rhs:{}",
                  lhs.shape(), rhs.shape());

        const Device device = rhs.device();
        NOA_CHECK(device == lhs.device(),
                  "The input arrays must be on the same device, but got lhs:{} and rhs:{}",
                  lhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::dot(lhs.get(), lhs.strides(), lhs.shape(),
                               rhs.get(), rhs.strides(), rhs.shape(),
                               threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::dot(lhs.get(), lhs.strides(), lhs.shape(),
                            rhs.get(), rhs.strides(), rhs.shape(),
                            cuda_stream);
            cuda_stream.enqueue_attach(lhs.share(), rhs.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the (batched) vector-vector dot product.
    /// \tparam Lhs, Rhs, Output    Array or View of i32, i64, u32, u64, f32, f64, c32, or c64.
    /// \param[in] lhs              (Batched) row or column vector.
    /// \param[in] rhs              (Batched) row or column vector.
    /// \param[out] output          Output contiguous vector with the dot products. One element per batch.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<typename Lhs, typename Rhs, typename Output, typename = std::enable_if_t<
             noa::traits::are_array_or_view_v<Lhs, Rhs, Output> &&
             noa::traits::are_almost_same_value_type_v<Lhs, Rhs, Output> &&
             details::is_valid_dot_t<noa::traits::value_type_t<Output>>>>
    void dot(const Lhs& lhs, const Rhs& rhs, const Output& output) {
        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(noa::indexing::is_vector(lhs.shape(), true) &&
                  noa::indexing::is_vector(rhs.shape(), true) &&
                  lhs.shape()[1] == 1 && rhs.shape()[1] == 1,
                  "The input should be (batched) column or row vectors, but got lhs:{} and rhs:{}",
                  lhs.shape(), rhs.shape());
        NOA_CHECK(lhs.shape()[2] * lhs.shape()[3] == rhs.shape()[2] * rhs.shape()[3],
                  "The input vectors don't have the same number of elements. Got lhs:{} and rhs:{}",
                  lhs.shape()[2] * lhs.shape()[3], rhs.shape()[2] * rhs.shape()[3]);

        NOA_CHECK(noa::indexing::is_vector(output.shape()) && output.is_contiguous(),
                  "The output should be a contiguous vector, but got shape {} and stride {}",
                  output.shape(), output.strides());

        const i64 batches = output.elements();
        auto lhs_stride = lhs.strides();
        if (!noa::indexing::broadcast(lhs.shape()[0], lhs_stride[0], batches))
            NOA_THROW("Cannot broadcast a size of {} into a size of {}", lhs.shape()[0], batches);
        auto rhs_stride = rhs.strides();
        if (!noa::indexing::broadcast(rhs.shape()[0], rhs_stride[0], batches))
            NOA_THROW("Cannot broadcast a size of {} into a size of {}", rhs.shape()[0], batches);

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::dot(lhs.get(), lhs_stride, lhs.shape(),
                               rhs.get(), rhs_stride, rhs.shape(),
                               output.get(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::dot(lhs.get(), lhs_stride, lhs.shape(),
                            rhs.get(), rhs_stride, rhs.shape(),
                            output.get(), cuda_stream);
            cuda_stream.enqueue_attach(lhs.share(), rhs.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes a matrix-matrix product, with general matrices.
    /// \details This function computes a matrix-matrix product, but it also accepts vectors.
    ///          As such, it can computes a matrix-vector product, a vector-matrix product and
    ///          the vector-vector outer-product or dot product.
    /// \tparam Lhs, Rhs, Output    Array or View of f32, f64, c32, or c64.
    /// \param[in] lhs              Dense {B,1,M,K} matrix.
    /// \param[in] rhs              Dense {B,1,K,N} matrix.
    /// \param[out] output          Dense {B,1,M,N} matrix.
    /// \note The memory layout is restricted: \p lhs, \p rhs and \p output should not overlap. All matrices should
    ///       either be row-major or column-major. The innermost dimension of the matrices should be contiguous and
    ///       the second-most dimension cannot be broadcast.
    template<typename Lhs, typename Rhs, typename Output, typename = std::enable_if_t<
             noa::traits::are_array_or_view_v<Lhs, Rhs, Output> &&
             noa::traits::are_almost_same_value_type_v<Lhs, Rhs, Output> &&
             details::is_valid_matmul_t<noa::traits::value_type_t<Output>>>>
    void matmul(const Lhs& lhs, const Rhs& rhs, const Output& output) {
        using value_t = noa::traits::value_type_t<Output>;
        matmul(lhs, rhs, output, value_t{1}, value_t{0});
    }

    /// Computes a scalar-matrix-matrix product, with general matrices and adds the result to a scalar-matrix product.
    /// \details This function computes a matrix-matrix product, but it also accepts vectors.
    ///          As such, it can computes a matrix-vector product, a vector-matrix product and
    ///          the vector-vector outer-product or dot product. The operation is defined as:
    ///          \p output = \p alpha * \p lhs * \p rhs + \p beta * \p output.
    /// \tparam Lhs, Rhs, Output    Array or View of f32, f64, c32, or c64.
    /// \param[in] lhs              Dense {B,1,M,K} matrix.
    /// \param[in] rhs              Dense {B,1,K,N} matrix.
    /// \param[out] output          Dense {B,1,M,N} matrix.
    /// \param alpha                Scalar for the scalar-matrix-matrix product.
    /// \param beta                 Scalar for the scalar-matrix product.
    ///                             If Value{0}, \p output doesn't need to be set.
    /// \param lhs_op               Whether \p lhs should be transposed before the operation.
    ///                             In this case, the matrix {B,1,K,M} is expected.
    /// \param rhs_op               Whether \p rhs should be transposed before the operation.
    ///                             In this case, the matrix {B,1,N,K} is expected.
    /// \note The memory layout is restricted: \p lhs and \p rhs should not overlap with \p output. All matrices should
    ///       either be row-major or column-major (before transposition). The innermost dimension of the matrices
    ///       (before transposition) should be contiguous and the second-most dimension cannot be broadcast.
    template<typename Lhs, typename Rhs, typename Output, typename Value, typename = std::enable_if_t<
             noa::traits::are_array_or_view_v<Lhs, Rhs, Output> &&
             noa::traits::are_almost_same_value_type_v<Lhs, Rhs, Output> &&
             details::is_valid_matmul_t<noa::traits::value_type_t<Output>> &&
             noa::traits::is_almost_same_v<Value, noa::traits::value_type_t<Output>>>>
    void matmul(const Lhs& lhs, const Rhs& rhs, const Output& output,
                Value alpha, Value beta, bool lhs_transpose = false, bool rhs_transpose = false) {
        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(lhs, output) &&
                  !noa::indexing::are_overlapped(rhs, output),
                  "Input and output arrays should not overlap");

        [[maybe_unused]] const bool is_col = noa::indexing::is_column_major(lhs.strides());
        NOA_CHECK(is_col == noa::indexing::is_column_major(rhs.strides()) &&
                  is_col == noa::indexing::is_column_major(output.strides()),
                  "All matrices should either be row-major or column-major");

        [[maybe_unused]] const i32 innermost = 3 - is_col;
        [[maybe_unused]] const i32 secondmost = 2 + is_col;
        NOA_CHECK(lhs.strides()[innermost] == 1 && lhs.strides()[secondmost] >= lhs.shape()[innermost],
                  "The innermost dimension of the left-hand side should be contiguous and "
                  "the second-most dimension cannot be broadcast, but got shape:{} and stride:{}",
                  lhs.shape(), lhs.strides());
        NOA_CHECK(rhs.strides()[innermost] == 1 && rhs.strides()[secondmost] >= rhs.shape()[innermost],
                  "The innermost dimension of the right-hand side should be contiguous and "
                  "the second-most dimension cannot be broadcast, but got shape:{} and stride:{}",
                  rhs.shape(), rhs.strides());
        NOA_CHECK(output.strides()[innermost] == 1 && output.strides()[secondmost] >= output.shape()[innermost],
                  "The innermost dimension of the output should be contiguous and "
                  "the second-most dimension cannot be broadcast, but got shape:{} and stride:{}",
                  output.shape(), output.strides());

        const Device device = output.device();
        NOA_CHECK(device == lhs.device() && device == rhs.device(),
                  "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
                  lhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::matmul(lhs.get(), lhs.strides(), lhs.shape(),
                                  rhs.get(), rhs.strides(), rhs.shape(),
                                  alpha, beta, lhs_transpose, rhs_transpose,
                                  output.get(), output.strides(), output.shape(),
                                  threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::matmul(lhs.get(), lhs.strides(), lhs.shape(),
                               rhs.get(), rhs.strides(), rhs.shape(),
                               alpha, beta, lhs_transpose, rhs_transpose,
                               output.get(), output.strides(), output.shape(),
                               cuda_stream);
            cuda_stream.enqueue_attach(lhs.share(), rhs.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
