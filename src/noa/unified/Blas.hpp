#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/Reduce.hpp"
#include "noa/cpu/Blas.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Blas.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/ReduceAxesEwise.hpp"
#include "noa/unified/ReduceEwise.hpp"
#include "noa/unified/Indexing.hpp"

namespace noa {
    /// Returns the vector-vector dot product.
    /// \param[in] lhs      Unbatched row or column vector.
    /// \param[in] rhs      Unbatched row or column vector.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<typename Lhs, typename Rhs>
    requires (nt::are_varray_v<Lhs, Rhs> and nt::are_almost_same_value_type_v<Lhs, Rhs>)
    [[nodiscard]] auto dot(const Lhs& lhs, const Rhs& rhs) {
        check(not lhs.shape().is_batched() and lhs.shape().ndim() <= 2 and
              not rhs.shape().is_batched() and rhs.shape().ndim() <= 2,
              "The inputs should be unbatched row or column vectors, but got lhs:shape={} and rhs:shape={}",
              lhs.shape(), rhs.shape());

        using value_t = nt::mutable_value_type_t<Lhs>;
        value_t output{};

        if constexpr (nt::is_real_or_complex_v<value_t>) {
            if (rhs.device().is_cpu()) {
                using op_t = ReduceAccurateSum<value_t>;
                using pair_t = op_t::pair_type;
                reduce_ewise(wrap(lhs.flat(), rhs.flat()), pair_t{}, output, op_t{});
                return output;
            }
        }
        reduce_ewise(wrap(lhs.flat(), rhs.flat()), value_t{}, output, ReduceSum{});
        return output;
    }

    /// Computes the (batched) vector-vector dot product.
    /// \param[in] lhs              (Batched) row or column vector.
    /// \param[in] rhs              (Batched) row or column vector.
    /// \param[out] output          Output contiguous vector with the dot products. One element per batch.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<typename Lhs, typename Rhs, typename Output>
    requires (nt::are_varray_v<Lhs, Rhs> and
              nt::are_almost_same_value_type_v<Lhs, Rhs> and
              nt::is_varray_of_mutable_v<Output>)
    void dot(const Lhs& lhs, const Rhs& rhs, const Output& output) {
        check(ni::is_vector(lhs.shape(), true) and
              ni::is_vector(rhs.shape(), true) and
              lhs.shape()[1] == 1 and rhs.shape()[1] == 1,
              "The input should be (batched) column or row vectors, but got lhs:shape={} and rhs:shape={}",
              lhs.shape(), rhs.shape());

        const i64 lhs_n_elements = lhs.shape()[2] * lhs.shape()[3];
        const i64 rhs_n_elements = rhs.shape()[2] * rhs.shape()[3];
        check(lhs_n_elements == rhs_n_elements,
              "The input vectors don't have the same number of elements. Got lhs:{} and rhs:{}",
              lhs_n_elements, rhs_n_elements);

        check(ni::is_contiguous_vector(output),
              "The output should be a contiguous vector, but got output:shape={} and output:stride={}",
              output.shape(), output.strides());

        const i64 batch = output.elements();
        const auto full_shape = Shape4<i64>{batch, 1, 1, lhs_n_elements};
        auto inputs = wrap(broadcast(lhs.flat(), full_shape),
                           broadcast(rhs.flat(), full_shape));

        using value_t = nt::mutable_value_type_t<Lhs>;
        if constexpr (nt::is_real_or_complex_v<value_t>) {
            if (rhs.device().is_cpu()) {
                using op_t = ReduceAccurateSum<value_t>;
                using pair_t = op_t::pair_type;
                reduce_axes_ewise(std::move(inputs), pair_t{}, output, op_t{});
            }
        }
        reduce_axes_ewise(std::move(inputs), value_t{}, output, ReduceSum{});
    }

    /// Computes a scalar-matrix-matrix product, with general matrices and adds the result to a scalar-matrix product.
    /// \details This function computes a matrix-matrix product, but it also accepts vectors.
    ///          As such, it can computes a matrix-vector product, a vector-matrix product and
    ///          the vector-vector outer-product or dot product. The operation is defined as:
    ///          \p output = \p alpha * \p lhs * \p rhs + \p beta * \p output.
    /// \tparam Lhs, Rhs, Output    VArray of f32, f64, c32, or c64.
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
    ///       (before transposition) should be contiguous and the second-most dimension cannot be broadcasted.
    template<typename Lhs, typename Rhs, typename Output, typename Value>
    requires (nt::are_varray_v<Lhs, Rhs, Output> and
              nt::are_almost_same_value_type_v<Lhs, Rhs, Output> and
              nt::is_any_v<nt::value_type_t<Output>, f32, f64, c32, c64> and
              nt::is_almost_same_v<Value, nt::value_type_t<Output>>)
    void matmul(
            const Lhs& lhs,
            const Rhs& rhs,
            const Output& output,
            Value alpha,
            Value beta,
            bool lhs_transpose = false,
            bool rhs_transpose = false
    ) {
        check(not lhs.is_empty() and not rhs.is_empty() and not output.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(lhs, output) and not ni::are_overlapped(rhs, output),
              "Input and output arrays should not overlap");

        const Device device = output.device();
        check(device == lhs.device() and device == rhs.device(),
              "The input and output arrays must be on the same device, but got lhs:{}, rhs:{} and output:{}",
              lhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::matmul(lhs.get(), lhs.strides(), lhs.shape(),
                                 rhs.get(), rhs.strides(), rhs.shape(),
                                 alpha, beta, lhs_transpose, rhs_transpose,
                                 output.get(), output.strides(), output.shape(),
                                 threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::matmul(lhs.get(), lhs.strides(), lhs.shape(),
                              rhs.get(), rhs.strides(), rhs.shape(),
                              alpha, beta, lhs_transpose, rhs_transpose,
                              output.get(), output.strides(), output.shape(),
                              cuda_stream);
            cuda_stream.enqueue_attach(lhs, rhs, output);
            #else
            panic("No GPU backend detected");
            #endif
        }
    }

    /// Computes a matrix-matrix product, with general matrices.
    /// \details This function computes a matrix-matrix product, but it also accepts vectors.
    ///          As such, it can computes a matrix-vector product, a vector-matrix product and
    ///          the vector-vector outer-product or dot product.
    /// \tparam Lhs, Rhs, Output    VArray of f32, f64, c32, or c64.
    /// \param[in] lhs              Dense {B,1,M,K} matrix.
    /// \param[in] rhs              Dense {B,1,K,N} matrix.
    /// \param[out] output          Dense {B,1,M,N} matrix.
    /// \note The memory layout is restricted: \p lhs, \p rhs and \p output should not overlap. All matrices should
    ///       either be row-major or column-major. The innermost dimension of the matrices should be contiguous and
    ///       the second-most dimension cannot be broadcast.
    template<typename Lhs, typename Rhs, typename Output>
    requires (nt::are_varray_v<Lhs, Rhs, Output> and
              nt::are_almost_same_value_type_v<Lhs, Rhs, Output> and
              nt::is_any_v<nt::value_type_t<Output>, f32, f64, c32, c64>)
    void matmul(const Lhs& lhs, const Rhs& rhs, const Output& output) {
        using value_t = nt::value_type_t<Output>;
        matmul(lhs, rhs, output, value_t{1}, value_t{0});
    }
}
#endif
