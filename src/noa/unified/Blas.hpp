#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Reduce.hpp"
#include "noa/cpu/Blas.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Blas.hpp"
#endif

#include "Reduce.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Reduce.hpp"
#include "noa/unified/Indexing.hpp"

namespace noa {
    /// Returns the vector-vector dot product.
    /// \param[in] lhs  Single row or column vector.
    /// \param[in] rhs  Single row or column vector.
    /// \param options  Sum options.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<typename Lhs, typename Rhs>
    requires nt::readable_varray_decay_of_almost_same_type<Lhs, Rhs>
    [[nodiscard]] auto dot(Lhs&& lhs, Rhs&& rhs, const SumOptions& options = {}) {
        check(not lhs.shape().is_batched() and lhs.shape().ndim() <= 2 and
              not rhs.shape().is_batched() and rhs.shape().ndim() <= 2,
              "The inputs should be single row or column vectors, but got lhs:shape={} and rhs:shape={}",
              lhs.shape(), rhs.shape());

        auto inputs = wrap(std::forward<Lhs>(lhs).flat(), std::forward<Rhs>(rhs).flat());
        using value_t = nt::mutable_value_type_t<Lhs>;
        using reduce_t = std::conditional_t<nt::real<value_t>, f64,
                         std::conditional_t<nt::complex<value_t>, c64,
                         value_t>>;

        value_t output;
        if constexpr (nt::real_or_complex<value_t>) {
            if (options.accurate) {
                reduce_ewise(std::move(inputs), Vec<reduce_t, 2>{}, output, ReduceSumKahan{});
                return output;
            }
        }
        reduce_ewise(std::move(inputs), reduce_t{}, output, ReduceSum{});
        return output;
    }

    /// Computes the (batched) vector-vector dot product.
    /// \param[in] lhs      (Batched) row or column vector.
    /// \param[in] rhs      (Batched) row or column vector.
    /// \param[out] output  Output contiguous vector with the dot products. One element per batch.
    /// \param options      Sum options.
    /// \note The input vector \p lhs and \p rhs are automatically reshaped in a row and column vector, respectively.
    template<nt::readable_varray_decay Lhs,
             nt::readable_varray_decay Rhs,
             nt::writable_varray_decay Output>
    requires nt::varray_decay_of_almost_same_type<Lhs, Rhs, Output>
    void dot(Lhs&& lhs, Rhs&& rhs, Output&& output, const SumOptions& options = {}) {
        check(ni::is_vector(lhs.shape(), true) and
              ni::is_vector(rhs.shape(), true) and
              lhs.shape()[1] == 1 and rhs.shape()[1] == 1,
              "The input should be (batched) column or row vectors, but got lhs:shape={} and rhs:shape={}",
              lhs.shape(), rhs.shape());

        const auto lhs_n_elements = lhs.shape()[2] * lhs.shape()[3];
        const auto rhs_n_elements = rhs.shape()[2] * rhs.shape()[3];
        check(lhs_n_elements == rhs_n_elements,
              "The input vectors don't have the same number of elements. Got lhs:size={} and rhs:size={}",
              lhs_n_elements, rhs_n_elements);

        check(ni::is_contiguous_vector(output),
              "The output should be a contiguous vector, but got output:shape={} and output:stride={}",
              output.shape(), output.strides());

        const auto batch = output.n_elements();
        const auto full_shape = Shape4{batch, 1, 1, lhs_n_elements};
        auto inputs = wrap(std::forward<Lhs>(lhs).reshape(full_shape),
                           std::forward<Rhs>(rhs).reshape(full_shape));

        using value_t = nt::mutable_value_type_t<Lhs>;
        using reduce_t = std::conditional_t<nt::real<value_t>, f64,
                         std::conditional_t<nt::complex<value_t>, c64,
                         value_t>>;

        if constexpr (nt::real_or_complex<value_t>) {
            if (options.accurate) {
                return reduce_axes_ewise(
                    std::move(inputs), Vec<reduce_t, 2>{},
                    std::forward<Output>(output), ReduceSumKahan{}
                );
            }
        }
        reduce_axes_ewise(
            std::move(inputs), reduce_t{},
            std::forward<Output>(output), ReduceSum{}
        );
    }

    template<nt::any_of<f32, f64, c32, c64> T>
    struct MatmulOptions {
        /// Scalar for the scalar-matrix-matrix product.
        T alpha{1};

        /// Scalar for the scalar-matrix product. If zero, the output doesn't need to be set.
        T beta{};

        /// Whether the lhs should be transposed before the operation. In this case, the {B,1,K,M} matrix is expected.
        bool lhs_transpose{};

        /// Whether the rhs should be transposed before the operation. In this case, the {B,1,N,K} matrix is expected.
        bool rhs_transpose{};
    };

    /// Computes a scalar-matrix-matrix product, with general matrices and adds the result to a scalar-matrix product.
    /// \details This function computes a matrix-matrix product, but it also accepts vectors.
    ///          As such, it can compute a matrix-vector product, a vector-matrix product and
    ///          the vector-vector outer-product or dot product. The operation is defined as:
    ///          \p output = \p options.alpha * \p lhs * \p rhs + \p options.beta * \p output.
    ///
    /// \param[in] lhs              Dense {B,1,M,K} matrix.
    /// \param[in] rhs              Dense {B,1,K,N} matrix.
    /// \param[out] output          Dense {B,1,M,N} matrix.
    /// \param options              Matmul options.
    ///
    /// \note The memory layout is restricted: \p lhs and \p rhs should not overlap with \p output. All matrices should
    ///       either be row-major or column-major (before transposition). The innermost dimension of the matrices
    ///       (before transposition) should be contiguous, and the second-most dimension cannot be broadcast.
    template<nt::readable_varray_decay Lhs,
             nt::readable_varray_decay Rhs,
             nt::writable_varray_decay Output>
    requires (nt::varray_decay_of_any<Output, f32, f64, c32, c64> and
              nt::varray_decay_of_almost_same_type<Lhs, Rhs, Output>)
    void matmul(
            Lhs&& lhs,
            Rhs&& rhs,
            Output&& output,
            MatmulOptions<nt::mutable_value_type_t<Output>> options = {}
    ) {
        check(not lhs.is_empty() and not rhs.is_empty() and not output.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(lhs, output) and not ni::are_overlapped(rhs, output),
              "Input and output arrays should not overlap");

        const Device device = output.device();
        check(device == lhs.device() and device == rhs.device(),
              "The input and output arrays must be on the same device, "
              "but got lhs:device={}, rhs:device={} and output:device={}",
              lhs.device(), rhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto n_threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=,
                l = std::forward<Lhs>(lhs),
                r = std::forward<Rhs>(rhs),
                o = std::forward<Output>(output)
            ] {
                noa::cpu::matmul(
                    l.get(), l.strides(), l.shape(),
                    r.get(), r.strides(), r.shape(),
                    options.alpha, options.beta, options.lhs_transpose, options.rhs_transpose,
                    o.get(), o.strides(), o.shape(),
                    n_threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::matmul(
                lhs.get(), lhs.strides(), lhs.shape(),
                rhs.get(), rhs.strides(), rhs.shape(),
                options.alpha, options.beta, options.lhs_transpose, options.rhs_transpose,
                output.get(), output.strides(), output.shape(),
                cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs), std::forward<Output>(output));
            #else
            panic_no_gpu_backend();
            #endif
        }
    }
}
