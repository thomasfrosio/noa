#pragma once

#include "noa/runtime/core/Reduce.hpp"

#include "noa/runtime/cpu/Blas.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/runtime/cuda/Blas.hpp"
#endif

#include "noa/runtime/Array.hpp"
#include "noa/runtime/Reduce.hpp"

namespace noa {
    /// Returns the vector-vector dot product.
    /// \param[in] lhs, rhs:
    ///     Input arrays.
    ///     Ignoring empty dimensions (which are squeezed out):
    ///     lhs:(n) x rhs:(n) -> output:1
    /// \param options  Sum options.
    template<typename Lhs, typename Rhs>
    requires nt::readable_array_decay_of_almost_same_type<Lhs, Rhs>
    [[nodiscard]] auto dot(Lhs&& lhs, Rhs&& rhs, const SumOptions& options = {}) {
        const auto lhs_n_axes = noa::sum(lhs.shape().cmp_gt(1));
        const auto rhs_n_axes = noa::sum(rhs.shape().cmp_gt(1));
        check(lhs_n_axes == 1 and rhs_n_axes == 1,
              "The inputs should be vectors, but got lhs:shape={} and rhs:shape={}",
              lhs.shape(), rhs.shape());

        auto inputs = noa::wrap(std::forward<Lhs>(lhs).template as_nd<1>(), std::forward<Rhs>(rhs).template as_nd<1>());
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
    /// \param[in] lhs, rhs, output:
    ///     Input and output arrays.
    ///     Ignoring empty dimensions (which are squeezed out):
    ///     lhs:(bn) x rhs:(bn) -> output:(b)
    ///     lhs: (n) x rhs: (n) -> output:(1)
    /// \param options:
    ///     Sum options.
    template<nt::readable_array_decay Lhs,
             nt::readable_array_decay Rhs,
             nt::writable_array_decay Output>
    requires nt::array_decay_of_almost_same_type<Lhs, Rhs, Output>
    void dot(Lhs&& lhs, Rhs&& rhs, Output&& output, const SumOptions& options = {}) {
        const auto get_bn = [&]<usize N>(Shape<isize, N> const& shape) {
            isize b{-1}, n{-1};
            for (isize i{N - 1}; i >= 0; --i) {
                if (shape[i] > 1) {
                    if (n == -1)
                        n = shape[i];
                    else if (b == - 1)
                        b = shape[i];
                    else
                        panic("The inputs should be compatible batched vectors, but got lhs:shape={} and rhs:shape={}",
                              lhs.shape(), rhs.shape());
                }
            }
            if (b == -1)
                b = 1;
            return Vec{b, n};
        };
        auto b = output.n_elements();
        auto lhs_bn = get_bn(lhs.shape());
        auto rhs_bn = get_bn(rhs.shape());
        check(lhs_bn == rhs_bn,
              "The inputs should be compatible batched vectors, but got lhs:shape={} and rhs:shape={}",
              lhs.shape(), rhs.shape());
        check(noa::sum(output.shape().cmp_gt(1)) <= 1,
              "The output should be a vector, but got output:shape={}", output.shape());
        check(b == lhs_bn[0],
              "The inputs and output batch dimension don't match: input:b={}, output:b={}",
              b, lhs_bn[0]);

        const auto shape_2d = Shape2{b, lhs_bn[1]};
        auto inputs = noa::wrap(
            std::forward<Lhs>(lhs).reshape(shape_2d),
            std::forward<Rhs>(rhs).reshape(shape_2d));
        auto output_2d = std::forward<Output>(output).reshape(Shape2{b, 1});

        using value_t = nt::mutable_value_type_t<Lhs>;
        using reduce_t = std::conditional_t<nt::real<value_t>, f64,
                         std::conditional_t<nt::complex<value_t>, c64,
                         value_t>>;

        if constexpr (nt::real_or_complex<value_t>)
            if (options.accurate)
                return reduce_axes_ewise(std::move(inputs), Vec<reduce_t, 2>{}, std::move(output_2d), ReduceSumKahan{});
        reduce_axes_ewise(std::move(inputs), reduce_t{}, std::move(output_2d), ReduceSum{});
    }

    template<nt::any_of<f32, f64, c32, c64> T>
    struct MatmulOptions {
        /// Scalar for the scalar-matrix-matrix product.
        T alpha{1};

        /// Scalar for the scalar-matrix product. If zero, the output doesn't need to be set.
        T beta{};

        /// Whether the lhs should be transposed before the operation.
        /// In this case, the {(b...,)k,m} matrix is expected.
        bool lhs_transpose{};

        /// Whether the rhs should be transposed before the operation.
        /// In this case, the {(b...,)n,k} matrix is expected.
        bool rhs_transpose{};
    };

    /// Computes a scalar-matrix-matrix product, with general matrices and adds the result to a scalar-matrix product.
    /// \details
    ///     This function computes a matrix-matrix product, but it also accepts vectors. As such, it can compute a
    ///     matrix-vector product, a vector-matrix product and the vector-vector outer-product or dot product.
    ///     The operation is defined as: output = options.alpha * lhs * rhs + options.beta * output.
    ///
    /// \param[in] lhs:
    ///     Dense {(b...,)m,k} matrix (or {(b...,)k,m} is options.lhs_transpose=true).
    ///     The batch dimensions b... (if any) should be collapsable.
    /// \param[in] rhs:
    ///     Dense {(b...,)k,n} matrix (or {(b...,)n,k} is options.rhs_transpose=true).
    ///     The batch dimensions b... (if any) should be collapsable.
    /// \param[out] output:
    ///     Dense {(b...,)m,n} matrix.
    ///     The batch dimensions b... (if any) should be collapsable.
    /// \param options:
    ///     Matmul options.
    ///
    /// \note
    ///     The memory layout is restricted: lhs and rhs should not overlap with output. All matrices should
    ///     either be row-major or column-major (before transposition). The innermost dimension of the matrices
    ///     (before transposition) should be contiguous, and the second-most dimension cannot be broadcast.
    template<nt::readable_array_decay Lhs,
             nt::readable_array_decay Rhs,
             nt::writable_array_decay Output>
    requires (nt::array_decay_of_any<Output, f32, f64, c32, c64> and
              nt::array_decay_of_almost_same_type<Lhs, Rhs, Output> and
              nt::array_size_v<Lhs> >= 2 and nt::array_size_v<Rhs> >= 2 and nt::array_size_v<Output> >= 2)
    void matmul(
        Lhs&& lhs,
        Rhs&& rhs,
        Output&& output,
        MatmulOptions<nt::mutable_value_type_t<Output>> options = {}
    ) {
        check(not lhs.is_empty() and not rhs.is_empty() and not output.is_empty(), "Empty array detected");
        check(not noa::are_overlapped(lhs, output) and not noa::are_overlapped(rhs, output),
              "Input and output arrays should not overlap");
        const Device device = output.device();
        check(device == lhs.device() and device == rhs.device(),
              "The input and output arrays must be on the same device, "
              "but got lhs:device={}, rhs:device={} and output:device={}",
              lhs.device(), rhs.device(), device);

        auto lhs_span = lhs.span().template as_nd<3>();
        auto rhs_span = rhs.span().template as_nd<3>();
        auto output_span = output.span().template as_nd<3>();
        check(output_span.shape()[0] == lhs_span.shape()[0] and
              output_span.shape()[0] == rhs_span.shape()[0],
              "The batch dimensions don't match, got shape lhs:shape={}, rhs:shape={} and output:shape={}",
              lhs.shape(), rhs.shape(), output.shape()
        );

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto n_threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=,
                handles = nd::extract_shared_handle_from_arrays(
                    noa::forward_as_tuple(
                        std::forward<Lhs>(lhs), std::forward<Rhs>(rhs), std::forward<Output>(output)))
            ] {
                noa::cpu::matmul(
                    lhs_span.get(), lhs_span.strides(), lhs_span.shape(),
                    rhs_span.get(), rhs_span.strides(), rhs_span.shape(),
                    options.alpha, options.beta, options.lhs_transpose, options.rhs_transpose,
                    output_span.get(), output_span.strides(), output_span.shape(),
                    n_threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::matmul(
                lhs_span.get(), lhs_span.strides(), lhs_span.shape(),
                rhs_span.get(), rhs_span.strides(), rhs_span.shape(),
                options.alpha, options.beta, options.lhs_transpose, options.rhs_transpose,
                output_span.get(), output_span.strides(), output_span.shape(),
                cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs), std::forward<Output>(output));
            #else
            panic_no_gpu_backend();
            #endif
        }
    }
}
