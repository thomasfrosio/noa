#include <cblas.h>
#include "noa/cpu/math/Blas.hpp"
#include "noa/cpu/utils/ReduceBinary.hpp"
#include "noa/algorithms/math/AccurateSum.hpp"

namespace {
    using namespace ::noa;

    // Extract size and strides from a column or row vector.
    std::pair<i64, i64> extract_vector_dim_(Strides2<i64> strides, Shape2<i64> shape) {
        NOA_ASSERT(shape.ndim() == 1);
        const bool is_column = shape[1] == 1;
        const i64 n = shape[1 - is_column];
        const i64 s = strides[1 - is_column];
        return {n, s};
    }

    template<typename T>
    T cblas_dot_(i64 n, const T* lhs, i64 lhs_strides, const T* rhs, i64 rhs_strides, i64 threads = 1) {
        // Trust that cblas uses a relatively accurate sum algorithm.
        // It is likely doing the sum in double-precision, but that's it.
        if constexpr (std::is_same_v<f32, T>) {
            return static_cast<f32>(cblas_dsdot(
                    safe_cast<blasint>(n),
                    lhs, safe_cast<blasint>(lhs_strides),
                    rhs, safe_cast<blasint>(rhs_strides)));
        } else if constexpr (std::is_same_v<f64, T>) {
            return cblas_ddot(
                    safe_cast<blasint>(n),
                    lhs, safe_cast<blasint>(lhs_strides),
                    rhs, safe_cast<blasint>(rhs_strides));
        } else if constexpr (noa::traits::is_complex_v<T>) {
            const auto shape = Shape4<i64>{1, 1, 1, n};
            const auto lhs_strides_4d = shape.strides() * lhs_strides;
            const auto rhs_strides_4d = shape.strides() * rhs_strides;
            c64 error{0};
            c64 sum{};
            noa::cpu::utils::reduce_binary(
                    lhs, lhs_strides_4d, rhs, rhs_strides_4d, shape,
                    &sum, Strides1<i64>{1}, c64{0},
                    [](T lhs_value, T rhs_value) { return static_cast<c64>(lhs_value * rhs_value); },
                    noa::algorithm::math::AccuratePlusComplex{&error}, {}, threads);
            return static_cast<T>(sum + error);
        } else {
            const auto shape = Shape4<i64>{1, 1, 1, n};
            const auto lhs_strides_4d = shape.strides() * lhs_strides;
            const auto rhs_strides_4d = shape.strides() * rhs_strides;
            T sum{};
            noa::cpu::utils::reduce_binary(
                    lhs, lhs_strides_4d, rhs, rhs_strides_4d, shape,
                    &sum, Strides1<i64>{1}, T{0},
                    [](T lhs_value, T rhs_value) { return lhs_value * rhs_value; },
                    noa::plus_t{}, {}, threads);
            return sum;
        }
    }

    template<typename T>
    void cblas_gemm_(CBLAS_ORDER order, bool lhs_transpose, bool rhs_transpose,
                     Vec3<blasint> mnk, Vec3<blasint> lb, T alpha, T beta,
                     const T* lhs, const T* rhs, T* output) {
        const auto lhs_op = lhs_transpose ? CblasTrans : CblasNoTrans;
        const auto rhs_op = rhs_transpose ? CblasTrans : CblasNoTrans;

        if constexpr (std::is_same_v<f32, T>) {
            cblas_sgemm(order, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], alpha,
                        lhs, lb[0], rhs, lb[1], beta, output, lb[2]);
        } else if constexpr (std::is_same_v<f64, T>) {
            cblas_dgemm(order, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                        alpha, lhs, lb[0], rhs, lb[1], beta, output, lb[2]);
        } else if constexpr (std::is_same_v<c32, T>) {
            cblas_cgemm3m(order, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                          &alpha, lhs, lb[0], rhs, lb[1], &beta, output, lb[2]);
        } else if constexpr (std::is_same_v<c64, T>) {
            cblas_zgemm3m(order, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], &alpha,
                          lhs, lb[0], rhs, lb[1], &beta, output, lb[2]);
        }
    }
}

namespace noa::cpu::math {
    template<typename T, typename>
    T dot(const T* lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
          const T* rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
          i64 threads) {
        NOA_ASSERT(lhs && rhs && all(lhs_shape > 0) && all(rhs_shape > 0));

        // Get vector shape:
        NOA_ASSERT(lhs_shape.ndim() == 1 && rhs_shape.ndim() == 1 &&
                   !lhs_shape.is_batched() && !rhs_shape.is_batched());
        auto [lhs_n, lhs_s] = extract_vector_dim_(lhs_strides.filter(2, 3), lhs_shape.filter(2, 3));
        auto [rhs_n, rhs_s] = extract_vector_dim_(rhs_strides.filter(2, 3), rhs_shape.filter(2, 3));
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        return cblas_dot_(lhs_n, lhs, lhs_s, rhs, rhs_s, threads);
    }

    template<typename T, typename>
    void dot(const T* lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
             const T* rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
             T* output, i64 threads) {
        NOA_ASSERT(lhs && rhs && output && all(lhs_shape > 0) && all(rhs_shape > 0));
        NOA_ASSERT(lhs_shape[0] == rhs_shape[0] && lhs_shape[1] == 1 && rhs_shape[1] == 1);
        const auto batches = lhs_shape[0];

        // Get vector shape: lhs should be a row vector, rhs can be a column or row vector
        i64 lhs_n{}, lhs_s{}, rhs_n{}, rhs_s{};
        std::tie(lhs_n, lhs_s) = extract_vector_dim_(lhs_strides.filter(2, 3), lhs_shape.filter(2, 3));
        std::tie(rhs_n, rhs_s) = extract_vector_dim_(rhs_strides.filter(2, 3), rhs_shape.filter(2, 3));
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        const auto lhs_batch_strides = lhs_strides[0];
        const auto rhs_batch_strides = rhs_strides[0];
        // If there's a lot of batches, reduce_binary might be faster.
        // Also, Intel's MKL might have a batched version.
        for (i64 batch = 0; batch < batches; ++batch) {
            output[batch] = cblas_dot_(
                    lhs_n,
                    lhs + lhs_batch_strides * batch, lhs_s,
                    rhs + rhs_batch_strides * batch, rhs_s,
                    threads);
        }
    }

    #define INSTANTIATE_DOT_(T)                                     \
    template T dot<T, void>(                                        \
        const T*, const Strides4<i64>&, const Shape4<i64>&,         \
        const T*, const Strides4<i64>&, const Shape4<i64>&, i64);   \
    template void dot<T, void>(                                     \
        const T*, const Strides4<i64>&, const Shape4<i64>&,         \
        const T*, const Strides4<i64>&, const Shape4<i64>&,         \
        T*, i64)

    INSTANTIATE_DOT_(i32);
    INSTANTIATE_DOT_(u32);
    INSTANTIATE_DOT_(i64);
    INSTANTIATE_DOT_(u64);
    INSTANTIATE_DOT_(f32);
    INSTANTIATE_DOT_(f64);
    INSTANTIATE_DOT_(c32);
    INSTANTIATE_DOT_(c64);
}

namespace noa::cpu::math {
    template<typename T, typename>
    void matmul(const T* lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
                const T* rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
                T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
                T* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                i64 threads) {
        NOA_ASSERT(lhs && rhs && output && all(lhs_shape > 0) && all(rhs_shape > 0));

        const auto [mnk, secondmost_strides, are_column_major] = noa::indexing::extract_matmul_layout(
                lhs_strides, lhs_shape, rhs_strides, rhs_shape, output_strides, output_shape,
                lhs_transpose, rhs_transpose);

        // dot is faster than gemm with OpenBLAS:
        if (mnk[0] == 1 && mnk[1] == 1 && alpha == T{1} && beta == T{0} &&
            noa::indexing::is_contiguous(output_strides, output_shape)[0]) {
            return dot(lhs, lhs_strides, lhs_shape, rhs, rhs_strides, rhs_shape, output, threads);
        }

        // TODO Intel MKL has gemm_batch...
        for (i64 batch = 0; batch < output_shape[0]; ++batch) {
            cblas_gemm_(are_column_major ? CblasColMajor : CblasRowMajor,
                        lhs_transpose, rhs_transpose,
                        mnk.vec().as_safe<blasint>(),
                        secondmost_strides.vec().as_safe<blasint>(),
                        alpha, beta,
                        lhs + lhs_strides[0] * batch,
                        rhs + rhs_strides[0] * batch,
                        output + output_strides[0] * batch);
        }
    }

    #define INSTANTIATE_GEMM_(T)                            \
    template void matmul<T, void>(                          \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        T, T, bool, bool,                                   \
        T*, const Strides4<i64>&, const Shape4<i64>&,       \
        i64)

    INSTANTIATE_GEMM_(f32);
    INSTANTIATE_GEMM_(f64);
    INSTANTIATE_GEMM_(c32);
    INSTANTIATE_GEMM_(c64);
}
