#include <cblas.h>
#include "noa/cpu/math/Blas.h"

namespace {
    using namespace ::noa;

    // Extract size and stride from a column or row vector.
    std::pair<size_t, size_t> extractDimFromVector_(size2_t stride, size2_t shape) {
        const bool is_column = shape.ndim() == 2;
        NOA_ASSERT(shape[is_column] == 1); // this is a vector
        const size_t n = shape[1 - is_column];
        const size_t s = stride[1 - is_column];
        return {n, s};
    }

    template<typename T>
    T dot_(size_t n, const T* lhs, size_t lhs_stride, const T* rhs, size_t rhs_stride) {
        if (n >= 2048) {
            if constexpr (std::is_same_v<float, T>) {
                return static_cast<float>(cblas_dsdot(static_cast<blasint>(n),
                                                      lhs, static_cast<blasint>(lhs_stride),
                                                      rhs, static_cast<blasint>(rhs_stride)));
            } else if constexpr (std::is_same_v<double, T>) {
                return cblas_ddot(static_cast<blasint>(n),
                                  lhs, static_cast<blasint>(lhs_stride),
                                  rhs, static_cast<blasint>(rhs_stride));
            }
        }

        T sum{0};
        for (size_t i = 0; i < n; ++i)
            sum += lhs[i * lhs_stride] * rhs[i * rhs_stride];
        return sum;
    }

    template<typename T>
    void cblasGEMM_(math::BlasTranspose lhs_transpose, math::BlasTranspose rhs_transpose,
                     Int3<blasint> mnk, Int3<blasint> lb, T alpha, T beta,
                     const T* lhs, const T* rhs, T* output) {
        auto to_cblas_transpose = [](math::BlasTranspose transpose) {
            switch (transpose) {
                case noa::math::BLAS_TRANSPOSE_NONE:
                    return CblasNoTrans;
                case noa::math::BLAS_TRANSPOSE:
                    return CblasTrans;
                case noa::math::BLAS_TRANSPOSE_CONJ:
                    return CblasConjTrans;
                default:
                    NOA_THROW_FUNC("math::matmul", "Invalid math::BlasTranspose ({})", transpose);
            }
        };

        const auto lhs_op = to_cblas_transpose(lhs_transpose);
        const auto rhs_op = to_cblas_transpose(rhs_transpose);

        if constexpr (std::is_same_v<float, T>) {
            cblas_sgemm(CblasRowMajor, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], alpha,
                        lhs, lb[0], rhs, lb[1], beta, output, lb[2]);
        } else if constexpr (std::is_same_v<double, T>) {
            cblas_dgemm(CblasRowMajor, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                        alpha, lhs, lb[0], rhs, lb[1], beta, output, lb[2]);
        } else if constexpr (std::is_same_v<cfloat_t, T>) {
            cblas_cgemm3m(CblasRowMajor, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                          &alpha, lhs, lb[0], rhs, lb[1], &beta, output, lb[2]);
        } else if constexpr (std::is_same_v<cdouble_t, T>) {
            cblas_zgemm3m(CblasRowMajor, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], &alpha,
                          lhs, lb[0], rhs, lb[1], &beta, output, lb[2]);
        }
    }
}

namespace noa::cpu::math {
    template<typename T, typename>
    T dot(const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
          const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
          Stream& stream) {

        // Get vector shape:
        NOA_ASSERT(lhs_shape.ndim() <= 2 && rhs_shape.ndim() <= 2);
        auto[lhs_n, lhs_s] = extractDimFromVector_(size2_t{lhs_stride.get(2)}, size2_t{lhs_shape.get(2)});
        auto[rhs_n, rhs_s] = extractDimFromVector_(size2_t{rhs_stride.get(2)}, size2_t{rhs_shape.get(2)});
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        stream.synchronize();
        return dot_(lhs_n, lhs.get(), lhs_s, rhs.get(), rhs_s);
    }

    template<typename T, typename>
    void dot(const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
             const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
             const std::shared_ptr<T[]>& output, Stream& stream) {
        NOA_ASSERT(lhs_shape[0] == rhs_shape[0] && lhs_shape[1] == 1 && rhs_shape[1] == 1);
        const size_t batches = lhs_shape[0];

        // Get vector shape: lhs should be a row vector, rhs can be a column or row vector
        size_t lhs_n, lhs_s, rhs_n, rhs_s;
        std::tie(lhs_n, lhs_s) = extractDimFromVector_(size2_t{lhs_stride.get(2)}, size2_t{lhs_shape.get(2)});
        std::tie(rhs_n, rhs_s) = extractDimFromVector_(size2_t{rhs_stride.get(2)}, size2_t{rhs_shape.get(2)});
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        const size_t lhs_batch_stride = lhs_stride[0];
        const size_t rhs_batch_stride = rhs_stride[0];
        stream.enqueue([=]() {
            T* ptr = output.get();
            for (size_t batch = 0; batch < batches; ++batch) {
                ptr[batch] = dot_(lhs_n,
                                  lhs.get() + lhs_batch_stride * batch, lhs_s,
                                  rhs.get() + rhs_batch_stride * batch, rhs_s);
            }
        });
    }

    #define INSTANTIATE_DOT_(T)                                                           \
    template T dot<T, void>(const std::shared_ptr<T[]>&, size4_t, size4_t,                \
                            const std::shared_ptr<T[]>&, size4_t, size4_t, Stream&); \
    template void dot<T, void>(const std::shared_ptr<T[]>&, size4_t, size4_t,             \
                               const std::shared_ptr<T[]>&, size4_t, size4_t,             \
                               const std::shared_ptr<T[]>&, Stream&)

    INSTANTIATE_DOT_(int32_t);
    INSTANTIATE_DOT_(uint32_t);
    INSTANTIATE_DOT_(int64_t);
    INSTANTIATE_DOT_(uint64_t);
    INSTANTIATE_DOT_(float);
    INSTANTIATE_DOT_(double);
    INSTANTIATE_DOT_(cfloat_t);
    INSTANTIATE_DOT_(cdouble_t);
}

namespace noa::cpu::math {
    using BlasTranspose = noa::math::BlasTranspose;

    template<typename T, typename>
    void matmul(BlasTranspose lhs_transpose, BlasTranspose rhs_transpose, T alpha,
                const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
                const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
                T beta, const std::shared_ptr<T[]>& output, size4_t output_stride, size4_t output_shape,
                Stream& stream) {

        // Get the shape: MxK @ KxN = MxN
        using blas3_t = Int3<blasint>;
        auto should_be_transposed = [](math::BlasTranspose transpose) {
            return transpose != noa::math::BLAS_TRANSPOSE_NONE;
        };
        const bool should_lhs_be_transposed = should_be_transposed(lhs_transpose);
        const bool should_rhs_be_transposed = should_be_transposed(rhs_transpose);
        const auto m = lhs_shape[2 + should_lhs_be_transposed];
        const auto n = rhs_shape[3 - should_rhs_be_transposed];
        const auto k = lhs_shape[3 - should_lhs_be_transposed];
        const blas3_t mnk{m, n, k};
        NOA_ASSERT(lhs_shape[1] == 1 && rhs_shape[1] == 1 && output_shape[1] == 1); // 2D matrices
        NOA_ASSERT(m == output_shape[2] && n == output_shape[3]); // output fits the expected shape
        NOA_ASSERT(k == rhs_shape[2 + should_rhs_be_transposed]); // left and right matrices have compatible shape

        // dot is faster than gemm:
        if (m == 1 && n == 1 && !should_lhs_be_transposed && !should_rhs_be_transposed &&
            indexing::isContiguous(output_stride, output_shape)[0] && alpha == T{1} && beta == T{0}) {
            return dot(lhs, lhs_stride, lhs_shape, rhs, rhs_stride, rhs_shape, output, stream);
        }

        // Get the pitch:
        const blas3_t lb{lhs_stride[2], rhs_stride[2], output_stride[2]};
        NOA_ASSERT(all(lb >= blas3_t{lhs_shape[3], rhs_shape[3], output_shape[3]})); // 2nd dim can be padded, no broadcast
        NOA_ASSERT(lhs_stride[3] == 1 && rhs_stride[3] == 1 && output_stride[3] == 1); // 1st dim is contiguous

        stream.enqueue([=](){
            // TODO Intel MKL has gemm_batch...
            for (size_t batch = 0; batch < output_shape[0]; ++batch) {
                cblasGEMM_(lhs_transpose, rhs_transpose, mnk, lb, alpha, beta,
                           lhs.get() + lhs_stride[0] * batch,
                           rhs.get() + rhs_stride[0] * batch,
                           output.get() + output_stride[0] * batch);
            }
        });
    }

    #define INSTANTIATE_BLAS_(T)                                                                                    \
    template void matmul<T, void>(BlasTranspose,  BlasTranspose, T, const std::shared_ptr<T[]>&, size4_t, size4_t,  \
                                  const std::shared_ptr<T[]>&, size4_t, size4_t, T, const std::shared_ptr<T[]>&, size4_t, size4_t, Stream&)

    INSTANTIATE_BLAS_(float);
    INSTANTIATE_BLAS_(double);
    INSTANTIATE_BLAS_(cfloat_t);
    INSTANTIATE_BLAS_(cdouble_t);
}
