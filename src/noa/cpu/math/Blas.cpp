#include <cblas.h>
#include "noa/cpu/math/Blas.h"

namespace {
    using namespace ::noa;

    // Extract size and strides from a column or row vector.
    std::pair<dim_t, dim_t> extractDimFromVector_(dim2_t strides, dim2_t shape) {
        const bool is_column = shape[1] == 1;
        NOA_ASSERT(shape.ndim() == 1);
        const dim_t n = shape[1 - is_column];
        const dim_t s = strides[1 - is_column];
        return {n, s};
    }

    template<typename T>
    T cblasDot_(dim_t n, const T* lhs, dim_t lhs_strides, const T* rhs, dim_t rhs_strides) {
        if (n >= 2048) {
            if constexpr (std::is_same_v<float, T>) {
                return static_cast<float>(cblas_dsdot(safe_cast<blasint>(n),
                                                      lhs, safe_cast<blasint>(lhs_strides),
                                                      rhs, safe_cast<blasint>(rhs_strides)));
            } else if constexpr (std::is_same_v<double, T>) {
                return cblas_ddot(safe_cast<blasint>(n),
                                  lhs, safe_cast<blasint>(lhs_strides),
                                  rhs, safe_cast<blasint>(rhs_strides));
            }
        }

        T sum{0};
        for (dim_t i = 0; i < n; ++i)
            sum += lhs[i * lhs_strides] * rhs[i * rhs_strides];
        return sum;
    }

    template<typename T>
    void cblasGEMM_(CBLAS_ORDER order, bool lhs_transpose, bool rhs_transpose,
                    Int3<blasint> mnk, Int3<blasint> lb, T alpha, T beta,
                    const T* lhs, const T* rhs, T* output) {
        const auto lhs_op = lhs_transpose ? CblasTrans : CblasNoTrans;
        const auto rhs_op = rhs_transpose ? CblasTrans : CblasNoTrans;

        if constexpr (std::is_same_v<float, T>) {
            cblas_sgemm(order, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], alpha,
                        lhs, lb[0], rhs, lb[1], beta, output, lb[2]);
        } else if constexpr (std::is_same_v<double, T>) {
            cblas_dgemm(order, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                        alpha, lhs, lb[0], rhs, lb[1], beta, output, lb[2]);
        } else if constexpr (std::is_same_v<cfloat_t, T>) {
            cblas_cgemm3m(order, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                          &alpha, lhs, lb[0], rhs, lb[1], &beta, output, lb[2]);
        } else if constexpr (std::is_same_v<cdouble_t, T>) {
            cblas_zgemm3m(order, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], &alpha,
                          lhs, lb[0], rhs, lb[1], &beta, output, lb[2]);
        }
    }
}

namespace noa::cpu::math {
    template<typename T, typename>
    T dot(const std::shared_ptr<T[]>& lhs, dim4_t lhs_strides, dim4_t lhs_shape,
          const std::shared_ptr<T[]>& rhs, dim4_t rhs_strides, dim4_t rhs_shape,
          Stream& stream) {

        // Get vector shape:
        NOA_ASSERT(lhs_shape.ndim() == 1 && rhs_shape.ndim() == 1);
        auto[lhs_n, lhs_s] = extractDimFromVector_(dim2_t(lhs_strides.get(2)), dim2_t(lhs_shape.get(2)));
        auto[rhs_n, rhs_s] = extractDimFromVector_(dim2_t(rhs_strides.get(2)), dim2_t(rhs_shape.get(2)));
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        stream.synchronize();
        return cblasDot_(lhs_n, lhs.get(), lhs_s, rhs.get(), rhs_s);
    }

    template<typename T, typename>
    void dot(const std::shared_ptr<T[]>& lhs, dim4_t lhs_strides, dim4_t lhs_shape,
             const std::shared_ptr<T[]>& rhs, dim4_t rhs_strides, dim4_t rhs_shape,
             const std::shared_ptr<T[]>& output, Stream& stream) {
        NOA_ASSERT(lhs_shape[0] == rhs_shape[0] && lhs_shape[1] == 1 && rhs_shape[1] == 1);
        const dim_t batches = lhs_shape[0];

        // Get vector shape: lhs should be a row vector, rhs can be a column or row vector
        dim_t lhs_n, lhs_s, rhs_n, rhs_s;
        std::tie(lhs_n, lhs_s) = extractDimFromVector_(dim2_t(lhs_strides.get(2)), dim2_t(lhs_shape.get(2)));
        std::tie(rhs_n, rhs_s) = extractDimFromVector_(dim2_t(rhs_strides.get(2)), dim2_t(rhs_shape.get(2)));
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        const dim_t lhs_batch_strides = lhs_strides[0];
        const dim_t rhs_batch_strides = rhs_strides[0];
        stream.enqueue([=]() {
            T* ptr = output.get();
            for (dim_t batch = 0; batch < batches; ++batch) {
                ptr[batch] = cblasDot_(lhs_n,
                                       lhs.get() + lhs_batch_strides * batch, lhs_s,
                                       rhs.get() + rhs_batch_strides * batch, rhs_s);
            }
        });
    }

    #define INSTANTIATE_DOT_(T)                                                     \
    template T dot<T, void>(const std::shared_ptr<T[]>&, dim4_t, dim4_t,            \
                            const std::shared_ptr<T[]>&, dim4_t, dim4_t, Stream&);  \
    template void dot<T, void>(const std::shared_ptr<T[]>&, dim4_t, dim4_t,         \
                               const std::shared_ptr<T[]>&, dim4_t, dim4_t,         \
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
    template<typename T, typename>
    void matmul(const std::shared_ptr<T[]>& lhs, dim4_t lhs_strides, dim4_t lhs_shape,
                const std::shared_ptr<T[]>& rhs, dim4_t rhs_strides, dim4_t rhs_shape,
                T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
                const std::shared_ptr<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                Stream& stream) {

        // Get the shape: MxK @ KxN = MxN
        using blas3_t = Int3<blasint>;
        const auto m = lhs_shape[2 + lhs_transpose];
        const auto n = rhs_shape[3 - rhs_transpose];
        const auto k = lhs_shape[3 - lhs_transpose];
        const auto mnk = safe_cast<blas3_t>(dim3_t{m, n, k});
        NOA_ASSERT(lhs_shape[1] == 1 && rhs_shape[1] == 1 && output_shape[1] == 1); // 2D matrices
        NOA_ASSERT(m == output_shape[2] && n == output_shape[3]); // output fits the expected shape
        NOA_ASSERT(k == rhs_shape[2 + rhs_transpose]); // left and right matrices have compatible shape

        // dot is faster than gemm:
        if (m == 1 && n == 1 && alpha == T{1} && beta == T{0} &&
            indexing::isContiguous(output_strides, output_shape)[0]) {
            return dot(lhs, lhs_strides, lhs_shape, rhs, rhs_strides, rhs_shape, output, stream);
        }

        // Select an order:
        const bool is_col = indexing::isColMajor(output_strides);
        const CBLAS_ORDER order = is_col ? CblasColMajor : CblasRowMajor;
        NOA_ASSERT(is_col == indexing::isColMajor(lhs_strides) &&
                   is_col == indexing::isColMajor(rhs_strides)); // same order for everyone

        // Get the pitch:
        const dim3_t ld{lhs_strides[2 + is_col], rhs_strides[2 + is_col], output_strides[2 + is_col]};
        NOA_ASSERT(all(ld >= dim3_t{lhs_shape[3 - is_col], rhs_shape[3 - is_col], output_shape[3 - is_col]}));
        NOA_ASSERT(lhs_strides[3 - is_col] == 1 && rhs_strides[3 - is_col] == 1 && output_strides[3 - is_col] == 1);

        stream.enqueue([=](){
            // TODO Intel MKL has gemm_batch...
            for (dim_t batch = 0; batch < output_shape[0]; ++batch) {
                cblasGEMM_(order, lhs_transpose, rhs_transpose, mnk, safe_cast<blas3_t>(ld), alpha, beta,
                           lhs.get() + lhs_strides[0] * batch,
                           rhs.get() + rhs_strides[0] * batch,
                           output.get() + output_strides[0] * batch);
            }
        });
    }

    #define INSTANTIATE_BLAS_(T)                                                                                            \
    template void matmul<T, void>(const std::shared_ptr<T[]>&, dim4_t, dim4_t, const std::shared_ptr<T[]>&, dim4_t, dim4_t, \
                                  T, T, bool, bool, const std::shared_ptr<T[]>&, dim4_t, dim4_t, Stream&)

    INSTANTIATE_BLAS_(float);
    INSTANTIATE_BLAS_(double);
    INSTANTIATE_BLAS_(cfloat_t);
    INSTANTIATE_BLAS_(cdouble_t);
}
