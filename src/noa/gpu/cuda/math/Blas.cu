#include <cublas_v2.h>

#include "noa/gpu/cuda/math/Blas.h"
#include "noa/gpu/cuda/utils/ReduceBinary.cuh"

// Add cublasStatus_t support for NOA_THROW_IF
namespace {
    using namespace ::noa;

    inline void throwIfCuBlas_(cublasStatus_t result, const char* file, const char* function, int line) {
        if (result != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            std::throw_with_nested(Exception(file, function, line, cublasGetStatusString(result)));
    }
    #define CUBLAS_THROW_IF_(result) throwIfCuBlas_(result, __FILE__, __FUNCTION__, __LINE__)

    class CuBlasHandle {
    public:
        cublasHandle_t handle{};
        CuBlasHandle() {
            CUBLAS_THROW_IF_(cublasCreate_v2(&handle));
        }
        ~CuBlasHandle() {
            cublasStatus_t err = cublasDestroy_v2(handle);
            NOA_ASSERT(err == CUBLAS_STATUS_SUCCESS);
            (void) err;
        }
    };

    std::unique_ptr<CuBlasHandle>& cublasCachedHandle_(int device) {
        constexpr size_t MAX_DEVICES = 16;
        thread_local std::unique_ptr<CuBlasHandle> g_cache[MAX_DEVICES];

        auto& cache = g_cache[device];
        if (!cache)
            cache = std::make_unique<CuBlasHandle>();
        return cache;
    }

    // Extract size and strides from a column or row vector.
    std::pair<dim_t, dim_t> extractDimFromVector_(dim2_t strides, dim2_t shape) {
        const bool is_column = shape[1] == 1;
        NOA_ASSERT(shape.ndim() == 1);
        const dim_t n = shape[1 - is_column];
        const dim_t s = strides[1 - is_column];
        return {n, s};
    }

    template<typename T>
    void cublasGEMM_(bool is_col, cublasHandle_t handle, bool lhs_transpose, bool rhs_transpose,
                     int3_t mnk, int3_t labc, long3_t sabc, int batches, T alpha, T beta,
                     const T* lhs, const T* rhs, T* output) {
        auto lhs_op = lhs_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
        auto rhs_op = rhs_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

        // cublas thinks everything is column-major (-_-), so if we are row-major we need to compute B.T @ A.T = C.T
        // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
        // https://peterwittek.com/cublas-matrix-c-style.html
        if (!is_col) {
            std::swap(lhs_op, rhs_op);
            std::swap(mnk[0], mnk[1]);
            std::swap(lhs, rhs);
            std::swap(labc[0], labc[1]);
            std::swap(sabc[0], sabc[1]);
        }

        if constexpr (std::is_same_v<float, T>) {
            cublasSgemmStridedBatched(handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], &alpha,
                                      lhs, labc[0], sabc[0],
                                      rhs, labc[1], sabc[1], &beta,
                                      output, labc[2], sabc[2], batches);
        } else if constexpr (std::is_same_v<double, T>) {
            cublasDgemmStridedBatched(handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], &alpha,
                                      lhs, labc[0], sabc[0],
                                      rhs, labc[1], sabc[1], &beta,
                                      output, labc[2], sabc[2], batches);
        } else if constexpr (std::is_same_v<cfloat_t, T>) {
            cublasCgemmStridedBatched(handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                                      reinterpret_cast<const cuComplex*>(&alpha),
                                      reinterpret_cast<const cuComplex*>(lhs), labc[0], sabc[0],
                                      reinterpret_cast<const cuComplex*>(rhs), labc[1], sabc[1],
                                      reinterpret_cast<const cuComplex*>(&beta),
                                      reinterpret_cast<cuComplex*>(output), labc[2], sabc[2], batches);
        } else if constexpr (std::is_same_v<cdouble_t, T>) {
            cublasZgemmStridedBatched(handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                                      reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                      reinterpret_cast<const cuDoubleComplex*>(lhs), labc[0], sabc[0],
                                      reinterpret_cast<const cuDoubleComplex*>(rhs), labc[1], sabc[1],
                                      reinterpret_cast<const cuDoubleComplex*>(&beta),
                                      reinterpret_cast<cuDoubleComplex*>(output), labc[2], sabc[2], batches);
        }
    }
}

namespace noa::cuda::math::details {
    void cublasClearCache(int device) {
        std::unique_ptr<CuBlasHandle>& cached_handle = cublasCachedHandle_(device);
        cached_handle = nullptr;
    }
}

namespace noa::cuda::math {
    template<typename T, typename>
    T dot(const std::shared_ptr<T[]>& lhs, dim4_t lhs_strides, dim4_t lhs_shape,
          const std::shared_ptr<T[]>& rhs, dim4_t rhs_strides, dim4_t rhs_shape,
          Stream& stream) {
        NOA_ASSERT(all(lhs_shape > 0) && all(rhs_shape > 0));

        // Get vector shape:
        NOA_ASSERT(lhs_shape.ndim() == 1 && rhs_shape.ndim() == 1);
        auto[lhs_n, lhs_s] = extractDimFromVector_(dim2_t(lhs_strides.get(2)), dim2_t(lhs_shape.get(2)));
        auto[rhs_n, rhs_s] = extractDimFromVector_(dim2_t(rhs_strides.get(2)), dim2_t(rhs_shape.get(2)));
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        T output{};
        using real_t = traits::value_type_t<T>;
        if constexpr (traits::is_float_v<T>) {
            NOA_ASSERT_DEVICE_PTR(lhs.get(), stream.device());
            NOA_ASSERT_DEVICE_PTR(rhs.get(), stream.device());

            cublasHandle_t handle = cublasCachedHandle_(stream.device().id())->handle;
            CUBLAS_THROW_IF_(cublasSetStream_v2(handle, stream.id()));
            CUBLAS_THROW_IF_(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST));

            cublasStatus_t err;
            const auto n = safe_cast<int>(lhs_n);
            const auto incx = safe_cast<int>(lhs_s);
            const auto incy = safe_cast<int>(rhs_s);
            if constexpr (std::is_same_v<T, float>) {
                err = cublasSdot_v2(handle, n, lhs.get(), incx, rhs.get(), incy, &output);
            } else if constexpr (std::is_same_v<T, double>) {
                err = cublasDdot_v2(handle, n, lhs.get(), incx, rhs.get(), incy, &output);
            } else if constexpr (std::is_same_v<T, cfloat_t>) {
                err = cublasCdotu_v2(handle, n,
                                     reinterpret_cast<const cuComplex*>(lhs.get()), incx,
                                     reinterpret_cast<const cuComplex*>(rhs.get()), incy,
                                     reinterpret_cast<cuComplex*>(&output));
            } else if constexpr (std::is_same_v<T, cdouble_t>) {
                err = cublasZdotu_v2(handle, n,
                                     reinterpret_cast<const cuDoubleComplex*>(lhs.get()), incx,
                                     reinterpret_cast<const cuDoubleComplex*>(rhs.get()), incy,
                                     reinterpret_cast<cuDoubleComplex*>(&output));
            }
            // These functions block the host thread until completion, so no need to sync the stream.
            CUBLAS_THROW_IF_(err);
        } else {
            T* null{};
            cuda::utils::reduce<false>( // sum(lhs * rhs)
                    "dot", lhs.get(), dim4_t(lhs_s), rhs.get(), dim4_t(rhs_s), dim4_t{1, 1, 1, lhs_n},
                    noa::math::copy_t{}, noa::math::copy_t{}, noa::math::multiply_t{}, noa::math::plus_t{}, T{0},
                    &output, 1, noa::math::copy_t{}, null, 1, noa::math::copy_t{}, true, stream);
            stream.synchronize();
        }
        return output;
    }

    template<typename T, typename>
    void dot(const std::shared_ptr<T[]>& lhs, dim4_t lhs_strides, dim4_t lhs_shape,
             const std::shared_ptr<T[]>& rhs, dim4_t rhs_strides, dim4_t rhs_shape,
             const std::shared_ptr<T[]>& output, Stream& stream) {
        NOA_ASSERT(all(lhs_shape > 0) && all(rhs_shape > 0));
        NOA_ASSERT(lhs_shape[0] == rhs_shape[0] && lhs_shape[1] == 1 && rhs_shape[1] == 1);
        const dim_t batches = lhs_shape[0];

        // Get vector shape: lhs should be a row vector, rhs can be a column or row vector
        dim_t lhs_n, lhs_s, rhs_n, rhs_s;
        std::tie(lhs_n, lhs_s) = extractDimFromVector_(dim2_t(lhs_strides.get(2)), dim2_t(lhs_shape.get(2)));
        std::tie(rhs_n, rhs_s) = extractDimFromVector_(dim2_t(rhs_strides.get(2)), dim2_t(rhs_shape.get(2)));
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        // While we could use cublas dot function, cuda::utils::reduce is batched, works with any type and
        // is as performant as cublas (although cublas is still expected to be more robust to different platforms).
        T* null{};
        cuda::utils::reduce<false>("dot", // sum(lhs * rhs)
                lhs.get(), dim4_t{lhs_strides[0], lhs_strides[0], lhs_strides[0], lhs_s},
                rhs.get(), dim4_t{rhs_strides[0], rhs_strides[0], rhs_strides[0], rhs_s}, dim4_t{batches, 1, 1, lhs_n},
                noa::math::copy_t{}, noa::math::copy_t{}, noa::math::multiply_t{}, noa::math::plus_t{}, T{0},
                output.get(), 1, noa::math::copy_t{}, null, 1, noa::math::copy_t{}, false, stream);
        stream.attach(lhs, rhs, output);
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

    template<typename T, typename>
    void matmul(const std::shared_ptr<T[]>& lhs, dim4_t lhs_strides, dim4_t lhs_shape,
                const std::shared_ptr<T[]>& rhs, dim4_t rhs_strides, dim4_t rhs_shape,
                T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
                const std::shared_ptr<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                Stream& stream) {
        NOA_ASSERT(all(lhs_shape > 0) && all(rhs_shape > 0));
        NOA_ASSERT_DEVICE_PTR(lhs.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(rhs.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        // Get the shape: MxK @ KxN = MxN
        const auto m = lhs_shape[2 + lhs_transpose];
        const auto n = rhs_shape[3 - rhs_transpose];
        const auto k = lhs_shape[3 - lhs_transpose];
        const auto mnk = safe_cast<int3_t>(dim3_t{m, n, k});
        NOA_ASSERT(lhs_shape[1] == 1 && rhs_shape[1] == 1 && output_shape[1] == 1); // 2D matrices
        NOA_ASSERT(m == output_shape[2] && n == output_shape[3]); // output fits the expected shape
        NOA_ASSERT(k == rhs_shape[2 + rhs_transpose]); // left and right matrices have compatible shape

        // In the CPU, in the case of dot products, OpenBlas GEMM is slower than its DOT function, so we check for
        // this condition and redirect to dot if necessary. Here, cublas GEMM is about as fast as the dot function,
        // so let it do a matrix-matrix product even if it is a dot product.

        // Select an order:
        const bool is_col = indexing::isColMajor(output_strides);
        NOA_ASSERT(is_col == indexing::isColMajor(lhs_strides) &&
                   is_col == indexing::isColMajor(rhs_strides)); // same order for everyone

        // Get the pitch:
        const dim3_t labc{lhs_strides[2 + is_col], rhs_strides[2 + is_col], output_strides[2 + is_col]};
        const dim3_t sabc{lhs_strides[0], rhs_strides[0], output_strides[0]};
        NOA_ASSERT(all(labc >= dim3_t{lhs_shape[3 - is_col], rhs_shape[3 - is_col], output_shape[3 - is_col]}));
        NOA_ASSERT(lhs_strides[3 - is_col] == 1 && rhs_strides[3 - is_col] == 1 && output_strides[3 - is_col] == 1);

        cublasHandle_t handle = cublasCachedHandle_(stream.device().id())->handle;
        CUBLAS_THROW_IF_(cublasSetStream_v2(handle, stream.id()));
        CUBLAS_THROW_IF_(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST));

        cublasGEMM_(is_col, handle, lhs_transpose, rhs_transpose,
                    mnk, safe_cast<int3_t>(labc), safe_cast<long3_t>(sabc), output_shape[0], alpha, beta,
                    lhs.get(), rhs.get(), output.get());
        stream.attach(lhs, rhs, output);
    }

    #define INSTANTIATE_BLAS_(T)                                                                                            \
    template void matmul<T,void>(const std::shared_ptr<T[]>&, dim4_t, dim4_t, const std::shared_ptr<T[]>&, dim4_t, dim4_t,  \
                                 T, T, bool, bool, const std::shared_ptr<T[]>&, dim4_t, dim4_t, Stream&)

    INSTANTIATE_BLAS_(float);
    INSTANTIATE_BLAS_(double);
    INSTANTIATE_BLAS_(cfloat_t);
    INSTANTIATE_BLAS_(cdouble_t);
}
