#include <cublas_v2.h>

#include "noa/gpu/cuda/math/Blas.h"
#include "noa/gpu/cuda/util/ReduceBinary.cuh"

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
            [[maybe_unused]] cublasStatus_t err = cublasDestroy_v2(handle);
            NOA_ASSERT(err == CUBLAS_STATUS_SUCCESS);
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

    // Extract size and stride from a column or row vector.
    std::pair<size_t, size_t> extractDimFromVector_(size2_t stride, size2_t shape) {
        const bool is_column = shape.ndim() == 2;
        NOA_ASSERT(shape[is_column] == 1); // this is a vector
        const size_t n = shape[1 - is_column];
        const size_t s = stride[1 - is_column];
        return {n, s};
    }

    template<typename T>
    void cublasGEMM_(cublasHandle_t handle, math::BlasTranspose lhs_transpose, math::BlasTranspose rhs_transpose,
                     int3_t mnk, int3_t labc, long3_t sabc, int batches, T alpha, T beta,
                     const T* lhs, const T* rhs, T* output) {
        auto to_cblas_transpose = [](math::BlasTranspose transpose) {
            switch (transpose) {
                case noa::math::BLAS_TRANSPOSE_NONE:
                    return CUBLAS_OP_N;
                case noa::math::BLAS_TRANSPOSE:
                    return CUBLAS_OP_T;
                case noa::math::BLAS_TRANSPOSE_CONJ:
                    return CUBLAS_OP_C;
                default:
                    NOA_THROW_FUNC("math::matmul", "Invalid math::BlasTranspose ({})", transpose);
            }
        };

        const auto lhs_op = to_cblas_transpose(lhs_transpose);
        const auto rhs_op = to_cblas_transpose(rhs_transpose);

        // Switch to row-major:
        // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
        // https://peterwittek.com/cublas-matrix-c-style.html
        if constexpr (std::is_same_v<float, T>) {
            cublasSgemmStridedBatched(handle, lhs_op, rhs_op, mnk[1], mnk[0], mnk[2], &alpha,
                                      rhs, labc[1], sabc[1],
                                      lhs, labc[0], sabc[0], &beta,
                                      output, labc[2], sabc[2], batches);
        } else if constexpr (std::is_same_v<double, T>) {
            cublasDgemmStridedBatched(handle, lhs_op, rhs_op, mnk[1], mnk[0], mnk[2], &alpha,
                                      rhs, labc[1], sabc[1],
                                      lhs, labc[0], sabc[0], &beta,
                                      output, labc[2], sabc[2], batches);
        } else if constexpr (std::is_same_v<cfloat_t, T>) {
            cublasCgemmStridedBatched(handle, lhs_op, rhs_op, mnk[1], mnk[0], mnk[2],
                                      reinterpret_cast<const cuComplex*>(&alpha),
                                      reinterpret_cast<const cuComplex*>(rhs), labc[1], sabc[1],
                                      reinterpret_cast<const cuComplex*>(lhs), labc[0], sabc[0],
                                      reinterpret_cast<const cuComplex*>(&beta),
                                      reinterpret_cast<cuComplex*>(output), labc[2], sabc[2], batches);
        } else if constexpr (std::is_same_v<cdouble_t, T>) {
            cublasZgemmStridedBatched(handle, lhs_op, rhs_op, mnk[1], mnk[0], mnk[2],
                                      reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                      reinterpret_cast<const cuDoubleComplex*>(rhs), labc[1], sabc[1],
                                      reinterpret_cast<const cuDoubleComplex*>(lhs), labc[0], sabc[0],
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
    template<typename T>
    T dot(const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
          const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
          Stream& stream) {

        // Get vector shape:
        NOA_ASSERT(lhs_shape.ndim() <= 2 && rhs_shape.ndim() <= 2);
        auto[lhs_n, lhs_s] = extractDimFromVector_(size2_t{lhs_stride.get(2)}, size2_t{lhs_shape.get(2)});
        auto[rhs_n, rhs_s] = extractDimFromVector_(size2_t{rhs_stride.get(2)}, size2_t{rhs_shape.get(2)});
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        T output{};
        using real_t = traits::value_type_t<T>;
        if constexpr (traits::is_float_v<T>) {
            cublasHandle_t handle = cublasCachedHandle_(stream.device().id())->handle;
            CUBLAS_THROW_IF_(cublasSetStream_v2(handle, stream.id()));
            CUBLAS_THROW_IF_(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST));

            cublasStatus_t err;
            if constexpr (std::is_same_v<T, float>) {
                err = cublasSdot_v2(handle, lhs_n, lhs.get(), lhs_s, rhs.get(), rhs_s, &output);
            } else if constexpr (std::is_same_v<T, double>) {
                err = cublasDdot_v2(handle, lhs_n, lhs.get(), lhs_s, rhs.get(), rhs_s, &output);
            } else if constexpr (std::is_same_v<T, cfloat_t>) {
                err = cublasCdotu_v2(handle, static_cast<int>(lhs_n),
                                     reinterpret_cast<const cuComplex*>(lhs.get()), static_cast<int>(lhs_s),
                                     reinterpret_cast<const cuComplex*>(rhs.get()), static_cast<int>(rhs_s),
                                     reinterpret_cast<cuComplex*>(&output));
            } else if constexpr (std::is_same_v<T, cdouble_t>) {
                err = cublasZdotu_v2(handle, static_cast<int>(lhs_n),
                                     reinterpret_cast<const cuDoubleComplex*>(lhs.get()), static_cast<int>(lhs_s),
                                     reinterpret_cast<const cuDoubleComplex*>(rhs.get()), static_cast<int>(rhs_s),
                                     reinterpret_cast<cuDoubleComplex*>(&output));
            }
            // These functions block the host thread until completion, so no need to sync the stream.
            CUBLAS_THROW_IF_(err);
        } else {
            T* null{};
            cuda::util::reduce<true, false>( // sum(lhs * rhs)
                    "dot", lhs.get(), uint4_t{lhs_s}, rhs.get(), uint4_t{rhs_s}, uint4_t{1, 1, 1, lhs_n},
                    noa::math::copy_t{}, noa::math::copy_t{}, noa::math::multiply_t{}, noa::math::plus_t{}, T{0},
                    &output, 1, noa::math::copy_t{}, null, 1, noa::math::copy_t{}, stream);
            stream.synchronize();
        }
        return output;
    }

    template<typename T>
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

        // While we could use cublas dot function, cuda::util::reduce is batched, works with any type and
        // is as performant as cublas (although cublas is still expected to be more robust to different platforms).
        T* null{};
        cuda::util::reduce<false, false>("dot", // sum(lhs * rhs)
                lhs.get(), uint4_t{lhs_stride[0], lhs_stride[0], lhs_stride[0], lhs_s},
                rhs.get(), uint4_t{rhs_stride[0], rhs_stride[0], rhs_stride[0], rhs_s}, uint4_t{batches, 1, 1, lhs_n},
                noa::math::copy_t{}, noa::math::copy_t{}, noa::math::multiply_t{}, noa::math::plus_t{}, T{0},
                output.get(), 1, noa::math::copy_t{}, null, 1, noa::math::copy_t{}, stream);
        stream.attach(lhs, rhs, output);
    }

    #define INSTANTIATE_DOT_(T)                                                     \
    template T dot<T>(const std::shared_ptr<T[]>&, size4_t, size4_t,                \
                      const std::shared_ptr<T[]>&, size4_t, size4_t, Stream&);      \
    template void dot<T>(const std::shared_ptr<T[]>&, size4_t, size4_t,             \
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

    template<typename T>
    void matmul(BlasTranspose lhs_transpose, BlasTranspose rhs_transpose, T alpha,
                const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
                const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
                T beta, const std::shared_ptr<T[]>& output, size4_t output_stride, size4_t output_shape,
                Stream& stream) {
        // Get the shape: MxK @ KxN = MxN
        auto should_be_transposed = [](math::BlasTranspose transpose) {
            return transpose == noa::math::BLAS_TRANSPOSE_NONE ? false : true;
        };
        const bool should_lhs_be_transposed = should_be_transposed(lhs_transpose);
        const bool should_rhs_be_transposed = should_be_transposed(rhs_transpose);
        const auto m = lhs_shape[2 + should_lhs_be_transposed];
        const auto n = rhs_shape[3 - should_rhs_be_transposed];
        const auto k = lhs_shape[3 - should_lhs_be_transposed];
        const int3_t mnk{m, n, k};
        NOA_ASSERT(lhs_shape[1] == 1 && rhs_shape[1] == 1 && output_shape[1] == 1); // 2D matrices
        NOA_ASSERT(m == output_shape[2] && n == output_shape[3]); // output fits the expected shape
        NOA_ASSERT(k == rhs_shape[2 + should_rhs_be_transposed]); // left and right matrices have compatible shape

        // In the CPU, in the case of dot products, OpenBlas GEMM is slower than its DOT function, so we check for
        // this condition and redirect to dot if necessary. Here, cublas GEMM is about as fast as the dot function,
        // so let it do a matrix-matrix product even if it is a dot product.

        // Get the pitch:
        const int3_t labc{lhs_stride[2], rhs_stride[2], output_stride[2]};
        const long3_t sabc{lhs_stride[0], rhs_stride[0], output_stride[0]};
        NOA_ASSERT(all(labc >= int3_t{lhs_shape[3], rhs_shape[3], output_shape[3]})); // 2nd dim can be padded, no broadcast
        NOA_ASSERT(lhs_stride[3] == 1 && rhs_stride[3] == 1 && output_stride[3] == 1); // 1st dim is contiguous

        cublasHandle_t handle = cublasCachedHandle_(stream.device().id())->handle;
        CUBLAS_THROW_IF_(cublasSetStream_v2(handle, stream.id()));
        CUBLAS_THROW_IF_(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST));

        cublasGEMM_(handle, lhs_transpose, rhs_transpose, mnk, labc, sabc, output_shape[0], alpha, beta,
                    lhs.get(), rhs.get(), output.get());
        stream.attach(lhs, rhs, output);
    }

    #define INSTANTIATE_BLAS_(T)\
    template void matmul<T>(BlasTranspose,  BlasTranspose, T, const std::shared_ptr<T[]>&, size4_t, size4_t,\
                            const std::shared_ptr<T[]>&, size4_t, size4_t, T, const std::shared_ptr<T[]>&, size4_t, size4_t, Stream&)

    INSTANTIATE_BLAS_(float);
    INSTANTIATE_BLAS_(double);
    INSTANTIATE_BLAS_(cfloat_t);
    INSTANTIATE_BLAS_(cdouble_t);
}
