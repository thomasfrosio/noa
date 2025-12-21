#include <cublas_v2.h>

#include "noa/runtime/core/indexing/Layout.hpp"
#include "noa/cuda/Blas.hpp"
#include "noa/cuda/Error.hpp"

namespace noa::cuda {
    void cublas_clear_cache(Device device);
}

namespace {
    using namespace noa::types;
    using noa::check;

    /// Throws an Exception if the result is not cudaSuccess.
    constexpr void check(cublasStatus_t result, const std::source_location& location = std::source_location::current()) {
        if (result == cublasStatus_t::CUBLAS_STATUS_SUCCESS) {
            /*do nothing*/
        } else {
            noa::panic_at_location(location, "cublas failed with error: {}", cublasGetStatusString(result));
        }
    }

    class CuBlasHandle {
    public:
        cublasHandle_t handle{};
        CuBlasHandle() {
            check(cublasCreate_v2(&handle));
        }
        ~CuBlasHandle() {
            const cublasStatus_t err = cublasDestroy_v2(handle);
            NOA_ASSERT(err == CUBLAS_STATUS_SUCCESS);
            (void) err;
        }
    };

    std::unique_ptr<CuBlasHandle>& cublas_cache_handle_(noa::cuda::Device device) {
        constexpr usize MAX_DEVICES = 16;
        thread_local std::unique_ptr<CuBlasHandle> g_cache[MAX_DEVICES];

        auto& cache = g_cache[device.id()];
        if (not cache) {
            cache = std::make_unique<CuBlasHandle>();
            Device::add_reset_callback(noa::cuda::cublas_clear_cache);
        }
        return cache;
    }

    template<typename T>
    void cublas_gemm_(
        bool is_column_major, bool lhs_transpose, bool rhs_transpose,
        Shape<int, 3> mnk, Vec<i32, 3> labc, Vec<isize, 3> sabc, int batches, T alpha, T beta,
        const T* lhs, const T* rhs, T* output, noa::cuda::Stream& stream
    ) {
        // OpenBlas GEMM is slower than its DOT function, so we check for this condition and
        // redirect to dot if necessary. Here, cublas GEMM is about as fast as the dot function,
        // so let it do a matrix-matrix product even if it is a dot product.
        cublasHandle_t handle = cublas_cache_handle_(stream.device())->handle;
        check(cublasSetStream_v2(handle, stream.id()));
        check(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST));

        auto lhs_op = lhs_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
        auto rhs_op = rhs_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

        // cublas thinks everything is column-major, so if we are row-major we need to compute B.T @ A.T = C.T
        // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
        // https://peterwittek.com/cublas-matrix-c-style.html
        if (not is_column_major) {
            std::swap(lhs_op, rhs_op);
            std::swap(mnk[0], mnk[1]);
            std::swap(lhs, rhs);
            std::swap(labc[0], labc[1]);
            std::swap(sabc[0], sabc[1]);
        }

        if constexpr (std::is_same_v<f32, T>) {
            check(cublasSgemmStridedBatched(
                handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], &alpha,
                lhs, labc[0], sabc[0],
                rhs, labc[1], sabc[1], &beta,
                output, labc[2], sabc[2], batches));
        } else if constexpr (std::is_same_v<f64, T>) {
            check(cublasDgemmStridedBatched(
                handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], &alpha,
                lhs, labc[0], sabc[0],
                rhs, labc[1], sabc[1], &beta,
                output, labc[2], sabc[2], batches));
        } else if constexpr (std::is_same_v<c32, T>) {
            check(cublasCgemmStridedBatched(
                handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                reinterpret_cast<const cuComplex*>(&alpha),
                reinterpret_cast<const cuComplex*>(lhs), labc[0], sabc[0],
                reinterpret_cast<const cuComplex*>(rhs), labc[1], sabc[1],
                reinterpret_cast<const cuComplex*>(&beta),
                reinterpret_cast<cuComplex*>(output), labc[2], sabc[2], batches));
        } else if constexpr (std::is_same_v<c64, T>) {
            check(cublasZgemmStridedBatched(
                handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                reinterpret_cast<const cuDoubleComplex*>(&alpha),
                reinterpret_cast<const cuDoubleComplex*>(lhs), labc[0], sabc[0],
                reinterpret_cast<const cuDoubleComplex*>(rhs), labc[1], sabc[1],
                reinterpret_cast<const cuDoubleComplex*>(&beta),
                reinterpret_cast<cuDoubleComplex*>(output), labc[2], sabc[2], batches));
        }
    }
}

namespace noa::cuda {
    void cublas_clear_cache(Device device) {
        std::unique_ptr<CuBlasHandle>& cached_handle = cublas_cache_handle_(device);
        cached_handle = nullptr;
    }

    template<typename T>
    void matmul(
        const T* lhs, const Strides4& lhs_strides, const Shape4& lhs_shape,
        const T* rhs, const Strides4& rhs_strides, const Shape4& rhs_shape,
        T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
        T* output, const Strides4& output_strides, const Shape4& output_shape,
        Stream& stream
    ) {
        auto [mnk, secondmost_strides, are_column_major] = ni::extract_matmul_layout(
            lhs_strides, lhs_shape, rhs_strides, rhs_shape, output_strides, output_shape,
            lhs_transpose, rhs_transpose);

        const auto labc = secondmost_strides.vec.as_safe<i32>();
        const auto sabc = Vec<isize, 3>{lhs_strides[0], rhs_strides[0], output_strides[0]};

        cublas_gemm_(are_column_major, lhs_transpose, rhs_transpose,
                     mnk.as_safe<i32>(), labc, sabc, static_cast<i32>(output_shape[0]), alpha, beta,
                     lhs, rhs, output, stream);
    }

    #define NOA_INSTANTIATE_MATMUL_(T)                          \
    template void matmul<T>(                                    \
        const T*, const Strides4&, const Shape4&, \
        const T*, const Strides4&, const Shape4&, \
        T, T, bool, bool,                                       \
        T*, const Strides4&, const Shape4&,       \
        Stream&)

    NOA_INSTANTIATE_MATMUL_(f32);
    NOA_INSTANTIATE_MATMUL_(f64);
    NOA_INSTANTIATE_MATMUL_(c32);
    NOA_INSTANTIATE_MATMUL_(c64);
}
