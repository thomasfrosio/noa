#include <cublas_v2.h>

#include "noa/gpu/cuda/math/Blas.h"
#include "noa/gpu/cuda/utils/ReduceBinary.cuh"

// TODO Add cublasStatus_t support for NOA_THROW_IF?
namespace {
    using namespace ::noa;

    inline void throw_if_cublas(cublasStatus_t result, const char* file, const char* function, int line) {
        if (result != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            std::throw_with_nested(Exception(file, function, line, cublasGetStatusString(result)));
    }
    #define CUBLAS_THROW_IF_(result) throw_if_cublas(result, __FILE__, __FUNCTION__, __LINE__)

    class CuBlasHandle {
    public:
        cublasHandle_t handle{};
        CuBlasHandle() {
            CUBLAS_THROW_IF_(cublasCreate_v2(&handle));
        }
        ~CuBlasHandle() {
            const cublasStatus_t err = cublasDestroy_v2(handle);
            NOA_ASSERT(err == CUBLAS_STATUS_SUCCESS);
            (void) err;
        }
    };

    std::unique_ptr<CuBlasHandle>& cublas_cache_handle_(int device) {
        constexpr size_t MAX_DEVICES = 16;
        thread_local std::unique_ptr<CuBlasHandle> g_cache[MAX_DEVICES];

        auto& cache = g_cache[device];
        if (!cache)
            cache = std::make_unique<CuBlasHandle>();
        return cache;
    }

    // Extract size and strides from a column or row vector.
    std::pair<i64, i64> extract_vector_dim_(Strides2<i64> strides, Shape2<i64> shape) {
        NOA_ASSERT(shape.ndim() == 1);
        const bool is_column = shape[1] == 1;
        const i64 n = shape[1 - is_column];
        const i64 s = strides[1 - is_column];
        return {n, s};
    }

    template<typename T>
    void cublas_gemm_(bool is_col, cublasHandle_t handle, bool lhs_transpose, bool rhs_transpose,
                      Shape3<int> mnk, Vec3<i32> labc, Vec3<i64> sabc, int batches, T alpha, T beta,
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

        if constexpr (std::is_same_v<f32, T>) {
            cublasSgemmStridedBatched(
                    handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], &alpha,
                    lhs, labc[0], sabc[0],
                    rhs, labc[1], sabc[1], &beta,
                    output, labc[2], sabc[2], batches);
        } else if constexpr (std::is_same_v<f64, T>) {
            cublasDgemmStridedBatched(
                    handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2], &alpha,
                    lhs, labc[0], sabc[0],
                    rhs, labc[1], sabc[1], &beta,
                    output, labc[2], sabc[2], batches);
        } else if constexpr (std::is_same_v<c32, T>) {
            cublasCgemmStridedBatched(
                    handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                    reinterpret_cast<const cuComplex*>(&alpha),
                    reinterpret_cast<const cuComplex*>(lhs), labc[0], sabc[0],
                    reinterpret_cast<const cuComplex*>(rhs), labc[1], sabc[1],
                    reinterpret_cast<const cuComplex*>(&beta),
                    reinterpret_cast<cuComplex*>(output), labc[2], sabc[2], batches);
        } else if constexpr (std::is_same_v<c64, T>) {
            cublasZgemmStridedBatched(
                    handle, lhs_op, rhs_op, mnk[0], mnk[1], mnk[2],
                    reinterpret_cast<const cuDoubleComplex*>(&alpha),
                    reinterpret_cast<const cuDoubleComplex*>(lhs), labc[0], sabc[0],
                    reinterpret_cast<const cuDoubleComplex*>(rhs), labc[1], sabc[1],
                    reinterpret_cast<const cuDoubleComplex*>(&beta),
                    reinterpret_cast<cuDoubleComplex*>(output), labc[2], sabc[2], batches);
        }
    }
}

namespace noa::cuda::math::details {
    void cublas_clear_cache(i32 device) {
        std::unique_ptr<CuBlasHandle>& cached_handle = cublas_cache_handle_(device);
        cached_handle = nullptr;
    }
}

namespace noa::cuda::math {
    template<typename T, typename>
    T dot(const T* lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
          const T* rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
          Stream& stream) {
        NOA_ASSERT(all(lhs_shape > 0) && all(rhs_shape > 0));

        // Get vector shape:
        NOA_ASSERT(lhs_shape.ndim() == 1 && rhs_shape.ndim() == 1 &&
                   !lhs_shape.is_batched() && !rhs_shape.is_batched());
        auto [lhs_n, lhs_s] = extract_vector_dim_(lhs_strides.filter(2, 3), lhs_shape.filter(2, 3));
        auto [rhs_n, rhs_s] = extract_vector_dim_(rhs_strides.filter(2, 3), rhs_shape.filter(2, 3));
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        T output{};
        using real_t = traits::value_type_t<T>;
        if constexpr (traits::is_real_v<T>) {
            NOA_ASSERT_DEVICE_PTR(lhs, stream.device());
            NOA_ASSERT_DEVICE_PTR(rhs, stream.device());

            cublasHandle_t handle = cublas_cache_handle_(stream.device().id())->handle;
            CUBLAS_THROW_IF_(cublasSetStream_v2(handle, stream.id()));
            CUBLAS_THROW_IF_(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST));

            cublasStatus_t err{};
            const auto n = safe_cast<i32>(lhs_n);
            const auto incx = safe_cast<i32>(lhs_s);
            const auto incy = safe_cast<i32>(rhs_s);
            if constexpr (std::is_same_v<T, f32>) {
                err = cublasSdot_v2(handle, n, lhs, incx, rhs, incy, &output);
            } else if constexpr (std::is_same_v<T, f64>) {
                err = cublasDdot_v2(handle, n, lhs, incx, rhs, incy, &output);
            } else if constexpr (std::is_same_v<T, c32>) {
                err = cublasCdotu_v2(handle, n,
                                     reinterpret_cast<const cuComplex*>(lhs), incx,
                                     reinterpret_cast<const cuComplex*>(rhs), incy,
                                     reinterpret_cast<cuComplex*>(&output));
            } else if constexpr (std::is_same_v<T, c64>) {
                err = cublasZdotu_v2(handle, n,
                                     reinterpret_cast<const cuDoubleComplex*>(lhs), incx,
                                     reinterpret_cast<const cuDoubleComplex*>(rhs), incy,
                                     reinterpret_cast<cuDoubleComplex*>(&output));
            }
            // These functions block the host thread until completion, so no need to sync the stream.
            CUBLAS_THROW_IF_(err);
        } else {
            cuda::utils::reduce_binary( // sum(lhs * rhs)
                    "dot",
                    lhs, Strides4<i64>{lhs_s},
                    rhs, Strides4<i64>{rhs_s},
                    Shape4<i64>{1, 1, 1, lhs_n},
                    &output, Strides1<i64>{1}, T{0},
                    noa::multiply_t{}, noa::plus_t{}, {},
                    true, false, stream);
            stream.synchronize();
        }
        return output;
    }

    template<typename T, typename>
    void dot(const T* lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
             const T* rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
             T* output, Stream& stream) {
        NOA_ASSERT(all(lhs_shape > 0) && all(rhs_shape > 0));
        NOA_ASSERT(lhs_shape[0] == rhs_shape[0] && lhs_shape[1] == 1 && rhs_shape[1] == 1);

        // Get vector shape: lhs should be a row vector, rhs can be a column or row vector
        i64 lhs_n{}, lhs_s{}, rhs_n{}, rhs_s{};
        std::tie(lhs_n, lhs_s) = extract_vector_dim_(lhs_strides.filter(2, 3), lhs_shape.filter(2, 3));
        std::tie(rhs_n, rhs_s) = extract_vector_dim_(rhs_strides.filter(2, 3), rhs_shape.filter(2, 3));
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        // reduce_binary is batched, works with any type and is as performant as cublas on my machine.
        // So while cublas is still expected to be more robust to different platforms, don't bother use it.
        const auto batches = lhs_shape[0];
        const auto lhs_strides_4d = Strides4<i64>{lhs_strides[0], lhs_strides[0], lhs_strides[0], lhs_s};
        const auto rhs_strides_4d = Strides4<i64>{rhs_strides[0], rhs_strides[0], rhs_strides[0], rhs_s};
        cuda::utils::reduce_binary(
                "dot", // sum(lhs * rhs)
                lhs, lhs_strides_4d,
                rhs, rhs_strides_4d,
                Shape4<i64>{batches, 1, 1, lhs_n},
                output, Strides1<i64>{1}, T{0},
                noa::multiply_t{}, noa::plus_t{}, {},
                false, false, stream);
    }

    #define INSTANTIATE_DOT_(T)                                         \
    template T dot<T, void>(                                            \
        const T*, const Strides4<i64>&, const Shape4<i64>&,             \
        const T*, const Strides4<i64>&, const Shape4<i64>&, Stream&);   \
    template void dot<T, void>(                                         \
        const T*, const Strides4<i64>&, const Shape4<i64>&,             \
        const T*, const Strides4<i64>&, const Shape4<i64>&,             \
        T*, Stream&)

    INSTANTIATE_DOT_(i32);
    INSTANTIATE_DOT_(u32);
    INSTANTIATE_DOT_(i64);
    INSTANTIATE_DOT_(u64);
    INSTANTIATE_DOT_(f32);
    INSTANTIATE_DOT_(f64);
    INSTANTIATE_DOT_(c32);
    INSTANTIATE_DOT_(c64);

    template<typename T, typename>
    void matmul(const T* lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
                const T* rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
                T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
                T* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                Stream& stream) {
        NOA_ASSERT(noa::all(lhs_shape > 0) && noa::all(rhs_shape > 0));
        NOA_ASSERT_DEVICE_PTR(lhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(rhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        // Get the shape: MxK @ KxN = MxN
        const auto m = lhs_shape[2 + lhs_transpose];
        const auto n = rhs_shape[3 - rhs_transpose];
        const auto k = lhs_shape[3 - lhs_transpose];
        const auto mnk = Shape3<i64>{m, n, k}.as_safe<i32>();
        NOA_ASSERT(lhs_shape[1] == 1 && rhs_shape[1] == 1 && output_shape[1] == 1); // 2D matrices
        NOA_ASSERT(m == output_shape[2] && n == output_shape[3]); // output fits the expected shape
        NOA_ASSERT(k == rhs_shape[2 + rhs_transpose]); // left and right matrices have compatible shape

        // In the CPU, in the case of dot products, OpenBlas GEMM is slower than its DOT function, so we check for
        // this condition and redirect to dot if necessary. Here, cublas GEMM is about as fast as the dot function,
        // so let it do a matrix-matrix product even if it is a dot product.

        // Select an order:
        const bool is_col = noa::indexing::is_column_major(output_strides);
        NOA_ASSERT(is_col == noa::indexing::is_column_major(lhs_strides) &&
                   is_col == noa::indexing::is_column_major(rhs_strides)); // same order for everyone

        // Get the pitch:
        const auto labc = Vec3<i64>{lhs_strides[2 + is_col], rhs_strides[2 + is_col], output_strides[2 + is_col]};
        const auto sabc = Vec3<i64>{lhs_strides[0], rhs_strides[0], output_strides[0]};
        NOA_ASSERT(noa::all(labc >= Vec3<i64>{lhs_shape[3 - is_col], rhs_shape[3 - is_col], output_shape[3 - is_col]}));
        NOA_ASSERT(lhs_strides[3 - is_col] == 1 && rhs_strides[3 - is_col] == 1 && output_strides[3 - is_col] == 1);

        cublasHandle_t handle = cublas_cache_handle_(stream.device().id())->handle;
        CUBLAS_THROW_IF_(cublasSetStream_v2(handle, stream.id()));
        CUBLAS_THROW_IF_(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST));

        cublas_gemm_(is_col, handle, lhs_transpose, rhs_transpose,
                     mnk, labc.as_safe<i32>(), sabc, output_shape[0], alpha, beta,
                     lhs, rhs, output);
    }

    #define INSTANTIATE_GEMM_(T)                                \
    template void matmul<T, void>(                              \
        const T*, const Strides4<i64>&, const Shape4<i64>&,     \
        const T*, const Strides4<i64>&, const Shape4<i64>&,     \
        T, T, bool, bool,                                       \
        T*, const Strides4<i64>&, const Shape4<i64>&,           \
        Stream&)

    INSTANTIATE_GEMM_(f32);
    INSTANTIATE_GEMM_(f64);
    INSTANTIATE_GEMM_(c32);
    INSTANTIATE_GEMM_(c64);
}
