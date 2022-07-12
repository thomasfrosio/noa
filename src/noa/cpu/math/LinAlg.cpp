#include <lapacke.h>
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/math/LinAlg.h"

namespace {
    using namespace noa;

    template<typename T, typename U>
    int gelsyOpenBlasLAPACKE_(int matrix_layout, int m, int n, int nrhs,
                              T* a, int lda, T* b, int ldb, int* jpvt, float rcond,
                              T* work, int lwork, U* rwork) {
        int rank;
        if constexpr(std::is_same_v<T, float>) {
            return LAPACKE_sgelsy_work(matrix_layout, m, n, nrhs,
                                       a, lda, b, ldb, jpvt,
                                       rcond, &rank, work, lwork);
        } else if constexpr(std::is_same_v<T, double>) {
            return LAPACKE_dgelsy_work(matrix_layout, m, n, nrhs,
                                       a, lda, b, ldb, jpvt,
                                       static_cast<double>(rcond), &rank, work, lwork);
        } else if constexpr(std::is_same_v<T, cfloat_t>) {
            return LAPACKE_cgelsy_work(matrix_layout, m, n, nrhs,
                                       reinterpret_cast<lapack_complex_float*>(a), lda,
                                       reinterpret_cast<lapack_complex_float*>(b), ldb, jpvt,
                                       rcond, &rank,
                                       reinterpret_cast<lapack_complex_float*>(work), lwork,
                                       reinterpret_cast<float*>(rwork));
        } else if constexpr(std::is_same_v<T, cdouble_t>) {
            return LAPACKE_zgelsy_work(matrix_layout, m, n, nrhs,
                                       reinterpret_cast<lapack_complex_double*>(a), lda,
                                       reinterpret_cast<lapack_complex_double*>(b), ldb, jpvt,
                                       static_cast<double>(rcond), &rank,
                                       reinterpret_cast<lapack_complex_double*>(work), lwork,
                                       reinterpret_cast<double*>(rwork));
        }
    }

    template<typename T, typename U, typename V>
    int gelsdOpenBlasLAPACKE_(int matrix_layout, int m, int n, int nrhs,
                              T* a, int lda, T* b, int ldb, U* s, float rcond,
                              T* work, int lwork, V* rwork, int* iwork) {
        int rank;
        if constexpr(std::is_same_v<T, float>) {
            return LAPACKE_sgelsd_work(matrix_layout, m, n, nrhs,
                                       a, lda, b, ldb, s,
                                       rcond, &rank, work, lwork, iwork);
        } else if constexpr(std::is_same_v<T, double>) {
            return LAPACKE_dgelsd_work(matrix_layout, m, n, nrhs,
                                       a, lda, b, ldb, s,
                                       static_cast<double>(rcond), &rank, work, lwork, iwork);
        } else if constexpr(std::is_same_v<T, cfloat_t>) {
            return LAPACKE_cgelsd_work(matrix_layout, m, n, nrhs,
                                       reinterpret_cast<lapack_complex_float*>(a), lda,
                                       reinterpret_cast<lapack_complex_float*>(b), ldb, s,
                                       rcond, &rank,
                                       reinterpret_cast<lapack_complex_float*>(work), lwork,
                                       reinterpret_cast<float*>(rwork), iwork);
        } else if constexpr(std::is_same_v<T, cdouble_t>) {
            return LAPACKE_zgelsd_work(matrix_layout, m, n, nrhs,
                                       reinterpret_cast<lapack_complex_double*>(a), lda,
                                       reinterpret_cast<lapack_complex_double*>(b), ldb, s,
                                       static_cast<double>(rcond), &rank,
                                       reinterpret_cast<lapack_complex_double*>(work), lwork,
                                       reinterpret_cast<double*>(rwork), iwork);
        }
    }

    template<typename T, typename U>
    int gelsyOpenBlasLAPACKEWorkSize_(int matrix_layout, int m, int n, int nrhs,
                                      T* a, int lda, T* b, int ldb, int* jpvt, float rcond, U* rwork) {
        T work_query;
        const auto info = gelsyOpenBlasLAPACKE_(
                matrix_layout, m, n, nrhs, a, lda, b, ldb,
                jpvt, rcond, &work_query, -1, rwork);
        if (info)
            return 0;
        return static_cast<int>(*reinterpret_cast<U*>(&work_query));
    }

    template<typename T, typename U>
    std::pair<int, int> gelsdOpenBlasLAPACKEWorkSize_(int matrix_layout, int m, int n, int nrhs,
                                                      T* a, int lda, T* b, int ldb, U* s, float rcond) {
        T work_query;
        int iwork_query;
        const auto info = gelsdOpenBlasLAPACKE_(
                matrix_layout, m, n, nrhs, a, lda, b, ldb,
                s, rcond, &work_query, -1, &work_query, &iwork_query);
        if (info)
            return {};
        return {static_cast<int>(*reinterpret_cast<U*>(&work_query)), iwork_query};
    }
}

namespace noa::cpu::math {
    template<typename T, typename U, typename>
    void lstsq(const shared_t<T[]>& a, size4_t a_stride, size4_t a_shape,
               const shared_t<T[]>& b, size4_t b_stride, size4_t b_shape,
               float cond, const shared_t<U[]>& svd,
               Stream& stream) {
        using real_t = traits::value_type_t<T>;

        NOA_ASSERT(a_shape[0] == b_shape[0]);
        NOA_ASSERT(a_shape[1] == 1 && b_shape[2] == 1);
        const size_t batches = a_shape[0];

        const int2_t a_shape_{a_shape.get(2)};
        const int2_t b_shape_{b_shape.get(2)};
        const int2_t a_stride_{a_stride.get(2)};
        const int2_t b_stride_{b_stride.get(2)};

        const int m = a_shape_[0];
        const int n = a_shape_[1];
        const int mn = noa::math::max({1, m, n});
        NOA_ASSERT(m > 0 && n > 0);

        // Check memory layout of a:
        const bool is_row_major = a_stride_[0] > a_stride_[1];
        const int matrix_layout = is_row_major ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
        const int lda = a_stride_[!is_row_major];
        NOA_ASSERT(is_row_major ?
                   (a_stride_[0] >= n && a_stride_[1] == 1) :
                   (a_stride_[1] >= m && a_stride_[0] == 1));

        // Check memory layout of b:
        NOA_ASSERT(is_row_major ? b_stride_[0] > b_stride_[1] : b_stride_[0] < b_stride_[1]);
        int nrhs, ldb;
        if (b_shape_[0] == 1) { // b is a row vector
            nrhs = 1;
            ldb = b_stride_[is_row_major]; // probably unused
            NOA_ASSERT(b_stride_[1] == 1 && b_shape_[1] == mn);
        } else { // b is a (stack of) column vector(s)
            nrhs = b_shape_[1];
            ldb = b_stride_[!is_row_major];
            NOA_ASSERT(b_stride_[is_row_major] == 1 && b_shape_[0] == mn);
        }
        NOA_ASSERT(is_row_major ? ldb >= nrhs : ldb >= mn);

        if (cond <= 0)
            cond = noa::math::Limits<real_t>::epsilon();

        stream.enqueue([=]() {
            LAPACKE_set_nancheck(0); // probably unused
            using namespace noa::cpu::memory;
            if (svd) {
                auto [lwork, liwork] = gelsdOpenBlasLAPACKEWorkSize_(
                        matrix_layout, m, n, nrhs, a.get(), lda, b.get(), ldb, svd.get(), cond);
                auto iwork = PtrHost<int>::alloc(static_cast<size_t>(liwork));
                auto work = PtrHost<T>::alloc(static_cast<size_t>(lwork));
                auto rwork = PtrHost<real_t>::alloc(traits::is_complex_v<T> ? static_cast<size_t>(lwork) : 0);

                int info;
                for (size_t batch = 0; batch < batches; ++batch) {
                    info = gelsdOpenBlasLAPACKE_(matrix_layout, m, n, nrhs,
                                                       a.get() + a_stride[0] * batch, lda,
                                                       b.get() + b_stride[0] * batch, lda,
                                                       svd.get(), cond, work.get(), lwork, rwork.get(), iwork.get());
                    if (info < 0)
                        NOA_THROW("Invalid value in the {}-th argument of internal gelsd", -info);
                    else if (info > 0)
                        NOA_THROW("SVD did not converge in Linear Least Square (gelsd)");
                }
            } else {
                auto jpvt = PtrHost<int>::calloc(static_cast<size_t>(n));
                auto rwork = PtrHost<real_t>::alloc(traits::is_complex_v<T> ? static_cast<size_t>(2 * n) : 0);
                const int lwork = gelsyOpenBlasLAPACKEWorkSize_(
                        matrix_layout, m, n, nrhs, a.get(), lda, b.get(), ldb, jpvt.get(), cond, rwork.get());
                auto work = PtrHost<T>::alloc(static_cast<size_t>(lwork));

                int info;
                for (size_t batch = 0; batch < batches; ++batch) {
                    info = gelsyOpenBlasLAPACKE_(matrix_layout, m, n, nrhs,
                                                 a.get() + a_stride[0] * batch, lda,
                                                 b.get() + b_stride[0] * batch, ldb,
                                                 jpvt.get(), cond, work.get(), lwork, rwork.get());
                    if (info < 0)
                        NOA_THROW("Invalid value in the {}-th argument of internal gelsy", -info);
                }
            }
        });
    }

    #define NOA_INSTANTIATE_LSTSQ_(T, U) \
    template void lstsq<T,U,void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float, const shared_t<U[]>&, Stream&)

    NOA_INSTANTIATE_LSTSQ_(float, float);
    NOA_INSTANTIATE_LSTSQ_(double, double);
    NOA_INSTANTIATE_LSTSQ_(cfloat_t, float);
    NOA_INSTANTIATE_LSTSQ_(cdouble_t, double);
}
