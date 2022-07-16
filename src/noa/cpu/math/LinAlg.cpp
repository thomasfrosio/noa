#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <lapacke.h>

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/math/LinAlg.h"

namespace {
    using namespace noa;

    enum LAPACKDriver { GELSY, GELSD };

    template<typename T>
    class LinearLeastSquareSolver {
    public:
        using input_t = T;
        using real_t = traits::value_type_t<T>;
        using lapack_complex_t = std::conditional_t<sizeof(real_t) <= 4, lapack_complex_float, lapack_complex_double>;
        using lapack_compute_t = std::conditional_t<traits::is_complex_v<input_t>, lapack_complex_t, input_t>;

        using work_unique_t = typename cpu::memory::PtrHost<input_t>::alloc_unique_t;
        using rwork_unique_t = typename cpu::memory::PtrHost<real_t>::alloc_unique_t;
        using jpvt_unique_t = typename cpu::memory::PtrHost<int>::calloc_unique_t;
        using iwork_unique_t = typename cpu::memory::PtrHost<int>::alloc_unique_t;

    public:
        LinearLeastSquareSolver(LAPACKDriver driver, int matrix_layout,
                                int m, int n, int nrhs, int lda, int ldb, float rcond)
                : m_driver(driver), m_matrix_layout(matrix_layout), m_m(m), m_n(n), m_nrhs(nrhs),
                  m_lda(lda), m_ldb(ldb), m_rcond(rcond <= 0 ? math::Limits<float>::epsilon() : rcond) {
            LAPACKE_set_nancheck(0); // probably unused
        }

        int solve(input_t* a, input_t* b, real_t* svd = nullptr) {
            int rank;

            // Check workspaces are allocated:
            if (m_driver == GELSY && !m_jpvt)
                m_jpvt = cpu::memory::PtrHost<int>::calloc(m_n);

            if (!m_work) {
                T work_query;

                if (m_driver == GELSY) {
                    const int info = gelsy_(
                            m_matrix_layout, m_m, m_n, m_nrhs, a, m_lda, b, m_ldb,
                            m_jpvt.get(), m_rcond, &rank, &work_query, m_lwork, m_rwork.get());
                    if (info)
                        NOA_THROW_FUNC("lstsq-gelsy",
                                       "Invalid value in the {}-th argument during workspace initialization", -info);
                } else if (m_driver == GELSD) {
                    int iwork_query;
                    const int info = gelsd_(
                            m_matrix_layout, m_m, m_n, m_nrhs, a, m_lda, b, m_ldb,
                            svd, m_rcond, &rank, &work_query, m_lwork, m_rwork.get(), &iwork_query);
                    if (info)
                        NOA_THROW_FUNC("lstsq-gelsd",
                                       "Invalid value in the {}-th argument during workspace initialization", -info);
                    m_iwork = cpu::memory::PtrHost<int>::alloc(std::max(1, iwork_query));
                }
                m_lwork = std::max(1, static_cast<int>(*reinterpret_cast<real_t*>(&work_query)));
                m_work = cpu::memory::PtrHost<T>::alloc(m_lwork);
            }

            if (traits::is_complex_v<T> && !m_rwork)
                m_rwork = cpu::memory::PtrHost<real_t>::alloc(std::max(1, 2 * m_n));

            // Decompose and solve:
            if (m_driver == GELSY) {
                const int info = gelsy_(
                        m_matrix_layout, m_m, m_n, m_nrhs, a, m_lda, b, m_ldb,
                        m_jpvt.get(), m_rcond, &rank, m_work.get(), m_lwork, m_rwork.get());
                if (info < 0)
                    NOA_THROW_FUNC("lstsq-gelsy", "Invalid value in the {}-th argument", -info);
            } else if (m_driver == GELSD) {
                const int info = gelsd_(
                        m_matrix_layout, m_m, m_n, m_nrhs, a, m_lda, b, m_ldb,
                        svd, m_rcond, &rank, m_work.get(), m_lwork, m_rwork.get(), m_iwork.get());
                if (info < 0)
                    NOA_THROW_FUNC("lstsq-gelsd", "Invalid value in the {}-th argument of internal gelsd", -info);
                else if (info > 0)
                    NOA_THROW_FUNC("lstsq-gelsd", "SVD did not converge in Linear Least Square (gelsd)");
            }
            return rank;
        }

    private:
        int gelsy_(int layout, int m, int n, int nrhs,
                   input_t* a, int lda, input_t* b, int ldb, int* jpvt, float rcond, int* rank,
                   input_t* work, int lwork, real_t* rwork) {
            auto* a_ = reinterpret_cast<lapack_compute_t*>(a);
            auto* b_ = reinterpret_cast<lapack_compute_t*>(b);
            auto* work_ = reinterpret_cast<lapack_compute_t*>(work);
            const auto rcond_ = static_cast<real_t>(rcond);

            if constexpr(std::is_same_v<T, float>)
                return LAPACKE_sgelsy_work(layout, m, n, nrhs, a_, lda, b_, ldb, jpvt, rcond_, rank, work_, lwork);
            else if constexpr(std::is_same_v<T, double>)
                return LAPACKE_dgelsy_work(layout, m, n, nrhs, a_, lda, b_, ldb, jpvt, rcond_, rank, work_, lwork);
            else if constexpr(std::is_same_v<T, cfloat_t>)
                return LAPACKE_cgelsy_work(layout, m, n, nrhs, a_, lda, b_, ldb, jpvt, rcond_, rank, work_, lwork, rwork);
            else if constexpr(std::is_same_v<T, cdouble_t>)
                return LAPACKE_zgelsy_work(layout, m, n, nrhs, a_, lda, b_, ldb, jpvt, rcond_, rank, work_, lwork, rwork);
        }

        int gelsd_(int layout, int m, int n, int nrhs,
                   input_t* a, int lda, input_t* b, int ldb, real_t* svd, float rcond, int* rank,
                   input_t* work, int lwork, real_t* rwork, int* iwork) {
            auto* a_ = reinterpret_cast<lapack_compute_t*>(a);
            auto* b_ = reinterpret_cast<lapack_compute_t*>(b);
            auto* work_ = reinterpret_cast<lapack_compute_t*>(work);
            const auto rcond_ = static_cast<real_t>(rcond);

            if constexpr(std::is_same_v<T, float>)
                return LAPACKE_sgelsd_work(layout, m, n, nrhs, a_, lda, b_, ldb, svd, rcond_, rank, work_, lwork, iwork);
            else if constexpr(std::is_same_v<T, double>)
                return LAPACKE_dgelsd_work(layout, m, n, nrhs, a_, lda, b_, ldb, svd, rcond_, rank, work_, lwork, iwork);
            else if constexpr(std::is_same_v<T, cfloat_t>)
                return LAPACKE_cgelsd_work(layout, m, n, nrhs, a_, lda, b_, ldb, svd, rcond_, rank, work_, lwork, rwork, iwork);
            else if constexpr(std::is_same_v<T, cdouble_t>)
                return LAPACKE_zgelsd_work(layout, m, n, nrhs, a_, lda, b_, ldb, svd, rcond_, rank, work_, lwork, rwork, iwork);
        }

    private:
        work_unique_t m_work{};
        rwork_unique_t m_rwork{};
        jpvt_unique_t m_jpvt{};
        iwork_unique_t m_iwork{};
        int m_lwork{-1};

        LAPACKDriver m_driver;
        int m_matrix_layout;
        int m_m, m_n, m_nrhs;
        int m_lda, m_ldb;
        float m_rcond;
    };

    template<LAPACKDriver DRIVER, typename T>
    class SurfaceFitter {
    public:
        using unique_t = typename cpu::memory::PtrHost<T>::alloc_unique_t;

    public:
        SurfaceFitter(const T* input, size4_t input_stride, size4_t input_shape, int order)
                : m_shape(input_shape[2], input_shape[3]),
                  m_m(m_shape.elements()),
                  m_n(nbParameters_(order)),
                  m_nrhs(input_shape[0]) {
            NOA_ASSERT(input_shape[1] == 1);

            // Compute the column-major m-by-n matrix representing the regular grid of the data:
            m_A = cpu::memory::PtrHost<T>::alloc(m_m * m_n);
            computeRegularGrid_();

            // Prepare the target column-major column vector(s):
            m_b = cpu::memory::PtrHost <T>::alloc(m_m * m_nrhs);
            cpu::memory::copy(input, input_stride, m_b.get(), input_shape.stride(), input_shape);

            // Solve x by minimizing (A @ x - b):
            const int m = static_cast<int>(m_m);
            const int n = static_cast<int>(m_n);
            const int nrhs = static_cast<int>(m_nrhs);
            LinearLeastSquareSolver<T> solver(DRIVER, LAPACK_COL_MAJOR, m, n, nrhs, m, m, 0);
            solver.solve(m_A.get(), m_b.get(), nullptr);
        }

        void saveParameters(T* parameters) {
            const size4_t b_stride{m_m * m_nrhs, m_m * m_nrhs, 1, m_m};
            const size4_t x_shape{1, 1, m_n, m_nrhs};
            const size4_t x_stride{m_n * m_nrhs, m_n * m_nrhs, 1, m_n};
            cpu::memory::copy(m_b.get(), b_stride, parameters, x_stride, x_shape);
        }

        void computeSurface(T* input, size4_t input_stride,
                            T* output, size4_t output_stride, size4_t shape) {
            const size2_t output_stride_2d(output_stride.get(2));
            const size2_t input_stride_2d(input_stride.get(2));

            std::array<T, 10> p{};
            for (size_t batch = 0; batch < shape[0]; ++batch) {
                T* output_ptr = output + output_stride[0] * batch;
                T* input_ptr = input + input_stride[0] * batch;

                const int order = order_();
                std::copy(m_b.get() + batch * m_m, m_b.get() + batch * m_m + m_n, p.data());

                T surface;
                for (size_t iy = 0; iy < shape[2]; ++iy) {
                    for (size_t ix = 0; ix < shape[3]; ++ix) {
                        const T y = static_cast<T>(iy);
                        const T x = static_cast<T>(ix);

                        surface = p[0] + x * p[1] + y * p[2];
                        if (order >= 2)
                            surface += x * y * p[3] + x * x * p[4] + y * y * p[5];
                        if (order == 3)
                            surface += x * x * y * p[6] + x * y * y * p[7] + x * x * x * p[8] + y * y * y * p[9];

                        output_ptr[indexing::at(iy, ix, output_stride_2d)] =
                                input ? input_ptr[indexing::at(iy, ix, input_stride_2d)] - surface : surface;
                    }
                }
            }
        }

    private:
        void computeRegularGrid_() {
            T* matrix = m_A.get();
            for (size_t y = 0; y < m_shape[0]; ++y) {
                for (size_t x = 0; x < m_shape[1]; ++x) {
                    matrix[m_m * 0 + y * m_shape[1] + x] = T{1};
                    matrix[m_m * 1 + y * m_shape[1] + x] = static_cast<T>(x);
                    matrix[m_m * 2 + y * m_shape[1] + x] = static_cast<T>(y);
                }
            }
            if (order_() >= 2) {
                for (size_t y = 0; y < m_shape[0]; ++y) {
                    for (size_t x = 0; x < m_shape[1]; ++x) {
                        matrix[m_m * 3 + y * m_shape[1] + x] = static_cast<T>(x * y);
                        matrix[m_m * 4 + y * m_shape[1] + x] = static_cast<T>(x * x);
                        matrix[m_m * 5 + y * m_shape[1] + x] = static_cast<T>(y * y);
                    }
                }
            }
            if (order_() == 3) {
                for (size_t y = 0; y < m_shape[0]; ++y) {
                    for (size_t x = 0; x < m_shape[1]; ++x) {
                        matrix[m_m * 6 + y * m_shape[1] + x] = static_cast<T>(x * x * y);
                        matrix[m_m * 7 + y * m_shape[1] + x] = static_cast<T>(x * y * y);
                        matrix[m_m * 8 + y * m_shape[1] + x] = static_cast<T>(x * x * x);
                        matrix[m_m * 9 + y * m_shape[1] + x] = static_cast<T>(y * y * y);
                    }
                }
            }
        }

        static size_t nbParameters_(int order) {
            return order == 3 ? 10 : static_cast<size_t>(order * 3);
        }

        int order_() {
            return m_n == 3 ? 1 : m_n == 6 ? 2 : 3;
        }

    private:
        unique_t m_A{};
        unique_t m_b{};
        size2_t m_shape;
        size_t m_m, m_n, m_nrhs;
    };
}

namespace noa::cpu::math {
    template<typename T, typename U, typename>
    void lstsq(const shared_t<T[]>& a, size4_t a_stride, size4_t a_shape,
               const shared_t<T[]>& b, size4_t b_stride, size4_t b_shape,
               float cond, const shared_t<U[]>& svd,
               Stream& stream) {
        NOA_ASSERT(a_shape[0] == b_shape[0]);
        NOA_ASSERT(a_shape[1] == 1 && b_shape[1] == 1);
        const size_t batches = a_shape[0];

        const int2_t a_shape_{a_shape.get(2)};
        const int2_t b_shape_{b_shape.get(2)};
        const int2_t a_stride_{a_stride.get(2)};
        const int2_t b_stride_{b_stride.get(2)};

        // Ax = b, where A is m-by-n, x is n-by-nrhs and b is m-by-nrhs. Most often, nrhs is 1.
        const int m = a_shape_[0];
        const int n = a_shape_[1];
        const int mn_max = std::max({1, m, n});
        const int mn_min = std::min(m, n);
        const int nrhs = b_shape_[1];
        NOA_ASSERT(m > 0 && n > 0 && nrhs > 0);

        // Check memory layout of a. Since most LAPACKE implementations are using column major and simply transposing
        // the row major matrices before and after decomposition, it is beneficial to detect the column major case
        // instead of assuming it's always row major.
        const bool is_row_major = indexing::isRowMajor(a_stride_);
        const int matrix_layout = is_row_major ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
        const int lda = a_stride_[!is_row_major];
        const int ldb = b_stride_[!is_row_major];

        // Check the size of the problem makes sense and that the innermost dimension of the matrices is contiguous.
        // Note that the secondmost dimension can be padded (i.e. lda and ldb).
        NOA_ASSERT(is_row_major ?
                   (lda >= n && ldb >= nrhs && a_stride_[1] == 1 && indexing::isRowMajor(b_stride_)) :
                   (lda >= m && ldb >= mn_max && a_stride_[0] == 1 && indexing::isColMajor(b_stride_)));
        NOA_ASSERT(b_stride_[is_row_major] == 1 && b_shape_[0] == mn_max);

        stream.enqueue([=]() {
            const LAPACKDriver driver = svd ? GELSD : GELSY;
            LinearLeastSquareSolver<T> solver(driver, matrix_layout, m, n, nrhs, lda, ldb, cond);
            for (size_t batch = 0; batch < batches; ++batch) {
                solver.solve(a.get() + a_stride[0] * batch,
                             b.get() + b_stride[0] * batch,
                             svd.get() + mn_min);
            }
        });
    }

    template<typename T, typename>
    void surface(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, bool subtract,
                 int order, const shared_t<T[]>& parameters, Stream& stream) {
        if (!output && !parameters)
            return;

        NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);
        NOA_ASSERT(order >= 1 && order <= 3);
        stream.enqueue([=]() {
            SurfaceFitter<GELSY, T> surface(input.get(), input_stride, input_shape, order);
            if (parameters)
                surface.saveParameters(parameters.get());
            if (output)
                surface.computeSurface(subtract ? input.get() : nullptr,
                                       input_stride, output.get(),  output_stride, output_shape);
        });
    }

    #define NOA_INSTANTIATE_LSTSQ_(T, U) \
    template void lstsq<T,U,void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float, const shared_t<U[]>&, Stream&)

    NOA_INSTANTIATE_LSTSQ_(float, float);
    NOA_INSTANTIATE_LSTSQ_(double, double);
    NOA_INSTANTIATE_LSTSQ_(cfloat_t, float);
    NOA_INSTANTIATE_LSTSQ_(cdouble_t, double);

    #define NOA_INSTANTIATE_SURFACE_(T) \
    template void surface<T,void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, bool, \
                                  int, const shared_t<T[]>&, Stream&)

    NOA_INSTANTIATE_SURFACE_(float);
    NOA_INSTANTIATE_SURFACE_(double);
}
