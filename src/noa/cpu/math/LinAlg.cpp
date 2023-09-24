#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <lapacke.h>

#include "noa/cpu/memory/AllocatorHeap.hpp"
#include "noa/cpu/memory/Copy.hpp"
#include "noa/cpu/math/LinAlg.hpp"

// Using Eigen is much cleaner and flexible than the old lapacke... It is also easier to package.
// Unfortunately, the SVD in Eigen takes >2min to compile, which is just not OK in my book.

namespace {
    using namespace noa;

    enum LAPACKDriver { GELSY, GELSD };

    template<typename T>
    class LinearLeastSquareSolver {
    public:
        using input_type = T;
        using real_type = traits::value_type_t<T>;
        using lapack_complex_type = std::conditional_t<sizeof(real_type) <= 4, lapack_complex_float, lapack_complex_double>;
        using lapack_compute_type = std::conditional_t<traits::is_complex_v<input_type>, lapack_complex_type, input_type>;

        using work_unique_type = typename cpu::memory::AllocatorHeap<input_type>::alloc_unique_type;
        using rwork_unique_type = typename cpu::memory::AllocatorHeap<real_type>::alloc_unique_type;
        using jpvt_unique_type = typename cpu::memory::AllocatorHeap<i32>::calloc_unique_type;
        using iwork_unique_type = typename cpu::memory::AllocatorHeap<i32>::alloc_unique_type;

    public:
        LinearLeastSquareSolver(
                LAPACKDriver driver, i32 matrix_layout,
                i32 m, i32 n, i32 nrhs,
                i32 lda, i32 ldb, f32 rcond)
                : m_driver(driver),
                  m_matrix_layout(matrix_layout),
                  m_m(m), m_n(n), m_nrhs(nrhs),
                  m_lda(lda), m_ldb(ldb),
                  m_rcond(rcond <= 0 ? noa::math::Limits<f32>::epsilon() : rcond) {
            LAPACKE_set_nancheck(0); // probably unused
        }

        i32 solve(input_type* a, input_type* b, real_type* svd = nullptr) {
            i32 rank{};

            // Check workspaces are allocated:
            if (m_driver == GELSY && !m_jpvt)
                m_jpvt = cpu::memory::AllocatorHeap<int>::calloc(m_n);

            if (!m_work) {
                T work_query;

                if (m_driver == GELSY) {
                    const i32 info = gelsy_(
                            m_matrix_layout, m_m, m_n, m_nrhs, a, m_lda, b, m_ldb,
                            m_jpvt.get(), m_rcond, &rank, &work_query, m_lwork, m_rwork.get());
                    if (info)
                        NOA_THROW_FUNC("lstsq-gelsy",
                                       "Invalid value in the {}-th argument during workspace initialization", -info);
                } else if (m_driver == GELSD) {
                    i32 iwork_query;
                    const i32 info = gelsd_(
                            m_matrix_layout, m_m, m_n, m_nrhs, a, m_lda, b, m_ldb,
                            svd, m_rcond, &rank, &work_query, m_lwork, m_rwork.get(), &iwork_query);
                    if (info)
                        NOA_THROW_FUNC("lstsq-gelsd",
                                       "Invalid value in the {}-th argument during workspace initialization", -info);
                    m_iwork = cpu::memory::AllocatorHeap<i32>::allocate(std::max(1, iwork_query));
                }
                m_lwork = std::max(1, static_cast<i32>(*reinterpret_cast<real_type*>(&work_query)));
                m_work = cpu::memory::AllocatorHeap<T>::allocate(m_lwork);
            }

            if (traits::is_complex_v<T> && !m_rwork)
                m_rwork = cpu::memory::AllocatorHeap<real_type>::allocate(std::max(1, 2 * m_n));

            // Decompose and solve:
            if (m_driver == GELSY) {
                const i32 info = gelsy_(
                        m_matrix_layout, m_m, m_n, m_nrhs, a, m_lda, b, m_ldb,
                        m_jpvt.get(), m_rcond, &rank, m_work.get(), m_lwork, m_rwork.get());
                if (info < 0)
                    NOA_THROW_FUNC("lstsq-gelsy", "Invalid value in the {}-th argument", -info);
            } else if (m_driver == GELSD) {
                const i32 info = gelsd_(
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
        i32 gelsy_(i32 layout, i32 m, i32 n, i32 nrhs,
                   input_type* a, i32 lda, input_type* b, i32 ldb,
                   i32* jpvt, f32 rcond, i32* rank,
                   input_type* work, i32 lwork, real_type* rwork) {
            auto* a_ = reinterpret_cast<lapack_compute_type*>(a);
            auto* b_ = reinterpret_cast<lapack_compute_type*>(b);
            auto* work_ = reinterpret_cast<lapack_compute_type*>(work);
            const auto rcond_ = static_cast<real_type>(rcond);

            if constexpr(std::is_same_v<T, f32>)
                return LAPACKE_sgelsy_work(layout, m, n, nrhs, a_, lda, b_, ldb, jpvt, rcond_, rank, work_, lwork);
            else if constexpr(std::is_same_v<T, f64>)
                return LAPACKE_dgelsy_work(layout, m, n, nrhs, a_, lda, b_, ldb, jpvt, rcond_, rank, work_, lwork);
            else if constexpr(std::is_same_v<T, c32>)
                return LAPACKE_cgelsy_work(layout, m, n, nrhs, a_, lda, b_, ldb, jpvt, rcond_, rank, work_, lwork, rwork);
            else if constexpr(std::is_same_v<T, c64>)
                return LAPACKE_zgelsy_work(layout, m, n, nrhs, a_, lda, b_, ldb, jpvt, rcond_, rank, work_, lwork, rwork);
        }

        int gelsd_(i32 layout, i32 m, i32 n, i32 nrhs,
                   input_type* a, i32 lda, input_type* b, i32 ldb,
                   real_type* svd, f32 rcond, i32* rank,
                   input_type* work, i32 lwork, real_type* rwork, int* iwork) {
            auto* a_ = reinterpret_cast<lapack_compute_type*>(a);
            auto* b_ = reinterpret_cast<lapack_compute_type*>(b);
            auto* work_ = reinterpret_cast<lapack_compute_type*>(work);
            const auto rcond_ = static_cast<real_type>(rcond);

            if constexpr(std::is_same_v<T, f32>)
                return LAPACKE_sgelsd_work(layout, m, n, nrhs, a_, lda, b_, ldb, svd, rcond_, rank, work_, lwork, iwork);
            else if constexpr(std::is_same_v<T, f64>)
                return LAPACKE_dgelsd_work(layout, m, n, nrhs, a_, lda, b_, ldb, svd, rcond_, rank, work_, lwork, iwork);
            else if constexpr(std::is_same_v<T, c32>)
                return LAPACKE_cgelsd_work(layout, m, n, nrhs, a_, lda, b_, ldb, svd, rcond_, rank, work_, lwork, rwork, iwork);
            else if constexpr(std::is_same_v<T, c64>)
                return LAPACKE_zgelsd_work(layout, m, n, nrhs, a_, lda, b_, ldb, svd, rcond_, rank, work_, lwork, rwork, iwork);
        }

    private:
        work_unique_type m_work{};
        rwork_unique_type m_rwork{};
        jpvt_unique_type m_jpvt{};
        iwork_unique_type m_iwork{};
        i32 m_lwork{-1};

        LAPACKDriver m_driver;
        i32 m_matrix_layout;
        i32 m_m, m_n, m_nrhs;
        i32 m_lda, m_ldb;
        f32 m_rcond;
    };

    template<LAPACKDriver DRIVER, typename T>
    class SurfaceFitter {
    public:
        using unique_type = typename cpu::memory::AllocatorHeap<T>::alloc_unique_type;

    public:
        SurfaceFitter(const T* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 order)
                : m_shape(shape.filter(2, 3)),
                  m_m(m_shape.elements()),
                  m_n(number_of_parameters_(order)),
                  m_nrhs(shape[0]) {
            NOA_ASSERT(shape[1] == 1);

            // Compute the column-major m-by-n matrix representing the regular grid of the data:
            m_A = cpu::memory::AllocatorHeap<T>::allocate(m_m * m_n);
            compute_regular_grid_();

            // Prepare the F-contiguous target column vector(s):
            // TODO The input is "transformed" into this column vector in the row-major order.
            //      Reordering is difficult here because it means the columns of the matrix A should
            //      be reordered as well since we probably want the solution to always be in the same order,
            //      i.e. p[0] + p[1]*x + p[2]*y + ...
            //      For now, the cpu::memory::copy will enforce the current row-major layout, which results
            //      in a permutation if the input is passed as column-major.
            m_b = cpu::memory::AllocatorHeap<T>::allocate(m_m * m_nrhs);
            cpu::memory::copy(input, strides, m_b.get(), shape.strides(), shape, 1);

            // Solve x by minimizing (A @ x - b):
            const i32 m = static_cast<i32>(m_m);
            const i32 n = static_cast<i32>(m_n);
            const i32 nrhs = static_cast<i32>(m_nrhs);
            LinearLeastSquareSolver<T> solver(DRIVER, LAPACK_COL_MAJOR, m, n, nrhs, m, m, 0);
            solver.solve(m_A.get(), m_b.get(), nullptr);
        }

        void save_parameters(T* parameters) {
            const auto b_strides = Strides4<i64>{m_m * m_nrhs, m_m * m_nrhs, 1, m_m};
            const auto x_shape = Shape4<i64>{1, 1, m_n, m_nrhs};
            const auto x_strides = Strides4<i64>{m_n * m_nrhs, m_n * m_nrhs, 1, m_n};
            cpu::memory::copy(m_b.get(), b_strides, parameters, x_strides, x_shape, 1);
        }

        void compute_surface(T* input, const Strides4<i64>& input_strides,
                             T* output, const Strides4<i64>& output_strides,
                             Shape4<i64> shape) {
            auto output_strides_2d = output_strides.filter(2, 3);
            auto input_strides_2d = input_strides.filter(2, 3);

            // If arrays are column-major, make sure to loop in the column major order.
            const bool swap = noa::indexing::is_column_major(output_strides) &&
                              (!input || noa::indexing::is_column_major(input_strides));
            if (swap) {
                std::swap(shape[2], shape[3]);
                std::swap(input_strides_2d[0], input_strides_2d[1]);
                std::swap(output_strides_2d[0], output_strides_2d[1]);
            }

            std::array<T, 10> p{};
            for (i64 batch = 0; batch < shape[0]; ++batch) {
                T* output_ptr = output + output_strides[0] * batch;
                T* input_ptr = input + input_strides[0] * batch;

                const i64 order = order_();
                std::copy(m_b.get() + batch * m_m, m_b.get() + batch * m_m + m_n, p.data());
                if (swap) {
                    std::swap(p[1], p[2]);
                    std::swap(p[4], p[5]);
                    std::swap(p[6], p[7]);
                    std::swap(p[8], p[9]);
                }

                T surface;
                for (i64 iy = 0; iy < shape[2]; ++iy) {
                    for (i64 ix = 0; ix < shape[3]; ++ix) {
                        const T y = static_cast<T>(iy);
                        const T x = static_cast<T>(ix);

                        surface = p[0] + x * p[1] + y * p[2];
                        if (order >= 2)
                            surface += x * y * p[3] + x * x * p[4] + y * y * p[5];
                        if (order == 3)
                            surface += x * x * y * p[6] + x * y * y * p[7] + x * x * x * p[8] + y * y * y * p[9];

                        const auto input_offset = noa::indexing::at(iy, ix, input_strides_2d);
                        const auto output_offset = noa::indexing::at(iy, ix, output_strides_2d);
                        output_ptr[output_offset] = input ? input_ptr[input_offset] - surface : surface;
                    }
                }
            }
        }

    private:
        void compute_regular_grid_() {
            T* matrix = m_A.get();
            for (i64 y = 0; y < m_shape[0]; ++y) {
                for (i64 x = 0; x < m_shape[1]; ++x) {
                    matrix[m_m * 0 + y * m_shape[1] + x] = T{1};
                    matrix[m_m * 1 + y * m_shape[1] + x] = static_cast<T>(x);
                    matrix[m_m * 2 + y * m_shape[1] + x] = static_cast<T>(y);
                }
            }
            if (order_() >= 2) {
                for (i64 y = 0; y < m_shape[0]; ++y) {
                    for (i64 x = 0; x < m_shape[1]; ++x) {
                        matrix[m_m * 3 + y * m_shape[1] + x] = static_cast<T>(x * y);
                        matrix[m_m * 4 + y * m_shape[1] + x] = static_cast<T>(x * x);
                        matrix[m_m * 5 + y * m_shape[1] + x] = static_cast<T>(y * y);
                    }
                }
            }
            if (order_() == 3) {
                for (i64 y = 0; y < m_shape[0]; ++y) {
                    for (i64 x = 0; x < m_shape[1]; ++x) {
                        matrix[m_m * 6 + y * m_shape[1] + x] = static_cast<T>(x * x * y);
                        matrix[m_m * 7 + y * m_shape[1] + x] = static_cast<T>(x * y * y);
                        matrix[m_m * 8 + y * m_shape[1] + x] = static_cast<T>(x * x * x);
                        matrix[m_m * 9 + y * m_shape[1] + x] = static_cast<T>(y * y * y);
                    }
                }
            }
        }

        static i64 number_of_parameters_(i64 order) {
            return order == 3 ? 10 : order * 3;
        }

        i64 order_() {
            return m_n == 3 ? 1 : m_n == 6 ? 2 : 3;
        }

    private:
        unique_type m_A{};
        unique_type m_b{};
        Shape2<i64> m_shape;
        i64 m_m, m_n, m_nrhs;
    };
}

namespace noa::cpu::math {
    template<typename T, typename U, typename>
    void lstsq(T* a, const Strides4<i64>& a_strides, const Shape4<i64>& a_shape,
               T* b, const Strides4<i64>& b_strides, const Shape4<i64>& b_shape,
               f32 cond, U* svd) {
        NOA_ASSERT(a && b && noa::all(a_shape > 0) && noa::all(b_shape > 0));
        NOA_ASSERT(a_shape[0] == b_shape[0]);
        NOA_ASSERT(a_shape[1] == 1 && b_shape[1] == 1);
        const i64 batches = a_shape[0];

        const auto a_shape_ = a_shape.filter(2, 3).as_safe<i32>();
        const auto b_shape_ = b_shape.filter(2, 3).as_safe<i32>();
        const auto a_strides_ = a_strides.filter(2, 3).as_safe<i32>();
        const auto b_strides_ = b_strides.filter(2, 3).as_safe<i32>();

        // Ax = b, where A is m-by-n, x is n-by-nrhs and b is m-by-nrhs. Most often, nrhs is 1.
        const i32 m = a_shape_[0];
        const i32 n = a_shape_[1];
        const i32 mn_max = std::max({1, m, n});
        const i32 mn_min = std::min(m, n);
        const i32 nrhs = b_shape_[1];
        NOA_ASSERT(m > 0 && n > 0 && nrhs > 0);

        // Check memory layout of a. Since most LAPACKE implementations are using column major and simply transposing
        // the row major matrices before and after decomposition, it is beneficial to detect the column major case
        // instead of assuming it's always row major.
        const bool is_row_major = noa::indexing::is_row_major(a_strides_);
        const int matrix_layout = is_row_major ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
        const int lda = a_strides_[!is_row_major];
        const int ldb = b_strides_[!is_row_major];

        // Check the size of the problem makes sense and that the innermost dimension of the matrices is contiguous.
        // Note that the second-most dimension can be padded (i.e. lda and ldb).
        NOA_ASSERT(is_row_major ?
                   (lda >= n && ldb >= nrhs && a_strides_[1] == 1 && noa::indexing::is_row_major(b_strides_)) :
                   (lda >= m && ldb >= mn_max && a_strides_[0] == 1 && noa::indexing::is_column_major(b_strides_)));
        NOA_ASSERT(b_strides_[is_row_major] == 1 && b_shape_[0] == mn_max);
        (void) mn_max;

        const LAPACKDriver driver = svd ? GELSD : GELSY;
        LinearLeastSquareSolver<T> solver(driver, matrix_layout, m, n, nrhs, lda, ldb, cond);
        for (i64 batch = 0; batch < batches; ++batch) {
            solver.solve(a + a_strides[0] * batch,
                         b + b_strides[0] * batch,
                         svd + mn_min);
        }
    }

    template<typename T, typename>
    void surface(T* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                 T* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                 bool subtract, i32 order, T* parameters) {
        NOA_ASSERT(input && all(input_shape > 0));
        if (!output && !parameters)
            return;

        NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);
        NOA_ASSERT(order >= 1 && order <= 3);
        SurfaceFitter<GELSY, T> surface(input, input_strides, input_shape, order);
        if (parameters)
            surface.save_parameters(parameters);
        if (output) {
            NOA_ASSERT(all(output_shape > 0));
            surface.compute_surface(subtract ? input : nullptr, input_strides,
                                    output, output_strides, output_shape);
        }
    }

    #define NOA_INSTANTIATE_LSTSQ_(T, U)                \
    template void lstsq<T,U,void>(                      \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        f32, U*)

    NOA_INSTANTIATE_LSTSQ_(f32, f32);
    NOA_INSTANTIATE_LSTSQ_(f64, f64);
    NOA_INSTANTIATE_LSTSQ_(c32, f32);
    NOA_INSTANTIATE_LSTSQ_(c64, f64);

    #define NOA_INSTANTIATE_SURFACE_(T)                 \
    template void surface<T,void>(                      \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        bool, i32, T*)

//    NOA_INSTANTIATE_SURFACE_(f32);
//    NOA_INSTANTIATE_SURFACE_(f64);
}
