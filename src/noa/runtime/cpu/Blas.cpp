#include "noa/base/Config.hpp"
#include "noa/base/Complex.hpp"
#include "noa/runtime/core/Utils.hpp"
#include "noa/runtime/cpu/Blas.hpp"

// Suppress Eigen warnings...
#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wsign-conversion"
#   pragma GCC diagnostic ignored "-Wnull-dereference"
#   if defined(NOA_COMPILER_GCC)
#       pragma GCC diagnostic ignored "-Wduplicated-branches"
#       pragma GCC diagnostic ignored "-Wuseless-cast"
#       pragma GCC diagnostic ignored "-Wclass-memaccess"
#   endif
#elif defined(NOA_COMPILER_MSVC)
#   pragma warning(push, 0)
#endif

#include <Eigen/Dense>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(pop)
#endif

namespace noa::cpu {
    template<typename T>
    void matmul(
        const T* lhs, const Strides4& lhs_strides, const Shape4& lhs_shape,
        const T* rhs, const Strides4& rhs_strides, const Shape4& rhs_shape,
        T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
        T* output, const Strides4& output_strides, const Shape4& output_shape,
        isize n_threads
    ) {
        auto [mnk, labc, are_column_major] = nd::extract_matmul_layout(
            lhs_strides, lhs_shape, rhs_strides, rhs_shape, output_strides, output_shape,
            lhs_transpose, rhs_transpose);
        auto sabc = Strides{lhs_strides[0], rhs_strides[0], output_strides[0]};

        // We use column major matrices in Eigen, so if the inputs are row-major, we need to compute B.T @ A.T = C.T
        if (not are_column_major) {
            std::swap(lhs_transpose, rhs_transpose);
            std::swap(mnk[0], mnk[1]);
            std::swap(lhs, rhs);
            std::swap(labc[0], labc[1]);
            std::swap(sabc[0], sabc[1]);
        }

        // Eigen doesn't support our complex types, but they have the same layout and alignment so reinterpret.
        using std_complex_t = std::complex<nt::value_type_t<T>>;
        using value_t = std::conditional_t<nt::complex<T>, std_complex_t, T>;
        auto* lhs_ = reinterpret_cast<const value_t*>(lhs);
        auto* rhs_ = reinterpret_cast<const value_t*>(rhs);
        auto* out_ = reinterpret_cast<value_t*>(output);
        auto cast_or_copy = [](T v) {
            if constexpr (nt::complex<T>)
                return std_complex_t{v[0], v[1]};
            else
                return v;
        };
        auto alpha_ = cast_or_copy(alpha);
        auto beta_ = cast_or_copy(beta);

        // TODO We cannot really guarantee the alignment, but we could check for it (as it is likely to be aligned)
        //      and compile for this case? Benchmarks would be required and this may end up taking too much time to
        //      compile.
        using matrix_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
        using strides_t = Eigen::Stride<Eigen::Dynamic, 1>;
        using imap_t = Eigen::Map<const matrix_t, Eigen::Unaligned, strides_t>;
        using omap_t = Eigen::Map<matrix_t, Eigen::Unaligned, strides_t>;

        Eigen::setNbThreads(static_cast<int>(n_threads));

        for (isize batch = 0; batch < output_shape[0]; ++batch) {
            imap_t lhs_matrix(lhs_ + sabc[0] * batch, mnk[0], mnk[2], strides_t(labc[0], 1));
            imap_t rhs_matrix(rhs_ + sabc[1] * batch, mnk[2], mnk[1], strides_t(labc[1], 1));
            omap_t out_matrix(out_ + sabc[2] * batch, mnk[0], mnk[1], strides_t(labc[2], 1));

            // FIXME Is there a better way to do this?
            if (beta == T{}) {
                if (lhs_transpose and rhs_transpose)
                    out_matrix.noalias() = (lhs_matrix.transpose() * rhs_matrix.transpose()) * alpha_;
                else if (lhs_transpose)
                    out_matrix.noalias() = (lhs_matrix.transpose() * rhs_matrix) * alpha_;
                else if (rhs_transpose)
                    out_matrix.noalias() = (lhs_matrix * rhs_matrix.transpose()) * alpha_;
                else
                    out_matrix.noalias() = (lhs_matrix * rhs_matrix) * alpha_;
            } else {
                out_matrix *= beta_;
                if (lhs_transpose and rhs_transpose)
                    out_matrix.noalias() += (lhs_matrix.transpose() * rhs_matrix.transpose()) * alpha_;
                else if (lhs_transpose)
                    out_matrix.noalias() += (lhs_matrix.transpose() * rhs_matrix) * alpha_;
                else if (rhs_transpose)
                    out_matrix.noalias() += (lhs_matrix * rhs_matrix.transpose()) * alpha_;
                else
                    out_matrix.noalias() += (lhs_matrix * rhs_matrix) * alpha_;
            }
        }
    }

    #define INSTANTIATE_GEMM_(T)                  \
    template void matmul<T>(                      \
        const T*, const Strides4&, const Shape4&, \
        const T*, const Strides4&, const Shape4&, \
        T, T, bool, bool,                         \
        T*, const Strides4&, const Shape4&,       \
        isize)

    INSTANTIATE_GEMM_(f32);
    INSTANTIATE_GEMM_(f64);
    INSTANTIATE_GEMM_(c32);
    INSTANTIATE_GEMM_(c64);
}
