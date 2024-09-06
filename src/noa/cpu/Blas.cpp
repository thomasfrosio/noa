// Eigen raises warnings...
#include "noa/core/Config.hpp"
#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wnull-dereference"
    #if defined(NOA_COMPILER_GCC)
    #pragma GCC diagnostic ignored "-Wduplicated-branches"
    #pragma GCC diagnostic ignored "-Wuseless-cast"
    #endif
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(push, 0)
#endif

#include <Eigen/Dense>
#include "noa/core/indexing/Layout.hpp"
#include "noa/cpu/Blas.hpp"

namespace noa::cpu {
    template<typename T>
    void matmul(
        const T* lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& lhs_shape,
        const T* rhs, const Strides4<i64>& rhs_strides, const Shape4<i64>& rhs_shape,
        T alpha, T beta, bool lhs_transpose, bool rhs_transpose,
        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
        i64 n_threads
    ) {
        const auto [mnk, secondmost_strides, are_column_major] = ni::extract_matmul_layout(
                lhs_strides, lhs_shape, rhs_strides, rhs_shape, output_strides, output_shape,
                lhs_transpose, rhs_transpose);
        const auto& [m, n, k] = mnk;
        const auto& [lhs_s, rhs_s, out_s] = secondmost_strides;

        Eigen::setNbThreads(static_cast<int>(n_threads));

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
        using matrix_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using strides_t = Eigen::Stride<Eigen::Dynamic, 1>;
        using imap_t = Eigen::Map<const matrix_t, Eigen::Unaligned, strides_t>;
        using omap_t = Eigen::Map<matrix_t, Eigen::Unaligned, strides_t>;

        for (i64 batch = 0; batch < output_shape[0]; ++batch) {
            imap_t lhs_matrix(lhs_ + lhs_strides[0]    * batch, m, k, strides_t(lhs_s, 1));
            imap_t rhs_matrix(rhs_ + rhs_strides[0]    * batch, k, n, strides_t(rhs_s, 1));
            omap_t out_matrix(out_ + output_strides[0] * batch, m, n, strides_t(out_s, 1));

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

    #define INSTANTIATE_GEMM_(T)                            \
    template void matmul<T>(                                \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        T, T, bool, bool,                                   \
        T*, const Strides4<i64>&, const Shape4<i64>&,       \
        i64)

    INSTANTIATE_GEMM_(f32);
    INSTANTIATE_GEMM_(f64);
    INSTANTIATE_GEMM_(c32);
    INSTANTIATE_GEMM_(c64);
}

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(pop)
#endif
