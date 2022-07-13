#include "noa/common/Definitions.h"

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wuseless-cast"
    #pragma GCC diagnostic ignored "-Wduplicated-branches"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

#ifdef NOA_ENABLE_BLAS
    #define EIGEN_USE_BLAS
#endif
#include <Eigen/Dense>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(pop)
#endif

#include "noa/cpu/math/Blas.h"
#include "noa/cpu/memory/PtrHost.h"

namespace {
    using namespace ::noa;

    // Extract size and stride from a column or row vector.
    template<typename T>
    std::pair<T, T> vectorDim_(size2_t stride, size2_t shape) {
        const bool is_column = shape.ndim() == 2;
        NOA_ASSERT(shape[is_column] == 1); // require vector
        const auto n = static_cast<T>(shape[1 - is_column]);
        const auto s = static_cast<T>(stride[1 - is_column]);
        return {n, s};
    }

    // Returns alignment. Only check for the most likely alignment since we don't necessarily want
    //  to instantiate for every possibility.
    template<typename T>
    bool isAligned_(const T* pointer) {
        const auto address = reinterpret_cast<uint64_t>(pointer);
        return !(address % Eigen::AlignedMax);
    }
}

namespace noa::cpu::math {
    template<typename T, typename>
    T dot(const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
          const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
          Stream& stream) {

        // Get vector shape:
        NOA_ASSERT(lhs_shape.ndim() <= 2 && rhs_shape.ndim() <= 2);
        auto[lhs_n, lhs_s] = vectorDim_<Eigen::Index>(size2_t{lhs_stride.get(2)}, size2_t{lhs_shape.get(2)});
        auto[rhs_n, rhs_s] = vectorDim_<Eigen::Index>(size2_t{rhs_stride.get(2)}, size2_t{rhs_shape.get(2)});
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        using namespace Eigen;
        using e_sca_t = std::conditional_t<traits::is_complex_v<T>, std::complex<traits::value_type_t<T>>, T>;
        const auto* lhs_ptr = reinterpret_cast<const e_sca_t*>(lhs.get());
        const auto* rhs_ptr = reinterpret_cast<const e_sca_t*>(rhs.get());

        // If contiguous and aligned, vectorize if possible:
        if (lhs_s == 1 && rhs_s == 1 && isAligned_(lhs_ptr) && isAligned_(rhs_ptr)) {
            using e_lhs_t = Map<const Matrix<e_sca_t, 1, Dynamic>, AlignedMax>;
            using e_rhs_t = Map<const Matrix<e_sca_t, Dynamic, 1>, AlignedMax>;
            e_lhs_t lhs_(lhs_ptr, lhs_n);
            e_rhs_t rhs_(rhs_ptr, rhs_n);
            stream.synchronize();
            return static_cast<T>((lhs_ * rhs_)(0));
        } else {
            using e_lhs_t = Map<const Matrix<e_sca_t, 1, Dynamic>, Unaligned, InnerStride<>>;
            using e_rhs_t = Map<const Matrix<e_sca_t, Dynamic, 1>, Unaligned, InnerStride<>>;
            e_lhs_t lhs_(lhs_ptr, lhs_n, InnerStride<>(lhs_s));
            e_rhs_t rhs_(rhs_ptr, rhs_n, InnerStride<>(rhs_s));
            stream.synchronize();
            return static_cast<T>((lhs_ * rhs_)(0));
        }
    }

    template<typename T, typename>
    void dot(const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
             const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
             const std::shared_ptr<T[]>& output, Stream& stream) {
        NOA_ASSERT(lhs_shape[0] == rhs_shape[0] && lhs_shape[1] == 1 && rhs_shape[1] == 1);

        // Get vector shape: lhs should be a row vector, rhs can be a column or row vector
        Eigen::Index lhs_n, lhs_s, rhs_n, rhs_s;
        std::tie(lhs_n, lhs_s) = vectorDim_<Eigen::Index>(size2_t{lhs_stride.get(2)}, size2_t{lhs_shape.get(2)});
        std::tie(rhs_n, rhs_s) = vectorDim_<Eigen::Index>(size2_t{rhs_stride.get(2)}, size2_t{rhs_shape.get(2)});
        NOA_ASSERT(lhs_n == rhs_n);
        (void) rhs_n;

        const size_t lhs_bs = lhs_stride[0];
        const size_t rhs_bs = rhs_stride[0];
        const size_t batches = lhs_shape[0];
        stream.enqueue([=]() {
            using namespace Eigen;
            using e_sca_t = std::conditional_t<traits::is_complex_v<T>, std::complex<traits::value_type_t<T>>, T>;
            const auto* lhs_ptr = reinterpret_cast<const e_sca_t*>(lhs.get());
            const auto* rhs_ptr = reinterpret_cast<const e_sca_t*>(rhs.get());

            // If contiguous and all batches are aligned, vectorize if possible:
            if (lhs_s == 1 && rhs_s == 1 && isAligned_(lhs_ptr) && isAligned_(rhs_ptr) &&
                (batches == 0 || (!(lhs_bs % AlignedMax) && !(rhs_bs % AlignedMax)))) {
                using e_row_t = Map<const Matrix<e_sca_t, 1, Dynamic>, AlignedMax>;
                using e_col_t = Map<const Matrix<e_sca_t, Dynamic, 1>, AlignedMax>;
                e_row_t lhs_(nullptr, lhs_n);
                e_col_t rhs_(nullptr, rhs_n);
                T* out_ptr = output.get();
                for (size_t batch = 0; batch < batches; ++batch) {
                    new (&lhs_) e_row_t(lhs_ptr + lhs_bs * batch, lhs_n);
                    new (&rhs_) e_col_t(rhs_ptr + rhs_bs * batch, rhs_n);
                    auto a = lhs_.dot(rhs_);
                    out_ptr[batch] = static_cast<T>((lhs_ * rhs_)(0));
                }
            } else {
                using e_row_t = Map<const Matrix<e_sca_t, 1, Dynamic>, Unaligned, InnerStride<>>;
                using e_col_t = Map<const Matrix<e_sca_t, Dynamic, 1>, Unaligned, InnerStride<>>;
                e_row_t lhs_(nullptr, lhs_n, InnerStride<>(lhs_s));
                e_col_t rhs_(nullptr, rhs_n, InnerStride<>(rhs_s));
                T* out_ptr = output.get();
                for (size_t batch = 0; batch < batches; ++batch) {
                    new (&lhs_) e_row_t(lhs_ptr + lhs_bs * batch, lhs_n, InnerStride<>(lhs_s));
                    new (&rhs_) e_col_t(rhs_ptr + rhs_bs * batch, rhs_n, InnerStride<>(rhs_s));
                    out_ptr[batch] = static_cast<T>((lhs_ * rhs_)(0));
                }
            }
        });
    }

    #define INSTANTIATE_DOT_(T)                                                         \
    template T dot<T, void>(const std::shared_ptr<T[]>&, size4_t, size4_t,              \
                            const std::shared_ptr<T[]>&, size4_t, size4_t, Stream&);    \
    template void dot<T, void>(const std::shared_ptr<T[]>&, size4_t, size4_t,           \
                               const std::shared_ptr<T[]>&, size4_t, size4_t,           \
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
    using BlasTranspose = noa::math::BlasTranspose;

    template<typename T, typename>
    void matmul(const std::shared_ptr<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape,
                const std::shared_ptr<T[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
                const std::shared_ptr<T[]>& output, size4_t output_stride, size4_t output_shape,
                Stream& stream) {
        // Get the shape: MxK @ KxN = MxN
        using eint3_t = Int3<Eigen::Index>;
        using eint2_t = Int2<Eigen::Index>;
        const eint3_t mnk{lhs_shape[2], rhs_shape[3], lhs_shape[3]};
        NOA_ASSERT(output.get() != lhs.get() && output.get() != rhs.get());
        NOA_ASSERT(lhs_shape[1] == 1 && rhs_shape[1] == 1 && output_shape[1] == 1); // 2D matrices
        NOA_ASSERT(mnk[0] == output_shape[2] && mnk[1] == output_shape[3]); // output fits the expected shape
        NOA_ASSERT(mnk[2] == rhs_shape[2]); // left and right matrices have compatible shape
        (void) output_shape;

        // Dot product is faster:
        if (mnk[0] == 1 && mnk[0] == 1 && indexing::isContiguous(output_stride, output_shape)[0])
            return dot(lhs, lhs_stride, lhs_shape, rhs, rhs_stride, rhs_shape, output, stream);

        const eint2_t lhs_s{lhs_stride[2], lhs_stride[3]};
        const eint2_t rhs_s{rhs_stride[2], rhs_stride[3]};
        const eint2_t out_s{output_stride[2], output_stride[3]};
        const size_t batches = lhs_shape[0];
        const size3_t batch_s{lhs_stride[0], rhs_stride[0], output_stride[0]};
        stream.enqueue([=]() {
            using namespace Eigen;
            using e_sca_t = std::conditional_t<traits::is_complex_v<T>, std::complex<traits::value_type_t<T>>, T>;
            const auto* lhs_ptr = reinterpret_cast<const e_sca_t*>(lhs.get());
            const auto* rhs_ptr = reinterpret_cast<const e_sca_t*>(rhs.get());
            auto* out_ptr = reinterpret_cast<e_sca_t*>(output.get());

            // If contiguous and all batches are aligned, vectorize if possible:
            if (isAligned_(lhs_ptr) && isAligned_(rhs_ptr) && isAligned_(out_ptr) &&
                all(indexing::isContiguous(lhs_s, eint2_t{mnk[0], mnk[2]})) &&
                all(indexing::isContiguous(rhs_s, eint2_t{mnk[2], mnk[1]})) &&
                all(indexing::isContiguous(out_s, eint2_t{mnk[0], mnk[1]})) &&
                (batches == 0 || all((batch_s % AlignedMax) == 0))) {
                using e_imat_t = Map<const Matrix<e_sca_t, Dynamic, Dynamic, RowMajor>, AlignedMax>;
                using e_omat_t = Map<Matrix<e_sca_t, Dynamic, Dynamic, RowMajor>, AlignedMax>;
                e_imat_t lhs_(nullptr, mnk[0], mnk[2]);
                e_imat_t rhs_(nullptr, mnk[2], mnk[1]);
                e_omat_t out_(nullptr, mnk[0], mnk[1]);

                for (size_t batch = 0; batch < batches; ++batch) {
                    new(&lhs_) e_imat_t(lhs_ptr + batch_s[0] * batch, mnk[0], mnk[2]);
                    new(&rhs_) e_imat_t(rhs_ptr + batch_s[1] * batch, mnk[2], mnk[1]);
                    new(&out_) e_omat_t(out_ptr + batch_s[2] * batch, mnk[0], mnk[1]);
                    out_.noalias() = lhs_ * rhs_;
                }
            } else {
                using e_stride_t = Stride<Dynamic, Dynamic>;
                using e_imat_t = Map<const Matrix<e_sca_t, Dynamic, Dynamic, RowMajor>, Unaligned, e_stride_t>;
                using e_omat_t = Map<Matrix<e_sca_t, Dynamic, Dynamic, RowMajor>, Unaligned, e_stride_t>;
                e_imat_t lhs_(nullptr, mnk[0], mnk[2], e_stride_t(lhs_s[0], lhs_s[1]));
                e_imat_t rhs_(nullptr, mnk[2], mnk[1], e_stride_t(rhs_s[0], rhs_s[1]));
                e_omat_t out_(nullptr, mnk[1], mnk[1], e_stride_t(out_s[0], out_s[1]));

                for (size_t batch = 0; batch < batches; ++batch) {
                    new(&lhs_) e_imat_t(lhs_ptr + batch_s[0] * batch, mnk[0], mnk[2], e_stride_t(lhs_s[0], lhs_s[1]));
                    new(&rhs_) e_imat_t(rhs_ptr + batch_s[1] * batch, mnk[2], mnk[1], e_stride_t(rhs_s[0], rhs_s[1]));
                    new(&out_) e_omat_t(out_ptr + batch_s[2] * batch, mnk[0], mnk[1], e_stride_t(out_s[0], out_s[1]));
                    out_.noalias() = lhs_ * rhs_;
                }
            }
        });
    }

    #define INSTANTIATE_BLAS_(T)                                                            \
    template void matmul<T, void>(const std::shared_ptr<T[]>&, size4_t, size4_t,            \
                                  const std::shared_ptr<T[]>&, size4_t, size4_t,            \
                                  const std::shared_ptr<T[]>&, size4_t, size4_t, Stream&)

    INSTANTIATE_BLAS_(float);
    INSTANTIATE_BLAS_(double);
    INSTANTIATE_BLAS_(cfloat_t);
    INSTANTIATE_BLAS_(cdouble_t);
}
