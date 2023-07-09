#include "noa/gpu/cuda/fft/Transforms.hpp"
#include "noa/gpu/cuda/memory/AllocatorManaged.hpp"
#include "noa/gpu/cuda/signal/fft/Correlate.hpp"
#include "noa/gpu/cuda/signal/fft/PhaseShift.hpp"

#include "noa/gpu/cuda/utils/EwiseBinary.cuh"
#include "noa/gpu/cuda/utils/ReduceUnary.cuh"
#include "noa/gpu/cuda/utils/ReduceBinary.cuh"

namespace noa::cuda::signal::fft {
    template<Remap REMAP, typename Real, typename _>
    void xmap(
            const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
            Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
            Real* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, CorrelationMode correlation_mode, Norm norm,
            Complex<Real>* tmp, Strides4<i64> tmp_strides, Stream& stream
    ) {
        if (!tmp) {
            tmp = rhs;
            tmp_strides = rhs_strides;
        }
        NOA_ASSERT(noa::all(tmp_strides > 0));

        constexpr auto EPSILON = static_cast<Real>(1e-13);
        constexpr bool IS_HALF = noa::traits::to_underlying(REMAP) & noa::fft::Layout::SRC_HALF;
        const auto shape_fft = IS_HALF ? shape.rfft() : shape;

        // TODO Identify auto-correlation case?
        switch (correlation_mode) {
            case CorrelationMode::CONVENTIONAL:
                noa::cuda::utils::ewise_binary(
                        lhs, lhs_strides, rhs, rhs_strides, tmp, tmp_strides,
                        shape.rfft(), stream, noa::multiply_conj_t{});
                break;
            case CorrelationMode::PHASE:
                noa::cuda::utils::ewise_binary(
                        lhs, lhs_strides, rhs, rhs_strides,
                        tmp, tmp_strides, shape.rfft(), stream,
                        []__device__(const Complex<Real>& l, const Complex<Real>& r) {
                            const Complex<Real> product = l * noa::math::conj(r);
                            const Real magnitude = noa::math::abs(product);
                            return product / (magnitude + EPSILON);
                            // The epsilon could be scaled by the max(abs(rhs)), but this seems to be useful only
                            // for input values close to zero (less than 1e-10). In most cases, this is fine.
                        });
                break;
            case CorrelationMode::DOUBLE_PHASE:
                noa::cuda::utils::ewise_binary(
                        lhs, lhs_strides, rhs, rhs_strides,
                        tmp, tmp_strides, shape.rfft(), stream,
                        []__device__(const Complex<Real>& l, const Complex<Real>& r) -> Complex<Real> {
                            const Complex<Real> product = l * noa::math::conj(r);
                            const Complex<Real> product_sqd = {product.real * product.real, product.imag * product.imag};
                            const Real magnitude = noa::math::sqrt(product_sqd.real + product_sqd.imag) + EPSILON;
                            return {(product_sqd.real - product_sqd.imag) / magnitude,
                                    (2 * product.real * product.imag) / magnitude};
                        });
                break;
            case CorrelationMode::MUTUAL:
                noa::cuda::utils::ewise_binary(
                        lhs, lhs_strides, rhs, rhs_strides,
                        tmp, tmp_strides, shape.rfft(), stream,
                        []__device__(const Complex<Real>& l, const Complex<Real>& r) {
                            const Complex<Real> product = l * noa::math::conj(r);
                            const Real magnitude_sqrt = noa::math::sqrt(noa::math::abs(product));
                            return product / (magnitude_sqrt + EPSILON);
                        });
                break;
        }

        if constexpr (REMAP == Remap::H2FC) {
            const auto shape_3d = shape.pop_front();
            if (shape_3d.ndim() == 3) {
                noa::cuda::signal::fft::phase_shift_3d<Remap::H2H>(
                        tmp, tmp_strides, tmp, tmp_strides, shape,
                        (shape_3d / 2).vec().as<f32>(), 1, stream);
            } else {
                noa::cuda::signal::fft::phase_shift_2d<Remap::H2H>(
                        tmp, tmp_strides, tmp, tmp_strides, shape,
                        (shape_3d.pop_front() / 2).vec().as<f32>(), 1, stream);
            }
        }

        noa::cuda::fft::c2r(tmp, tmp_strides, output, output_strides, shape, norm, true, stream);
    }

    #define INSTANTIATE_XMAP(R, T)                      \
    template void xmap<R, T, void>(                     \
        const Complex<T>*, const Strides4<i64>&,        \
        Complex<T>*, const Strides4<i64>&,              \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        CorrelationMode, Norm, Complex<T>*, Strides4<i64>, Stream&)

    #define INSTANTIATE_XMAP_ALL(T)         \
    INSTANTIATE_XMAP(noa::fft::H2F, T);     \
    INSTANTIATE_XMAP(noa::fft::H2FC, T)

    INSTANTIATE_XMAP_ALL(f32);
    INSTANTIATE_XMAP_ALL(f64);
}

namespace noa::cuda::signal::fft::details {
    template<typename Real>
    void xcorr(
            const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
            const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
            const Shape4<i64>& shape, Real* coefficients, Stream& stream
    ) {
        const auto batches = shape[0];
        const auto buffer = noa::cuda::memory::AllocatorManaged<Real>::allocate(batches * 3, stream);
        const auto numerator = buffer.get();
        const auto denominator_lhs = buffer.get() + batches;
        const auto denominator_rhs = buffer.get() + batches * 2;

        noa::cuda::utils::reduce_unary(
                lhs, lhs_strides, shape,
                denominator_lhs, Strides1<i64>{1}, Real{0},
                noa::abs_squared_t{}, noa::plus_t{}, {},
                false, true, stream);
        noa::cuda::utils::reduce_unary(
                rhs, rhs_strides, shape,
                denominator_rhs, Strides1<i64>{1}, Real{0},
                noa::abs_squared_t{}, noa::plus_t{}, {},
                false, true, stream);

        auto preprocess_op = []__device__(const Complex<Real>& l, const Complex<Real>& r) {
            return noa::math::real(l * noa::math::conj(r));
        };
        noa::cuda::utils::reduce_binary(
                lhs, lhs_strides, rhs, rhs_strides, shape,
                numerator, Strides1<i64>{1}, Real{0},
                preprocess_op, noa::plus_t{}, {},
                false, true, stream);

        stream.synchronize(); // FIXME Add callback?
        for (i64 batch = 0; batch < batches; ++batch)
            coefficients[batch] = numerator[batch] / noa::math::sqrt(denominator_lhs[batch] * denominator_rhs[batch]);
    }

    template<typename Real>
    Real xcorr(
            const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
            const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
            const Shape4<i64>& shape, Stream& stream
    ) {
        NOA_ASSERT(shape[0] == 1);

        Real numerator{}, denominator_lhs{}, denominator_rhs{};
        noa::cuda::utils::reduce_unary(
                lhs, lhs_strides, shape,
                &denominator_lhs, Strides1<i64>{1}, Real{0},
                noa::abs_squared_t{}, noa::plus_t{}, {},
                true, true, stream);
        noa::cuda::utils::reduce_unary(
                rhs, rhs_strides, shape,
                &denominator_rhs, Strides1<i64>{1}, Real{0},
                noa::abs_squared_t{}, noa::plus_t{}, {},
                true, true, stream);

        auto preprocess_op = []__device__(const Complex<Real>& l, const Complex<Real>& r) {
            return noa::math::real(l * noa::math::conj(r));
        };
        noa::cuda::utils::reduce_binary(
                lhs, lhs_strides, rhs, rhs_strides, shape,
                &numerator, Strides1<i64>{1}, Real{0},
                preprocess_op, noa::plus_t{}, {},
                true, true, stream);

        stream.synchronize();
        return numerator / noa::math::sqrt(denominator_lhs * denominator_rhs);
    }

    #define INSTANTIATE_XCORR(T)                    \
    template void xcorr<T>(                         \
        const Complex<T>*, const Strides4<i64>&,    \
        const Complex<T>*, const Strides4<i64>&,    \
        const Shape4<i64>&, T*, Stream&);           \
    template T xcorr<T>(                            \
        const Complex<T>*, const Strides4<i64>&,    \
        const Complex<T>*, const Strides4<i64>&,    \
        const Shape4<i64>&, Stream&)

    INSTANTIATE_XCORR(f32);
    INSTANTIATE_XCORR(f64);
}
