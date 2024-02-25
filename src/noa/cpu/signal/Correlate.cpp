#include "noa/cpu/fft/Transforms.hpp"
#include "noa/cpu/signal/fft/Correlate.hpp"
#include "noa/cpu/signal/fft/PhaseShift.hpp"
#include "noa/cpu/utils/EwiseBinary.hpp"

namespace noa::cpu::signal::fft {
    template<Remap REMAP, typename Real, typename>
    void xmap(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
              Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
              Real* output, const Strides4<i64>& output_strides,
              const Shape4<i64>& shape, CorrelationMode correlation_mode, Norm norm,
              Complex<Real>* tmp, Strides4<i64> tmp_strides, i64 threads) {
        if (!tmp) {
            tmp = rhs;
            tmp_strides = rhs_strides;
        }
        NOA_ASSERT(noa::all(tmp_strides > 0));

        constexpr auto EPSILON = static_cast<Real>(1e-13);
        constexpr bool IS_HALF = noa::to_underlying(REMAP) & noa::fft::Layout::SRC_HALF;
        const auto shape_fft = IS_HALF ? shape.rfft() : shape;

        // TODO Identify auto-correlation case?
        switch (correlation_mode) {
            case CorrelationMode::CONVENTIONAL:
                noa::cpu::utils::ewise_binary(
                        lhs, lhs_strides, rhs, rhs_strides, tmp, tmp_strides,
                        shape.rfft(), noa::multiply_conj_t{}, threads);
                break;
            case CorrelationMode::PHASE:
                noa::cpu::utils::ewise_binary(
                        lhs, lhs_strides, rhs, rhs_strides, tmp, tmp_strides, shape.rfft(),
                        [](const Complex<Real>& l, const Complex<Real>& r) {
                            const Complex<Real> product = l * noa::math::conj(r);
                            const Real magnitude = noa::math::abs(product);
                            return product / (magnitude + EPSILON);
                            // The epsilon could be scaled by the max(abs(rhs)), but this seems to be useful only
                            // for input values close to zero (less than 1e-10). In most cases, this is fine.
                        }, threads);
                break;
            case CorrelationMode::DOUBLE_PHASE:
                noa::cpu::utils::ewise_binary(
                        lhs, lhs_strides, rhs, rhs_strides, tmp, tmp_strides, shape.rfft(),
                        [](const Complex<Real>& l, const Complex<Real>& r) -> Complex<Real> {
                            const Complex<Real> product = l * noa::math::conj(r);
                            const Complex<Real> product_sqd = {product.real * product.real, product.imag * product.imag};
                            const Real magnitude = noa::math::sqrt(product_sqd.real + product_sqd.imag) + EPSILON;
                            return {(product_sqd.real - product_sqd.imag) / magnitude,
                                    (2 * product.real * product.imag) / magnitude};
                        }, threads);
                break;
            case CorrelationMode::MUTUAL:
                noa::cpu::utils::ewise_binary(
                        lhs, lhs_strides, rhs, rhs_strides, tmp, tmp_strides, shape_fft,
                        [](const Complex<Real>& l, const Complex<Real>& r) {
                            const Complex<Real> product = l * noa::math::conj(r);
                            const Real magnitude_sqrt = noa::math::sqrt(noa::math::abs(product));
                            return product / (magnitude_sqrt + EPSILON);
                        }, threads);
                break;
        }

        if constexpr (REMAP == Remap::H2FC) {
            const auto shape_3d = shape.pop_front();
            if (shape_3d.ndim() == 3) {
                noa::cpu::signal::fft::phase_shift_3d<Remap::H2H>(
                        tmp, tmp_strides, tmp, tmp_strides, shape,
                        (shape_3d / 2).vec().as<f32>(), 1, threads);
            } else {
                noa::cpu::signal::fft::phase_shift_2d<Remap::H2H>(
                        tmp, tmp_strides, tmp, tmp_strides, shape,
                        (shape_3d.pop_front() / 2).vec().as<f32>(), 1, threads);
            }
        }

        noa::cpu::fft::c2r(
                tmp, tmp_strides, output, output_strides,
                shape, cpu::fft::ESTIMATE, norm, threads);
    }

    #define INSTANTIATE_XMAP(R, T)                      \
    template void xmap<R, T, void>(                     \
        const Complex<T>*, const Strides4<i64>&,        \
        Complex<T>*, const Strides4<i64>&,              \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        CorrelationMode, Norm, Complex<T>*, Strides4<i64>, i64)

    #define INSTANTIATE_XMAP_ALL(T)         \
    INSTANTIATE_XMAP(noa::fft::H2F, T);     \
    INSTANTIATE_XMAP(noa::fft::H2FC, T)

    INSTANTIATE_XMAP_ALL(f32);
    INSTANTIATE_XMAP_ALL(f64);
}

namespace noa::cpu::signal::fft::details {
    // TODO We could use the same approach as in CUDA, with 3 separate reductions,
    //      but this approach is likely to be more efficient here. Benchmark?
    template<typename Real>
    Real xcorr(const Complex<Real>* lhs, const Strides3<i64>& lhs_strides,
               const Complex<Real>* rhs, const Strides3<i64>& rhs_strides,
               const Shape3<i64>& shape, i64 threads) {

        auto accurate_sum = [](f64 value, f64& s, f64& e) {
            auto t = s + value;
            e += noa::math::abs(s) >= noa::math::abs(value) ? (s - t) + value : (value - t) + s;
            s = t;
        };

        const auto lhs_accessor = Accessor<const Complex<Real>, 3, i64>(lhs, lhs_strides);
        const auto rhs_accessor = Accessor<const Complex<Real>, 3, i64>(rhs, rhs_strides);

        f64 sum = 0, sum_lhs = 0, sum_rhs = 0;
        f64 err = 0, err_lhs = 0, err_rhs = 0;

        #pragma omp parallel for collapse(3) num_threads(threads) if(shape.elements() > 1'048'576)  \
        reduction(+:sum, sum_lhs, sum_rhs, err, err_lhs, err_rhs) default(none)                     \
        shared(lhs_accessor, rhs_accessor, shape, accurate_sum)

        for (i64 j = 0; j < shape[0]; ++j) {
            for (i64 k = 0; k < shape[1]; ++k) {
                for (i64 l = 0; l < shape[2]; ++l) {

                    const auto lhs_value = static_cast<c64>(lhs_accessor(j, k, l));
                    const auto rhs_value = static_cast<c64>(rhs_accessor(j, k, l));

                    accurate_sum(noa::abs_squared_t{}(lhs_value), sum_lhs, err_lhs);
                    accurate_sum(noa::abs_squared_t{}(rhs_value), sum_rhs, err_rhs);
                    accurate_sum(noa::math::real(lhs_value * noa::math::conj(rhs_value)), sum, err);
                }
            }
        }

        const f64 numerator = sum + err;
        const f64 denominator = noa::math::sqrt((sum_lhs + err_lhs) * (sum_rhs + err_rhs));
        return static_cast<Real>(numerator / denominator);
    }

    #define INSTANTIATE_XCORR(T)                    \
    template T xcorr<T>(                            \
        const Complex<T>*, const Strides3<i64>&,    \
        const Complex<T>*, const Strides3<i64>&,    \
        const Shape3<i64>&, i64)

    INSTANTIATE_XCORR(f32);
    INSTANTIATE_XCORR(f64);
}
