#include "noa/cpu/fft/Transforms.h"
#include "noa/cpu/math/Ewise.h"
#include "noa/cpu/signal/fft/Correlate.h"
#include "noa/cpu/signal/fft/Shift.h"

namespace noa::cpu::signal::fft {
    template<Remap REMAP, typename Real, typename>
    void xmap(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
              const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
              const shared_t<Real[]>& output, dim4_t output_strides,
              dim4_t shape, CorrelationMode correlation_mode, Norm norm, Stream& stream,
              const shared_t<Complex<Real>[]>& tmp, dim4_t tmp_strides) {

        using complex_t = Complex<Real>;
        using real_t = Real;
        const shared_t<complex_t[]>& buffer = tmp ? tmp : rhs;
        const dim4_t& buffer_strides = tmp ? tmp_strides : rhs_strides;
        NOA_ASSERT(all(buffer_strides > 0));

        constexpr auto EPSILON = static_cast<real_t>(1e-13);
        switch (correlation_mode) {
            case noa::signal::CONVENTIONAL_CORRELATION:
                cpu::math::ewise(lhs, lhs_strides, rhs, rhs_strides, buffer, buffer_strides,
                                 shape.fft(), noa::math::multiply_conj_t{}, stream);
                break;
            case noa::signal::PHASE_CORRELATION:
                cpu::math::ewise(
                        lhs, lhs_strides, rhs, rhs_strides, buffer, buffer_strides, shape.fft(),
                        [](complex_t l, complex_t r) {
                            const complex_t product = l * noa::math::conj(r);
                            const real_t magnitude = noa::math::abs(product);
                            return product / (magnitude + EPSILON);
                            // The epsilon could be scaled by the max(abs(rhs)), but this seems to be useful only
                            // for input values close to zero (less than 1e-10). In most cases, this is fine.
                        }, stream);
                break;
            case noa::signal::DOUBLE_PHASE_CORRELATION:
                cpu::math::ewise(
                        lhs, lhs_strides, rhs, rhs_strides, buffer, buffer_strides, shape.fft(),
                        [](complex_t l, complex_t r) -> complex_t {
                            const complex_t product = l * noa::math::conj(r);
                            const complex_t product_sqd = {product.real * product.real, product.imag * product.imag};
                            const real_t magnitude = noa::math::sqrt(product_sqd.real + product_sqd.imag) + EPSILON;
                            return {(product_sqd.real - product_sqd.imag) / magnitude,
                                    (2 * product.real * product.imag) / magnitude};
                        }, stream);
                break;
            case noa::signal::MUTUAL_CORRELATION:
                cpu::math::ewise(
                        lhs, lhs_strides, rhs, rhs_strides, buffer, buffer_strides, shape.fft(),
                        [](complex_t l, complex_t r) {
                            const complex_t product = l * noa::math::conj(r);
                            const real_t magnitude_sqrt = noa::math::sqrt(noa::math::abs(product));
                            return product / (magnitude_sqrt + EPSILON);
                        }, stream);
                break;
        }

        if constexpr (REMAP == Remap::H2FC) {
            const dim3_t shape_3d(shape.get(1));
            if (shape_3d.ndim() == 3) {
                cpu::signal::fft::shift3D<Remap::H2H>(
                        buffer, buffer_strides, buffer, buffer_strides, shape,
                        float3_t(shape_3d / 2), 1, stream);
            } else {
                cpu::signal::fft::shift2D<Remap::H2H>(
                        buffer, buffer_strides, buffer, buffer_strides, shape,
                        float2_t{shape_3d[1] / 2, shape_3d[2] / 2}, 1, stream);
            }
        }

        cpu::fft::c2r(buffer, buffer_strides, output, output_strides, shape, cpu::fft::ESTIMATE, norm, stream);
    }

    #define INSTANTIATE_XMAP(R, T)              \
    template void xmap<R, T, void>(             \
        const shared_t<Complex<T>[]>&, dim4_t,  \
        const shared_t<Complex<T>[]>&, dim4_t,  \
        const shared_t<T[]>&, dim4_t, dim4_t,   \
        CorrelationMode, Norm, Stream&,         \
        const shared_t<Complex<T>[]>&, dim4_t)

    #define INSTANTIATE_XMAP_ALL(T)     \
    INSTANTIATE_XMAP(Remap::H2F, T);    \
    INSTANTIATE_XMAP(Remap::H2FC, T)

    INSTANTIATE_XMAP_ALL(float);
    INSTANTIATE_XMAP_ALL(double);
}

namespace noa::cpu::signal::fft::details {
    template<typename Real>
    Real xcorr(const Complex<Real>* lhs, dim3_t lhs_strides,
               const Complex<Real>* rhs, dim3_t rhs_strides,
               dim3_t shape, dim_t threads) {
        double sum = 0, sum_lhs = 0, sum_rhs = 0;
        double err = 0, err_lhs = 0, err_rhs = 0;

        auto abs_sqd = noa::math::abs_squared_t{};
        auto accurate_sum = [](double value, double& s, double& e) {
            auto t = s + value;
            e += noa::math::abs(s) >= noa::math::abs(value) ? (s - t) + value : (value - t) + s;
            s = t;
        };

        #pragma omp parallel for collapse(3) num_threads(threads) if(shape.elements() > 1048576)    \
        reduction(+:sum, sum_lhs, sum_rhs, err, err_lhs, err_rhs) default(none)                     \
        shared(lhs, lhs_strides, rhs, rhs_strides, shape, abs_sqd, accurate_sum)

        for (dim_t j = 0; j < shape[0]; ++j) {
            for (dim_t k = 0; k < shape[1]; ++k) {
                for (dim_t l = 0; l < shape[2]; ++l) {

                    const auto lhs_value = static_cast<cdouble_t>(lhs[indexing::at(j, k, l, lhs_strides)]);
                    const auto rhs_value = static_cast<cdouble_t>(rhs[indexing::at(j, k, l, rhs_strides)]);

                    accurate_sum(abs_sqd(lhs_value), sum_lhs, err_lhs);
                    accurate_sum(abs_sqd(rhs_value), sum_rhs, err_rhs);
                    accurate_sum(noa::math::real(lhs_value * noa::math::conj(rhs_value)), sum, err);
                }
            }
        }

        const double numerator = sum + err;
        const double denominator = noa::math::sqrt((sum_lhs + err_lhs) * (sum_rhs + err_rhs));
        return static_cast<Real>(numerator / denominator);
    }

    #define INSTANTIATE_XCORR(T) \
    template T xcorr<T>(const Complex<T>*, dim3_t, const Complex<T>*, dim3_t, dim3_t, dim_t)

    INSTANTIATE_XCORR(float);
    INSTANTIATE_XCORR(double);
}
