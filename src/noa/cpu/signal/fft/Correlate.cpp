#include "noa/cpu/fft/Transforms.h"
#include "noa/cpu/math/Ewise.h"
#include "noa/cpu/signal/fft/Correlate.h"
#include "noa/cpu/signal/fft/Shift.h"

namespace noa::cpu::signal::fft {
    template<Remap REMAP, typename T, typename>
    void xmap(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_strides,
              const shared_t<Complex<T>[]>& rhs, dim4_t rhs_strides,
              const shared_t<T[]>& output, dim4_t output_strides,
              dim4_t shape, bool normalize, Norm norm, Stream& stream,
              const shared_t<Complex<T>[]>& tmp, dim4_t tmp_strides) {

        const shared_t<Complex<T>[]>& buffer = tmp ? tmp : rhs;
        const dim4_t& buffer_strides = tmp ? tmp_strides : rhs_strides;
        NOA_ASSERT(all(buffer_strides > 0));

        if (normalize) {
            cpu::math::ewise(
                    lhs, lhs_strides, rhs, rhs_strides, buffer, buffer_strides, shape.fft(),
                    [](Complex<T> l, Complex<T> r) {
                        const Complex<T> product = l * noa::math::conj(r);
                        const T magnitude = noa::math::abs(product);
                        return product / (magnitude + static_cast<T>(1e-13));
                        // The epsilon could be scaled by the max(abs(rhs)), but this seems to be useful only
                        // for input values close to zero (less than 1e-10). In most cases, this is fine.
                        // Note that the normalization can sharpen the peak considerably.
                    }, stream);
        } else {
            cpu::math::ewise(lhs, lhs_strides, rhs, rhs_strides, buffer, buffer_strides,
                             shape.fft(), noa::math::multiply_conj_t{}, stream);
        }

        if constexpr (REMAP == Remap::H2FC) {
            const dim3_t shape_3d(shape.get(1));
            if (shape_3d.ndim() == 3) {
                cpu::signal::fft::shift3D<Remap::H2H>(buffer, buffer_strides, buffer, buffer_strides, shape,
                                                      float3_t(shape_3d / 2), 1, stream);
            } else {
                cpu::signal::fft::shift2D<Remap::H2H>(buffer, buffer_strides, buffer, buffer_strides, shape,
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
        bool, Norm, Stream&, const shared_t<Complex<T>[]>&, dim4_t)

    #define INSTANTIATE_XMAP_ALL(T)     \
    INSTANTIATE_XMAP(Remap::H2F, T);    \
    INSTANTIATE_XMAP(Remap::H2FC, T)

    INSTANTIATE_XMAP_ALL(float);
    INSTANTIATE_XMAP_ALL(double);
}

namespace noa::cpu::signal::fft::details {
    template<typename T>
    T xcorr(const Complex<T>* lhs, dim3_t lhs_strides,
            const Complex<T>* rhs, dim3_t rhs_strides,
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
        return static_cast<T>(numerator / denominator);
    }

    #define INSTANTIATE_XCORR(T) \
    template T xcorr<T>(const Complex<T>*, dim3_t, const Complex<T>*, dim3_t, dim3_t, dim_t)

    INSTANTIATE_XCORR(float);
    INSTANTIATE_XCORR(double);
}
