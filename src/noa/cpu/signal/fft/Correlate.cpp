#include "noa/cpu/fft/Transforms.h"
#include "noa/cpu/math/Ewise.h"
#include "noa/cpu/math/Find.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/signal/fft/Correlate.h"
#include "noa/cpu/signal/fft/Shift.h"

namespace {
    using namespace ::noa;
    using long1_t = std::array<int64_t, 1>;
    using float1_t = std::array<float, 1>;

    // From the DC-centered frequency to a valid index in the non-centered output.
    // The input frequency should be in-bound, i.e. -n/2 <= frequency <= (n-1)/2
    inline constexpr int64_t getIndex_(int64_t frequency, int64_t volume_dim) noexcept {
        return frequency < 0 ? volume_dim + frequency : frequency;
    }

    // From a valid index to the DC-centered frequency.
    inline constexpr long3_t getFrequency_(long3_t index, long3_t shape) noexcept {
        return {index[0] < (shape[0] + 1) / 2 ? index[0] : index[0] - shape[0],
                index[1] < (shape[1] + 1) / 2 ? index[1] : index[1] - shape[1],
                index[2] < (shape[2] + 1) / 2 ? index[2] : index[2] - shape[2]};
    }

    // From a valid index to the DC-centered frequency.
    inline constexpr long2_t getFrequency_(long2_t index, long2_t shape) noexcept {
        return {index[0] < (shape[0] + 1) / 2 ? index[0] : index[0] - shape[0],
                index[1] < (shape[1] + 1) / 2 ? index[1] : index[1] - shape[1]};
    }

    inline constexpr long1_t getFrequency_(long1_t index, long1_t shape) noexcept {
        return long1_t{index[0] < (shape[0] + 1) / 2 ? index[0] : index[0] - shape[0]};
    }

    // Given values at three successive positions, y[0], y[1], y[2], where
    // y[1] is the peak value, this fits a parabola to the values and returns the
    // offset (from -0.5 to 0.5) from the center position.
    template<typename T>
    inline constexpr T getParabolicVertex_(T* y) noexcept {
        // From IMOD/libcfshr/filtxcorr.c::parabolicFitPosition
        const T d = 2 * (y[0] + y[2] - 2 * y[1]);
        T x = 0;
        if (math::abs(d) > math::abs(static_cast<T>(1e-2) * (y[0] - y[2])))
            x = (y[0] - y[2]) / d;
        if (x > T{0.5})
            x = T{0.5};
        if (x < T{-0.5})
            x = T{-0.5};
        return x;
    }

    template<dim_t NDIM, bool IS_CENTERED, typename T, typename U>
    inline constexpr auto getSinglePeak(const T* input, U strides, U shape, U peak) {
        static_assert((NDIM == 1 && traits::is_almost_same_v<U, long1_t>) ||
                      (NDIM == 2 && traits::is_almost_same_v<U, long2_t>) ||
                      (NDIM == 3 && traits::is_almost_same_v<U, long3_t>));
        constexpr dim_t ELEMENTS = 3 * NDIM;

        std::array<T, ELEMENTS> output{0};
        T* output_ptr = output.data();

        if constexpr (!IS_CENTERED) {
            const U peak_ = getFrequency_(peak, shape);
            for (dim_t dim = 0; dim < NDIM; ++dim) {
                const T* input_ = input;

                // Add peak offset in the other dimensions.
                for (dim_t i = 0; i < NDIM; ++i)
                    input_ += peak[i] * strides[i] * (dim != i);

                // Here the 2 adjacent points might be in a separate quadrant.
                // Retrieve the frequency, add the offset, and if it's a valid
                // frequency, convert back to an index.
                for (int l = -1; l < 2; ++l, ++output_ptr) {
                    const int64_t il = peak_[dim] + l;
                    if (-shape[dim] / 2 <= il && il <= (shape[dim] - 1) / 2)
                        *output_ptr = input_[getIndex_(il, shape[dim]) * strides[dim]];
                }
            }
            // The CC map was not centered, so center it here.
            for (dim_t dim = 0; dim < NDIM; ++dim)
                peak[dim] = math::FFTShift(peak[dim], shape[dim]);

        } else {
            if constexpr (NDIM == 1)
                input += peak[0] * strides[0];
            else
                input += indexing::at(peak, strides);
            for (dim_t dim = 0; dim < NDIM; ++dim) {
                for (int l = -1; l < 2; ++l, ++output_ptr) {
                    const int64_t il = peak[dim] + l;
                    if (il >= 0 && il < shape[dim])
                        *output_ptr = input[l * strides[dim]];
                }
            }
        }

        // Add sub-pixel position by fitting a 1D parabola to the peak and its 2 adjacent points.
        using peak_type = std::conditional_t<NDIM == 1, float1_t, std::conditional_t<NDIM == 2, float2_t, float3_t>>;
        peak_type final_peak{0};
        for (dim_t dim = 0; dim < NDIM; ++dim)
            final_peak[dim] = static_cast<float>(peak[dim]) +
                              static_cast<float>(getParabolicVertex_(output.data() + 3 * dim));
        return final_peak;
    }
}

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

    template<Remap REMAP, typename T, typename>
    void xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                 const shared_t<float[]>& peaks, Stream& stream) {
        NOA_ASSERT(dim3_t(shape.get(1)).ndim() == 1);
        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        stream.enqueue([=]() mutable {
            cpu::memory::PtrHost<int64_t> offsets(shape[0]);
            math::find(noa::math::first_max_t{}, xmap, strides, shape, offsets.share(), true, true, stream);

            const bool is_column = shape[3] == 1;
            NOA_ASSERT(strides[3 - is_column] > 0);
            const long1_t shape_1d{safe_cast<int64_t>(shape[3 - is_column])};
            const long1_t strides_1d{safe_cast<int64_t>(strides[3 - is_column])};
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const long1_t peak{offsets[batch] / strides_1d[0]};
                const T* imap = xmap.get() + strides[0] * batch;
                peaks.get()[batch] = getSinglePeak<1, IS_CENTERED>(imap, strides_1d, shape_1d, peak)[0];
            }
        });
    }

    template<Remap REMAP, typename T, typename>
    float xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, Stream& stream) {
        const bool is_column = shape[3] == 1;
        NOA_ASSERT(shape.ndim() == 1);
        NOA_ASSERT(strides[3 - is_column] > 0);
        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        const auto offset = math::find<int64_t>(noa::math::first_max_t{}, xmap, strides, shape, true, stream);

        const long1_t peak{offset / safe_cast<int64_t>(strides[3 - is_column])};
        const long1_t shape_1d{safe_cast<int64_t>(shape[3 - is_column])};
        const long1_t strides_1d{safe_cast<int64_t>(strides[3 - is_column])};
        return getSinglePeak<1, IS_CENTERED>(xmap.get(), strides_1d, shape_1d, peak)[0];
    }

    template<Remap REMAP, typename T, typename>
    void xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                 const shared_t<float2_t[]>& peaks, Stream& stream) {
        NOA_ASSERT(shape[1] == 1);
        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        stream.enqueue([=]() mutable {
            cpu::memory::PtrHost<int64_t> offsets(shape[0]);
            math::find(noa::math::first_max_t{}, xmap, strides, shape, offsets.share(), true, true, stream);
            const auto shape_2d = safe_cast<long2_t>(dim2_t(shape.get(2)));
            const auto strides_2d = safe_cast<long2_t>(dim2_t(strides.get(2)));
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const long2_t peak = indexing::indexes(offsets[batch], strides_2d, shape_2d);
                const T* imap = xmap.get() + strides[0] * batch;
                peaks.get()[batch] = getSinglePeak<2, IS_CENTERED>(imap, strides_2d, shape_2d, peak);
            }
        });
    }

    template<Remap REMAP, typename T, typename>
    float2_t xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT(shape.ndim() == 2);
        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        const auto offset = math::find<int64_t>(noa::math::first_max_t{}, xmap, strides, shape, true, stream);
        const auto shape_2d = safe_cast<long2_t>(dim2_t(shape.get(2)));
        const auto strides_2d = safe_cast<long2_t>(dim2_t(strides.get(2)));
        const long2_t peak = indexing::indexes(offset, strides_2d, shape_2d);
        return getSinglePeak<2, IS_CENTERED>(xmap.get(), strides_2d, shape_2d, peak);
    }

    template<Remap REMAP, typename T, typename>
    void xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                 const shared_t<float3_t[]>& peaks, Stream& stream) {
        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        stream.enqueue([=]() mutable {
            cpu::memory::PtrHost<int64_t> offsets(shape[0]);
            math::find(noa::math::first_max_t{}, xmap, strides, shape, offsets.share(), true, true, stream);
            const auto shape_3d = safe_cast<long3_t>(dim3_t(shape.get(1)));
            const auto strides_3d = safe_cast<long3_t>(dim3_t(strides.get(1)));
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const long3_t peak = indexing::indexes(offsets[batch], strides_3d, shape_3d);
                const T* imap = xmap.get() + strides[0] * batch;
                peaks.get()[batch] = getSinglePeak<3, IS_CENTERED>(imap, strides_3d, shape_3d, peak);
            }
        });
    }

    template<Remap REMAP, typename T, typename>
    float3_t xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT(shape.ndim() == 3);
        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        const auto offset = math::find<int64_t>(noa::math::first_max_t{}, xmap, strides, shape, true, stream);
        const auto shape_3d = safe_cast<long3_t>(dim3_t(shape.get(1)));
        const auto strides_3d = safe_cast<long3_t>(dim3_t(strides.get(1)));
        const long3_t peak = indexing::indexes(offset, strides_3d, shape_3d);
        return getSinglePeak<3, IS_CENTERED>(xmap.get(), strides_3d, shape_3d, peak);
    }

    #define INSTANTIATE_XMAP(T) \
    template void xmap<Remap::H2F, T, void>(const shared_t<Complex<T>[]>&, dim4_t, const shared_t<Complex<T>[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, bool, Norm, Stream&, const shared_t<Complex<T>[]>&, dim4_t);   \
    template void xmap<Remap::H2FC, T, void>(const shared_t<Complex<T>[]>&, dim4_t, const shared_t<Complex<T>[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, bool, Norm, Stream&, const shared_t<Complex<T>[]>&, dim4_t);  \
    template void xpeak1D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float[]>&, Stream&);      \
    template void xpeak1D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float[]>&, Stream&);    \
    template void xpeak2D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float2_t[]>&, Stream&);   \
    template void xpeak2D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float2_t[]>&, Stream&); \
    template void xpeak3D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float3_t[]>&, Stream&);   \
    template void xpeak3D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float3_t[]>&, Stream&); \
    template float xpeak1D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, Stream&);       \
    template float xpeak1D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, Stream&);     \
    template float2_t xpeak2D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, Stream&);    \
    template float2_t xpeak2D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, Stream&);  \
    template float3_t xpeak3D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, Stream&);    \
    template float3_t xpeak3D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, Stream&)

    INSTANTIATE_XMAP(float);
    INSTANTIATE_XMAP(double);
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

        #pragma omp parallel for collapse(3) num_threads(threads)               \
        reduction(+:sum, sum_lhs, sum_rhs, err, err_lhs, err_rhs) default(none) \
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
