#pragma once

#include "noa/common/Types.h"
#include "noa/common/math/LeastSquare.h"

namespace noa::signal::fft::details {
    template<typename SInt, typename = std::enable_if_t<traits::is_sint_v<SInt> || traits::is_intX_v<SInt>>>
    constexpr NOA_FHD auto nonCenteredIndex2Frequency(SInt index, SInt dim_size) noexcept {
        static_assert(traits::is_sint_v<traits::value_type_t<SInt>>);
        if constexpr (traits::is_sint_v<SInt>) {
            return index < (dim_size + 1) / 2 ? index : index - dim_size;
        } else {
            SInt out;
            for (size_t dim = 0; dim < SInt::COUNT; ++dim)
                out[dim] = index[dim] < (dim_size[dim] + 1) / 2 ? index[dim] : index[dim] - dim_size[dim];
            return out;
        }
    }

    template<typename SInt, typename = std::enable_if_t<traits::is_sint_v<SInt> || traits::is_intX_v<SInt>>>
    constexpr NOA_FHD auto frequency2NonCenteredIndex(SInt frequency, SInt dim_size) noexcept {
        static_assert(traits::is_sint_v<traits::value_type_t<SInt>>);
        if constexpr (traits::is_sint_v<SInt>) {
            return frequency < 0 ? dim_size + frequency : frequency;
        } else {
            SInt out;
            for (size_t dim = 0; dim < SInt::COUNT; ++dim)
                out[dim] = frequency[dim] < 0 ? dim_size[dim] + frequency[dim] : frequency[dim];
            return out;
        }
    }

    // FIXME Return Pair<Vec<N,f64>, Vec<N,f64>> instead.
    template<int64_t NDIM, typename Real, typename SIntVector, typename CoordVector>
    constexpr NOA_HD double vertexParabola1D(const Real* peak_window,
                                             SIntVector peak_radius,
                                             CoordVector& peak_coordinate) {
        using coord_t = traits::value_type_t<CoordVector>;
        using sint_t = traits::value_type_t<SIntVector>;
        static_assert(traits::is_float_v<coord_t> && traits::is_sint_v<sint_t>);

        // Add sub-pixel position by fitting a 1D parabola to the peak and its two adjacent points.
        double peak_value{0};
        const Real* current_window = peak_window;
        for (sint_t dim = 0; dim < NDIM; ++dim) {
            const auto window_radius = static_cast<sint_t>(peak_radius[dim]);
            const auto window_size = window_radius * 2 + 1;

            if (window_radius == 1) {
                const auto [x, y] = noa::math::lstsqFitQuadraticVertex3Points(
                        current_window[0], current_window[1], current_window[2]);
                peak_coordinate[dim] += static_cast<coord_t>(x);
                peak_value += static_cast<double>(y);
            } else {
                double a{0}, b{0}, c{0};
                noa::math::lstsqFitQuadratic(current_window, window_size, &a, &b, &c);
                if (a != 0) { // This can fail if all values in output are equal.
                    const auto x = noa::math::clamp(-b / (2 * a), 0.5, static_cast<double>(window_size) - 1.5);
                    const auto y = a * x * x + b * x + c;
                    peak_coordinate[dim] += static_cast<coord_t>(x - static_cast<double>(window_radius));
                    peak_value += y;
                }
            }
            current_window += window_size;
        }
        return peak_value / NDIM;
    }
}
