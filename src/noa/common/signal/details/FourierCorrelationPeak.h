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

    template<int64_t NDIM, typename Real, typename SIntVector, typename RealVector>
    constexpr NOA_HD void addSubpixelCoordParabola1D(const Real* peak_window,
                                                     SIntVector peak_radius,
                                                     RealVector& peak_coordinate) {
        using real_t = traits::value_type_t<RealVector>;
        using sint_t = traits::value_type_t<SIntVector>;
        static_assert(traits::is_float_v<real_t> && traits::is_sint_v<sint_t>);

        // Add sub-pixel position by fitting a 1D parabola to the peak and its adjacent points.
        const Real* current_window = peak_window;
        for (sint_t dim = 0; dim < NDIM; ++dim) {
            const auto window_radius = static_cast<sint_t>(peak_radius[dim]);
            const auto window_size = window_radius * 2 + 1;

            real_t vertex_offset{0};
            if (window_radius == 1) {
                vertex_offset = static_cast<real_t>(noa::math::lstsqFitQuadraticVertex3Points(
                        current_window[0], current_window[1], current_window[2]));
            } else {
                double a{}, b{};
                noa::math::lstsqFitQuadratic(current_window, window_size, &a, &b, nullptr);
                if (a != 0) { // This can fail if all values in output are equal.
                    const auto d_radius = static_cast<double>(window_radius);
                    const auto vertex = -b / (2 * a) - d_radius;
                    vertex_offset = static_cast<real_t>(noa::math::clamp(vertex, -d_radius + 0.5, d_radius - 0.5));
                }
            }
            peak_coordinate[dim] += vertex_offset;
            current_window += window_size;
        }
    }
}
