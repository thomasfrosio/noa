#pragma once

#include "noa/core/Types.hpp"

namespace noa::algorithm::geometry {
    template<bool IS_CENTERED, typename SInt>
    [[nodiscard]] NOA_FHD SInt index2frequency(SInt index, SInt size) {
        static_assert(noa::traits::is_sint_v<SInt>);
        if constexpr (IS_CENTERED)
            return index - size / 2;
        else
            return index < (size + 1) / 2 ? index : index - size;
        return SInt{}; // unreachable, remove nvcc spurious warning
    }

    // The input frequency should be in-bound, i.e. -size/2 <= frequency <= (size-1)/2
    template<bool IS_CENTERED, typename SInt>
    [[nodiscard]] NOA_FHD SInt frequency2index(SInt frequency, SInt size) {
        static_assert(noa::traits::is_sint_v<SInt>);
        if constexpr (IS_CENTERED)
            return frequency + size / 2;
        else
            return frequency < 0 ? frequency + size : frequency;
        return SInt{}; // unreachable, remove nvcc spurious warning
    }

    template<typename Complex, typename Coord>
    [[nodiscard]] NOA_FHD Complex phase_shift(Coord shift, Coord freq) {
        static_assert(noa::traits::is_real2_v<Coord> || noa::traits::is_real3_v<Coord>);
        static_assert(noa::traits::is_complex_v<Complex>);
        using real_t = typename Complex::value_type;
        const auto factor = static_cast<real_t>(-math::dot(shift, freq));
        Complex phase_shift;
        noa::math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }
}
