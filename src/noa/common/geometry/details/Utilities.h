#pragma once

#include "noa/common/Types.h"

namespace noa::geometry::fft::details {
    using Remap = ::noa::fft::Remap;

    template<bool IS_CENTERED, typename SInt>
    [[nodiscard]] NOA_FHD SInt index2frequency(SInt index, SInt size) {
        static_assert(traits::is_sint_v<SInt>);
        if constexpr (IS_CENTERED)
            return index - size / 2;
        else
            return index < (size + 1) / 2 ? index : index - size;
        return SInt{}; // unreachable, remove nvcc spurious warning
    }

    // The input frequency should be in-bound, i.e. -size/2 <= frequency <= (size-1)/2
    template<bool IS_CENTERED, typename SInt>
    [[nodiscard]] NOA_FHD SInt frequency2index(SInt frequency, SInt size) {
        static_assert(traits::is_sint_v<SInt>);
        if constexpr (IS_CENTERED)
            return frequency + size / 2;
        else
            return frequency < 0 ? frequency + size : frequency;
        return SInt{}; // unreachable, remove nvcc spurious warning
    }

    template<typename ComplexValue, typename Coord>
    [[nodiscard]] NOA_FHD ComplexValue phaseShift(Coord shift, Coord freq) {
        static_assert(traits::is_float2_v<Coord> || traits::is_float3_v<Coord>);
        using real_t = traits::value_type_t<ComplexValue>;
        const auto factor = static_cast<real_t>(-math::dot(shift, freq));
        ComplexValue phase_shift;
        noa::math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }
}
