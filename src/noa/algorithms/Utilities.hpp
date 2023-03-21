#pragma once

#include "noa/core/Types.hpp"

namespace noa::algorithm {
    template<bool IS_CENTERED, typename SInt>
    [[nodiscard]] NOA_FHD SInt index2frequency(SInt index, SInt size) {
        static_assert(noa::traits::is_sint_v<SInt>);
        if constexpr (IS_CENTERED)
            return index - size / 2;
        else
            return index < (size + 1) / 2 ? index : index - size;
        return SInt{}; // unreachable, remove nvcc spurious warning
    }

    template<bool IS_CENTERED, typename SInt, size_t N>
    [[nodiscard]] constexpr NOA_FHD auto index2frequency(
            const Vec<SInt, N>& indexes, const Shape<SInt, N>& shape
    ) noexcept {
        static_assert(noa::traits::is_sint_v<SInt>);
        Vec<SInt, N> out;
        for (size_t i = 0; i < N; ++i)
            out[i] = index2frequency<IS_CENTERED>(indexes[i], shape[i]);
        return out;
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

    template<bool IS_CENTERED, typename SInt, size_t N>
    [[nodiscard]] constexpr NOA_FHD auto frequency2index(
            const Vec<SInt, N>& indexes, const Shape<SInt, N>& shape
    ) noexcept {
        static_assert(noa::traits::is_sint_v<SInt>);
        Vec<SInt, N> out;
        for (size_t i = 0; i < N; ++i)
            out[i] = frequency2index<IS_CENTERED>(indexes[i], shape[i]);
        return out;
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

    template<noa::fft::Remap REMAP, typename Int>
    NOA_FHD Int to_centered_index(Int index, Int shape) {
        constexpr auto u8_REMAP = static_cast<u8>(REMAP);
        constexpr bool IS_SRC_CENTERED = u8_REMAP & noa::fft::Layout::SRC_CENTERED;
        if constexpr (IS_SRC_CENTERED)
            return index;
        else
            return noa::math::fft_shift(index, shape);
    }

    template<noa::fft::Remap REMAP, typename Int, size_t N>
    NOA_FHD Vec<Int, N> to_centered_indexes(const Vec<Int, N>& indexes, const Shape<Int, N>& shape) {
        constexpr auto u8_REMAP = static_cast<u8>(REMAP);
        constexpr bool IS_SRC_CENTERED = u8_REMAP & noa::fft::Layout::SRC_CENTERED;
        if constexpr (IS_SRC_CENTERED)
            return indexes;
        else
            return noa::math::fft_shift(indexes, shape);
    }

    template<noa::fft::Remap REMAP, typename Int>
    NOA_FHD Int to_output_index(Int index, Int shape) {
        constexpr auto u_REMAP = static_cast<u8>(REMAP);
        constexpr bool IS_SRC_CENTERED = u_REMAP & noa::fft::Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = u_REMAP & noa::fft::Layout::DST_CENTERED;
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED) {
            return index;
        } else if constexpr (IS_SRC_CENTERED) {
            return noa::math::ifft_shift(index, shape);
        } else { // F2FC
            return noa::math::fft_shift(index, shape);
        }
    }

    template<noa::fft::Remap REMAP, typename Int, size_t N>
    NOA_FHD Vec<Int, N> to_output_indexes(const Vec<Int, N>& indexes, const Shape<Int, N>& shape) {
        constexpr auto u_REMAP = static_cast<u8>(REMAP);
        constexpr bool IS_SRC_CENTERED = u_REMAP & noa::fft::Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = u_REMAP & noa::fft::Layout::DST_CENTERED;
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED) {
            return indexes;
        } else if constexpr (IS_SRC_CENTERED) {
            return noa::math::ifft_shift(indexes, shape);
        } else { // F2FC
            return noa::math::fft_shift(indexes, shape);
        }
    }
}
