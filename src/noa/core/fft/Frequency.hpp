#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/Enums.hpp"

namespace noa::fft {
    /// Returns the highest normalized frequency (in cycle/pix) that a dimension with a given size can have.
    template<std::floating_point T>
    [[nodiscard]] NOA_FHD constexpr T highest_normalized_frequency(std::integral auto size) noexcept {
        // even: the highest frequency is always 0.5, i.e. Nyquist. E.g. size=64, 32/64=0.5
        // odd: Nyquist cannot be reached. Eg. size=63, 31/63 = 0.49206
        const auto max_index = size / 2; // integer division
        return static_cast<T>(max_index) / static_cast<T>(size);
    }

    /// Returns the highest normalized frequency (in cycle/pix) that a dimension with a given size can have.
    template<std::floating_point T>
    [[nodiscard]] NOA_FHD constexpr auto highest_fftfreq(std::integral auto size) noexcept {
        return highest_normalized_frequency(size);
    }

    /// Returns the fft centered index of the corresponding fft non-centered index.
    /// Should satisfy `0 <= index < size`.
    template<typename Int> requires std::is_integral_v<Int>
    [[nodiscard]] NOA_FHD constexpr Int fftshift(Int index, Int size) noexcept {
        // n=10: [0, 1, 2, 3, 4,-5,-4,-3,-2,-1] -> [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4]
        // n=11: [0, 1, 2, 3, 4, 5,-5,-4,-3,-2,-1] -> [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5]
        return (index < (size + 1) / 2) ? index + size / 2 : index - (size + 1) / 2; // or (index + size / 2) % size
    }
    template<typename VecInt> requires nt::is_vec_int_v<VecInt>
    [[nodiscard]] NOA_FHD constexpr VecInt fftshift(VecInt indexes, VecInt sizes) noexcept {
        VecInt shifted_indexes;
        for (size_t i = 0; i < VecInt::SIZE; ++i)
            shifted_indexes[i] = fftshift(indexes[i], sizes[i]);
        return shifted_indexes;
    }
    template<std::integral Int, size_t N>
    [[nodiscard]] NOA_FHD constexpr Vec<Int, N> fftshift(
            const Vec<Int, N>& indexes,
            const Shape<Int, N>& shape
    ) noexcept {
        return fftshift(indexes, shape.vec);
    }

    /// Returns the fft non-centered index of the corresponding centered fft index.
    /// Should be within `0 <= index < size`.
    template<typename Int> requires std::is_integral_v<Int>
    [[nodiscard]] NOA_FHD constexpr Int ifftshift(Int index, Int size) noexcept {
        // n=10: [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4] -> [0, 1, 2, 3, 4,-5,-4,-3,-2,-1]
        // n=11: [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5] -> [0, 1, 2, 3, 4, 5,-5,-4,-3,-2,-1]
        return (index < size / 2) ? index + (size + 1) / 2 : index - size / 2; // or (index + (size + 1) / 2) % size
    }
    template<typename VecInt> requires nt::is_vec_int_v<VecInt>
    [[nodiscard]] NOA_FHD constexpr VecInt ifftshift(VecInt indexes, VecInt sizes) {
        VecInt shifted_indexes;
        for (size_t i = 0; i < VecInt::SIZE; ++i)
            shifted_indexes[i] = ifftshift(indexes[i], sizes[i]);
        return shifted_indexes;
    }
    template<std::integral Int, size_t N>
    [[nodiscard]] NOA_FHD constexpr Vec<Int, N> ifftshift(
            const Vec<Int, N>& indexes,
            const Shape<Int, N>& shape
    ) noexcept {
        return ifftshift(indexes, shape.vec);
    }

    /// Returns the frequency (in samples) at a given index in range [0, size).
    /// \warning This function is only intended to be used for full dimensions!
    ///          For a rfft's half dimension (which is not supported by this function),
    ///          the index is equal to the frequency, regardless of the centering.
    template<bool IS_CENTERED, std::signed_integral SInt>
    [[nodiscard]] NOA_FHD constexpr SInt index2frequency(SInt index, SInt size) noexcept {
        // n=5: [0, 1, 2, 3, 4] -> centered=[-2,-1, 0, 1, 2], non-centered=[0, 1, 2,-2,-1]
        // n=6: [0, 1, 2, 3, 4, 5] -> centered=[-3,-2,-1, 0, 1, 2], non-centered=[0, 1, 2,-3,-2,-1]
        if constexpr (IS_CENTERED)
            return index - size / 2;
        return index < (size + 1) / 2 ? index : index - size;
    }

    /// Returns the multidimensional frequency at a given multidimensional index, assuming ((D)H)W order.
    /// For the rfft, the index along the width (ie rightmost dimension) is the frequency, so the width
    /// value is ignored and can be omitted from the shape.
    template<bool IS_CENTERED, bool IS_RFFT = false, std::signed_integral SInt, size_t N0, size_t N1>
    requires ((N0 == N1) or (IS_RFFT and N0 == N1 - 1))
    [[nodiscard]] NOA_FHD constexpr auto index2frequency(
            const Vec<SInt, N0>& indexes,
            const Shape<SInt, N1>& shape
    ) noexcept {
        Vec<SInt, N0> out;
        for (size_t i = 0; i < N0; ++i) {
            out[i] = (IS_RFFT and i == N0 - 1) ?
                     static_cast<SInt>(i) : // if width of rfft, frequency == index
                     index2frequency<IS_CENTERED>(indexes[i], shape[i]);
        }
        return out;
    }

    /// Returns the index at a given frequency (in samples).
    /// The frequency should be in range [-size/2, (size-1)/2]
    /// \warning This function is only intended to be used for full dimensions!
    ///          For a rfft's half dimension (which is not supported by this function),
    ///          the index is equal to the frequency, regardless of the centering.
    template<bool IS_CENTERED, std::signed_integral SInt>
    [[nodiscard]] NOA_FHD constexpr SInt frequency2index(SInt frequency, SInt size) noexcept {
        // n=5: [0, 1, 2, 3, 4] -> centered=[-2,-1, 0, 1, 2], non-centered=[0, 1, 2,-2,-1]
        // n=6: [0, 1, 2, 3, 4, 5] -> centered=[-3,-2,-1, 0, 1, 2], non-centered=[0, 1, 2,-3,-2,-1]
        if constexpr (IS_CENTERED)
            return frequency + size / 2;
        return frequency < 0 ? frequency + size : frequency;
    }

    /// Returns the multidimensional index at a given multidimensional frequency, assuming ((D)H)W order.
    /// For the rfft, the index along the width (ie rightmost dimension) is the frequency, so the width
    /// value is ignored and can be omitted from the shape.
    template<bool IS_CENTERED, bool IS_RFFT = false, std::signed_integral SInt, size_t N0, size_t N1>
    requires ((N0 == N1) or (IS_RFFT and N0 == N1 - 1))
    [[nodiscard]] NOA_FHD constexpr auto frequency2index(
            const Vec<SInt, N0>& indexes,
            const Shape<SInt, N1>& shape
    ) noexcept {
        Vec<SInt, N0> out;
        for (size_t i = 0; i < N0; ++i) {
            out[i] = (IS_RFFT && i == N0 - 1) ?
                     static_cast<SInt>(i) : // if width of rfft, frequency == index
                     frequency2index<IS_CENTERED>(indexes[i], shape[i]);
        }
        return out;
    }

    /// Returns the centered index, given the (non-)centered index.
    /// \warning This function only supports full-dimensions.
    template<bool IS_SRC_CENTERED, std::integral Int>
    NOA_FHD constexpr Int to_centered_index(Int index, Int shape) noexcept {
        if constexpr (IS_SRC_CENTERED)
            return index;
        return fftshift(index, shape);
    }

    /// Returns the multidimensional centered index given the multidimensional (non-)centered index, assuming ((D)H)W order.
    /// For rffts, the centering doesn't apply to the width (ie rightmost dimension) and the index is left unchanged.
    template<bool IS_CENTERED, bool IS_RFFT = false, std::signed_integral Int, size_t N0, size_t N1>
    requires ((N0 == N1) or (IS_RFFT and N0 == N1 - 1))
    NOA_FHD constexpr auto to_centered_indexes(
            const Vec<Int, N0>& indexes,
            const Shape<Int, N1>& shape
    ) noexcept -> Vec<Int, N0> {
        Vec<Int, N0> out;
        for (size_t i = 0; i < N0; ++i) {
            if constexpr (IS_CENTERED) {
                out[i] = indexes[i];
            } else {
                out[i] = (IS_RFFT and i == N0 - 1) ? indexes[i] : fftshift(indexes[i], shape[i]);
            }
        }
        return out;
    }

    /// Returns the output index, given the input index.
    /// \warning This function only supports full-dimensions.
    template<Remap REMAP, bool FLIP_REMAP = false, typename Int> requires std::is_integral_v<Int>
    NOA_FHD constexpr Int remap_index(Int index, Int shape) noexcept {
        constexpr auto u_REMAP = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = u_REMAP & (FLIP_REMAP ? Layout::DST_CENTERED : Layout::SRC_CENTERED);
        constexpr bool IS_DST_CENTERED = u_REMAP & (FLIP_REMAP ? Layout::SRC_CENTERED : Layout::DST_CENTERED);
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED) {
            return index;
        } else if constexpr (IS_SRC_CENTERED) {
            return ifftshift(index, shape);
        } else {
            return fftshift(index, shape);
        }
    }

    /// Returns the output indexes given input indexes. This does the (i)fftshift remap operation.
    /// This function is limited to input/output being both rffts or both ffts.
    /// For rffts, the centering doesn't apply to the width (ie rightmost dimension) and the index is left unchanged.
    template<Remap REMAP, bool FLIP_REMAP = false, std::integral Int, size_t N0, size_t N1>
    NOA_FHD constexpr auto remap_indexes(
            const Vec<Int, N0>& indexes,
            const Shape<Int, N1>& shape
    ) noexcept -> Vec<Int, N0> {
        constexpr auto u_REMAP = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_RFFT = u_REMAP & Layout::SRC_HALF;
        constexpr bool IS_DST_RFFT = u_REMAP & Layout::DST_HALF;
        static_assert(IS_SRC_RFFT == IS_DST_RFFT);
        static_assert((N0 == N1) || (IS_DST_RFFT && N0 == N1 - 1));

        constexpr bool IS_SRC_CENTERED = u_REMAP & (FLIP_REMAP ? Layout::DST_CENTERED : Layout::SRC_CENTERED);
        constexpr bool IS_DST_CENTERED = u_REMAP & (FLIP_REMAP ? Layout::SRC_CENTERED : Layout::DST_CENTERED);
        Vec<Int, N0> out;
        for (size_t i = 0; i < N0; ++i) {
            if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED) {
                out[i] = indexes[i];
            } else if constexpr (IS_SRC_CENTERED) {
                out[i] = (IS_DST_RFFT and i == N0 - 1) ? indexes[i] : ifftshift(indexes[i], shape[i]);
            } else {
                out[i] = (IS_DST_RFFT and i == N0 - 1) ? indexes[i] : fftshift(indexes[i], shape[i]);
            }
        }
        return out;
    }
}
