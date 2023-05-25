#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/utils/Atomic.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::algorithm::geometry {
    // Rotational average implementation.
    // * A lerp is used to add frequencies in its two neighbour shells, instead of rounding to the nearest shell.
    // * The frequencies are normalized, so rectangular volumes can be passed.
    // * The number of shells is fixed by the shape: min(shape) // 2 + 1
    // * The input/output can be complex/real, respectively, resulting in the abs() of the input being used instead.
    template<noa::fft::Remap REMAP, size_t N,
             typename Coord, typename Index, typename Offset,
             typename Input, typename Output>
    class RotationalAverage {
    public:
        static_assert(REMAP == noa::fft::H2H || REMAP == noa::fft::HC2H ||
                      REMAP == noa::fft::F2H || REMAP == noa::fft::FC2H);
        static_assert(noa::traits::is_sint_v<Index>);
        static_assert(noa::traits::is_int_v<Offset>);
        static_assert(noa::traits::is_real_v<Coord>);
        static_assert((noa::traits::are_all_same_v<Input, Output> &&
                       noa::traits::are_real_or_complex_v<Input, Output>) ||
                      (noa::traits::is_complex_v<Input> &&
                       noa::traits::is_real_v<Output>));

        static constexpr bool IS_CENTERED = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_CENTERED;
        static constexpr bool IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_HALF;

        using index_type = Index;
        using offset_type = Offset;
        using coord_type = Coord;
        using input_type = Input;
        using output_type = Output;
        using real_type = noa::traits::value_type_t<output_type>;

        using shape_type = Shape<index_type, N - IS_HALF>;
        using coord_nd_type = Vec<coord_type, N>;
        using shape_nd_type = Shape<index_type, N>;
        using input_accessor_type = AccessorRestrict<const input_type, (N + 1), offset_type>;
        using output_accessor_type = AccessorRestrictContiguous<output_type, 2, offset_type>;
        using weight_accessor_type = AccessorRestrictContiguous<real_type, 2, offset_type>;

    public:
        RotationalAverage(const input_accessor_type& input,
                          const shape_nd_type& shape,
                          const output_accessor_type& output,
                          const weight_accessor_type& weight)
                : m_input(input), m_output(output), m_weight(weight),
                  m_norm(coord_type{1} / static_cast<coord_nd_type>(shape.vec())),
                  m_scale(static_cast<coord_type>(noa::math::min(shape))),
                  m_max_shell_index(noa::math::min(shape) / 2) {
            if constexpr (IS_HALF)
                m_shape = shape.pop_back();
            else
                m_shape = shape;
        }

        // Batched 2d.
        template<typename Void = void, typename = std::enable_if_t<(N == 2) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            // Compute the normalized frequency.
            auto frequency = coord_nd_type{
                    noa::fft::index2frequency<IS_CENTERED>(y, m_shape[0]),
                    IS_HALF ? x : noa::fft::index2frequency<IS_CENTERED>(x, m_shape[1])};
            frequency *= m_norm;

            // Shortcut for everything past Nyquist.
            const auto radius_sqd = noa::math::dot(frequency, frequency);
            if (radius_sqd > coord_type{0.25})
                return;

            const auto output = input_to_output_(m_input(batch, y, x));
            add_to_output_(batch, radius_sqd, output);
        }

        // Batched 3d.
        template<typename Void = void, typename = std::enable_if_t<(N == 3) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept {
            // Compute the normalized frequency.
            auto frequency = coord_nd_type{
                    noa::fft::index2frequency<IS_CENTERED>(z, m_shape[0]),
                    noa::fft::index2frequency<IS_CENTERED>(y, m_shape[1]),
                    IS_HALF ? x : noa::fft::index2frequency<IS_CENTERED>(x, m_shape[2])};
            frequency *= m_norm;

            // Shortcut for everything past Nyquist.
            const auto radius_sqd = noa::math::dot(frequency, frequency);
            if (radius_sqd > coord_type{0.25})
                return;

            const auto output = input_to_output_(m_input(batch, z, y, x));
            add_to_output_(batch, radius_sqd, output);
        }

    private:
        NOA_HD static output_type input_to_output_(input_type input) noexcept {
            if constexpr (noa::traits::is_complex_v<input_type> &&
                          noa::traits::is_real_v<output_type>) {
                return noa::math::abs(input);
            } else {
                return input;
            }
        }

        NOA_HD void add_to_output_(index_type batch, coord_type radius_sqd, output_type value) const noexcept {
            const auto radius = noa::math::sqrt(radius_sqd) * m_scale;
            const auto fraction = noa::math::floor(radius);

            // Compute lerp weights.
            const auto shell_low = static_cast<index_type>(noa::math::floor(radius));
            const auto shell_high = noa::math::min(m_max_shell_index, shell_low + 1); // clamp for oob
            const auto fraction_high = static_cast<real_type>(radius - fraction);
            const auto fraction_low = 1 - fraction_high;

            // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers.
            noa::details::atomic_add(m_output, value * fraction_low, batch, shell_low);
            noa::details::atomic_add(m_output, value * fraction_high, batch, shell_high);
            if (m_weight) {
                noa::details::atomic_add(m_weight, fraction_low, batch, shell_low);
                noa::details::atomic_add(m_weight, fraction_high, batch, shell_high);
            }
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        weight_accessor_type m_weight;

        coord_nd_type m_norm;
        coord_type m_scale;
        shape_type m_shape; // width is removed
        index_type m_max_shell_index;
    };

    template<noa::fft::Remap REMAP, typename Coord = f32,
             typename Index, typename Offset,
             typename Input, typename Output, typename Weight>
    auto rotational_average_2d(
            const Input* input, const Strides4<Offset>& strides,
            const Shape4<Index>& shape,
            Output* output, Weight* weight, Index shell_count
    ) {
        const auto shell_strides = Strides2<Offset>{shell_count, 1};
        const auto input_accessor = AccessorRestrict<const Input, 3, Offset>(input, strides.filter(0, 2, 3));
        const auto output_accessor = AccessorRestrictContiguous<Output, 2, Offset>(output, shell_strides);
        const auto weight_accessor = AccessorRestrictContiguous<Weight, 2, Offset>(weight, shell_strides);

        return RotationalAverage<REMAP, 2, Coord, Index, Offset, Input, Output>(
                input_accessor, shape.filter(2, 3), output_accessor, weight_accessor);
    }

    template<noa::fft::Remap REMAP, typename Coord = f32,
             typename Index, typename Offset,
             typename Input, typename Output, typename Weight>
    auto rotational_average_3d(
            const Input* input, const Strides4<Offset>& strides,
            const Shape4<Index>& shape,
            Output* output, Weight* weight, Index shell_count
    ) {
        const auto shell_strides = Strides2<Offset>{shell_count, 1};
        const auto input_accessor = AccessorRestrict<const Input, 4, Offset>(input, strides);
        const auto output_accessor = AccessorRestrictContiguous<Output, 2, Offset>(output, shell_strides);
        const auto weight_accessor = AccessorRestrictContiguous<Weight, 2, Offset>(weight, shell_strides);

        return RotationalAverage<REMAP, 3, Coord, Index, Offset, Input, Output>(
                input_accessor, shape.filter(1, 2, 3), output_accessor, weight_accessor);
    }
}

namespace noa::algorithm::geometry::fft {

}
