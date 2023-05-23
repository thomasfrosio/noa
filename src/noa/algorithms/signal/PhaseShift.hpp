#pragma once

#include "noa/core/Types.hpp"
#include "noa/algorithms/Utilities.hpp"

namespace noa::algorithm::signal {
    // Phase shift each dimension by size / 2 (floating-point division).
    template<noa::fft::Remap REMAP, typename Index, typename Offset, typename Value>
    class PhaseShiftHalf {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SRC_CENTERED = u8_REMAP & Layout::SRC_CENTERED;
        static_assert(u8_REMAP & Layout::SRC_HALF && u8_REMAP & Layout::DST_HALF);

        using index_type = Index;
        using offset_type = Offset;
        using value_type = Value;
        using index3_type = Vec3<index_type>;
        using shape2_type = Shape2<index_type>;
        using shape4_type = Shape4<index_type>;
        using input_accessor_type = Accessor<const value_type, 4, offset_type>;
        using output_accessor_type = Accessor<value_type, 4, offset_type>;
        using real_type = noa::traits::value_type_t<value_type>;

    public:
        PhaseShiftHalf(const input_accessor_type& input,
                       const output_accessor_type& output,
                       const shape4_type& shape)
                : m_input(input), m_output(output), m_bh_shape(shape.filter(1, 2)) {}

        NOA_HD constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const noexcept {
            const auto freq = index3_type{index2frequency<IS_SRC_CENTERED>(ij, m_bh_shape[0]),
                                          index2frequency<IS_SRC_CENTERED>(ik, m_bh_shape[1]),
                                          il};
            const auto phase_shift = static_cast<real_type>(noa::math::product(1 - 2 * noa::math::abs(freq % 2)));

            const auto oj = to_output_index<REMAP>(ij, m_bh_shape[0]);
            const auto ok = to_output_index<REMAP>(ik, m_bh_shape[1]);
            m_output(ii, oj, ok, il) = m_input ? m_input(ii, ij, ik, il) * phase_shift : phase_shift;
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        shape2_type m_bh_shape;
    };

    template<noa::fft::Remap REMAP, size_t NDIM,
             typename Index, typename Offset, typename Value, typename Coord, typename Shift>
    class PhaseShift {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SRC_CENTERED = u8_REMAP & Layout::SRC_CENTERED;
        static_assert(u8_REMAP & Layout::SRC_HALF && u8_REMAP & Layout::DST_HALF);
        static_assert((NDIM == 2 && noa::traits::is_any_v<Shift, Vec2<Coord>, const Vec2<Coord>*>) ||
                      (NDIM == 3 && noa::traits::is_any_v<Shift, Vec3<Coord>, const Vec3<Coord>*>));

        using index_type = Index;
        using offset_type = Offset;
        using value_type = Value;
        using coord_type = Coord;
        using shift_type = Shift;

        using vec_nd_type = Vec<coord_type, NDIM>;
        using shape_nd_type = Shape<index_type, NDIM>;
        using shape_type = Shape<index_type, NDIM - 1>;
        using preshift_type = std::conditional_t<std::is_pointer_v<Shift>, vec_nd_type, Empty>;

        using input_accessor_type = Accessor<const value_type, NDIM + 1, offset_type>;
        using output_accessor_type = Accessor<value_type, NDIM + 1, offset_type>;

    public:
        PhaseShift(const input_accessor_type& input,
                   const output_accessor_type& output,
                   const shape_nd_type& shape,
                   const shift_type& shift,
                   coord_type cutoff)
                : m_input(input), m_output(output),
                  m_norm(coord_type{1} / vec_nd_type(shape.vec())),
                  m_shape(shape.pop_back()),
                  m_shift(shift),
                  m_cutoff_sqd(cutoff * cutoff) {
            const vec_nd_type pre_shift = 2 * noa::math::Constant<coord_type>::PI / vec_nd_type(shape.vec());
            if constexpr (!std::is_pointer_v<shift_type>)
                m_shift *= pre_shift;
            else
                m_preshift = pre_shift;
        }

        template<typename Void = void, typename = std::enable_if_t<NDIM == 2 && std::is_void_v<Void>>>
        NOA_HD constexpr void operator()(index_type ii, index_type ik, index_type il) const noexcept {
            const vec_nd_type frequency{index2frequency<IS_SRC_CENTERED>(ik, m_shape[0]),
                                        il};

            const vec_nd_type norm_freq = frequency * m_norm;
            const value_type phase_shift =
                    noa::math::dot(norm_freq, norm_freq) <= m_cutoff_sqd ?
                    phase_shift_(ii, frequency) : value_type{1, 0};

            const auto ok = to_output_index<REMAP>(ik, m_shape[0]);
            m_output(ii, ok, il) = m_input ? m_input(ii, ik, il) * phase_shift : phase_shift;
        }

        template<typename Void = void, typename = std::enable_if_t<NDIM == 3 && std::is_void_v<Void>>>
        NOA_HD constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const noexcept {
            const auto frequency = vec_nd_type{index2frequency<IS_SRC_CENTERED>(ij, m_shape[0]),
                                               index2frequency<IS_SRC_CENTERED>(ik, m_shape[1]),
                                               il};

            const vec_nd_type norm_freq = frequency * m_norm;
            const value_type phase_shift =
                    noa::math::dot(norm_freq, norm_freq) <= m_cutoff_sqd ?
                    phase_shift_(ii, frequency) : value_type{1, 0};

            // FIXME If even, the real nyquist should stay real, so add the conjugate pair?

            const auto oj = to_output_index<REMAP>(ij, m_shape[0]);
            const auto ok = to_output_index<REMAP>(ik, m_shape[1]);
            m_output(ii, oj, ok, il) = m_input ? m_input(ii, ij, ik, il) * phase_shift : phase_shift;
        }

    private:
        NOA_HD constexpr value_type phase_shift_(index_type batch, const vec_nd_type& frequency) const noexcept {
            if constexpr (std::is_pointer_v<shift_type>)
                return phase_shift<value_type>(m_shift[batch] * m_preshift, frequency);
            else
                return phase_shift<value_type>(m_shift, frequency);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        vec_nd_type m_norm;
        shape_type m_shape;
        shift_type m_shift;
        coord_type m_cutoff_sqd;
        NOA_NO_UNIQUE_ADDRESS preshift_type m_preshift{};
    };

    template<noa::fft::Remap REMAP, typename Index, typename Offset, typename Value>
    auto phase_shift_half(const Value* input, const Strides4<Offset>& input_strides,
                          Value* output, const Strides4<Offset>& output_strides,
                          const Shape4<Index>& shape) {
        const auto input_accessor = Accessor<const Value, 4, Offset>(input, input_strides);
        const auto output_accessor = Accessor<Value, 4, Offset>(output, output_strides);
        return PhaseShiftHalf<REMAP, Index, Offset, Value>(input_accessor, output_accessor, shape);
    }

    template<noa::fft::Remap REMAP, size_t NDIM,
             typename Index, typename Offset, typename Value, typename Coord, typename Shift>
    auto phase_shift(const Value* input, const Strides4<Offset>& input_strides,
                     Value* output, const Strides4<Offset>& output_strides,
                     const Shape4<Index>& shape, const Shift& shift, Coord cutoff) {
        if constexpr (NDIM == 2) {
            const auto input_accessor = Accessor<const Value, 3, Offset>(input, input_strides.filter(0, 2, 3));
            const auto output_accessor = Accessor<Value, 3, Offset>(output, output_strides.filter(0, 2, 3));
            return PhaseShift<REMAP, NDIM, Index, Offset, Value, Coord, Shift>(
                    input_accessor, output_accessor, shape.filter(2, 3), shift, cutoff);
        } else {
            const auto input_accessor = Accessor<const Value, 4, Offset>(input, input_strides);
            const auto output_accessor = Accessor<Value, 4, Offset>(output, output_strides);
            return PhaseShift<REMAP, NDIM, Index, Offset, Value, Coord, Shift>(
                    input_accessor, output_accessor, shape.filter(1, 2, 3), shift, cutoff);
        }
    }
}
