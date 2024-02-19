#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/indexing/Offset.hpp"

namespace noa {
    template<typename Accessor, typename Index>
    requires (nt::is_accessor_nd<Accessor, 4>::value and std::is_integral_v<Index>)
    class Iota4d {
    public:
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using index_type = Index;
        using index4_type = Vec4<Index>;
        using shape4_type = Shape4<Index>;
        using strides4_type = Strides4<Index>;

    public:
        constexpr Iota4d(
                const accessor_type& accessor,
                const shape4_type& shape,
                const index4_type& tile
        ) : m_output(accessor),
            m_contiguous_strides(shape.strides()),
            m_tile(tile) {}

        NOA_HD constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const noexcept {
            const auto iota = ni::offset_at(
                    ii % m_tile[0],
                    ij % m_tile[1],
                    ik % m_tile[2],
                    il % m_tile[3],
                    m_contiguous_strides);
            m_output(ii, ij, ik, il) = static_cast<value_type>(iota);
        }

    private:
        accessor_type m_output;
        strides4_type m_contiguous_strides;
        index4_type m_tile;
    };

    template<typename Accessor, typename Index>
    class Iota1d {
    public:
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using index_type = Index;

    public:
        constexpr Iota1d(const accessor_type& accessor, index_type tile) : m_output(accessor), m_tile(tile) {}

        NOA_HD constexpr void operator()(index_type i) const noexcept {
            m_output[i] = static_cast<value_type>(i % m_tile);
        }

    private:
        accessor_type m_output;
        index_type m_tile;
    };
}
