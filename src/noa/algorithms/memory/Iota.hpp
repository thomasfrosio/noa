#pragma once

#include "noa/core/Types.hpp"

namespace noa::algorithm::memory {
    template<typename Index, typename Offset, typename Value>
    class Iota4D {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using index4_type = Vec4<Index>;
        using shape4_type = Shape4<Index>;
        using strides4_type = Strides4<Index>;
        using accessor_type = Accessor<value_type, 4, offset_type>;

    public:
        Iota4D(const accessor_type& accessor,
               const shape4_type& shape,
               const index4_type& tile)
                : m_output(accessor),
                  m_contiguous_strides(shape.strides()),
                  m_tile(tile) {}

        NOA_HD constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const {
            m_output(ii, ij, ik, il) = static_cast<value_type>(noa::indexing::at(
                    ii % m_tile[0],
                    ij % m_tile[1],
                    ik % m_tile[2],
                    il % m_tile[3],
                    m_contiguous_strides));
        }

    private:
        accessor_type m_output;
        strides4_type m_contiguous_strides;
        index4_type m_tile;
    };

    template<typename Index, typename Offset, typename Value>
    auto iota_4d(Value* output, const Strides4<i64>& strides, const Shape4<i64>& shape,
                 const Vec4<i64>& tile) {
        return Iota4D<Index, Offset, Value>(
                Accessor<Value, 4, Offset>(output, strides.as_safe<Offset>()),
                shape.as_safe<Index>(), tile);
    }

    // Optimization for C-contiguous arrays.
    template<typename Index, typename Offset, typename Value>
    class Iota1D {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using accessor_type = AccessorContiguous<value_type, 1, offset_type>;

    public:
        Iota1D(const accessor_type& accessor, index_type tile)
                : m_output(accessor), m_tile(tile) {}

        NOA_HD constexpr void operator()(index_type i) const {
            m_output[i] = static_cast<value_type>(i % m_tile);
        }

    private:
        accessor_type m_output;
        index_type m_tile;
    };

    template<typename Index, typename Offset, typename Value>
    auto iota_1d(Value* output, i64 tile) {
        return Iota1D<Index, Offset, Value>(AccessorContiguous<Value, 1, Offset>(output), safe_cast<Index>(tile));
    }
}
