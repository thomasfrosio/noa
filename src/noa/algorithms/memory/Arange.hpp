#pragma once

#include "noa/core/Types.hpp"

namespace noa::algorithm::memory {
    template<typename Index, typename Offset, typename Value>
    class Arange4D {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using shape4_type = Shape4<Index>;
        using strides4_type = Strides4<Index>;
        using accessor_type = Accessor<value_type, 4, offset_type>;

    public:
        Arange4D(const accessor_type& accessor,
                 const shape4_type& shape,
                 value_type start, value_type step)
                : m_output(accessor),
                  m_contiguous_strides(shape.strides()),
                  m_start(start),
                  m_step(step) {}

        NOA_HD constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const {
            const auto offset = indexing::at(ii, ij, ik, il, m_contiguous_strides);
            m_output(ii, ij, ik, il) = m_start + static_cast<value_type>(offset) * m_step;
        }

    private:
        accessor_type m_output;
        strides4_type m_contiguous_strides;
        value_type m_start;
        value_type m_step;
    };

    template<typename Index, typename Offset, typename Value>
    auto arange_4d(Value* output, const Strides4<i64>& strides, const Shape4<i64>& shape,
                   Value start, Value step) {
        return Arange4D<Index, Offset, Value>(
                Accessor<Value, 4, Offset>(output, strides.as_safe<Offset>()),
                shape.as_safe<Index>(), start, step);
    }

    // Optimization for C-contiguous arrays.
    template<typename Index, typename Offset, typename Value>
    class Arange1D {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using accessor_type = AccessorContiguous<value_type, 1, offset_type>;

    public:
        Arange1D(const accessor_type& accessor,
                 value_type start, value_type step)
                : m_output(accessor),
                  m_start(start),
                  m_step(step) {}

        NOA_HD constexpr void operator()(index_type i) const {
            m_output(i) = m_start + static_cast<value_type>(i) * m_step;
        }

    private:
        accessor_type m_output;
        value_type m_start;
        value_type m_step;
    };

    template<typename Index, typename Offset, typename Value>
    auto arange_1d(Value* output, Value start, Value step) {
        return Arange1D<Index, Offset, Value>(AccessorContiguous<Value, 1, Offset>(output), start, step);
    }
}
