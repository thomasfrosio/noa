#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/indexing/Offset.hpp"

namespace noa {
    /// Defines evenly spaced values within a given interval.
    template<typename Value> requires nt::is_numeric_v<Value>
    struct Arange {
        Value start{0};
        Value step{1};

        template<typename Index> requires std::is_integral_v<Index>
        [[nodiscard]] NOA_HD constexpr Value operator()(Index index) const noexcept {
            using real_t = nt::value_type_t<Value>;
            return start + static_cast<real_t>(index) * step;
        }
    };

    /// Arange index-wise operator for 1d ranges.
    template<typename AccessorLike>
    class Arange1d {
    public:
        using accessor_type = AccessorLike;
        using value_type = accessor_type::value_type;

    public:
        constexpr explicit Arange1d(const accessor_type& accessor) : m_output(accessor) {}

        constexpr Arange1d(const accessor_type& accessor, value_type start, value_type step)
                : m_output(accessor), m_arange{start, step} {}

        constexpr void operator()(auto i) const { m_output[i] = m_arange(i); }

    private:
        accessor_type m_output;
        Arange<value_type> m_arange;
    };

    /// Arange index-wise operator for 4d ranges.
    template<typename AccessorLike, typename Index>
    class Arange4d {
    public:
        using accessor_type = AccessorLike;
        using value_type = accessor_type::value_type;
        using index_type = Index;
        using shape4_type = Shape4<index_type>;
        using strides4_type = Strides4<index_type>;

    public:
        constexpr Arange4d(const accessor_type& accessor, const shape4_type& shape)
                : m_output(accessor), m_contiguous_strides(shape.strides()) {}

        constexpr Arange4d(
                const accessor_type& accessor,
                const shape4_type& shape,
                value_type start, value_type step
        ) : m_output(accessor),
            m_contiguous_strides(shape.strides()),
            m_arange{start, step} {}

        NOA_HD constexpr void operator()(auto indices) const {
            m_output(indices) = m_arange(ni::offset_at(indices, m_contiguous_strides));
        }

    private:
        accessor_type m_output;
        strides4_type m_contiguous_strides;
        Arange<value_type> m_arange;
    };
}
