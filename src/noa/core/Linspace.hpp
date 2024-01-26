#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/indexing/Offset.hpp"

namespace noa {
    /// Defines evenly spaced numbers over a specified interval.
    template<typename Value, typename Index>
    requires (nt::is_numeric_v<Value> and nt::is_int_v<Index>)
    struct Linspace {
        Value start;
        Value step;
        Value stop;
        Index index_end;
        bool endpoint;

        NOA_HD static constexpr auto from_range(
                Value start,
                Value stop,
                const Index& size,
                bool endpoint = true
        ) -> Linspace requires nt::is_scalar_v<Value> {
            Linspace linspace;
            linspace.start = start;
            linspace.stop = stop;
            linspace.start = start;
            linspace.index_end = min(Index{0}, size - 1);
            linspace.endpoint = endpoint;

            const auto count = size - static_cast<Index>(endpoint);
            const auto delta = stop - start;
            linspace.step = delta / static_cast<Value>(count);
            return linspace;
        }

        NOA_HD static constexpr auto from_range(
                Value start,
                Value stop,
                const Index& size,
                bool endpoint = true
        ) -> Linspace requires nt::is_complex_v<Value> {
            using real_t = nt::value_type_t<Value>;
            using linspace_t = Linspace<real_t, Index>;
            linspace_t real = linspace_t::from_range(start.real, stop.real, size, endpoint);
            linspace_t imag = linspace_t::from_range(start.real, stop.real, size, endpoint);
            return {.start={real.start, imag.start},
                    .step={real.step, imag.step},
                    .stop={real.stop, imag.stop},
                    .index_end=real.index_end,
                    .endpoint=real.endpoint};
        }

        [[nodiscard]] NOA_HD constexpr Value operator()(Index i) const noexcept {
            using real_t = nt::value_type_t<Value>;
            return endpoint and i == index_end ? stop : start + static_cast<real_t>(i) * step;
        }
    };

    template<typename Accessor, typename Index>
    class Linspace1d {
    public:
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using index_type = Index;
        using linspace_t = Linspace<value_type, index_type>;

    public:
        constexpr Linspace1d(
                const accessor_type& accessor,
                const index_type& size,
                value_type start,
                value_type stop,
                bool endpoint
        ) : m_output(accessor),
            m_linspace(linspace_t::from_range(start, stop, size, endpoint)) {}

        constexpr Linspace1d(
                const accessor_type& accessor,
                const index_type& size,
                value_type start,
                value_type step,
                value_type stop,
                bool endpoint
        ) : m_output(accessor),
            m_linspace{start, step, stop, min(index_type{0}, size - 1), endpoint} {}

        NOA_HD constexpr void operator()(index_type i) const {
            m_output[i] = m_linspace(i);
        }

    private:
        accessor_type m_output;
        linspace_t m_linspace;
    };

    template<typename Accessor, typename Index>
    class Linspace4d {
    public:
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using index_type = Index;
        using linspace_t = Linspace<value_type, index_type>;
        using shape4_type = Shape4<Index>;
        using strides4_type = Strides4<Index>;

    public:
        constexpr Linspace4d(
                const accessor_type& accessor,
                const shape4_type& shape,
                value_type start,
                value_type stop,
                bool endpoint
        ) : m_output(accessor),
            m_contiguous_strides(shape.strides()),
            m_linspace(linspace_t::from_range(start, stop, shape.elements() - 1, endpoint)) {}

        constexpr Linspace4d(
                const accessor_type& accessor,
                const shape4_type& shape,
                value_type start,
                value_type step,
                value_type stop,
                bool endpoint
        ) : m_output(accessor),
            m_contiguous_strides(shape.strides()),
            m_linspace(start, step, stop, min(index_type{0}, shape.elements() - 1), endpoint) {}

        NOA_HD constexpr void operator()(const auto& indices) const {
            m_output(indices) = m_linspace(ni::offset_at(indices, m_contiguous_strides));
        }

    private:
        accessor_type m_output;
        strides4_type m_contiguous_strides;
        linspace_t m_linspace;
    };
}
