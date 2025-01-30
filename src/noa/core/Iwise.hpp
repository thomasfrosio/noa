#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/indexing/Offset.hpp"

namespace noa {
    /// Defines evenly spaced values within a given interval.
    template<nt::numeric Value>
    struct Arange {
        using value_type = Value;
        using real_type = nt::value_type_t<value_type>;

        Value start{0};
        Value step{1};

        [[nodiscard]] NOA_FHD constexpr Value operator()(nt::integer auto index) const noexcept {
            return start + static_cast<real_type>(index) * step;
        }

        template<nt::numeric U>
        [[nodiscard]] NOA_FHD constexpr auto as() const -> Arange<U> {
            return {static_cast<U>(start), static_cast<U>(step)};
        }
    };

    /// Defines evenly spaced numbers over a specified interval.
    template<nt::numeric T>
    struct Linspace {
        using value_type = T;
        using real_type = nt::value_type_t<value_type>;

        value_type start;
        value_type stop;
        bool endpoint{true};

        template<nt::integer I>
        struct Op {
            using value_type = T;
            using index_type = I;

            value_type start;
            value_type step;
            value_type stop;
            index_type index_end;
            bool endpoint;

            [[nodiscard]] NOA_FHD constexpr value_type operator()(index_type i) const noexcept {
                return endpoint and i == index_end ? stop : start + static_cast<real_type>(i) * step;
            }
        };

        template<nt::numeric U>
        [[nodiscard]] NOA_FHD constexpr auto as() const -> Linspace<U> {
            return {static_cast<U>(start), static_cast<U>(stop), endpoint};
        }

        template<nt::integer I>
        [[nodiscard]] NOA_FHD constexpr auto for_size(const I& size) const -> Op<I> requires nt::scalar<value_type> {
            Op<I> op;
            op.start = start;
            op.stop = stop;
            op.index_end = max(I{}, size - 1);
            op.endpoint = endpoint;

            const auto count = max(I{1}, size - static_cast<I>(endpoint));
            const auto delta = stop - start;
            op.step = delta / static_cast<value_type>(count);
            return op;
        }

        template<nt::integer I>
        [[nodiscard]] NOA_FHD constexpr auto for_size(const I& size) const -> Op<I> requires nt::complex<value_type> {
            auto real = Linspace<real_type>{start.real, stop.real, endpoint}.for_size(size);
            auto imag = Linspace<real_type>{start.imag, stop.imag, endpoint}.for_size(size);
            return Op{
                .start = {real.start, imag.start},
                .step = {real.step, imag.step},
                .stop = {real.stop, imag.stop},
                .index_end = real.index_end,
                .endpoint = real.endpoint
            };
        }
    };
}

namespace noa::guts {
    /// Arange index-wise operator for nd ranges.
    template<size_t N, nt::writable_nd<N> T, nt::integer I, typename R>
    class IwiseRange {
    public:
        using output_type = T;
        using index_type = I;
        using range_type = R;
        using value_type = nt::value_type_t<output_type>;
        using shape_type = Shape<index_type, N>;

        static constexpr bool CONTIGUOUS_1D = N == 1 and nt::marked_contiguous<T>;
        using strides_type = std::conditional_t<CONTIGUOUS_1D, Empty, Strides<index_type, N>>;

    public:
        constexpr IwiseRange(
            const output_type& accessor,
            const shape_type& shape,
            const range_type& range
        ) requires (not CONTIGUOUS_1D) :
            m_output(accessor),
            m_contiguous_strides(shape.strides()),
            m_range{range} {}

        constexpr IwiseRange(
            const output_type& accessor,
            const shape_type&, // for CTAD
            const range_type& range
        ) requires CONTIGUOUS_1D :
            m_output(accessor),
            m_range{range} {}

        template<typename... U> requires nt::iwise_core_indexing<N, index_type, U...>
        NOA_HD constexpr void operator()(U... indices) const {
            if constexpr (CONTIGUOUS_1D)
                m_output(indices...) = static_cast<value_type>(m_range(indices...));
            else
                m_output(indices...) = static_cast<value_type>(m_range(ni::offset_at(m_contiguous_strides, indices...)));
        }

    private:
        output_type m_output;
        NOA_NO_UNIQUE_ADDRESS strides_type m_contiguous_strides;
        range_type m_range;
    };

    template<size_t N, nt::writable_nd<N> T, nt::integer I>
    class Iota {
    public:
        using output_type = T;
        using index_type = I;
        using indices_type = Vec<I, N>;
        using shape_type = Shape<I, N>;
        using value_type = nt::value_type_t<output_type>;

        static constexpr bool CONTIGUOUS_1D = N == 1 and nt::marked_contiguous<T>;
        using strides_type = std::conditional_t<CONTIGUOUS_1D, Empty, Strides<index_type, N>>;

    public:
        constexpr Iota(
            const output_type& accessor,
            const shape_type& shape,
            const indices_type& tile
        ) requires (not CONTIGUOUS_1D) :
            m_output(accessor),
            m_tile(tile),
            m_contiguous_strides(shape.strides()) {}

        constexpr Iota(
            const output_type& accessor,
            const shape_type&, // for CTAD
            const indices_type& tile
        ) requires CONTIGUOUS_1D :
            m_output(accessor),
            m_tile(tile) {}

        NOA_HD constexpr void operator()(const indices_type& indices) const {
            if constexpr (CONTIGUOUS_1D) {
                m_output(indices) = static_cast<value_type>(indices % m_tile);
            } else {
                const auto iota = ni::offset_at(m_contiguous_strides, indices % m_tile);
                m_output(indices) = static_cast<value_type>(iota);
            }
        }

    private:
        output_type m_output;
        indices_type m_tile;
        NOA_NO_UNIQUE_ADDRESS strides_type m_contiguous_strides;
    };
}
