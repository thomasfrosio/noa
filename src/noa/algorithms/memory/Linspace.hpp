#pragma once

#include "noa/core/Types.hpp"

namespace noa::algorithm::memory {
    template<typename Value>
    std::tuple<i64, Value, Value> linspace_range(i64 elements, Value start, Value stop, bool endpoint = true) {
        using compute_t =
                std::conditional_t<nt::is_complex_v<Value>, c64,
                std::conditional_t<nt::is_real_v<Value>, f64, Value>>;
        const auto count = elements - static_cast<i64>(endpoint);
        const auto delta = static_cast<compute_t>(stop) - static_cast<compute_t>(start);
        const auto step = delta / static_cast<compute_t>(count);
        return {count, static_cast<Value>(delta), static_cast<Value>(step)};
    }

    template<typename Value>
    Value linspace_step(i64 elements, Value start, Value stop, bool endpoint = true) {
        const auto [count, delta, step] = linspace_range(elements, start, stop, endpoint);
        return step;
    }

    template<typename Index, typename Offset, typename Value>
    class Linspace4D {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using shape4_type = Shape4<Index>;
        using strides4_type = Strides4<Index>;
        using accessor_type = Accessor<value_type, 4, offset_type>;
        using compute_type =
                std::conditional_t<std::is_same_v<value_type, f16>, f32,
                std::conditional_t<std::is_same_v<value_type, c16>, c32, value_type>>;

    public:
        Linspace4D(const accessor_type& accessor,
                   const shape4_type& shape,
                   value_type start, value_type step,
                   value_type stop, bool endpoint)
                : m_output(accessor),
                  m_index_end(shape - 1),
                  m_contiguous_strides(shape.strides()),
                  m_start(static_cast<compute_type>(start)),
                  m_step(static_cast<compute_type>(step)),
                  m_stop(static_cast<compute_type>(stop)),
                  m_endpoint(endpoint) {}

        NOA_HD constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const {
            if (m_endpoint &&
                ii == m_index_end[0] &&
                ii == m_index_end[1] &&
                ii == m_index_end[2] &&
                ii == m_index_end[3]) {
                m_output(ii, ij, ik, il) = static_cast<value_type>(m_stop);
            } else {
                const auto offset = static_cast<compute_type>(indexing::at(ii, ij, ik, il, m_contiguous_strides));
                m_output(ii, ij, ik, il) = static_cast<value_type>(m_start + offset * m_step);
            }
        }

    private:
        accessor_type m_output;
        shape4_type m_index_end;
        strides4_type m_contiguous_strides;
        compute_type m_start;
        compute_type m_step;
        compute_type m_stop;
        bool m_endpoint;
    };

    template<typename Index, typename Offset, typename Value>
    auto linspace_4d(Value* output, const Strides4<i64>& strides, const Shape4<i64>& shape,
                     Value start, Value stop, bool endpoint) {
        const auto step = linspace_step(shape.elements(), start, stop, endpoint);
        return std::pair{
                Linspace4D<Index, Offset, Value>(
                        Accessor<Value, 4, Offset>(output, strides.as_safe<Offset>()),
                        shape.as_safe<Index>(), start, step, stop, endpoint),
                step
        };
    }

    // Optimization for C-contiguous arrays.
    template<typename Index, typename Offset, typename Value>
    class Linspace1D {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using accessor_type = AccessorContiguous<value_type, 1, offset_type>;

    public:
        Linspace1D(const accessor_type& accessor,
                   const index_type& size,
                   value_type start, value_type step,
                   value_type stop, bool endpoint)
                : m_output(accessor),
                  m_index_end(size - 1),
                  m_start(start),
                  m_step(step),
                  m_stop(stop),
                  m_endpoint(endpoint) {}

        NOA_HD constexpr void operator()(index_type i) const {
            m_output[i] = m_endpoint && i == m_index_end ?
                          m_stop : m_start + static_cast<value_type>(i) * m_step;
        }

    private:
        accessor_type m_output;
        index_type m_index_end;
        value_type m_start;
        value_type m_step;
        value_type m_stop;
        bool m_endpoint;
    };

    template<typename Index, typename Offset, typename Value>
    auto linspace_1d(Value* output, i64 size, Value start, Value stop, bool endpoint) {
        const auto step = linspace_step(size, start, stop, endpoint);
        return std::pair{
                Linspace1D<Index, Offset, Value>(
                        AccessorContiguous<Value, 1, Offset>(output),
                        safe_cast<Index>(size), start, step, stop, endpoint),
                step
        };
    }
}
