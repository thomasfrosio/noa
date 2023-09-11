#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/algorithms/Algorithms.hpp"

#include "noa/core/geometry/Quaternion.hpp"

namespace noa::geometry {
    template<bool POST_CORRECTION, typename Coord, typename Index, typename Value>
    class GriddingCorrectionOp {
        static_assert(nt::is_int_v<Index>);
        static_assert(nt::are_real_v<Value, Coord>);

        using index_type = Index;
        using value_type = Value;
        using coord_type = Coord;

        using input_accessor_type = Accessor<const value_type, 4, index_type>;
        using output_accessor_type = Accessor<value_type, 4, index_type>;
        using shape3_type = Shape3<index_type>;
        using coord3_type = Vec3<coord_type>;

    public:
        GriddingCorrectionOp(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const Shape4<index_type>& shape
        )
                : m_input(input), m_output(output)
        {
            const shape3_type l_shape = shape.pop_front();
            m_f_shape = coord3_type(l_shape.vec());
            m_half = m_f_shape / 2 * coord3_type(l_shape != 1); // if size == 1, half should be 0
        }

        NOA_HD void operator()(index_type i, index_type j, index_type k, index_type l) const noexcept {
            coord3_type dist{j, k, l};
            dist -= m_half;
            dist /= m_f_shape;

            constexpr coord_type PI = noa::math::Constant<coord_type>::PI;
            const coord_type radius = noa::math::sqrt(noa::math::dot(dist, dist));
            const coord_type sinc = noa::math::sinc(PI * radius);
            const auto sinc2 = static_cast<value_type>(sinc * sinc); // > 0.05

            const value_type value = m_input(i, j, k, l);
            m_output(i, j, k, l) = POST_CORRECTION ? value / sinc2 : value * sinc2;
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        coord3_type m_f_shape;
        coord3_type m_half;
    };
}

namespace noa::geometry {
    template<bool POST_CORRECTION, typename Coord = f32, typename Index, typename Value>
    auto gridding_correction_op(
            const Accessor<const Value, 4, Index>& input,
            const Accessor<Value, 4, Index>& output,
            const Shape4<Index>& shape
    ) {
        return GriddingCorrectionOp<POST_CORRECTION, Coord, Index, Value>(input, output, shape);
    }
}
