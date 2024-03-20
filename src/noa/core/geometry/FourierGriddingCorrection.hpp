#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/geometry/Quaternion.hpp"

namespace noa::geometry {
    /// Pre/post gridding correction, assuming linear interpolation.
    template<bool POST_CORRECTION, typename Coord, typename InputAccessor, typename OutputAccessor>
    requires (nt::are_accessor_pure_nd<4, InputAccessor, OutputAccessor>::value and nt::is_real<Coord>::value)
    class GriddingCorrection {
        using coord_type = Coord;
        using coord3_type = Vec3<coord_type>;
        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using output_value_type = nt::value_type_t<output_accessor_type>;
        using input_value_type = nt::value_type_t<input_accessor_type>;
        static_assert(nt::are_real_v<input_value_type, output_value_type>);

    public:
        template<typename T>
        GriddingCorrection(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const Shape4<T>& shape
        ) : m_input(input), m_output(output)
        {
            const auto l_shape = shape.pop_front();
            m_f_shape = coord3_type::from_vec(l_shape.vec);
            m_half = m_f_shape / 2 * coord3_type::from_vec(l_shape != 1); // if size == 1, half should be 0
        }

        template<std::integral T>
        NOA_HD void operator()(T batch, T j, T k, T l) const noexcept {
            auto dist = coord3_type::from_values(j, k, l);
            dist -= m_half;
            dist /= m_f_shape;

            constexpr coord_type PI = Constant<coord_type>::PI;
            const coord_type radius = sqrt(dot(dist, dist));
            const coord_type sinc = sinc(PI * radius);
            const auto sinc2 = static_cast<input_value_type>(sinc * sinc); // > 0.05

            const auto value = m_input(batch, j, k, l);
            if constexpr (POST_CORRECTION) {
                m_output(batch, j, k, l) = static_cast<output_value_type>(value / sinc2);
            } else {
                m_output(batch, j, k, l) = static_cast<output_value_type>(value * sinc2);
            }
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        coord3_type m_f_shape;
        coord3_type m_half;
    };
}
