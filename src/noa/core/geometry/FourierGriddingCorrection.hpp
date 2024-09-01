#pragma once

#include "noa/core/types/Shape.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::geometry {
    /// Pre/post gridding correction, assuming linear interpolation.
    template<bool POST_CORRECTION,
             nt::real Coord,
             nt::readable_nd<4> Input,
             nt::writable_nd<4> Output>
    class GriddingCorrection {
        using coord_type = Coord;
        using coord3_type = Vec3<coord_type>;
        using input_type = Input;
        using output_type = Output;
        using output_value_type = nt::value_type_t<output_type>;
        using input_value_type = nt::value_type_t<input_type>;
        static_assert(nt::are_real_v<input_value_type, output_value_type>);

    public:
        template<typename T>
        constexpr GriddingCorrection(
                const input_type& input,
                const output_type& output,
                const Shape4<T>& shape
        ) :
                m_input(input),
                m_output(output)
        {
            const auto l_shape = shape.pop_front();
            m_f_shape = coord3_type::from_vec(l_shape.vec);
            m_half = m_f_shape / 2 * coord3_type::from_vec(l_shape != 1); // if size == 1, half should be 0
        }

        template<nt::integer T>
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
        input_type m_input;
        output_type m_output;
        coord3_type m_f_shape;
        coord3_type m_half;
    };
}
