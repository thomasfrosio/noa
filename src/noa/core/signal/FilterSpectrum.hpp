#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::signal::guts {
    template<typename Input, typename Filter = Empty,
             typename InputReal = nt::mutable_value_type_twice_t<Input>,
             typename InputCoord = std::conditional_t<(sizeof(InputReal) < 4), f32, InputReal>>
    using filter_spectrum_default_coord_t =
        std::conditional_t<nt::has_value_type_v<Filter>, nt::value_type_t<Filter>, InputCoord>;

    template<typename Filter, size_t N, typename Input,
             typename Coord = filter_spectrum_default_coord_t<Input, Filter>>
    concept filterable_nd =
        (not nt::has_value_type_v<Filter> or nt::any_of<nt::value_type_t<Filter>, f32, f64>) and
        nt::real_or_complex<std::invoke_result_t<const Filter&, const Vec<Coord, N>&, i64>>; // nvcc bug - use requires

    template<size_t N, Remap REMAP,
             nt::integer Index,
             nt::real Coord,
             nt::readable_nd_optional<4> Input,
             nt::writable_nd<4> Output,
             filterable_nd<N, Input> Filter>
    requires (nt::accessor_pure<Input> and (REMAP.is_hx2hx() or REMAP.is_fx2fx()))
    class FilterSpectrum {
    public:
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static constexpr bool IS_SRC_CENTERED = REMAP.is_xc2xx();

        using index_type = Index;
        using coord_type = Coord;
        using coord_nd_type = Vec<coord_type, N>;
        using shape_nd_type = Shape<index_type, N + 1>;
        using shape_type = Shape<index_type, N - REMAP.is_hx2hx()>;

        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using filter_type = Filter;

        using filter_value_type = std::decay_t<std::invoke_result_t<const filter_type&, coord_nd_type, index_type>>;
        using filter_real_type = nt::value_type_t<filter_value_type>;
        static_assert(nt::real_or_complex<filter_value_type>);

        using real_type = std::conditional_t<nt::any_of<f64, filter_real_type, input_real_type>, f64, f32>;
        using filter_result_type = std::conditional_t<nt::complex<filter_value_type>, Complex<real_type>, real_type>;
        using input_result_type = std::conditional_t<nt::complex<input_value_type>, Complex<real_type>, real_type>;

    public:
        constexpr FilterSpectrum(
            const input_type& input,
            const output_type& output,
            const shape_nd_type& shape,
            const filter_type& filter
        ) :
            m_input(input),
            m_output(output),
            m_norm(coord_type{1} / coord_nd_type::from_vec(shape.pop_front().vec)),
            m_shape(shape.pop_front().template pop_back<REMAP.is_hx2hx()>()),
            m_filter(std::move(filter)) {}

        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        constexpr void operator()(index_type batch, I... indices) const {
            const auto frequency = noa::fft::index2frequency<IS_SRC_CENTERED, IS_RFFT>(Vec{indices...}, m_shape);
            const auto fftfreq = coord_nd_type::from_vec(frequency) * m_norm;

            const auto filter = m_filter(fftfreq, batch);
            const auto output_indices = noa::fft::remap_indices<REMAP>(Vec{indices...}, m_shape);
            auto& output = m_output(output_indices.push_front(batch));

            if (m_input) {
                output = cast_or_abs_squared<output_value_type>(
                    static_cast<input_result_type>(m_input(batch, indices...)) *
                    static_cast<filter_result_type>(filter));
            } else {
                output = cast_or_abs_squared<output_value_type>(filter);
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        coord_nd_type m_norm;
        shape_type m_shape;
        filter_type m_filter;
    };
}
