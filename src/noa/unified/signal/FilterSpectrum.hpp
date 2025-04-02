#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/Iwise.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"

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
             nt::readable_nd_optional<N + 1> Input,
             nt::writable_nd<N + 1> Output,
             filterable_nd<N, Input> Filter>
    requires (nt::accessor_pure<Input> and (REMAP.is_hx2hx() or REMAP.is_fx2fx()))
    class FilterSpectrum {
    public:
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static constexpr bool IS_SRC_CENTERED = REMAP.is_xc2xx();

        using index_type = Index;
        using coord_type = Coord;
        using coord2_type = Vec<coord_type, 2>;
        using coord_nd_type = Vec<coord_type, N>;
        using shape_nd_type = Shape<index_type, N>;
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
            const filter_type& filter,
            coord2_type fftfreq_range,
            bool fftfreq_endpoint
        ) :
            m_input(input),
            m_output(output),
            m_fftfreq_start(fftfreq_range[0]),
            m_shape(shape.template pop_back<REMAP.is_hx2hx()>()),
            m_filter(std::move(filter))
        {
            // If frequency.end is negative, defaults to the highest frequency.
            // In this case, and if the frequency.start is 0, this results in the full frequency range.
            for (size_t i{}; i < N; ++i) {
                const auto max_sample_size = shape[i] / 2 + 1;
                const auto frequency_end =
                    fftfreq_range[1] <= 0 ?
                    noa::fft::highest_fftfreq<coord_type>(shape[i]) :
                    fftfreq_range[1];
                m_fftfreq_step[i] = Linspace{
                    .start = fftfreq_range[0],
                    .stop = frequency_end,
                    .endpoint = fftfreq_endpoint
                }.for_size(max_sample_size).step;
            }
        }

        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        constexpr void operator()(index_type batch, I... indices) const {
            const auto frequency = noa::fft::index2frequency<IS_SRC_CENTERED, IS_RFFT>(Vec{indices...}, m_shape);
            const auto fftfreq = coord_nd_type::from_vec(frequency) * m_fftfreq_step + m_fftfreq_start;

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
        coord_nd_type m_fftfreq_step;
        coord_type m_fftfreq_start;
        shape_type m_shape;
        filter_type m_filter;
    };

    template<size_t N, Remap REMAP, typename Input, typename Output>
    void check_filter_spectrum_parameters(const Input& input, const Output& output, const Shape4<i64>& shape) {
        check(not output.is_empty(), "Empty array detected");

        if constexpr (N == 1)
            check(shape[1] == 1 and shape[2] == 1, "1d spectra are expected, but got shape={}", shape);
        else if constexpr (N == 2)
            check(shape[1] == 1, "2d spectra are expected, but got shape={}", shape);

        const auto expected_output_shape = REMAP.is_xx2hx() ? shape.rfft() : shape;
        check(vall(Equal{}, output.shape(), expected_output_shape),
              "Given the logical shape {} and {} remap, the expected output shape should be {}, but got {}",
              shape, REMAP, expected_output_shape, output.shape());

        if (not input.is_empty()) {
            check(output.device() == input.device(),
                  "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
                  input.device(), output.device());

            const auto expected_input_shape = REMAP.is_hx2xx() ? shape.rfft() : shape;
            check(vall(Equal{}, input.shape(), expected_input_shape),
                  "Given the logical shape {} and {} remap, the expected input shape should be {}, but got {}",
                  shape, REMAP, expected_input_shape, input.shape());

            check(not REMAP.has_layout_change() or not ni::are_overlapped(input, output),
                  "In-place remapping is not allowed");
        }
    }
}

namespace noa::signal {
    struct FilterSpectrumOptions {
        /// Frequency [start, end] range of the input and output, from the zero, along the cartesian axes.
        /// If the end is negative or zero, it is set to the highest frequencies for the given dimensions,
        /// i.e. the entire rfft/fft range is selected. For even dimensions, this is equivalent to {0, 0.5}.
        Vec2<f64> fftfreq_range{0, -1};

        /// Whether the frequency_range's end should be included in the range.
        bool fftfreq_endpoint{true};
    };

    /// Filters a nd spectrum(s).
    ///
    /// \tparam N           Dimensionality of the spectrum. 1, 2, or 3.
    /// \tparam REMAP       Input and output layout.
    /// \param[in] input    Spectrum to filter. If empty, the filter is written into the output.
    /// \param[out] output  Filtered spectrum. Can be equal to the input (in-place filtering) if there's no remapping.
    ///                     If real and the filtered input is complex, the power spectrum of the filter input is saved.
    /// \param[in] shape    BDHW logical shape.
    /// \param[in] filter   Filter operator: filter(Vec<value_type, N> fftfreq, i64 batch) -> return_type.
    ///                     The Filter type can specialize value_type, otherwise, it defaults to f64 if the input is
    ///                     f64|c64, or to f32 if the input is f16|f32|c16|c32. The return_type should be real or
    ///                     complex.
    /// \param[in] options  Spectrum options.
    ///
    /// \note Like an iwise operator, each computing thread holds a copy of the given filter object.
    template<Remap REMAP, size_t N = 3,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>,
             guts::filterable_nd<N, Input> Filter>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and (REMAP.is_hx2hx() or REMAP.is_fx2fx()))
    void filter_spectrum(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        const Filter& filter,
        FilterSpectrumOptions options = {}
    ) {
        guts::check_filter_spectrum_parameters<N, REMAP>(input, output, shape);

        auto input_accessor = AccessorI64<nt::const_value_type_t<Input>, N + 1>(
            input.get(), input.strides().template filter_nd<N>());
        auto output_accessor = AccessorI64<nt::value_type_t<Output>, N + 1>(
            output.get(), output.strides().template filter_nd<N>());

        using coord_t = guts::filter_spectrum_default_coord_t<Input, Filter>;
        using op_t = guts::FilterSpectrum<
            N, REMAP, i64, coord_t, decltype(input_accessor), decltype(output_accessor), std::decay_t<Filter>>;
        auto op = op_t(input_accessor, output_accessor, shape.filter_nd<N>().pop_front(), filter,
                       options.fftfreq_range.as<coord_t>(), options.fftfreq_endpoint);

        iwise(output.shape().template filter_nd<N>(), output.device(), op,
              std::forward<Input>(input), std::forward<Output>(output));
    }

    /// Filters 1d spectrum(s).
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>,
             guts::filterable_nd<1, Input> Filter>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void filter_spectrum_1d(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        const Filter& filter,
        FilterSpectrumOptions options = {}
    ) {
        filter_spectrum<REMAP, 1>(std::forward<Input>(input), std::forward<Output>(output), shape, filter, options);
    }

    /// Filters 1|2d spectrum(s).
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>,
             guts::filterable_nd<2, Input> Filter>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void filter_spectrum_2d(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        const Filter& filter,
        FilterSpectrumOptions options = {}
    ) {
        filter_spectrum<REMAP, 2>(std::forward<Input>(input), std::forward<Output>(output), shape, filter, options);
    }

    /// Filters 1|2|3d spectrum(s).
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>,
             guts::filterable_nd<3, Input> Filter>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void filter_spectrum_3d(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        const Filter& filter,
        FilterSpectrumOptions options = {}
    ) {
        filter_spectrum<REMAP, 3>(std::forward<Input>(input), std::forward<Output>(output), shape, filter, options);
    }
}
