#pragma once

#include "noa/fft/core/Frequency.hpp"
#include "noa/fft/core/Layout.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/core/Iwise.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/Iwise.hpp"

namespace noa::signal::details {
    template<typename Input, typename Filter = Empty,
             typename InputReal = nt::mutable_value_type_twice_t<Input>,
             typename InputCoord = std::conditional_t<(sizeof(InputReal) < 4), f32, InputReal>>
    using filter_spectrum_default_coord_t =
        std::conditional_t<nt::has_value_type_v<Filter>, nt::value_type_t<Filter>, InputCoord>;

    template<typename Filter = Empty>
    using filter_spectrum_default_index_t =
        std::conditional_t<nt::has_index_type_v<Filter>, nt::index_type_t<Filter>, isize>;

    struct FilterSpectrumChecker {
        template<typename Op, usize B, usize R, typename Coord, typename Index>
        static consteval bool check_0 () {
            return requires {
                { std::declval<Op&>()(std::declval<const Vec<Coord, R>&>(), std::declval<const Vec<Index, B>&>()) } -> nt::real_or_complex;
            };
        }

        template<typename Op, usize B, usize R, typename Coord, typename Index>
        static consteval bool check_1() {
            return []<Index... I>(std::integer_sequence<Index, I...>) {
                return requires {
                    { std::declval<Op&>()(std::declval<const Vec<Coord, R>&>(), std::declval<I>()...) } -> nt::real_or_complex;
                };
            }(std::make_integer_sequence<Index, B>{});
        }

        template<typename Op, usize, usize R, typename Coord, typename>
        static consteval bool check_2() {
            return requires { { std::declval<Op&>()(std::declval<const Vec<Coord, R>&>()) } -> nt::real_or_complex; };
        }

        template<typename Op, usize B, usize R, typename Coord, typename Index>
        static consteval bool is_valid() {
            return check_0<Op, B, R, Coord, Index>() or
                   check_1<Op, B, R, Coord, Index>() or
                   check_2<Op, B, R, Coord, Index>();
        }
    };

    template<typename Input, typename Filter, usize B, usize R,
             typename Coord = filter_spectrum_default_coord_t<Input, Filter>,
             typename Index = filter_spectrum_default_index_t<Filter>>
    concept filterable_nd =
        nt::any_of<Coord, f32, f64> and nt::integer<Index> and
        FilterSpectrumChecker::is_valid<std::decay_t<Filter>, B, R, Coord, Index>();

    template<nf::Layout REMAP, usize R, typename Input, typename Output, usize N>
    concept filter_spectrum_able =
        nt::readable_array_decay<Input> and
        nt::writable_array_decay<Output> and
        nt::array_decay_with_spectrum_types<Input, Output> and
        nt::array_size_v<Input> == N and nt::array_size_v<Output> == N and
        N >= R and (REMAP.is_hx2hx() or REMAP.is_fx2fx());

    template<nf::Layout REMAP,
             nt::integer Index, nt::real Coord,
             usize B, usize R,
             nt::readable_nd_optional<B + R> Input,
             nt::writable_nd<B + R> Output,
             typename Filter>
    class FilterSpectrum {
    public:
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static constexpr bool IS_SRC_CENTERED = REMAP.is_xc2xx();

        using index_type = Index;
        using coord_type = Coord;
        using coord_nd_type = Vec<coord_type, R>;
        using shape_nd_type = Shape<index_type, R>;
        using shape_type = Shape<index_type, R - REMAP.is_hx2hx()>;
        using coord_or_empty_type = std::conditional_t<R == 1, coord_type, Empty>;

        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using filter_type = Filter;

        using filter_value_type = std::decay_t<std::invoke_result_t<filter_type&, coord_nd_type, index_type>>;
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
            filter_type filter,
            const Linspace<coord_type>& fftfreq_range
        ) :
            m_input(input),
            m_output(output),
            m_shape(shape.template pop_back<REMAP.is_hx2hx()>()),
            m_filter(std::move(filter))
        {
            // If frequency.end is negative, defaults to the highest frequency.
            // In this case, and if the frequency.start is 0, this results in the full frequency range.
            for (usize i{}; i < R; ++i) {
                const auto max_sample_size = shape[i] / 2 + 1;
                const auto frequency_end =
                    fftfreq_range.stop <= 0 ?
                    nf::highest_fftfreq<coord_type>(shape[i]) :
                    fftfreq_range.stop;
                m_fftfreq_step[i] = Linspace{
                    .start = fftfreq_range.start,
                    .stop = frequency_end,
                    .endpoint = fftfreq_range.endpoint
                }.for_size(max_sample_size).step;
            }
            if constexpr (R == 1)
                m_fftfreq_start = fftfreq_range.start;
        }

        constexpr auto get_filter(const coord_nd_type& fftfreq, const Vec<index_type, B>& batches) {
            if constexpr (FilterSpectrumChecker::check_0<filter_type, B, R, coord_nd_type, index_type>()) {
                return m_filter(fftfreq, batches);
            } else if constexpr (FilterSpectrumChecker::check_0<filter_type, B, R, coord_nd_type, index_type>()) {
                return [&]<Index... I>(std::integer_sequence<Index, I...>) {
                    return m_filter(fftfreq, batches[I]...);
                }(std::make_integer_sequence<Index, B>{});
            } else {
                return m_filter(fftfreq);
            }
        }

        constexpr void operator()(const Vec<index_type, R>& batched_indices) {
            const auto [batches, indices] = batched_indices.template split<B>();
            const auto frequency = nf::index2frequency<IS_SRC_CENTERED, IS_RFFT>(indices, m_shape);
            auto fftfreq = coord_nd_type::from_vec(frequency) * m_fftfreq_step;
            if constexpr (R == 1)
                fftfreq += m_fftfreq_start;

            const auto filter = get_filter(fftfreq, batches);
            const auto output_indices = nf::remap_indices<REMAP>(indices, m_shape);
            auto& output = m_output(output_indices.push_front(batches));

            if (m_input) {
                output = cast_or_abs_squared<output_value_type>(
                    static_cast<input_result_type>(m_input(batched_indices)) *
                    static_cast<filter_result_type>(filter));
            } else {
                output = cast_or_abs_squared<output_value_type>(filter);
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        coord_nd_type m_fftfreq_step;
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_fftfreq_start;
        shape_type m_shape;
        filter_type m_filter;
    };

    template<usize N, nf::Layout REMAP, typename Input, typename Output>
    void check_filter_spectrum_parameters(
        const Input& input,
        const Output& output,
        const Shape4& shape,
        noa::Linspace<f64>& fftfreq_range
    ) {
        check(not output.is_empty(), "Empty array detected");

        const auto expected_output_shape = REMAP.is_xx2hx() ? shape.rfft() : shape;
        check(output.shape() == expected_output_shape,
              "Given the logical shape {} and {} remap, the expected output shape should be {}, but got {}",
              shape, REMAP, expected_output_shape, output.shape());

        if (not input.is_empty()) {
            check(output.device() == input.device(),
                  "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
                  input.device(), output.device());

            const auto expected_input_shape = REMAP.is_hx2xx() ? shape.rfft() : shape;
            check(input.shape() == expected_input_shape,
                  "Given the logical shape {} and {} remap, the expected input shape should be {}, but got {}",
                  shape, REMAP, expected_input_shape, input.shape());

            check(not REMAP.has_layout_change() or not are_overlapped(input, output),
                  "In-place remapping is not allowed");
        }
        check(N == 1 or allclose(fftfreq_range.start, 0.),
              "For multidimensional cases, the starting fftfreq should be 0, but got {}", fftfreq_range.start);
    }
}

namespace noa::signal {
    struct FilterSpectrumOptions {
        /// Frequency range of the input and output, from zero, along the cartesian axes.
        /// If the end is negative or zero, it is set to the highest frequencies for the given dimensions,
        /// meaning the entire rfft/fft range is selected. For even dimensions, this is equivalent to {0, 0.5}.
        /// For 2d and 3d cases, the start should be 0, otherwise, an error will be thrown.
        noa::Linspace<f64> fftfreq_range{.start = 0, .stop = -1, .endpoint = true};
    };

    /// Filters a nd spectrum(s).
    /// \tparam R:
    ///     Rank of the spectrum. 1, 2, or 3.
    /// \tparam REMAP:
    ///     Input and output layout.
    /// \param[in] input:
    ///     Spectrum to filter.
    ///     R=1 ((B..,)W)
    ///     R=2 ((B..,)HW)
    ///     R=3 ((B..,)DHW)
    ///     If empty, the filter is written into the output.
    /// \param[out] output
    ///     Filtered spectrum, with the same shape (or half-shape depending on REMAP) as the input.
    ///     Can be equal to the input (in-place filtering) if there's no remapping.
    ///     If real and the filtered input is complex, the power spectrum of the filter input is saved.
    /// \param[in] shape
    ///     Logical shape of the input and output.
    /// \param[in] filter:
    ///     Filter operator:
    ///         filter(Vec<value_type, R> fftfreq, const Vec<index_type, N-R>& batches) -> real|complex, or
    ///         filter(Vec<value_type, R> fftfreq, const index_type& batches...) -> real|complex, or
    ///         filter(Vec<value_type, R> fftfreq) -> real|complex.
    ///     The Filter type can specialize value_type, otherwise, it defaults to f64 if the input is f64|c64,
    ///     or to f32 if the input is f16|f32|c16|c32. Similarly, it can specialize the index_type, otherwise,
    ///     isize is used. Like an iwise operator, each computing thread holds a copy of the given filter object.
    /// \param[in] options:
    ///     Spectrum options.
    template<nf::Layout REMAP, usize R = 3, typename Output, usize N, typename Input = Output, typename Filter>
        requires (details::filter_spectrum_able<REMAP, R, Input, Output, N> and
                  details::filterable_nd<Input, Filter, N - R, R>)
    void filter_spectrum(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        Filter&& filter,
        FilterSpectrumOptions options = {}
    ) {
        details::check_filter_spectrum_parameters<R, REMAP>(input, output, shape, options.fftfreq_range);

        constexpr auto B = N - R;
        using coord_t = details::filter_spectrum_default_coord_t<Input, Filter>;
        using index_t = details::filter_spectrum_default_index_t<Filter>;
        using input_t = Accessor<nt::const_value_type_t<Input>, N, index_t>;
        using output_t = Accessor<nt::value_type_t<Output>, N, index_t>;
        auto input_br = input.template span<nt::const_value_type_t<Input>, N, index_t>().accessor();
        auto output_br = output.template span<nt::value_type_t<Input>, N, index_t>().accessor();
        auto shape_r = shape.template pop_front<B>().template as<index_t>();

        using op_t = details::FilterSpectrum<REMAP, index_t, coord_t, B, R, input_t, output_t, std::decay_t<Filter>>;
        auto op = op_t(input_br, output_br, shape_r, std::forward<Filter>(filter), options.fftfreq_range.as<coord_t>());

        iwise(output.shape(), output.device(), std::move(op),
              std::forward<Input>(input), std::forward<Output>(output));
    }

    /// Filters 1d spectrum(s).
    template<nf::Layout REMAP,
             typename Output, typename Filter, usize N,
             typename Input = Array<nt::value_type_t<Output>, N, ArrayOwnership::VIEW>>
    void filter_spectrum_1d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        Filter&& filter,
        FilterSpectrumOptions options = {}
    ) {
        filter_spectrum<REMAP, 1>(
            std::forward<Input>(input), std::forward<Output>(output), shape,
            std::forward<Filter>(filter), options
        );
    }

    /// Filters 1|2d spectrum(s).
    template<nf::Layout REMAP,
             typename Output, typename Filter, usize N,
             typename Input = Array<nt::value_type_t<Output>, N, ArrayOwnership::VIEW>>
    void filter_spectrum_2d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        Filter&& filter,
        FilterSpectrumOptions options = {}
    ) {
        filter_spectrum<REMAP, 2>(
            std::forward<Input>(input), std::forward<Output>(output), shape,
            std::forward<Filter>(filter), options
        );
    }

    /// Filters 1|2|3d spectrum(s).
    template<nf::Layout REMAP,
             typename Output, typename Filter, usize N,
             typename Input = Array<nt::value_type_t<Output>, N, ArrayOwnership::VIEW>>
    void filter_spectrum_3d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        Filter&& filter,
        FilterSpectrumOptions options = {}
    ) {
        filter_spectrum<REMAP, 3>(
            std::forward<Input>(input), std::forward<Output>(output), shape,
            std::forward<Filter>(filter), options
        );
    }
}
