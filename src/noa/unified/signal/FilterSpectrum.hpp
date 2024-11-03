#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/signal/FilterSpectrum.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::signal::guts {
    template<size_t N, Remap REMAP, typename Input, typename Output>
    void check_filter_spectrum_parameters(const Input& input, const Output& output, const Shape4<i64>& shape) {
        check(not output.is_empty(), "Empty array detected");

        if constexpr (N == 1)
            check(shape[1] == 0 and shape[2] == 0, "1d spectra are expected, but got shape={}", shape);
        else if constexpr (N == 2)
            check(shape[1] == 0, "2d spectra are expected, but got shape={}", shape);

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
    ///
    /// \note Like an iwise operator, each computing thread holds a copy of the given filter object.
    template<Remap REMAP, size_t N = 3,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>,
             guts::filterable_nd<N, Input> Filter>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void filter_spectrum(Input&& input, Output&& output, const Shape4<i64>& shape, const Filter& filter) {
        guts::check_filter_spectrum_parameters<N, REMAP>(input, output, shape);

        auto input_accessor = AccessorI64<nt::const_value_type_t<Input>, N + 1>(
            input.get(), input.strides().template filter_nd<N>());
        auto output_accessor = AccessorI64<nt::value_type_t<Output>, N + 1>(
            output.get(), output.strides().template filter_nd<N>());

        using coord_t = guts::filter_spectrum_default_coord_t<Input, Filter>;
        using op_t = guts::FilterSpectrum<
            N, REMAP, i64, coord_t, decltype(input_accessor), decltype(output_accessor), std::decay_t<Filter>>;
        auto op = op_t(input_accessor, output_accessor, shape.filter_nd<N>(), filter);

        iwise(output.shape().template filter_nd<N>(), output.device(), op,
              std::forward<Input>(input), std::forward<Output>(output));
    }

    /// Filters 1d spectrum(s).
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>,
             guts::filterable_nd<1, Input> Filter>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void filter_spectrum_1d(Input&& input, Output&& output, const Shape4<i64>& shape, const Filter& filter) {
        filter_spectrum<1, REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, filter);
    }

    /// Filters 1|2d spectrum(s).
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>,
             guts::filterable_nd<2, Input> Filter>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void filter_spectrum_2d(Input&& input, Output&& output, const Shape4<i64>& shape, const Filter& filter) {
        filter_spectrum<2, REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, filter);
    }

    /// Filters 1|2|3d spectrum(s).
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>,
             guts::filterable_nd<3, Input> Filter>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void filter_spectrum_3d(Input&& input, Output&& output, const Shape4<i64>& shape, const Filter& filter) {
        filter_spectrum<3, REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, filter);
    }
}
