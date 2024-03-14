#pragma once

#include "noa/core/fft/RemapInterface.hpp"
#include "noa/core/signal/PhaseShift.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Factory.hpp"
#include "noa/unified/fft/Remap.hpp"

namespace noa::signal::guts {
    template<noa::fft::Remap REMAP, typename Input, typename Output, typename Shift>
    void check_phase_shift_parameters(
            const Input& input, const Output& output,
            const Shape4<i64>& shape, const Shift& shifts
    ) {
        check(not output.is_empty(), "Empty array detected");
        check(all(output.shape() == shape.rfft()),
              "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
              shape, shape.rfft(), output.shape());

        if (not input.is_empty()) {
            check(output.device() == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), output.device());
            check(REMAP == noa::fft::Remap::H2H or
                  REMAP == noa::fft::Remap::HC2HC or
                  input.get() != output.get(),
                  "In-place remapping is not allowed");
        }

        if constexpr (nt::is_varray_v<Shift>) {
            check(ni::is_contiguous_vector(shifts) and shifts.elements() == output.shape()[0],
                  "The input shift(s) should be entered as a 1D contiguous vector, with one shift per output batch, "
                  "but got shift {} and output {}", shifts.shape(), output.shape());
            check(output.device() == shifts.device(),
                  "The shift and output arrays must be on the same device, but got shifts:{}, output:{}",
                  shifts.device(), output.device());
        }
    }

    template<noa::fft::Remap REMAP, typename Input, typename Output>
    void no_phase_shift(const Input& input, const Output& output, const Shape4<i64>& shape) {
        using Layout = noa::fft::Layout;
        constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        constexpr bool NO_REMAP = bool(u8_REMAP & Layout::SRC_CENTERED) == bool(u8_REMAP & Layout::DST_CENTERED);

        if (input.is_empty()) {
            using value_t = nt::value_type_t<Output>;
            fill(output, value_t{1, 0});
        } else {
            if constexpr (NO_REMAP)
                copy(input, output);
            else
                noa::fft::remap(REMAP, input, output, shape);
        }
    }

    template<typename Shift>
    auto extract_shift(const Shift& shift) {
        if constexpr (nt::is_vec_v<Shift>) {
            return shift;
        } else {
            using ptr_t = const nt::value_type_t<Shift>*;
            return ptr_t(shift.get());
        }
    }
}

namespace noa::signal {
    /// Phase-shifts 2d rfft(s).
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \param[in] input        2d rfft to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output      Phase-shifted 2d rfft.
    /// \param shape            BDHW logical shape.
    /// \param[in] shifts       HW 2d phase-shift to apply.
    ///                         A single value or a contiguous vector with one shift per batch.
    /// \param fftfreq_cutoff   Maximum output frequency to consider, in cycle/pix.
    ///                         Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<noa::fft::RemapInterface REMAP, typename Output, typename Shift,
             typename Input = View<const nt::value_type_t<Output>>>
    requires (nt::are_varray_of_complex_v<Input, Output> and
              (nt::is_varray_of_almost_any_v<Shift, Vec2<f32>, Vec2<f64>> or nt::is_vec_real_size_v<Shift, 2>) and
              (REMAP.remap == noa::fft::Remap::H2H or
               REMAP.remap == noa::fft::Remap::H2HC or
               REMAP.remap == noa::fft::Remap::HC2H or
               REMAP.remap == noa::fft::Remap::HC2HC))
    void phase_shift_2d(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const Shift& shifts,
            f64 fftfreq_cutoff = 1
    ) {
        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        guts::check_phase_shift_parameters<REMAP.remap>(input, output, shape, shifts);

        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        if constexpr (nt::is_varray_v<Shift>) {
            if (all(shifts == 0))
                return guts::no_phase_shift<REMAP.remap>(input, output, shape);

            const bool is_half_shift = all(allclose(abs(shifts), shape.filter(2, 3).vec.as<f32>() / 2));
            if (is_half_shift and fftfreq_cutoff >= sqrt(0.5)) {
                auto input_accessor = Accessor<const input_value_t, 4, i64>(input, input_strides);
                auto output_accessor = Accessor<output_value_t, 4, i64>(output, output.strides());
                using op_t = PhaseShiftHalf<REMAP.remap, i64, decltype(input_accessor), decltype(output_accessor)>;
                auto op = op_t(input_accessor, output_accessor, shape);
                return iwise(shape.rfft(), output.device(), op, input, output, shifts);
            }
        }

        using input_accessor_t = Accessor<const input_value_t, 3, i64>;
        using output_accessor_t = Accessor<output_value_t, 3, i64>;
        using shift_t = decltype(guts::extract_shift(shifts));
        using coord_t = nt::mutable_value_type_twice_t<Shift>;
        using op_t = PhaseShift<REMAP.remap, 2, i64, shift_t, input_accessor_t, output_accessor_t>;
        auto op = op_t(input_accessor_t(input, input_strides.filter(0, 2, 3)),
                       output_accessor_t(output, output.strides().filter(0, 2, 3)),
                       shape.filter(2, 3), guts::extract_shift(shifts),
                       static_cast<coord_t>(fftfreq_cutoff));
        iwise(shape.filter(0, 2, 3).rfft(), output.device(), op, input, output, shifts);
    }

    /// Phase-shifts 3d rfft(s).
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \param[in] input        3d rfft to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output      Phase-shifted 3d rfft.
    /// \param shape            BDHW logical shape.
    /// \param[in] shifts       HW 3d phase-shift to apply.
    ///                         A single value or a contiguous vector with one shift per batch.
    /// \param fftfreq_cutoff   Maximum output frequency to consider, in cycle/pix.
    ///                         Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<noa::fft::RemapInterface REMAP, typename Output, typename Shift,
             typename Input = View<const nt::value_type_t<Output>>>
    requires (nt::are_varray_of_complex_v<Input, Output> and
              (nt::is_varray_of_almost_any_v<Shift, Vec3<f32>, Vec3<f64>> or nt::is_vec_real_size_v<Shift, 3>) and
              (REMAP.remap == noa::fft::Remap::H2H or
               REMAP.remap == noa::fft::Remap::H2HC or
               REMAP.remap == noa::fft::Remap::HC2H or
               REMAP.remap == noa::fft::Remap::HC2HC))
    void phase_shift_3d(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const Shift& shifts,
            f64 fftfreq_cutoff = 1
    ) {
        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        guts::check_phase_shift_parameters<REMAP.remap>(input, output, shape, shifts);

        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        if constexpr (nt::is_varray_v<Shift>) {
            if (all(shifts == 0))
                return guts::no_phase_shift<REMAP.remap>(input, output, shape);

            const bool is_half_shift = all(allclose(abs(shifts), shape.filter(1, 2, 3).vec.as<f32>() / 2));
            if (is_half_shift and fftfreq_cutoff >= sqrt(0.5)) {
                auto input_accessor = Accessor<const input_value_t, 4, i64>(input, input_strides);
                auto output_accessor = Accessor<output_value_t, 4, i64>(output, output.strides());
                using op_t = PhaseShiftHalf<REMAP.remap, i64, decltype(input_accessor), decltype(output_accessor)>;
                auto op = op_t(input_accessor, output_accessor, shape);
                return iwise(shape.rfft(), output.device(), op, input, output, shifts);
            }
        }

        using input_accessor_t = Accessor<const input_value_t, 4, i64>;
        using output_accessor_t = Accessor<output_value_t, 4, i64>;
        using shift_t = decltype(guts::extract_shift(shifts));
        using coord_t = nt::mutable_value_type_twice_t<Shift>;
        using op_t = PhaseShift<REMAP.remap, 2, i64, shift_t, input_accessor_t, output_accessor_t>;
        auto op = op_t(input_accessor_t(input, input_strides),
                       output_accessor_t(output, output.strides()),
                       shape.filter(1, 2, 3), guts::extract_shift(shifts),
                       static_cast<coord_t>(fftfreq_cutoff));
        iwise(shape.rfft(), output.device(), op, input, output, shifts);
    }
}
