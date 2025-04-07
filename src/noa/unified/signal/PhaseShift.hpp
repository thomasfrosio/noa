#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/utils/BatchedParameter.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Factory.hpp"
#include "noa/unified/fft/Remap.hpp"

namespace noa::signal::guts {
    /// 4d iwise operator to phase shift each dimension by size / 2 (floating-point division).
    template<Remap REMAP, size_t N,
             nt::sinteger Index,
             nt::readable_nd_or_empty<N + 1> Input,
             nt::writable_nd<N + 1> Output>
    requires (N == 2 or N == 3)
    class PhaseShiftHalf {
    public:
        static constexpr bool IS_SRC_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static_assert(REMAP.is_hx2hx() or REMAP.is_fx2fx());

        using index_type = Index;
        using shape_nd_type = Shape<index_type, N>;
        using shape_type = Shape<index_type, N - IS_RFFT>;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::complex<output_value_type, input_value_type>);

    public:
        constexpr PhaseShiftHalf(
            const input_type& input,
            const output_type& output,
            const shape_nd_type& shape
        ) :
            m_input(input),
            m_output(output),
            m_shape(shape.template pop_back<IS_RFFT>()) {}

        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        constexpr void operator()(index_type batch, I... indices) const {
            const auto frequency = noa::fft::index2frequency<IS_SRC_CENTERED, IS_RFFT>(Vec{indices...}, m_shape);
            const auto phase_shift = static_cast<input_real_type>(product(1 - 2 * abs(frequency % 2)));

            const auto output_indices = noa::fft::remap_indices<REMAP>(Vec{indices...}, m_shape);
            auto& output = m_output(output_indices.push_front(batch));

            if (m_input)
                output = static_cast<output_value_type>(m_input(batch, indices...) * phase_shift);
            else
                output = static_cast<output_value_type>(phase_shift);
        }

    private:
        input_type m_input;
        output_type m_output;
        shape_type m_shape;
    };

    /// 3d or 4d iwise operator to phase shift 2d or 3d array(s).
    template<Remap REMAP, size_t N,
             nt::sinteger Index,
             nt::batched_parameter Shift,
             nt::readable_nd_optional<N + 1> Input,
             nt::writable_nd<N + 1> Output>
    requires (N == 2 or N == 3)
    class PhaseShift {
    public:
        static constexpr bool IS_SRC_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static_assert(REMAP.is_hx2hx() or REMAP.is_fx2fx());

        using index_type = Index;
        using shape_nd_type = Shape<index_type, N>;
        using shape_type = Shape<index_type, N - IS_RFFT>;

        using shift_parameter_type = Shift;
        using vec_nd_type = nt::value_type_t<shift_parameter_type>;
        using coord_type = nt::value_type_t<vec_nd_type>;
        static_assert(nt::vec_real_size<vec_nd_type, N>);

        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::complex<output_value_type, input_value_type>);

    public:
        constexpr PhaseShift(
            const input_type& input,
            const output_type& output,
            const shape_nd_type& shape,
            const shift_parameter_type& shift,
            coord_type cutoff
        ) :
            m_input(input), m_output(output),
            m_norm(coord_type{1} / vec_nd_type::from_vec(shape.vec)),
            m_shape(shape.template pop_back<IS_RFFT>()),
            m_shift(shift),
            m_cutoff_fftfreq_sqd(cutoff * cutoff) {}

        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        constexpr void operator()(index_type batch, I... indices) const {
            const auto frequency = noa::fft::index2frequency<IS_SRC_CENTERED, IS_RFFT>(Vec{indices...}, m_shape);
            const auto fftfreq = vec_nd_type::from_vec(frequency) * m_norm;

            input_value_type phase_shift{1, 0};
            if (dot(fftfreq, fftfreq) <= m_cutoff_fftfreq_sqd)
                phase_shift = noa::fft::phase_shift<input_value_type>(m_shift[batch], fftfreq);
            // TODO If even, the real nyquist should stay real, so add the conjugate pair?

            const auto output_indices = noa::fft::remap_indices<REMAP>(Vec{indices...}, m_shape);
            auto& output = m_output(output_indices.push_front(batch));
            if (m_input)
                output = static_cast<output_value_type>(m_input(batch, indices...) * phase_shift);
            else
                output = static_cast<output_value_type>(phase_shift);
        }

    private:
        input_type m_input;
        output_type m_output;
        vec_nd_type m_norm;
        shape_type m_shape;
        shift_parameter_type m_shift;
        coord_type m_cutoff_fftfreq_sqd;
    };

    template<Remap REMAP, typename Input, typename Output, typename Shift>
    void check_phase_shift_parameters(
        const Input& input, const Output& output,
        const Shape4<i64>& shape, const Shift& shifts
    ) {
        check(not output.is_empty(), "Empty array detected");
        check(vall(Equal{}, output.shape(), REMAP.is_hx2hx() ? shape.rfft() : shape),
              "Given the logical shape {} and FFT layout {}, the expected physical shape should be {}, but got {}",
              shape, REMAP, REMAP.is_hx2hx() ? shape.rfft() : shape, output.shape());

        if (not input.is_empty()) {
            check(output.device() == input.device(),
                  "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
                  input.device(), output.device());
            check(not REMAP.has_layout_change() or input.get() != output.get(),
                  "In-place remapping is not allowed");
        }

        if constexpr (nt::varray<Shift>) {
            check(ni::is_contiguous_vector(shifts) and shifts.n_elements() == output.shape()[0],
                  "The input shift(s) should be entered as a 1d contiguous vector, with one shift per output batch, "
                  "but got shift {} and output {}", shifts.shape(), output.shape());
            check(output.device() == shifts.device(),
                  "The shift and output arrays must be on the same device, but got shifts:device={}, output:device={}",
                  shifts.device(), output.device());
        }
    }

    template<Remap REMAP, typename Input, typename Output>
    void no_phase_shift(Input&& input, Output&& output, const Shape4<i64>& shape) {
        if (input.is_empty()) {
            using value_t = nt::value_type_t<Output>;
            fill(std::forward<Output>(output), value_t{1, 0});
        } else {
            if constexpr (not REMAP.has_layout_change()) {
                if (input.get() != output.get())
                    copy(std::forward<Input>(input), std::forward<Output>(output));
            } else {
                noa::fft::remap(REMAP, std::forward<Input>(input), std::forward<Output>(output), shape);
            }
        }
    }

    template<typename T>
    auto extract_shift(const T& shift) {
        if constexpr (nt::vec<T>) {
            return BatchedParameter{shift};
        } else {
            using ptr_t = nt::const_value_type_t<T>*;
            return BatchedParameter<ptr_t>{shift.get()};
        }
    }
}

namespace noa::signal {
    /// Phase-shifts 2d rfft(s).
    /// \tparam REMAP           Remap operation. Should be HX2HX or FX2FX.
    /// \param[in] input        2d rfft to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output      Phase-shifted 2d rfft.
    /// \param shape            BDHW logical shape.
    /// \param[in] shifts       HW 2d phase-shift to apply.
    ///                         A single value or a contiguous vector with one shift per batch.
    /// \param fftfreq_cutoff   Maximum output frequency to consider, in cycle/pix.
    ///                         Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal as long as the layout is unchanged.
    template<Remap REMAP,
             nt::writable_varray_decay_of_complex Output,
             nt::readable_varray_decay_of_complex Input = View<nt::const_value_type_t<Output>>,
             nt::varray_decay_or_value_of_almost_any<Vec2<f32>, Vec2<f64>> Shift>
    requires (REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void phase_shift_2d(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        Shift&& shifts,
        f64 fftfreq_cutoff = 1
    ) {
        using coord_t = nt::mutable_value_type_twice_t<Shift>;
        using input_accessor_t = Accessor<nt::const_value_type_t<Input>, 3, i64>;
        using output_accessor_t = Accessor<nt::value_type_t<Output>, 3, i64>;

        guts::check_phase_shift_parameters<REMAP>(input, output, shape, shifts);

        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }
        const auto iwise_shape = REMAP.is_hx2hx() ? shape.filter(0, 2, 3).rfft() : shape.filter(0, 2, 3);
        const auto input_accessor = input_accessor_t(input.get(), input_strides.filter(0, 2, 3));
        const auto output_accessor = output_accessor_t(output.get(), output.strides().filter(0, 2, 3));

        if constexpr (not nt::varray_decay<Shift>) { // single shift
            if (vall(IsZero{}, shifts))
                return guts::no_phase_shift<REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape);

            const auto shape_2d = shape.filter(2, 3);
            const auto is_half_shift = [](auto shift, auto half_shift) { return allclose(abs(shift), half_shift); };
            if (vall(is_half_shift, shifts, shape_2d.vec.as<coord_t>() / 2) and fftfreq_cutoff >= sqrt(0.5)) {
                using op_t = guts::PhaseShiftHalf<REMAP, 2, i64, input_accessor_t, output_accessor_t>;
                return iwise(iwise_shape, output.device(), op_t(input_accessor, output_accessor, shape_2d),
                             std::forward<Input>(input), std::forward<Output>(output));
            }
        }

        using shift_t = decltype(guts::extract_shift(shifts));
        using op_t = guts::PhaseShift<REMAP, 2, i64, shift_t, input_accessor_t, output_accessor_t>;
        auto op = op_t(input_accessor, output_accessor,
                       shape.filter(2, 3), guts::extract_shift(shifts),
                       static_cast<coord_t>(fftfreq_cutoff));
        iwise(iwise_shape, output.device(), op,
              std::forward<Input>(input),
              std::forward<Output>(output),
              std::forward<Shift>(shifts));
    }

    /// Phase-shifts 3d rfft(s).
    /// \tparam REMAP           Remap operation. Should be HX2HX or FX2FX.
    /// \param[in] input        3d rfft to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output      Phase-shifted 3d rfft.
    /// \param shape            BDHW logical shape.
    /// \param[in] shifts       HW 3d phase-shift to apply.
    ///                         A single value or a contiguous vector with one shift per batch.
    /// \param fftfreq_cutoff   Maximum output frequency to consider, in cycle/pix.
    ///                         Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal as long as the layout is unchanged.
    template<Remap REMAP,
             nt::writable_varray_decay_of_complex Output,
             nt::readable_varray_decay_of_complex Input = View<nt::const_value_type_t<Output>>,
             nt::varray_decay_or_value_of_almost_any<Vec3<f32>, Vec3<f64>> Shift>
    requires (REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void phase_shift_3d(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        Shift&& shifts,
        f64 fftfreq_cutoff = 1
    ) {
        using coord_t = nt::mutable_value_type_twice_t<Shift>;
        using input_accessor_t = Accessor<nt::const_value_type_t<Input>, 4, i64>;
        using output_accessor_t = Accessor<nt::value_type_t<Output>, 4, i64>;

        guts::check_phase_shift_parameters<REMAP>(input, output, shape, shifts);

        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }
        const auto iwise_shape = REMAP.is_hx2hx() ? shape.rfft() : shape;
        const auto input_accessor = input_accessor_t(input.get(), input_strides);
        const auto output_accessor = output_accessor_t(output.get(), output.strides());

        if constexpr (not nt::varray_decay<Shift>) { // single shift
            if (vall(IsZero{}, shifts))
                return guts::no_phase_shift<REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape);

            const auto shape_3d = shape.filter(1, 2, 3);
            const auto is_half_shift = [](auto shift, auto half_shift) { return allclose(abs(shift), half_shift); };
            if (vall(is_half_shift, shifts, shape_3d.vec.as<coord_t>() / 2) and fftfreq_cutoff >= sqrt(0.5)) {
                using op_t = guts::PhaseShiftHalf<REMAP, 3, i64, input_accessor_t, output_accessor_t>;
                return iwise(iwise_shape, output.device(), op_t(input_accessor, output_accessor, shape_3d),
                             std::forward<Input>(input), std::forward<Output>(output));
            }
        }

        using shift_t = decltype(guts::extract_shift(shifts));
        using op_t = guts::PhaseShift<REMAP, 3, i64, shift_t, input_accessor_t, output_accessor_t>;
        auto op = op_t(input_accessor, output_accessor,
                       shape.filter(1, 2, 3), guts::extract_shift(shifts),
                       static_cast<coord_t>(fftfreq_cutoff));
        iwise(iwise_shape, output.device(), op,
              std::forward<Input>(input),
              std::forward<Output>(output),
              std::forward<Shift>(shifts));
    }
}
