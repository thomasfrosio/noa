#pragma once

#include "noa/fft/core/Frequency.hpp"
#include "noa/fft/Remap.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/core/Utils.hpp"
#include "noa/runtime/Factory.hpp"
#include "noa/runtime/Iwise.hpp"

namespace noa::signal::details {
    template<nf::Layout REMAP, usize B, usize R,
             nt::sinteger Index,
             nt::readable_nd_or_empty<B + R> Input,
             nt::writable_nd<B + R> Output>
    class PhaseShiftHalf {
    public:
        static constexpr bool IS_SRC_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static_assert(REMAP.is_hx2hx() or REMAP.is_fx2fx());

        using index_type = Index;
        using shape_nd_type = Shape<index_type, R>;
        using shape_type = Shape<index_type, R - IS_RFFT>;
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

        constexpr void operator()(const Vec<index_type, B + R>& batched_indices) const {
            const auto& [batches, indices] = batched_indices.template split<B>();
            const auto frequency = nf::index2frequency<IS_SRC_CENTERED, IS_RFFT>(indices, m_shape);
            const auto phase_shift = static_cast<input_real_type>(product(1 - 2 * abs(frequency % 2))); // shifts by size / 2 (fp division).

            const auto output_indices = nf::remap_indices<REMAP>(indices, m_shape);
            auto& output = m_output(output_indices.push_front(batches));
            if (m_input)
                output = static_cast<output_value_type>(m_input(batched_indices) * phase_shift);
            else
                output = static_cast<output_value_type>(phase_shift);
        }

    private:
        input_type m_input;
        output_type m_output;
        shape_type m_shape;
    };

    template<nf::Layout REMAP,
             nt::sinteger Index, usize B, usize R,
             nt::readable_nd<B> Shift,
             nt::readable_nd_optional<B + R> Input,
             nt::writable_nd<B + R> Output>
    class PhaseShift {
    public:
        static constexpr bool IS_SRC_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static_assert(REMAP.is_hx2hx() or REMAP.is_fx2fx());

        using index_type = Index;
        using shape_nd_type = Shape<index_type, R>;
        using shape_type = Shape<index_type, R - IS_RFFT>;

        using shift_parameter_type = Shift;
        using shift_type = nt::value_type_t<shift_parameter_type>;
        using coord_type = nt::value_type_t<shift_type>;
        using vec_nd_type = Vec<coord_type, R>;
        static_assert(nt::vec_real_size<shift_type, R>);

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

        constexpr void operator()(const Vec<index_type, B + R>& batched_indices) const {
            const auto& [batches, indices] = batched_indices.template split<B>();
            const auto frequency = nf::index2frequency<IS_SRC_CENTERED, IS_RFFT>(indices, m_shape);
            const auto fftfreq = vec_nd_type::from_vec(frequency) * m_norm;

            input_value_type phase_shift{1, 0};
            if (dot(fftfreq, fftfreq) <= m_cutoff_fftfreq_sqd)
                phase_shift = nf::phase_shift<input_value_type>(m_shift(batches), fftfreq);
            // TODO If even, the real nyquist should stay real, so add the conjugate pair?

            const auto output_indices = nf::remap_indices<REMAP>(indices, m_shape);
            auto& output = m_output(output_indices.push_front(batches));
            if (m_input)
                output = static_cast<output_value_type>(m_input(batched_indices) * phase_shift);
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

    template<nf::Layout REMAP, usize B, usize N, typename Input, typename Output, usize S, typename Shift>
    void check_phase_shift_parameters(
        const Input& input, const Output& output,
        const Shape<isize, S>& shape, const Shift& shifts
    ) {
        check(not output.is_empty(), "Empty array detected");
        check(output.shape() == (REMAP.is_hx2hx() ? shape.rfft() : shape),
              "Given the logical shape {} and FFT layout {}, the expected physical shape should be {}, but got {}",
              shape, REMAP, REMAP.is_hx2hx() ? shape.rfft() : shape, output.shape());

        if (not input.is_empty()) {
            check(output.device() == input.device(),
                  "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
                  input.device(), output.device());
            check(not REMAP.has_layout_change() or input.get() != output.get(),
                  "In-place remapping is not allowed");
        }

        if constexpr (nt::array<Shift>) {
            const auto expected_batch_shape = output.shape().template pop_back<N - B>();
            check(shifts.shape() == expected_batch_shape,
                  "Given the output:shape={} with {} batch dimensions, the expected shift shape is {}, but got shifts:shape={}",
                  output.shape(), B, expected_batch_shape, shifts.shape());
            check(shifts.is_contiguous(),
                  "The input shift(s) should be entered as a 1d contiguous vector, with one shift per output batch, "
                  "but got shift {} and output {}", shifts.shape(), output.shape());
            check(output.device() == shifts.device(),
                  "The shift and output arrays must be on the same device, but got shifts:device={}, output:device={}",
                  shifts.device(), output.device());
        }
    }

    template<nf::Layout REMAP, i32 RANK, typename Input, typename Output, usize N>
    void no_phase_shift(Input&& input, Output&& output, const Shape<isize, N>& shape) {
        if (input.is_empty()) {
            using value_t = nt::value_type_t<Output>;
            fill(std::forward<Output>(output), value_t{1, 0});
        } else {
            if constexpr (not REMAP.has_layout_change()) {
                if (input.get() != output.get())
                    copy(std::forward<Input>(input), std::forward<Output>(output));
            } else {
                nf::remap(REMAP, std::forward<Input>(input), std::forward<Output>(output), shape, {.rank = RANK});
            }
        }
    }

    template<typename T>
    auto extract_shift_accessor(const T& shift) {
        if constexpr (nt::vec<T>) {
            return AccessorValue(shift);
        } else if constexpr (nt::array<T>) {
            using value_type = nt::const_value_type_t<T>;
            return shift.template span_contiguous<value_type>().accessor();
        } else {
            static_assert(nt::always_false<T>);
        }
    }

    template<typename ShiftVec, usize N>
    concept phase_shift_vec =
        nt::vec<ShiftVec> and nt::any_of<nt::value_type_t<ShiftVec>, f32, f64> and ShiftVec::SIZE <= N and
        (ShiftVec::SIZE == 1 or ShiftVec::SIZE == 2 or ShiftVec::SIZE == 3);

    template<typename ShiftArray, usize N, typename ShiftVec = nt::value_type_t<ShiftArray>>
    concept phase_shift_array_vec =
        nt::array<ShiftArray> and phase_shift_vec<ShiftVec, N> and
        nt::array_size_v<ShiftArray> == (N - ShiftVec::SIZE);

    template<typename Input, typename Output, usize N, typename Shift>
    concept phase_shiftable =
        nt::readable_array_decay_of_complex<Input> and
        nt::writable_array_decay_of_complex<Output> and
        nt::array_decay_with_same_nd<Input, Output> and nt::array_size_v<Input> == N and
        (phase_shift_vec<std::decay_t<Shift>, N> or phase_shift_array_vec<std::decay_t<Shift>, N>);

    template<typename Shift, usize N, typename ShiftDecay = std::decay_t<Shift>>
    concept phase_shift_nd =
        nt::vec_of_size<ShiftDecay, N> or
        (nt::array<ShiftDecay> and nt::vec_of_size<nt::value_type_t<ShiftDecay>, N>);
}

namespace noa::signal {
    /// Phase-shifts rFFT(s).
    /// \tparam REMAP:
    ///     Remap operation.
    ///     Should be HX2HX or FX2FX.
    ///     The input and output can be equal as long as the layout is unchanged.
    /// \param[in] input:
    ///     rFFT(s) to phase-shift.
    ///     The rank of the transform, therefore which dimensions are batch dimensions, depends on options.rank.
    ///     If empty, the phase-shifts are saved into the output.
    /// \param[out] output:
    ///     The rank of the transform, therefore which dimensions are batch dimensions, depends on options.rank.
    ///     Phase-shifted rFFT or phase-shifts if the input is empty.
    /// \param shape:
    ///     Logical shape of the input and output.
    /// \param[in] shifts:
    ///     Phase-shift to apply.
    ///     A single Vec<f32|f64, R> or a contiguous (B..) array of this type.
    ///     The dimensionality of the shift (R=1|2|3) sets the rank of the transform and therefore which dimensions
    ///     are considered batch dimensions. An additional constraint of this function is that it enforces the input
    ///     and output to have at least N dimensions, such as:
    ///         R=1: ((B..,)W),
    ///         R=2: ((B..,)H,W),
    ///         R=3: ((B..,)D,H,W).
    ///     Consequently, it is perfectly fine to pass, for instance, a 3D transform and 2D shifts, in which case the
    ///     3D transform (DHW) will be interpreted as multiple 2D transforms (BHW), which is the expected behavior. If
    ///     multiple 3D transforms (BDHW) are passed, the BD dimensions are both batch dimensions and if an array
    ///     of shifts is passed it should be of shape BD.
    ///     // TODO For the case where a single zero-vector is passed, if the layout changes, remap is called.
    ///     //      Since remap takes the rank as a runtime parameter, it requires the batch dimensions to be collapsible.
    /// \param fftfreq_cutoff:
    ///     Maximum output frequency to consider, in cycle/pix.
    ///     Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///     Frequencies higher than this value are not phase-shifted.
    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N, typename Shift>
        requires ((REMAP.is_hx2hx() or REMAP.is_fx2fx()) and details::phase_shiftable<Input, Output, N, Shift>)
    void phase_shift(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        Shift&& shifts,
        f64 fftfreq_cutoff = 1
    ) {
        check(not output.is_empty(), "Empty array detected");
        check(output.shape() == (REMAP.is_hx2hx() ? shape.rfft() : shape),
              "Given the logical shape {} and FFT layout {}, the expected physical shape should be {}, but got {}",
              shape, REMAP, REMAP.is_hx2hx() ? shape.rfft() : shape, output.shape());

        if (not input.is_empty()) {
            check(output.device() == input.device(),
                  "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
                  input.device(), output.device());
            check(not REMAP.has_layout_change() or input.get() != output.get(),
                  "In-place remapping is not allowed");
        }

        // Get the batched spans (B..,R..).
        // This only expands with one empty left dimension for cases with N <= 3.
        using shift_t = std::decay_t<Shift>;
        using shift_vec_t = std::conditional_t<nt::array<shift_t>, nt::value_type_t<shift_t>, shift_t>;
        constexpr auto R = shift_vec_t::SIZE;
        constexpr auto B = std::max(usize{1}, N - R);
        constexpr auto BR = B + R;
        const auto input_bn = input.span().template as_nd<BR>();
        const auto output_bn = output.span().template as_nd<BR>();
        const auto shape_bn = shape.template as_nd<BR>();
        const auto [shape_b, shape_n] = shape_bn.template split<B>();

        if constexpr (nt::array<Shift>) {
            check(shifts.shape() == shape_b,
                  "Given the output:shape={} with {}D phase-shifts, the expected shifts shape is {}, but got shifts:shape={}",
                  output.shape(), R, shape_b, shifts.shape());
            check(shifts.is_contiguous(),
                  "The input shift(s) should be contiguous, but got shifts:shape={} and shifts:strides={}",
                  shifts.shape(), shifts.strides());
            check(output.device() == shifts.device(),
                  "The shift and output arrays must be on the same device, but got shifts:device={}, output:device={}",
                  shifts.device(), output.device());
        }

        // Implicit broadcast of the input, if any.
        auto input_bn_strides = input_bn.strides();
        if (not input.is_empty() and not broadcast(input_bn.shape(), input_bn_strides, output_bn.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input_bn.shape(), output_bn.shape());
        }

        using coord_t = nt::mutable_value_type_twice_t<Shift>;
        using iaccessor_t = Accessor<nt::const_value_type_t<Input>, BR, isize>;
        using oaccessor_t = Accessor<nt::value_type_t<Output>, BR, isize>;
        const auto iwise_shape = REMAP.is_hx2hx() ? shape_bn.rfft() : shape_bn;
        const auto iaccessor = iaccessor_t(input_bn.get(), input_bn_strides);
        const auto oaccessor = output_bn.accessor();

        if constexpr (nt::vec<shift_t>) {
            if (noa::allclose(shifts, 0))
                return details::no_phase_shift<REMAP, R>(
                    std::forward<Input>(input), std::forward<Output>(output), shape);

            if (fftfreq_cutoff >= std::sqrt(0.5)) {
                const auto half_shifts = shape_n.vec.template as<coord_t>() / 2;
                if (noa::allclose(abs(shifts), half_shifts)) {
                    using op_t = details::PhaseShiftHalf<REMAP, B, R, isize, iaccessor_t, oaccessor_t>;
                    return iwise(
                        iwise_shape, output.device(), op_t(iaccessor, oaccessor, shape_n),
                        std::forward<Input>(input), std::forward<Output>(output)
                    );
                }
            }
        }

        auto saccessor = details::extract_shift_accessor(shifts);
        using saccessor_t = decltype(saccessor);
        using op_t = details::PhaseShift<REMAP, isize, B, R, saccessor_t, iaccessor_t, oaccessor_t>;
        auto op = op_t(iaccessor, oaccessor, shape_n, saccessor, static_cast<coord_t>(fftfreq_cutoff));
        iwise(
            iwise_shape, output.device(), op,
            std::forward<Input>(input),
            std::forward<Output>(output),
            std::forward<Shift>(shifts)
        );
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N, typename Shift>
        requires details::phase_shift_nd<Shift, 1>
    void phase_shift_1d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        Shift&& shifts,
        f64 fftfreq_cutoff = 1
    ) {
        phase_shift<REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, shifts, fftfreq_cutoff);
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N, typename Shift>
        requires details::phase_shift_nd<Shift, 2>
    void phase_shift_2d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        Shift&& shifts,
        f64 fftfreq_cutoff = 1
    ) {
        phase_shift<REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, shifts, fftfreq_cutoff);
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N, typename Shift>
        requires details::phase_shift_nd<Shift, 3>
    void phase_shift_3d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        Shift&& shifts,
        f64 fftfreq_cutoff = 1
    ) {
        phase_shift<REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, shifts, fftfreq_cutoff);
    }
}
