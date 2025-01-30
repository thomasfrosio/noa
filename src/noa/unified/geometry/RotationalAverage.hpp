#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/Iwise.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/signal/CTF.hpp"
#include "noa/core/utils/Atomic.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Factory.hpp"

namespace noa::geometry::guts {
    struct RotationalAverageUtils {
        template<typename T, typename U, typename C, typename... I>
        NOA_FHD static void lerp_to_output(const T& op, const U& value, C fftfreq, I... batch) noexcept {
            // fftfreq to output index.
            const C scaled_fftfreq = (fftfreq - op.m_output_fftfreq_start) / op.m_output_fftfreq_span;
            const C radius = scaled_fftfreq * static_cast<C>(op.m_max_shell_index);
            const C radius_floor = floor(radius);
            const auto shell_low = static_cast<T::index_type>(radius_floor);
            const auto shell_high = shell_low + 1; // shell_low can be the last index

            // Compute lerp weights.
            const C fraction_high = radius - radius_floor;
            const C fraction_low = 1 - fraction_high;

            // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers?
            if (shell_low >= 0 and shell_low <= op.m_max_shell_index) {
                ng::atomic_add(op.m_output, value * static_cast<T::output_real_type>(fraction_low), batch..., shell_low);
                if (op.m_weight)
                    ng::atomic_add(op.m_weight, static_cast<T::weight_value_type>(fraction_low), batch..., shell_low);
            }

            if (shell_high >= 0 and shell_high <= op.m_max_shell_index) {
                ng::atomic_add(op.m_output, value * static_cast<T::output_real_type>(fraction_high), batch..., shell_high);
                if (op.m_weight)
                    ng::atomic_add(op.m_weight, static_cast<T::weight_value_type>(fraction_high), batch..., shell_high);
            }
        }
    };

    /// 3d or 4d iwise operator to compute a rotational average of 2d or 3d array(s).
    /// - The output layout is noted "H", since often the number of output shells is min(shape) // 2 + 1
    ///   Otherwise, the input can be any of the for layouts (H, HC, F or FC).
    /// - A lerp is used to add frequencies in its two neighbor shells, instead of rounding to the nearest shell.
    /// - The frequencies are normalized, so input dimensions don't have to be equal.
    /// - The user sets the number of output shells, as well as the output frequency range.
    /// - If input is complex and output real, the input is preprocessed to abs(input)^2.
    /// - The 2d distortion from the anisotropic ctf can be corrected.
    template<Remap REMAP,
             size_t N,
             nt::real Coord,
             nt::sinteger Index,
             nt::readable_nd<N + 1> Input,
             nt::atomic_addable_nd<2> Output,
             nt::atomic_addable_nd_optional<2> Weight,
             nt::batched_parameter Ctf>
    class RotationalAverage {
    public:
        static_assert((N == 2 or N == 3) and REMAP.is_xx2h());
        static constexpr bool IS_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_RFFT = REMAP.is_hx2xx();

        using index_type = Index;
        using coord_type = Coord;
        using shape_type = Shape<index_type, N - IS_RFFT>;
        using coord_nd_type = Vec<coord_type, N>;
        using coord2_type = Vec2<coord_type>;
        using shape_nd_type = Shape<index_type, N>;

        using input_type = Input;
        using output_type = Output;
        using weight_type = Weight;
        using output_value_type = nt::value_type_t<output_type>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using weight_value_type = nt::value_type_t<weight_type>;
        static_assert(nt::spectrum_types<nt::value_type_t<input_type>, output_value_type>);
        static_assert(nt::same_as<weight_value_type, output_real_type>);

        using batched_ctf_type = Ctf;
        using ctf_type = nt::mutable_value_type_t<batched_ctf_type>;
        static_assert(nt::empty<ctf_type> or (N == 2 and nt::ctf_anisotropic<ctf_type>));

        friend RotationalAverageUtils;

    public:
        constexpr RotationalAverage(
            const input_type& input,
            const shape_nd_type& input_shape,
            const batched_ctf_type& input_ctf,
            const output_type& output,
            const weight_type& weight,
            index_type n_shells,
            Linspace<coord_type> input_fftfreq,
            Linspace<coord_type> output_fftfreq
        ) :
            m_input(input),
            m_output(output),
            m_weight(weight),
            m_ctf(input_ctf),
            m_shape(input_shape.template pop_back<IS_RFFT>())
        {
            // If input_fftfreq.stop is negative, defaults to the highest frequency.
            // In this case, and if the frequency.start is 0, this results in the full frequency range.
            // The input is N-d, so we have to handle each axis separately.
            m_input_fftfreq_start = input_fftfreq.start;
            coord_type max_input_fftfreq{-1};
            for (size_t i{}; i < N; ++i) {
                const auto max_sample_size = input_shape[i] / 2 + 1;
                const auto fftfreq_end =
                    input_fftfreq.stop <= 0 ?
                    noa::fft::highest_fftfreq<coord_type>(input_shape[i]) :
                    input_fftfreq.stop;
                max_input_fftfreq = max(max_input_fftfreq, fftfreq_end);
                m_input_fftfreq_step[i] = Linspace{
                    .start = input_fftfreq.start,
                    .stop = fftfreq_end,
                    .endpoint = input_fftfreq.endpoint
                }.for_size(max_sample_size).step;
            }

            // The output defaults to the input range. Of course, it is a reduction to 1d, so take the max fftfreq.
            if (output_fftfreq.start < 0)
                output_fftfreq.start = input_fftfreq.start;
            if (output_fftfreq.stop <= 0)
                output_fftfreq.stop = max_input_fftfreq;

            // Transform to inclusive range so that we only have to deal with one case.
            if (not output_fftfreq.endpoint) {
                output_fftfreq.stop -= output_fftfreq.for_size(n_shells).step;
                output_fftfreq.endpoint = true;
            }
            m_output_fftfreq_start = output_fftfreq.start;
            m_output_fftfreq_span = output_fftfreq.stop - output_fftfreq.start;
            m_max_shell_index = n_shells - 1;

            // To shortcut early, compute the fftfreq cutoffs where we know the output isn't affected.
            auto output_fftfreq_step = output_fftfreq.for_size(n_shells).step;
            m_fftfreq_cutoff[0] = output_fftfreq.start + -1 * output_fftfreq_step;
            m_fftfreq_cutoff[1] = output_fftfreq.stop + static_cast<coord_type>(m_max_shell_index) * output_fftfreq_step;
        }

        // 2d or 3d rotational average, with an optional anisotropic field correction.
        template<nt::same_as<index_type>... I> requires (N == sizeof...(I))
        NOA_HD void operator()(index_type batch, I... indices) const noexcept {
            // Input indices to fftfreq.
            const auto frequency = noa::fft::index2frequency<IS_CENTERED, IS_RFFT>(Vec{indices...}, m_shape);
            const auto fftfreq_nd = m_input_fftfreq_start + coord_nd_type::from_vec(frequency) * m_input_fftfreq_step;

            coord_type fftfreq;
            if constexpr (nt::empty<ctf_type>) {
                fftfreq = sqrt(dot(fftfreq_nd, fftfreq_nd));
            } else {
                // Correct for anisotropic field (pixel size and defocus).
                fftfreq = static_cast<coord_type>(m_ctf[batch].isotropic_fftfreq(fftfreq_nd));
            }

            // Remove most out-of-bounds asap.
            if (fftfreq < m_fftfreq_cutoff[0] or fftfreq > m_fftfreq_cutoff[1])
                return;

            const auto value = cast_or_abs_squared<output_value_type>(m_input(batch, indices...));
            RotationalAverageUtils::lerp_to_output(*this, value, fftfreq, batch);
        }

    private:
        input_type m_input;
        output_type m_output;
        weight_type m_weight;
        NOA_NO_UNIQUE_ADDRESS batched_ctf_type m_ctf;

        shape_type m_shape;
        coord_nd_type m_input_fftfreq_step;
        coord_type m_input_fftfreq_start;
        coord2_type m_fftfreq_cutoff;
        coord_type m_output_fftfreq_start;
        coord_type m_output_fftfreq_span;
        index_type m_max_shell_index;
    };

    template<nt::real Coord,
             nt::sinteger Index,
             nt::readable_nd<2> Input,
             nt::atomic_addable_nd<1> Output,
             nt::atomic_addable_nd_optional<1> Weight,
             nt::batched_parameter InputCtf,
             nt::ctf_isotropic OutputCtf>
    class FuseRotationalAverages {
    public:
        using index_type = Index;
        using coord_type = Coord;
        using coord_nd_type = Vec<coord_type, 1>;
        using coord2_type = Vec2<coord_type>;

        using input_type = Input;
        using output_type = Output;
        using weight_type = Weight;
        using output_value_type = nt::value_type_t<output_type>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using weight_value_type = nt::value_type_t<weight_type>;
        static_assert(nt::spectrum_types<nt::value_type_t<input_type>, output_value_type>);
        static_assert(nt::same_as<weight_value_type, output_real_type>);

        using batched_input_ctf_type = InputCtf;
        using input_ctf_type = nt::mutable_value_type_t<batched_input_ctf_type>;
        using output_ctf_type = OutputCtf;
        static_assert(nt::ctf_isotropic<input_ctf_type>);

        friend RotationalAverageUtils;

    public:
        constexpr FuseRotationalAverages(
            const input_type& input,
            const Linspace<coord_type>& input_fftfreq,
            const batched_input_ctf_type& input_ctf,
            index_type n_input_shells,
            const output_type& output,
            Linspace<coord_type> output_fftfreq,
            const output_ctf_type& output_ctf,
            index_type n_output_shells,
            const weight_type& weight
        ) :
            m_input(input),
            m_output(output),
            m_weight(weight),
            m_input_ctf(input_ctf),
            m_output_ctf(output_ctf)
        {
            m_input_fftfreq_start = input_fftfreq.start;
            m_input_fftfreq_step = input_fftfreq.for_size(n_input_shells).step;

            // Transform to inclusive range so that we only have to deal with one case.
            if (not output_fftfreq.endpoint) {
                output_fftfreq.stop -= output_fftfreq.for_size(n_output_shells).step;
                output_fftfreq.endpoint = true;
            }
            m_output_fftfreq_start = output_fftfreq.start;
            m_output_fftfreq_span = output_fftfreq.stop - output_fftfreq.start;
            m_max_shell_index = n_output_shells - 1;

            // To shortcut early, compute the fftfreq cutoffs where we know the output isn't affected.
            auto output_fftfreq_step = output_fftfreq.for_size(n_output_shells).step;
            m_fftfreq_cutoff[0] = output_fftfreq.start + -1 * output_fftfreq_step;
            m_fftfreq_cutoff[1] = output_fftfreq.stop + static_cast<coord_type>(m_max_shell_index) * output_fftfreq_step;
        }

        NOA_HD void operator()(index_type batch, index_type index) const noexcept {
            const auto input_fftfreq = m_input_fftfreq_start + static_cast<coord_type>(index) * m_input_fftfreq_step;
            const auto input_phase = m_input_ctf[batch].phase_at(input_fftfreq);
            const auto fftfreq = static_cast<coord_type>(m_output_ctf.fftfreq_at(input_phase));

            // Remove most out-of-bounds asap.
            if (fftfreq < m_fftfreq_cutoff[0] or fftfreq > m_fftfreq_cutoff[1])
                return;

            const auto value = cast_or_abs_squared<output_value_type>(m_input(batch, index));
            RotationalAverageUtils::lerp_to_output(*this, value, fftfreq);
        }

    private:
        input_type m_input;
        output_type m_output;
        weight_type m_weight;
        batched_input_ctf_type m_input_ctf;
        output_ctf_type m_output_ctf;

        coord2_type m_fftfreq_cutoff;
        coord_type m_input_fftfreq_start;
        coord_type m_input_fftfreq_step;
        coord_type m_output_fftfreq_start;
        coord_type m_output_fftfreq_span;
        index_type m_max_shell_index;
    };

    template<Remap REMAP, typename Input, typename Output, typename Weight, typename Ctf = Empty>
    auto check_parameters_rotational_average(
        const Input& input,
        const Shape4<i64>& shape,
        const Ctf& input_ctf,
        const Output& output,
        const Weight& weights
    ) -> i64 {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        const bool weights_is_empty = weights.is_empty();

        check(all(input.shape() == (REMAP.is_hx2xx() ? shape.rfft() : shape)),
              "The input array does not match the logical shape. Got input:shape={}, shape={}, remap={}",
              input.shape(), shape, REMAP);

        check(shape[0] == output.shape()[0] and
              (weights_is_empty or shape[0] == weights.shape()[0]),
              "The numbers of batches between arrays do not match. Got batch={}, output:batch={}{}",
              shape[0], output.shape()[0],
              weights_is_empty ? "" : fmt::format(" and weights:batch={}", weights.shape()[0]));

        check(ni::is_contiguous_vector_batched_strided(output),
              "The output must be a (batch of) contiguous vector(s), but got output:shape={} and output:strides={}",
              output.shape(), output.strides());

        const i64 n_shells = output.shape().pop_front().n_elements();
        if (not weights_is_empty) {
            check(ni::is_contiguous_vector_batched_strided(weights),
                  "The weights must be a (batch of) contiguous vector(s), "
                  "but got weights:shape={} and weights:strides={}",
                  weights.shape(), weights.strides());

            const i64 weights_n_shells = weights.shape().pop_front().n_elements();
            check(n_shells == weights_n_shells,
                  "The number of shells does not match the output shape. "
                  "Got output:n_shells={} and weights:n_shells={}",
                  n_shells, weights_n_shells);
        }

        check(input.device() == output.device() and
              (weights_is_empty or weights.device() == output.device()),
              "The arrays must be on the same device, but got input:device={}, output:device={}{}",
              input.device(), output.device(),
              weights_is_empty ? "" : fmt::format(" and weights:device={}", weights.device()));

        if constexpr (not nt::empty<Ctf>) {
            check(shape.ndim() == 2,
                  "Only (batched) 2d arrays are supported with anisotropic CTFs, but got shape={}",
                  shape);
        }
        if constexpr (nt::varray<Ctf>) {
            check(ni::is_contiguous_vector(input_ctf) and input_ctf.n_elements() == shape[0],
                  "The anisotropic input CTFs, specified as a contiguous vector, should have the same batch size "
                  "as the input. Got input_ctf:strides={}, input_ctf:shape={}, input:batch={}",
                  input_ctf.strides(), input_ctf.shape(), shape[0]);
            check(input_ctf.device() == output.device(),
                  "The input and output arrays must be on the same device, "
                  "but got input_ctf:device={} and output:device={}",
                  input_ctf.device(), output.device());
        }

        return n_shells;
    }

    template<
        Remap REMAP, bool IS_GPU = false,
        typename Input, typename Index, typename Ctf,
        typename Output, typename Weight, typename Options>
    void launch_rotational_average(
        Input&& input, const Shape4<Index>& input_shape, Ctf&& input_ctf,
        Output&& output, Weight&& weight, i64 n_shells, const Options& options
    ) {
        using input_value_t = nt::const_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using weight_value_t = nt::value_type_t<Weight>;
        using coord_t = nt::value_type_t<output_value_t>;
        constexpr auto IWISE_OPTION = IwiseOptions{.generate_cpu = not IS_GPU, .generate_gpu = IS_GPU};
        constexpr auto EWISE_OPTION = EwiseOptions{.generate_cpu = not IS_GPU, .generate_gpu = IS_GPU};

        // Output must be zeroed out.
        const auto output_view = output.view();
        if (not options.add_to_output)
            ewise<EWISE_OPTION>({}, output_view, Zero{});

        // When computing the average, the weights must be valid.
        auto weight_view = weight.view();
        Array<weight_value_t> weight_buffer;
        if (options.average) {
            if (weight_view.is_empty()) {
                weight_buffer = zeros<weight_value_t>(output_view.shape(), ArrayOption{output.device(), Allocator::DEFAULT_ASYNC});
                weight_view = weight_buffer.view();
            } else if (not options.add_to_output) {
                ewise<EWISE_OPTION>({}, weight_view, Zero{});
            }
        }

        using output_accessor_t = AccessorRestrictContiguous<output_value_t, 2, Index>;
        using weight_accessor_t = AccessorRestrictContiguous<weight_value_t, 2, Index>;
        auto output_accessor = output_accessor_t(output_view.get(), Strides1<Index>::from_value(output_view.strides()[0]));
        auto weight_accessor = weight_accessor_t(weight_view.get(), Strides1<Index>::from_value(weight_view.strides()[0]));

        const auto input_fftfreq = options.input_fftfreq.template as<coord_t>();
        const auto output_fftfreq = options.output_fftfreq.template as<coord_t>();
        const auto iwise_shape = input.shape().template as<Index>();
        const auto input_strides = input.strides().template as<Index>();

        if (input_shape.ndim() == 2) {
            // Retrieve the CTF(s).
            auto ctf = [&] {
                if constexpr (nt::varray_decay<Ctf>) {
                    return BatchedParameter{input_ctf.get()};
                } else if constexpr (nt::empty<Ctf>) {
                    return BatchedParameter<Empty>{};
                } else { // ctf_anisotropic
                    return BatchedParameter{input_ctf};
                }
            }();

            using input_accessor_t = AccessorRestrict<input_value_t, 3, Index>;
            auto op = RotationalAverage
                <REMAP, 2, coord_t, Index, input_accessor_t, output_accessor_t, weight_accessor_t, decltype(ctf)>(
                input_accessor_t(input.get(), input_strides.filter(0, 2, 3)), input_shape.filter(2, 3),
                ctf, output_accessor, weight_accessor, static_cast<Index>(n_shells),
                input_fftfreq, output_fftfreq);

            iwise<IWISE_OPTION>(
                iwise_shape.filter(0, 2, 3), output.device(), op,
                std::forward<Input>(input), output, weight, std::forward<Ctf>(input_ctf));

        } else {
            using input_accessor_t = AccessorRestrict<input_value_t, 4, Index>;
            auto op = RotationalAverage<
                REMAP, 3, coord_t, Index,
                input_accessor_t, output_accessor_t, weight_accessor_t, BatchedParameter<Empty>
            >(input_accessor_t(input.get(), input_strides), input_shape.filter(1, 2, 3),
              {}, output_accessor, weight_accessor, static_cast<Index>(n_shells),
              input_fftfreq, output_fftfreq);

            iwise<IWISE_OPTION>(iwise_shape, output.device(), op, std::forward<Input>(input), output, weight);
        }

        // Some shells can be 0, so use DivideSafe.
        if (options.average) {
            if (weight_buffer.is_empty()) {
                ewise<EWISE_OPTION>(wrap(output_view, weight), std::forward<Output>(output), DivideSafe{});
            } else {
                ewise<EWISE_OPTION>(wrap(output_view, std::move(weight_buffer)), std::forward<Output>(output), DivideSafe{});
            }
        }
    }

    template<
        bool IS_GPU = false, typename Index,
        typename Input, typename Output, typename Weight,
        typename InputCtf, typename OutputCtf, typename Options>
    void launch_fuse_rotational_averages(
        Input&& input, const Linspace<f64>& input_fftfreq, InputCtf&& input_ctf,
        Output&& output, const Linspace<f64>& output_fftfreq, const OutputCtf& output_ctf,
        Weight&& weight, const Options& options
    ) {
        using input_value_t = nt::const_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using weight_value_t = nt::value_type_t<Weight>;
        using coord_t = nt::value_type_t<output_value_t>;
        constexpr auto IWISE_OPTION = IwiseOptions{.generate_cpu = not IS_GPU, .generate_gpu = IS_GPU};
        constexpr auto EWISE_OPTION = EwiseOptions{.generate_cpu = not IS_GPU, .generate_gpu = IS_GPU};

        // Output must be zeroed out.
        const auto output_view = output.view();
        if (not options.add_to_output)
            ewise<EWISE_OPTION>({}, output_view, Zero{});

        // When computing the average, the weights must be valid.
        auto weight_view = weight.view();
        Array<weight_value_t> weight_buffer;
        if (options.average) {
            if (weight_view.is_empty()) {
                weight_buffer = zeros<weight_value_t>(output_view.shape(), ArrayOption{output.device(), Allocator::DEFAULT_ASYNC});
                weight_view = weight_buffer.view();
            } else if (not options.add_to_output) {
                ewise<EWISE_OPTION>({}, weight_view, Zero{});
            }
        }

        using input_accessor_t = AccessorRestrictContiguous<input_value_t, 2, Index>;
        using output_accessor_t = AccessorRestrictContiguous<output_value_t, 1, Index>;
        using weight_accessor_t = AccessorRestrictContiguous<weight_value_t, 1, Index>;
        auto input_accessor = input_accessor_t(input.get(), input.strides().filter(0).template as<Index>());
        auto output_accessor = output_accessor_t(output_view.get());
        auto weight_accessor = weight_accessor_t(weight_view.get());

        const auto input_fftfreq_f = input_fftfreq.as<coord_t>();
        const auto output_fftfreq_f = output_fftfreq.as<coord_t>();
        const auto n_input_shells = static_cast<Index>(input.shape()[3]);
        const auto n_output_shells = static_cast<Index>(output.shape()[3]);
        const auto iwise_shape = Shape{static_cast<Index>(input.shape()[0]), n_input_shells};

        // Retrieve the CTF(s).
        auto batched_input_ctf = [&] {
            if constexpr (nt::varray_decay<InputCtf>) {
                return BatchedParameter{input_ctf.get()};
            } else if constexpr (nt::empty<InputCtf>) {
                return BatchedParameter<Empty>{};
            } else { // ctf_isotropic
                return BatchedParameter{input_ctf};
            }
        }();

        using op_t = FuseRotationalAverages<
            coord_t, Index, input_accessor_t, output_accessor_t,
            weight_accessor_t, decltype(batched_input_ctf), OutputCtf>;
        auto op = op_t(
            input_accessor, input_fftfreq_f, batched_input_ctf, n_input_shells,
            output_accessor, output_fftfreq_f, output_ctf, n_output_shells, weight_accessor
        );
        iwise<IWISE_OPTION>(
            iwise_shape, output.device(), op,
            std::forward<Input>(input), output, weight, std::forward<InputCtf>(input_ctf)
        );

        // Some shells can be 0, so use DivideSafe.
        if (options.average) {
            if (weight_buffer.is_empty()) {
                ewise<EWISE_OPTION>(wrap(output_view, weight), std::forward<Output>(output), DivideSafe{});
            } else {
                ewise<EWISE_OPTION>(wrap(output_view, std::move(weight_buffer)), std::forward<Output>(output), DivideSafe{});
            }
        }
    }

    template<typename Ctf>
    concept rotational_average_anisotropic_ctf =
        nt::ctf_anisotropic<std::decay_t<Ctf>> or
        (nt::varray_decay<Ctf> and nt::ctf_anisotropic<nt::value_type_t<Ctf>>);

    template<typename Ctf>
    concept rotational_average_isotropic_ctf =
        nt::ctf_isotropic<std::decay_t<Ctf>> or
        (nt::varray_decay<Ctf> and nt::ctf_isotropic<nt::value_type_t<Ctf>>);
}

// TODO Add rotation_average() for 2d only with frequency and angle range.
//      This should be able to take multiple angle ranges for the same input,
//      to "extract" multiple wedges efficiently.

namespace noa::geometry {
    struct RotationalAverageOptions {
        /// Input [start, end] fftfreq range. If the end-frequency is negative or zero, it defaults the highest
        /// fftfreq along the cartesian axes (thus requiring to know the logical shape).
        /// \warning For multidimensional spectra with both odd and even dimensions, the default should be used since
        ///          it is the only way to correctly map the spectrum (because the stop frequency is different for
        ///          odd and even axes). If all dimensions are even, this is equivalent to entering 0.5 and this can
        ///          be ignored.
        Linspace<f64> input_fftfreq{.start = 0., .stop = -1, .endpoint = true};

        /// Output [start, end] fftfreq range. The output shells span over this range.
        /// A negative value (or zero for the stop) defaults to the corresponding value of the input fftfreq range.
        /// If the input fftfreq is defaulted, this would be equal to [0, max(noa::fft::highest_fftfreq(input_shape))].
        Linspace<f64> output_fftfreq{.start = -1., .stop = -1, .endpoint = true};

        /// Whether the rotational average should be computed instead of the rotational sum.
        bool average{true};

        /// Whether the outputs (including the optional weights) are initialized.
        /// If so, the function can skip the extra zeroing.
        bool add_to_output{false};
    };

    /// Computes the rotational sum/average of a 2d or 3d spectrum.
    /// \tparam REMAP           Should be either H2H, HC2H, F2H or FC2H. The output layout is "H" for no particular
    ///                         good reasons other than the number of output shells is often (but not limited to) equal
    ///                         to the half-dimension, i.e. min(shape) // 2 + 1.
    /// \param[in] input        Input spectrum to reduce. Can be real or complex.
    /// \param input_shape      BDHW logical shape of input.
    /// \param[in,out] output   Rotational sum/average. Should be a (batch of) contiguous vector(s).
    ///                         If real, and the input is complex, the power spectrum is computed.
    /// \param[in,out] weights  Rotational weights. Can be empty, or be a (batch of) contiguous vector(s) with the same
    ///                         shape as the output. If valid, the output weights are also saved in this array. If empty
    ///                         and options.average is true, a temporary vector like output is allocated.
    /// \param options          Rotational average and frequency range options.
    template<
        Remap REMAP,
        nt::readable_varray_decay Input,
        nt::writable_varray_decay Output,
        nt::writable_varray_decay_of_any<nt::value_type_twice_t<Output>> Weight = View<nt::value_type_twice_t<Output>>>
    requires (REMAP.is_xx2h() and nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    [[gnu::noinline]] void rotational_average(
        Input&& input,
        const Shape4<i64>& input_shape,
        Output&& output,
        Weight&& weights = {},
        RotationalAverageOptions options = {}
    ) {
        const auto n_shells = guts::check_parameters_rotational_average<REMAP>(
            input, input_shape, Empty{}, output, weights);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            check(ng::is_accessor_access_safe<i32>(input, input.shape()) and
                  ng::is_accessor_access_safe<i32>(output, output.shape()) and
                  ng::is_accessor_access_safe<i32>(weights, weights.shape()),
                  "i64 indexing not instantiated for GPU devices");
            guts::launch_rotational_average<REMAP, true>(
                std::forward<Input>(input), input_shape.as<i32>(), Empty{},
                std::forward<Output>(output),
                std::forward<Weight>(weights),
                n_shells, options);
            #else
            panic_no_gpu_backend();
            #endif
        } else {
            guts::launch_rotational_average<REMAP>(
                std::forward<Input>(input), input_shape.as<i64>(), Empty{},
                std::forward<Output>(output),
                std::forward<Weight>(weights),
                n_shells, options);
        }
    }

    /// Computes the rotational sum/average of a 2d DFT, while correcting for the distortion from the anisotropic ctf.
    /// \tparam REMAP       Should be either H2H, HC2H, F2H or FC2H. The output layout is "H" for no particularly good
    ///                     reasons other than the fact that the number of output shells is often (but not limited to)
    ///                     the half-dimension size, i.e. min(shape) // 2 + 1.
    /// \param[in] input    Input spectrum to reduce. Can be real or complex.
    /// \param input_shape  BDHW logical shape of input.
    /// \param input_ctf    Anisotropic CTF(s). The anisotropic sampling rate and astigmatic field of the defocus are
    ///                     accounted for, resulting in an isotropic rotational average(s). If a varray is passed,
    ///                     there should be one CTF per input batch. Otherwise, the same CTF is assigned to every batch.
    /// \param[out] output  Rotational sum/average. Should be a (batch of) contiguous vector(s).
    ///                     If real and input is complex, the power spectrum is computed.
    /// \param[out] weights Rotational weights. Can be empty, or be a (batch of) contiguous vector(s) with the same
    ///                     shape as the output. If valid, the output weights are also saved in this array.
    /// \param options      Rotational average options.
    /// \note If weights is empty and options.average is true, a temporary vector like output is allocated.
    template<Remap REMAP,
             nt::readable_varray_decay Input,
             nt::writable_varray_decay Output,
             guts::rotational_average_anisotropic_ctf Ctf,
             nt::writable_varray_decay_of_any<nt::value_type_twice_t<Output>> Weight =
                 View<nt::value_type_twice_t<Output>>>
    requires (REMAP.is_xx2h() and nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    [[gnu::noinline]] void rotational_average_anisotropic(
        Input&& input,
        const Shape4<i64>& input_shape,
        Ctf&& input_ctf,
        Output&& output,
        Weight&& weights = {},
        RotationalAverageOptions options = {}
    ) {
        const auto n_shells = guts::check_parameters_rotational_average<REMAP>(
            input, input_shape, input_ctf, output, weights);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            check(ng::is_accessor_access_safe<i32>(input, input.shape()) and
                  ng::is_accessor_access_safe<i32>(output, output.shape()) and
                  ng::is_accessor_access_safe<i32>(weights, weights.shape()),
                  "i64 indexing not instantiated for GPU devices");
            guts::launch_rotational_average<REMAP, true>(
                std::forward<Input>(input), input_shape.as<i32>(),
                std::forward<Ctf>(input_ctf),
                std::forward<Output>(output),
                std::forward<Weight>(weights),
                n_shells, options);
            #else
            panic_no_gpu_backend();
            #endif
        } else {
            guts::launch_rotational_average<REMAP>(
                std::forward<Input>(input), input_shape.as<i64>(),
                std::forward<Ctf>(input_ctf),
                std::forward<Output>(output),
                std::forward<Weight>(weights),
                n_shells, options);
        }
    }

    struct FuseRotationalAveragesOptions {
        /// Whether the average of the input spectra should be computed instead of their sum.
        bool average{true};

        /// Whether the output values (including the optional weights) are initialized.
        /// If so, the function can skip the extra zeroing.
        bool add_to_output{false};
    };

    /// Scale rotational averages (or any 1d spectra) so that their CTF phases match with a target CTF, then, average them.
    /// \param[in] input        Input 1d spectra to reduce. Can be real or complex.
    /// \param input_fftfreq    Frequency range of the input spectra.
    /// \param[in] input_ctf    Isotropic CTFs of the 1d input spectra. One per spectrum.
    /// \param[out] output      Output spectrum. If real, and the input is complex, the power spectrum is computed.
    /// \param output_fftfreq   Frequency range of the output spectrum.
    /// \param output_ctf       Target isotropic CTF. Inputs are rescaled to match the phases of this CTF.
    /// \param[out] weights     Averaging weights. Can be empty, or be a contiguous vector with the same
    ///                         shape as the output. If valid, the output weights are saved in this array.
    ///                         If empty and options.average is true, a temporary vector is allocated.
    /// \param options          Spectrum and averaging options.
    template<
        nt::readable_varray_decay Input,
        nt::writable_varray_decay Output,
        guts::rotational_average_isotropic_ctf InputCtf,
        nt::ctf_isotropic OutputCtf,
        nt::writable_varray_decay_of_any<nt::value_type_twice_t<Output>> Weight = View<nt::value_type_twice_t<Output>>>
    requires (nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    [[gnu::noinline]] void fuse_rotational_averages(
        Input&& input,
        const Linspace<f64>& input_fftfreq,
        InputCtf&& input_ctf,
        Output&& output,
        const Linspace<f64>& output_fftfreq,
        const OutputCtf& output_ctf,
        Weight&& weights = {},
        FuseRotationalAveragesOptions options = {}
    ) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        const bool weights_is_empty = weights.is_empty();

        check(input_fftfreq.start >= 0 and input_fftfreq.start < input_fftfreq.stop and
              output_fftfreq.start >= 0 and output_fftfreq.start < output_fftfreq.stop,
              "Invalid input/output fftfreq range");

        // For simplicity, enforce contiguous row vectors for now.
        check(ni::is_contiguous_vector_batched_strided(input) and noa::all(input.shape().filter(1, 2) == 1),
              "The input must be a (batch of) contiguous row vector(s), but got input:shape={} and input:strides={}",
              input.shape(), input.strides());
        check(ni::is_contiguous_vector(output) and noa::all(output.shape().filter(0, 1, 2) == 1),
              "The output must be a contiguous row vector, but got output:shape={} and output:strides={}",
              output.shape(), output.strides());

        if (not weights_is_empty) {
            check(ni::is_contiguous_vector(weights) and noa::all(weights.shape().filter(0, 1, 2) == 1),
                  "The weights must be a contiguous row vector, but got weights:shape={} and weights:strides={}",
                  weights.shape(), weights.strides());
            check(output.shape()[3] == weights.shape()[3],
                  "The output and weights should have the same size, but got output:n_shells={} and weights:n_shells={}",
                  output.shape()[3], weights.shape()[3]);
        }

        check(input.device() == output.device() and (weights_is_empty or weights.device() == output.device()),
              "The arrays must be on the same device, but got input:device={}, output:device={}{}",
              input.device(), output.device(),
              weights_is_empty ? "" : fmt::format(" and weights:device={}", weights.device()));

        if constexpr (nt::varray_decay<InputCtf>) {
            check(
                ni::is_contiguous_vector(input_ctf) and input_ctf.n_elements() == input.shape()[0],
                "The input CTFs, specified as a contiguous vector, should have the same size "
                "as the input batch size. Got input_ctf:strides={}, input_ctf:shape={}, input:batch={}",
                input_ctf.strides(), input_ctf.shape(), input.shape()[0]
            );
            check(
                input_ctf.device() == output.device(),
                "The input and output arrays must be on the same device, "
                "but got input_ctf:device={} and output:device={}",
                input_ctf.device(), output.device()
            );
        }

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            check(
                ng::is_accessor_access_safe<i32>(input, input.shape()) and
                ng::is_accessor_access_safe<i32>(output, output.shape()) and
                ng::is_accessor_access_safe<i32>(weights, weights.shape()),
                "i64 indexing not instantiated for GPU devices"
            );
            guts::launch_fuse_rotational_averages<true, i32>(
                std::forward<Input>(input), input_fftfreq, std::forward<InputCtf>(input_ctf),
                std::forward<Output>(output), output_fftfreq, output_ctf,
                std::forward<Weight>(weights), options
            );
            #else
            panic_no_gpu_backend();
            #endif
        } else {
            guts::launch_fuse_rotational_averages<false, i64>(
                std::forward<Input>(input), input_fftfreq, std::forward<InputCtf>(input_ctf),
                std::forward<Output>(output), output_fftfreq, output_ctf,
                std::forward<Weight>(weights), options
            );
        }
    }
}
