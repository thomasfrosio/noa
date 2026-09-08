#pragma once

#include "noa/runtime/core/Iwise.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Ewise.hpp"
#include "noa/runtime/Iwise.hpp"
#include "noa/runtime/Factory.hpp"

#include "noa/fft/core/Frequency.hpp"
#include "noa/xform/Utils.hpp"

namespace noa::xform::details {
    struct LerpAndAddAtomicallyToOutput {
        template<typename G, typename T, typename U, typename C, typename I>
        NOA_FHD static void call(
            const G& cg, const T& op, const U& value, C fftfreq, I batches
        ) noexcept {
            // fftfreq to output index.
            const C scaled_fftfreq = (fftfreq - op.m_output_fftfreq_start) / op.m_output_fftfreq_span;
            const C radius = scaled_fftfreq * static_cast<C>(op.m_max_shell_index);
            const C radius_floor = floor(radius);
            const auto shell_low = static_cast<nt::value_type_t<I>>(radius_floor);
            const auto shell_high = shell_low + 1; // shell_low can be the last index

            // Compute lerp weights.
            const C fraction_high = radius - radius_floor;
            const C fraction_low = 1 - fraction_high;

            // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers?
            // TODO Add option for nearest interp.

            auto output = op.m_output[batches];
            auto weight = op.m_weight[batches];
            if (shell_low >= 0 and shell_low <= op.m_max_shell_index) {
                cg.atomic_add(value * static_cast<T::output_real_type>(fraction_low), output, shell_low);
                if (op.m_weight)
                    cg.atomic_add(static_cast<T::weight_value_type>(fraction_low), weight, shell_low);
            }

            if (shell_high >= 0 and shell_high <= op.m_max_shell_index) {
                cg.atomic_add(value * static_cast<T::output_real_type>(fraction_high), output, shell_high);
                if (op.m_weight)
                    cg.atomic_add(static_cast<T::weight_value_type>(fraction_high), weight, shell_high);
            }
        }
    };

    template<nf::Layout REMAP,
             usize B, usize R,
             nt::real Coord,
             nt::sinteger Index,
             nt::readable_nd<B + R> Input,
             nt::readable_nd<B> InputCtf,
             nt::atomic_addable_nd<B + 1> Output,
             nt::atomic_addable_nd_optional<B + 1> Weight>
    class RotationalAverage {
    public:
        static_assert((R == 2 or R == 3) and REMAP.is_xx2h());
        static constexpr bool IS_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_RFFT = REMAP.is_hx2xx();

        using coord_rd_type = Vec<Coord, R>;
        using coord_2d_type = Vec<Coord, 2>;
        using shape_rd_truncated_type = Shape<Index, R - IS_RFFT>;

        using output_value_type = nt::value_type_t<Output>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using weight_value_type = nt::value_type_t<Weight>;
        static_assert(nt::spectrum_types<nt::value_type_t<Input>, output_value_type>);
        static_assert(nt::same_as<weight_value_type, output_real_type>);

        using ctf_type = nt::mutable_value_type_t<InputCtf>;
        static_assert(nt::empty<ctf_type> or (R == 2 and nt::ctf_anisotropic<ctf_type>));

        friend LerpAndAddAtomicallyToOutput;

    public:
        constexpr RotationalAverage(
            const Input& input,
            const Shape<Index, R>& input_shape,
            const InputCtf& input_ctf,
            const Output& output,
            const Weight& weight,
            Index n_shells,
            Linspace<Coord> input_fftfreq,
            Linspace<Coord> output_fftfreq
        ) :
            m_input(input),
            m_output(output),
            m_weight(weight),
            m_input_ctf(input_ctf),
            m_shape(input_shape.template pop_back<IS_RFFT>())
        {
            // If input_fftfreq.stop is negative, defaults to the highest frequency.
            // In this case, and if the frequency.start is 0, this results in the full frequency range.
            // The input is R-d, so we have to handle each axis separately.
            Coord max_input_fftfreq{-1};
            for (usize i{}; i < R; ++i) {
                const auto max_sample_size = input_shape[i] / 2 + 1;
                const auto fftfreq_end =
                    input_fftfreq.stop <= 0 ?
                    nf::highest_fftfreq<Coord>(input_shape[i]) :
                    input_fftfreq.stop;
                max_input_fftfreq = max(max_input_fftfreq, fftfreq_end);
                m_input_fftfreq_step[i] = Linspace<Coord>{
                    .start = 0,
                    .stop = fftfreq_end,
                    .endpoint = input_fftfreq.endpoint
                }.for_size(max_sample_size).step;
            }

            // The output defaults to the input range. Of course, it is a reduction to 1d, so take the max fftfreq.
            if (output_fftfreq.start < 0)
                output_fftfreq.start = 0;
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
            m_fftfreq_cutoff[1] = output_fftfreq.stop + static_cast<Coord>(m_max_shell_index) * output_fftfreq_step;
        }

        // 2d or 3d rotational average, with an optional anisotropic field correction.
        template<nt::compute_handle C>
        NOA_HD void operator()(const C& ch, const Vec<Index, B + R>& batched_indices) const noexcept {
            // Input indices to fftfreq.
            const auto& [batches, indices] = batched_indices.template split<B>();
            const auto frequency = nf::index2frequency<IS_CENTERED, IS_RFFT>(indices, m_shape);
            const auto fftfreq_nd = coord_rd_type::from_vec(frequency) * m_input_fftfreq_step;

            Coord fftfreq;
            if constexpr (nt::empty<ctf_type>) {
                fftfreq = sqrt(dot(fftfreq_nd, fftfreq_nd));
            } else {
                // Correct for anisotropic field (pixel size and defocus).
                fftfreq = static_cast<Coord>(m_input_ctf[batches].isotropic_fftfreq(fftfreq_nd));
            }

            // Remove most out-of-bounds asap.
            if (fftfreq < m_fftfreq_cutoff[0] or fftfreq > m_fftfreq_cutoff[1])
                return;

            const auto value = cast_or_abs_squared<output_value_type>(m_input[batched_indices]);
            LerpAndAddAtomicallyToOutput::call(ch.grid(), *this, value, fftfreq, batches);
        }

    private:
        Input m_input;
        Output m_output;
        Weight m_weight;
        NOA_NO_UNIQUE_ADDRESS InputCtf m_input_ctf;

        shape_rd_truncated_type m_shape;
        coord_rd_type m_input_fftfreq_step;
        coord_2d_type m_fftfreq_cutoff;
        Coord m_output_fftfreq_start;
        Coord m_output_fftfreq_span;
        Index m_max_shell_index;
    };

    template<usize B, usize R, nt::real Coord, nt::sinteger Index,
             nt::readable_nd<B + R> Input,
             nt::atomic_addable_nd<B + R> Output,
             nt::atomic_addable_nd_optional<B + R> Weight,
             nt::readable_nd<B> InputCtf,
             nt::readable_nd<B> OutputCtf>
    class EquiphaseRescaleAndAverage {
    public:
        static_assert(R == 1); // TODO Add support for higher ranks

        using output_value_type = nt::value_type_t<Output>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using weight_value_type = nt::value_type_t<Weight>;
        static_assert(nt::spectrum_types<nt::value_type_t<Input>, output_value_type>);
        static_assert(nt::same_as<weight_value_type, output_real_type>);

        using input_ctf_type = nt::mutable_value_type_t<InputCtf>;
        using output_ctf_type = nt::mutable_value_type_t<OutputCtf>;
        static_assert(nt::ctf_isotropic<input_ctf_type, output_ctf_type>);

        friend LerpAndAddAtomicallyToOutput;

    public:
        constexpr EquiphaseRescaleAndAverage(
            const Input& input,
            const Linspace<Coord>& input_fftfreq,
            const InputCtf& input_ctf,
            Index n_input_shells,
            const Output& output,
            Linspace<Coord> output_fftfreq,
            const OutputCtf& output_ctf,
            Index n_output_shells,
            const Weight& weight
        ) :
            m_input(input),
            m_output(output),
            m_weight(weight),
            m_input_ctf(input_ctf),
            m_output_ctf(output_ctf),
            m_max_shell_index(n_output_shells - 1)
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

            // To shortcut early, compute the fftfreq cutoffs where we know the output isn't affected.
            auto output_fftfreq_step = output_fftfreq.for_size(n_output_shells).step;
            m_fftfreq_cutoff[0] = output_fftfreq.start + -1 * output_fftfreq_step;
            m_fftfreq_cutoff[1] = output_fftfreq.stop + static_cast<Coord>(m_max_shell_index) * output_fftfreq_step;
        }

        NOA_HD void operator()(nt::compute_handle auto& ch, const Vec<Index, B + 1>& batched_indices) const noexcept {
            const auto& [batches, index] = batched_indices.template split<B>();
            const auto input_fftfreq = m_input_fftfreq_start + static_cast<Coord>(index[0]) * m_input_fftfreq_step;
            const auto input_phase = m_input_ctf[batches].phase_at(input_fftfreq);
            const auto fftfreq = m_output_ctf[batches].template fftfreq_at<Coord>(input_phase);

            // Remove most out-of-bounds asap.
            if (not fftfreq.has_value() or *fftfreq < m_fftfreq_cutoff[0] or *fftfreq > m_fftfreq_cutoff[1])
                return;

            const auto value = cast_or_abs_squared<output_value_type>(m_input[batched_indices]);
            const auto weight = static_cast<output_real_type>(m_input_ctf[batches].scale());
            LerpAndAddAtomicallyToOutput::call(ch.grid(), *this, value * weight, *fftfreq, batches);
        }

    private:
        Input m_input;
        Output m_output;
        Weight m_weight;
        InputCtf m_input_ctf;
        OutputCtf m_output_ctf;

        Vec<Coord, 2> m_fftfreq_cutoff;
        Coord m_input_fftfreq_start;
        Coord m_input_fftfreq_step;
        Coord m_output_fftfreq_start;
        Coord m_output_fftfreq_span;
        Index m_max_shell_index;
        Index m_chunk_size;
    };

    template<usize B, usize R,
             nt::real Coord,
             nt::sinteger Index,
             nt::interpolator_spectrum_nd<R> Interpolator,
             nt::readable_nd<B + R> Input,
             nt::writable_nd<B + R> Output,
             nt::readable_nd<B> InputCtf,
             nt::readable_nd<B> OutputCtf>
    class EquiphaseRescale {
    public:
        static_assert(R == 1); // TODO Add support for higher ranks

        using output_value_type = nt::value_type_t<Output>;
        static_assert(nt::spectrum_types<nt::value_type_t<Input>, output_value_type>);

        using input_ctf_type = nt::mutable_value_type_t<InputCtf>;
        using output_ctf_type = nt::mutable_value_type_t<OutputCtf>;
        static_assert(nt::ctf_isotropic<input_ctf_type, output_ctf_type>);

    public:
        constexpr EquiphaseRescale(
            const Input& input,
            const Interpolator& interpolator,
            const Linspace<Coord>& input_fftfreq,
            const InputCtf& input_ctf,
            Index n_input_shells,
            const Output& output,
            Linspace<Coord> output_fftfreq,
            const OutputCtf& output_ctf,
            Index n_output_shells
        ) :
            m_input{input},
            m_output{output},
            m_input_ctf{input_ctf},
            m_output_ctf{output_ctf},
            m_interpolator{interpolator},
            m_input_fftfreq_start{input_fftfreq.start},
            m_input_fftfreq_step{input_fftfreq.for_size(n_input_shells).step},
            m_output_fftfreq_start{output_fftfreq.start},
            m_output_fftfreq_step{output_fftfreq.for_size(n_output_shells).step}
        {}

        NOA_HD void operator()(const Vec<Index, B + R>& batched_indices) const noexcept {
            const auto& [batches, index] = batched_indices.template split<B>();
            const auto output_fftfreq = m_output_fftfreq_start + static_cast<Coord>(index[0]) * m_output_fftfreq_step;
            const auto phase = m_output_ctf[batches].phase_at(output_fftfreq);
            const auto input_fftfreq = m_input_ctf[batches].template fftfreq_at<Coord>(phase);
            if (not input_fftfreq.has_value()) {
                m_output[batched_indices] = 0;
                return;
            }

            const auto input_frequency = (*input_fftfreq - m_input_fftfreq_start) / m_input_fftfreq_step;
            const auto interpolated_value = m_interpolator.get(m_input[batches], Vec{input_frequency});
            m_output[batched_indices] = cast_or_abs_squared<output_value_type>(interpolated_value);
        }

    private:
        Input m_input;
        Output m_output;
        InputCtf m_input_ctf;
        OutputCtf m_output_ctf;
        NOA_NO_UNIQUE_ADDRESS Interpolator m_interpolator;

        Coord m_input_fftfreq_start;
        Coord m_input_fftfreq_step;
        Coord m_output_fftfreq_start;
        Coord m_output_fftfreq_step;
    };

    template<bool CONTIGUOUS = true, typename T, usize B>
    auto check_parameters_ctf(
        const T& ctf, const Shape<isize, B>& shape_b, Device device,
        const std::source_location& location = std::source_location::current()
    ) {
        if constexpr (nt::array<T>) {
            check_at_location(
                location, (not CONTIGUOUS or ctf.is_contiguous()) and ctf.shape() == shape_b,
                "The CTFs, specified as a array, should have a shape matching the corresponding array batch axes. Got ctf:strides={}, ctf:shape={}, batches={}",
                ctf.strides(), ctf.shape(), shape_b
            );
            check_at_location(
                location, ctf.device() == device,
                "The input and output arrays must be on the same device, but got ctf:device={} and output:device={}",
                ctf.device(), device
            );
        }
    }

    template<nf::Layout REMAP, typename Input, usize N, typename InputCtf, typename Output, typename Weight>
    void check_parameters_rotational_average(
        const Input& input,
        const Shape<isize, N>& shape,
        const InputCtf& input_ctf,
        const Output& output,
        const Weight& weights,
        const Linspace<f64>& input_fftfreq
    ) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");

        check(input.shape() == (REMAP.is_hx2xx() ? shape.rfft() : shape),
              "The input array does not match the logical shape. Got input:shape={}, shape={}, remap={}",
              input.shape(), shape, REMAP);
        check(input.device() == output.device(),
              "The arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), output.device());

        // Check for batch axes match.
        constexpr auto B = nt::array_size_v<Output> - 1;
        constexpr auto R = N - B;
        auto input_shape_b = input.shape().template pop_back<R>();
        auto output_shape_b = output.shape().template pop_back<1>();
        if constexpr (B >= 1) {
            for (usize i{}; i < B; ++i)
                if (output_shape_b[i] == 1)
                    output_shape_b[i] = input_shape_b[i];
            check(input_shape_b == output_shape_b,
                  "Each output batch axis should match the corresponding input batch axis or be 1, but got input:batches={}, output:batches={}",
                  input_shape_b, output.shape().template pop_back<1>());
        }

        check(output.contiguity()[B] == true,
              "The output must be an array of contiguous vector(s), but got output:shape={} and output:strides={}",
              output.shape(), output.strides());

        if constexpr (not nt::empty<Weight>) {
            if (not weights.is_empty()) {
                check(weights.contiguity()[B] == true,
                      "The weights must be an array of contiguous vector(s), but got weights:shape={} and weights:strides={}",
                      weights.shape(), weights.strides());
                check(weights.shape() == output.shape(),
                      "The weights should have the same shape as the output array, but got weights:shape={} and output:shape={}",
                      weights.shape(), output.shape());
                check(input.device() == weights.device(),
                      "The arrays must be on the same device, but got input:device={}, weights:device={}",
                      input.device(), weights.device());
            }
        }

        check_parameters_ctf<true>(input_ctf, input_shape_b, input.device());
        check(allclose(input_fftfreq.start, 0.), "The starting fftfreq should be 0, but got {}", input_fftfreq.start);
    }

    template<bool REDUCE, typename Input, typename InputCtf, typename Output, typename OutputCtf, typename Weight = Empty>
    void check_parameters_equiphase_rescale(
        const Input& input,
        const Linspace<f64>& input_fftfreq,
        const InputCtf& input_ctf,
        const Output& output,
        const Linspace<f64>& output_fftfreq,
        const OutputCtf& output_ctf,
        const Weight& weights = Weight{}
    ) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");

        check(input_fftfreq.start >= 0 and input_fftfreq.start < input_fftfreq.stop and
              output_fftfreq.start >= 0 and output_fftfreq.start < output_fftfreq.stop,
              "Invalid input/output fftfreq range");

        // For simplicity, enforce contiguous row vectors for now.
        constexpr auto B = nt::array_size_v<Output> - 1;
        constexpr auto R = 1;
        check(input.contiguity()[B] == true,
              "The input must be contiguous vector(s), but got input:shape={} and input:strides={}",
              input.shape(), input.strides());
        check(output.contiguity()[B] == true,
              "The output must be contiguous vector(s), but got output:shape={} and output:strides={}",
              output.shape(), output.strides());

        // Check for batch axes match.
        auto input_shape_b = input.shape().template pop_back<R>();
        auto output_shape_b = output.shape().template pop_back<R>();
        if constexpr (B >= 1) {
            if constexpr (REDUCE) {
                for (usize i{}; i < B; ++i)
                    if (output_shape_b[i] == 1)
                        output_shape_b[i] = input_shape_b[i];
                check(input_shape_b == output_shape_b,
                      "Each output batch axis should match the corresponding input batch axis or be 1, but got input:batches={}, output:batches={}",
                      input_shape_b, output.shape().template pop_back<1>());
            } else {
                for (usize i{}; i < B; ++i)
                    if (input_shape_b[i] == 1)
                        input_shape_b[i] = output_shape_b[i];
                check(input_shape_b == output_shape_b,
                      "Each input batch axis should match the corresponding output batch axis or be 1, but got input:batches={}, output:batches={}",
                      input.shape().template pop_back<R>(), output_shape_b);
            }
        }

        check(input.device() == output.device(),
              "The arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), output.device());

        if constexpr (not nt::empty<Weight>) {
            if (not weights.is_empty()) {
                check(weights.contiguity()[B] == true,
                      "The weights must be an array of contiguous vector(s), but got weights:shape={} and weights:strides={}",
                      weights.shape(), weights.strides());
                check(weights.shape() == output.shape(),
                      "The weights should have the same shape as the output array, but got weights:shape={} and output:shape={}",
                      weights.shape(), output.shape());
                check(input.device() == weights.device(),
                      "The arrays must be on the same device, but got input:device={}, weights:device={}",
                      input.device(), weights.device());
            }
        }

        check_parameters_ctf<REDUCE>(input_ctf, input_shape_b, output.device());
        check_parameters_ctf<not REDUCE>(output_ctf, output_shape_b, output.device());
    }

    template<bool ALLOW_EMPTY = false, bool CONTIGUOUS = true, typename T>
    constexpr auto ctf_to_accessor(const T& value) {
        using value_t = nt::const_value_type_t<T>;
        if constexpr (nt::empty<T>) {
            if constexpr (ALLOW_EMPTY)
                return AccessorValue<Empty>{};
            else
                static_assert(nt::always_false<T>);
        } else if constexpr (nt::array<T>) {
            NOA_ASSERT(value.is_contiguous());
            constexpr auto B = nt::array_size_v<T>;
            constexpr auto STRIDE_TRAIT = CONTIGUOUS ? StridesTraits::CONTIGUOUS : StridesTraits::CONTIGUOUS;
            auto strides = value.strides();
            if constexpr (CONTIGUOUS and B >= 1)
                for (usize i{}; i < B; ++i)
                    if (value.shape()[i] == 1)
                        strides[i] = 0;
            return AccessorRestrict<value_t, B, isize, STRIDE_TRAIT>{value.get(), strides};
        } else {
            return AccessorValue{value};
        }
    }

    template<nf::Layout REMAP,
             typename Input, typename Index, typename Ctf, usize N,
             typename Output, typename Weight, typename Options>
    void launch_rotational_average(
        Input&& input, const Shape<Index, N>& input_shape, Ctf&& input_ctf,
        Output&& output, Weight&& weight, const Options& options
    ) {
        using input_value_t = nt::const_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using weight_value_t = nt::value_type_t<output_value_t>;
        using coord_t = nt::largest_type_t<f32, nt::value_type_t<output_value_t>>;

        constexpr usize B = nt::array_size_v<Output> - 1;
        constexpr usize R = nt::array_size_v<Input> - B;

        // Output must be zeroed out.
        const auto output_view = output.view();
        if (not options.add_to_output)
            ewise({}, output_view, Zero{});

        // When computing the average, the weights must be valid.
        using weight_view_t = Array<weight_value_t, B + 1, ArrayOwnership::VIEW>;
        constexpr bool HAS_WEIGHT = not nt::empty<std::decay_t<Weight>>;
        auto weight_view = weight_view_t{};
        if constexpr (HAS_WEIGHT)
            weight_view = weight.view();

        auto weight_buffer = Array<weight_value_t, B + 1>{};
        if (options.average) {
            if (weight_view.is_empty()) {
                weight_buffer = zeros<weight_value_t>(output_view.shape(), {
                    .device = output.device(),
                    .allocator = Allocator::DEFAULT_ASYNC,
                });
                weight_view = weight_buffer.view();
            } else if (not options.add_to_output) {
                ewise({}, weight_view, Zero{});
            }
        }

        // Allow reduction of output batch axes by broadcasting them.
        auto output_strides = output_view.strides().template as<Index>();
        auto weight_strides = weight_view.strides().template as<Index>();
        if constexpr (B >= 1) {
            for (usize i{}; i < B; ++i) {
                if (output.shape()[i] == 1) {
                    output_strides[i] = 0;
                    weight_strides[i] = 0;
                }
            }
        }

        using input_accessor_t = AccessorRestrict<input_value_t, N, Index>;
        using output_accessor_t = AccessorRestrictContiguous<output_value_t, B + 1, Index>;
        using weight_accessor_t = AccessorRestrictContiguous<weight_value_t, B + 1, Index>;
        auto input_accessor = input_accessor_t(input.get(), input.strides().template as<Index>());
        auto output_accessor = output_accessor_t(output_view.get(), output_strides);
        auto weight_accessor = weight_accessor_t(weight_view.get(), weight_strides);
        auto input_ctf_accessor = ctf_to_accessor<true, true>(input_ctf);
        using input_ctf_accessor_t = decltype(input_ctf_accessor);

        using op_t = RotationalAverage<
            REMAP, B, R, coord_t, Index,
            input_accessor_t, input_ctf_accessor_t,
            output_accessor_t, weight_accessor_t>;

        const auto input_shape_br = input.shape().template as<Index>();
        const auto input_shape_r = input_shape.template pop_front<B>();
        const auto n_output_shells = output_view.shape()[B];
        auto op = op_t(
            input_accessor, input_shape_r, input_ctf_accessor,
            output_accessor, weight_accessor, static_cast<Index>(n_output_shells),
            options.input_fftfreq.template as<coord_t>(),
            options.output_fftfreq.template as<coord_t>()
        );
        iwise(input_shape_br, output.device(), op, NOA_FWD(input), output, weight, NOA_FWD(input_ctf));

        // Some shells can be 0, so use DivideSafe.
        if (options.average) {
            if (weight_buffer.is_empty()) {
                if constexpr (HAS_WEIGHT)
                    ewise(wrap(output_view, weight), NOA_FWD(output), DivideSafe{});
            } else {
                ewise(wrap(output_view, std::move(weight_buffer)), NOA_FWD(output), DivideSafe{});
            }
        }
    }

    template<typename Index,
             typename Input, typename Output, typename Weight,
             typename InputCtf, typename OutputCtf, typename Options>
    void launch_equiphase_rescale_and_average(
        Input&& input, const Linspace<f64>& input_fftfreq, InputCtf&& input_ctf,
        Output&& output, const Linspace<f64>& output_fftfreq, OutputCtf&& output_ctf,
        Weight&& weight, const Options& options
    ) {
        using input_value_t = nt::const_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using weight_value_t = nt::value_type_t<output_value_t>;
        using coord_t = nt::largest_type_t<f32, nt::value_type_t<output_value_t>>;

        constexpr usize B = nt::array_size_v<Output> - 1;
        constexpr usize R = nt::array_size_v<Input> - B;
        static_assert(R == 1);

        // Output must be zeroed out.
        const auto output_view = output.view();
        if (not options.add_to_output)
            ewise({}, output_view, Zero{});

        // When computing the average, the weights must be valid.
        using weight_view_t = Array<weight_value_t, B + 1, ArrayOwnership::VIEW>;
        constexpr bool HAS_WEIGHT = not nt::empty<std::decay_t<Weight>>;
        auto weight_view = weight_view_t{};
        if constexpr (HAS_WEIGHT)
            weight_view = weight.view();

        auto weight_buffer = Array<weight_value_t, B + 1>{};
        if (options.average) {
            if (weight_view.is_empty()) {
                weight_buffer = zeros<weight_value_t>(output_view.shape(), {
                    .device = output.device(),
                    .allocator = Allocator::DEFAULT_ASYNC,
                });
                weight_view = weight_buffer.view();
            } else if (not options.add_to_output) {
                ewise({}, weight_view, Zero{});
            }
        }

        // Allow reduction of output batch axes by broadcasting them.
        auto output_strides = output_view.strides().template as<Index>();
        auto weight_strides = weight_view.strides().template as<Index>();
        if constexpr (B >= 1) {
            for (usize i{}; i < B; ++i) {
                if (output.shape()[i] == 1) {
                    output_strides[i] = 0;
                    weight_strides[i] = 0;
                }
            }
        }

        using input_accessor_t = AccessorRestrictContiguous<input_value_t, B + 1, Index>;
        using output_accessor_t = AccessorRestrictContiguous<output_value_t, B + 1, Index>;
        using weight_accessor_t = AccessorRestrictContiguous<weight_value_t, B + 1, Index>;
        auto input_accessor = input_accessor_t(input.get(), input.strides().template as<Index>());
        auto output_accessor = output_accessor_t(output_view.get(), output_strides);
        auto weight_accessor = weight_accessor_t(weight_view.get(), weight_strides);
        auto input_ctf_accessor = ctf_to_accessor<false, true>(input_ctf);
        auto output_ctf_accessor = ctf_to_accessor<false, false>(output_ctf); // strided for broadcasting
        using input_ctf_accessor_t = decltype(input_ctf_accessor);
        using output_ctf_accessor_t = decltype(output_ctf_accessor);

        const auto n_input_shells = static_cast<Index>(input.shape()[B]);
        const auto n_output_shells = static_cast<Index>(output.shape()[B]);

        using op_t = EquiphaseRescaleAndAverage<
            B, R, coord_t, Index, input_accessor_t, output_accessor_t,
            weight_accessor_t, input_ctf_accessor_t, output_ctf_accessor_t>;
        auto op = op_t(
            input_accessor, input_fftfreq.as<coord_t>(), input_ctf_accessor, n_input_shells,
            output_accessor, output_fftfreq.as<coord_t>(), output_ctf_accessor, n_output_shells,
            weight_accessor
        );
        iwise(
            input.shape().template as<Index>(), output.device(), op,
            NOA_FWD(input), output, weight, NOA_FWD(input_ctf), NOA_FWD(output_ctf)
        );

        // Some shells can be 0, so use DivideSafe.
        if (options.average) {
            if (weight_buffer.is_empty()) {
                if constexpr (HAS_WEIGHT)
                    ewise(wrap(output_view, weight), NOA_FWD(output), DivideSafe{});
            } else {
                ewise(wrap(output_view, std::move(weight_buffer)), NOA_FWD(output), DivideSafe{});
            }
        }
    }

    template<typename Index,
             typename Input, typename Output,
             typename InputCtf, typename OutputCtf, typename Options>
    void launch_equiphase_rescale(
        Input&& input, const Linspace<f64>& input_fftfreq, InputCtf&& input_ctf,
        Output&& output, const Linspace<f64>& output_fftfreq, OutputCtf&& output_ctf,
        const Options& options
    ) {
        using output_value_t = nt::value_type_t<Output>;
        using coord_t = nt::largest_type_t<f32, nt::value_type_t<output_value_t>>;

        constexpr usize B = nt::array_size_v<Output> - 1;
        constexpr usize R = nt::array_size_v<Input> - B;
        static_assert(R == 1);

        const auto n_input_shells = static_cast<Index>(input.shape()[B]);
        const auto n_output_shells = static_cast<Index>(output.shape()[B]);

        using output_accessor_t = AccessorRestrictContiguous<output_value_t, 2, Index>;
        auto output_accessor = output_accessor_t(output.get(), output.strides().template as<Index>());

        auto input_ctf_accessor = ctf_to_accessor<false, false>(input_ctf); // strided for broadcasting
        auto output_ctf_accessor = ctf_to_accessor<false, true>(output_ctf);
        using input_ctf_accessor_t = decltype(input_ctf_accessor);
        using output_ctf_accessor_t = decltype(output_ctf_accessor);

        auto launch_iwise = [&](auto interp) {
            // Because it is 1D and RFFT, this logical shape is correct even for odd logical sizes.
            // frequency_bounds returns [0, logical_size / 2], so for:
            //  n_shells=5 -> logical=(5-1)*2=8  -> bond=8/2=4 (expected=5-1=4).
            //  n_shells=6 -> logical=(6-1)*2=10 -> bound=10/2=5 (expected=6-1=5).
            const auto logical_shape_r = Shape<Index, 1>{(n_input_shells - 1) * 2};
            auto result = prepare_interpolation_spectrum_inputs<1, nf::Layout::H2H, interp(), false, coord_t, false>(
                input.span_contiguous(), logical_shape_r);
            using interpolator_t = decltype(result)::interpolator_type;
            using input_accessor_t = decltype(result)::accessor_type;

            using op_t = EquiphaseRescale<
                B, R, coord_t, Index,
                interpolator_t, input_accessor_t, output_accessor_t,
                input_ctf_accessor_t, output_ctf_accessor_t>;
            auto op = op_t(
                result.accessor, result.interpolator, input_fftfreq.as<coord_t>(), input_ctf_accessor, n_input_shells,
                output_accessor, output_fftfreq.as<coord_t>(), output_ctf_accessor, n_output_shells
            );
            iwise(
                output.shape().template as<Index>(), output.device(), op,
                NOA_FWD(input), output, NOA_FWD(input_ctf), NOA_FWD(output_ctf)
            );
        };

        auto interp = options.interp.erase_fast();
        switch (options.interp) {
            case Interp::NEAREST:       return launch_iwise(WrapInterp<Interp::NEAREST>{});
            case Interp::LINEAR:        return launch_iwise(WrapInterp<Interp::LINEAR>{});
            case Interp::CUBIC:         return launch_iwise(WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_BSPLINE: return launch_iwise(WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::LANCZOS4:      return launch_iwise(WrapInterp<Interp::LANCZOS4>{});
            case Interp::LANCZOS6:      return launch_iwise(WrapInterp<Interp::LANCZOS6>{});
            case Interp::LANCZOS8:      return launch_iwise(WrapInterp<Interp::LANCZOS8>{});
            default: panic("interp={} is not supported", interp);
        }
    }

    template<nf::Layout REMAP, typename Input, typename Output, typename Weight, usize N,
             usize NO = nt::array_size_v<Output>>
    concept rotational_averageable = REMAP.is_xx2h() and
        nt::readable_array_decay<Input> and nt::writable_array_decay<Output> and
        nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>> and
        nt::array_size_v<Input> == N and N > NO and is_within(N - (NO - 1), 2, 3) and
        (nt::empty<std::decay_t<Weight>> or (
            nt::writable_array_decay_of_any<Weight, nt::value_type_twice_t<Output>> and
            nt::array_size_v<Weight> == NO));

    template<typename Ctf, usize B, usize R>
    concept equiphase_average_anisotropic_ctf =
        R == 2 and
        (nt::ctf_anisotropic<std::decay_t<Ctf>> or
         (nt::array_decay_nd<Ctf, B> and nt::ctf_anisotropic<nt::value_type_t<Ctf>>));

    template<nf::Layout REMAP, typename Input, typename InputCtf, typename Output, typename Weight, usize N,
             usize NO = nt::array_size_v<Output>>
    concept equiphase_averageable =
        rotational_averageable<REMAP, Input, Output, Weight, N> and
        equiphase_average_anisotropic_ctf<InputCtf, NO - 1, N + 1 - NO>; // rank 2 only

    template<typename Ctf, usize B>
    concept equiphase_rescale_isotropic_ctf =
        nt::ctf_isotropic<std::decay_t<Ctf>> or
        (nt::array_decay_nd<Ctf, B> and nt::ctf_isotropic<nt::value_type_t<Ctf>>);

    template<typename Input, typename InputCtf, typename Output, typename OutputCtf, typename Weight,
             usize N = nt::array_size_v<Output>>
    concept equiphase_rescale_and_averageable =
        nt::readable_array_decay<Input> and nt::writable_array_decay<Output> and
        nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>> and
        N == nt::array_size_v<Output> and
        (nt::empty<std::decay_t<Weight>> or (
            nt::writable_array_decay_of_any<Weight, nt::value_type_twice_t<Output>> and
            N == nt::array_size_v<Output>)) and
        equiphase_rescale_isotropic_ctf<InputCtf, N - 1> and
        equiphase_rescale_isotropic_ctf<OutputCtf, N - 1>;
}

// TODO Add rotation_average() for 2d only with frequency and angle range.
//      This should be able to take multiple angle ranges for the same input,
//      to "extract" multiple wedges efficiently.

namespace noa::xform {
    struct RotationalAverageOptions {
        /// Input [0, end] fftfreq range. If the end-frequency is negative or zero, it defaults the highest
        /// fftfreq along the cartesian axes (thus requiring to know the logical shape).
        /// Note that the start frequency must be zero, otherwise an error will be thrown.
        /// \warning For multidimensional spectra with both odd and even dimensions, the default should be used since
        ///          it is the only way to correctly map the spectrum (because the stop frequency is different for
        ///          odd and even axes). If all dimensions are even, this is equivalent to entering 0.5 and this can
        ///          be ignored.
        Linspace<f64> input_fftfreq{.start = 0., .stop = -1, .endpoint = true};

        /// Output [start, end] fftfreq range. The output shells span over this range.
        /// A negative value (or zero for the stop) defaults to the corresponding value of the input fftfreq range.
        /// If the input fftfreq is defaulted, this would be equal to [0, max(noa::fft::highest_fftfreq(input_shape))].
        Linspace<f64> output_fftfreq{.start = 0., .stop = -1, .endpoint = true};

        /// Whether the rotational average should be computed instead of the rotational sum.
        bool average{true};

        /// Whether the outputs (including the optional weights) are initialized.
        /// If so, the function can skip the extra zeroing.
        bool add_to_output{false};
    };

    /// Computes the rotational sum/average of a 2d or 3d spectrum.
    /// \tparam REMAP:
    ///     Should be either H2H, HC2H, F2H or FC2H.
    ///     The output layout is "H" for no particular good reasons other than the number of output shells is often
    ///     (but not limited to) equal to the half-dimension, i.e. min(shape) // 2 + 1.
    /// \param[in] input:
    ///     2D: ((Bi..,)Hi,Wi) or 3D: ((Bi..,)Di,Hi,Wi).
    ///     Input spectrum to reduce. Can be real or complex.
    /// \param input_shape:
    ///     Logical shape of the input.
    /// \param[in,out] output:
    ///     ((Bo..,)Wo). Rotational sum/average(s).
    ///     The number of batch axes sets the rank of the input.
    ///     Each (Bo..) axis should match the corresponding input (Bi..) or be 1, in which case the axis is reduced.
    ///     Should be contiguous vector(s). If real, and the input is complex, the power spectrum is computed.
    /// \param[in,out] weights:
    ///     Rotational weights.
    ///     Can be empty, or be contiguous vector(s) with the same shape as the output.
    ///     If valid, the output weights are also saved in this array.
    ///     If empty and options.average is true, a temporary array is allocated.
    /// \param options:
    ///     Rotational average and frequency range options.
    template<nf::Layout REMAP, typename Input, typename Output, typename Weight = Empty, usize N>
        requires details::rotational_averageable<REMAP, Input, Output, Weight, N>
    void rotational_average(
        Input&& input,
        const Shape<isize, N>& input_shape,
        Output&& output,
        Weight&& weights = {},
        const RotationalAverageOptions& options = {}
    ) {
        details::check_parameters_rotational_average<REMAP>(
            input, input_shape, Empty{}, output, weights, options.input_fftfreq
        );
        details::launch_rotational_average<REMAP>(
            NOA_FWD(input), input_shape, Empty{},
            NOA_FWD(output), NOA_FWD(weights), options
        );
    }

    template<nf::Layout REMAP, typename Input, typename Output, typename Weight = Empty>
    void rotational_average_2d(
        Input&& input,
        const Shape<isize, 2>& input_shape,
        Output&& output,
        Weight&& weights = {},
        const RotationalAverageOptions& options = {}
    ) {
        rotational_average<REMAP>(NOA_FWD(input), input_shape, NOA_FWD(output), NOA_FWD(weights), options);
    }
    template<nf::Layout REMAP, typename Input, typename Output, typename Weight = Empty>
    void rotational_average_3d(
        Input&& input,
        const Shape<isize, 3>& input_shape,
        Output&& output,
        Weight&& weights = {},
        const RotationalAverageOptions& options = {}
    ) {
        rotational_average<REMAP>(NOA_FWD(input), input_shape, NOA_FWD(output), NOA_FWD(weights), options);
    }

    /// Computes the rotational sum/average of a 2d DFT, while correcting for the distortion from the anisotropic ctf.
    /// \tparam REMAP:
    ///     Should be either H2H, HC2H, F2H or FC2H.
    ///     Same as rotational_average.
    /// \param[in] input:
    ///     ((Bi..,)Hi,Wi) Input spectrum to reduce.
    ///     Same as rotational_average.
    /// \param input_shape:
    ///     Logical shape of input.
    /// \param input_ctf:
    ///     Anisotropic CTF(s).
    ///     A contiguous (Bi..) array, or a single value, in which case the CTF is assigned to every input batch.
    ///     The anisotropic sampling rate and astigmatic field of the defocus are accounted for, resulting in an
    ///     isotropic rotational average(s).
    /// \param[out] output:
    ///     Rotational sum/average.
    ///     Same as rotational_average.
    /// \param[out] weights:
    ///     Rotational weights.
    ///     Same as rotational_average.
    /// \param options:
    ///     Rotational average options.
    template<nf::Layout REMAP, typename Input, typename InputCtf, typename Output, typename Weight = Empty, usize N>
        requires (details::equiphase_averageable<REMAP, Input, InputCtf, Output, Weight, N>)
    void equiphase_average(
        Input&& input,
        const Shape<isize, N>& input_shape,
        InputCtf&& input_ctf,
        Output&& output,
        Weight&& weights = {},
        const RotationalAverageOptions& options = {}
    ) {
        details::check_parameters_rotational_average<REMAP>(
            input, input_shape, input_ctf, output, weights, options.input_fftfreq
        );
        details::launch_rotational_average<REMAP>(
            NOA_FWD(input), input_shape, NOA_FWD(input_ctf),
            NOA_FWD(output), NOA_FWD(weights), options
        );
    }

    template<nf::Layout REMAP, typename Input, typename InputCtf, typename Output, typename Weight = Empty>
    void equiphase_average_2d(
        Input&& input,
        const Shape<isize, 2>& input_shape,
        InputCtf&& input_ctf,
        Output&& output,
        Weight&& weights = {},
        const RotationalAverageOptions& options = {}
    ) {
        equiphase_average<REMAP>(
            NOA_FWD(input), input_shape, NOA_FWD(input_ctf),
            NOA_FWD(output), NOA_FWD(weights), options
        );
    }

    struct FuseSpectraOptions {
        /// Whether the average of the input spectra should be computed instead of their sum.
        bool average{true};

        /// Whether the output values (including the optional weights) are initialized.
        /// If so, the function can skip the extra zeroing.
        bool add_to_output{false};
    };

    /// Scale 1d rfft spectra so that their CTF phases match with the target CTF, then, average them.
    /// \param[in] input:
    ///     ((Bi..,)Wi) Input 1d spectra of contiguous vectors to reduce. Can be real or complex.
    /// \param input_fftfreq:
    ///     Frequency range of the input spectra.
    /// \param[in] input_ctf:
    ///     Isotropic CTF(s) of the 1d input spectra.
    ///     A contiguous (Bi..) array, or a single value, in which case the CTF is assigned to every input batch.
    ///     The CTF scale is used to assign a weight to each input spectrum.
    /// \param[out] output:
    ///     ((Bo..,)Wo) Output spectra.
    ///     Each (Bo..) axis should match the corresponding input (Bi..) or be 1, in which case the axis is reduced
    ///     by averaging (or summing) the corresponding rescaled input spectra. If (Bi..) == (Bo..), no reduction
    ///     is computed. In this case, using equiphase_rescale_1d instead is best.
    ///     Should be contiguous vector(s). If real, and the input is complex, the power spectrum is computed.
    /// \param output_fftfreq:
    ///     Frequency range of the output spectra.
    /// \param[in] output_ctf:
    ///     Target isotropic CTFs. Inputs are rescaled to match the phases of these CTFs.
    ///     A contiguous (Bo..) array, or a single value, in which case the CTF is assigned to every output batch.
    /// \param[out] weights:
    ///     Averaging weights.
    ///     Same as rotational_average.
    /// \param options:
    ///     Spectrum and averaging options.
    template<typename Input, typename Output, typename InputCtf, typename OutputCtf, typename Weight = Empty>
        requires details::equiphase_rescale_and_averageable<Input, InputCtf, Output, OutputCtf, Weight>
    void equiphase_rescale_and_average_1d(
        Input&& input,
        const Linspace<f64>& input_fftfreq,
        InputCtf&& input_ctf,
        Output&& output,
        const Linspace<f64>& output_fftfreq,
        OutputCtf&& output_ctf,
        Weight&& weights = {},
        const FuseSpectraOptions& options = {}
    ) {
        details::check_parameters_equiphase_rescale<true>(
            input, input_fftfreq, input_ctf, output, output_fftfreq, output_ctf, weights
        );
        details::launch_equiphase_rescale_and_average<isize>(
            NOA_FWD(input), input_fftfreq, NOA_FWD(input_ctf),
            NOA_FWD(output), output_fftfreq, NOA_FWD(output_ctf),
            NOA_FWD(weights), options
        );
    }

    struct PhaseSpectraOptions {
        /// Interpolation used for the scaling.
        Interp interp{Interp::LINEAR};
    };

    /// Scale 1d rfft spectra so that their CTF phases match with the target CTF.
    /// \param[in] input:
    ///     ((Bi..,)Wi) Input 1d spectra. Can be real or complex.
    ///     Batch axes are broadcast to the output batches.
    /// \param input_fftfreq:
    ///     Frequency range of the input spectra.
    /// \param[in] input_ctf:
    ///     Isotropic CTF(s) of the 1d input spectra.
    ///     A (Bi..) array, or a single value, in which case the CTF is assigned to every input batch.
    ///     The CTF scale is used to assign a weight to each input spectrum.
    /// \param[out] output:
    ///     ((Bo..,)Wo) Output spectra as contiguous vectors.
    ///     If real, and the input is complex, the power spectrum is computed.
    /// \param output_fftfreq:
    ///     Frequency range of the output spectra.
    /// \param[in] output_ctf:
    ///     Target isotropic CTFs. Inputs are rescaled to match the phases of these CTFs.
    ///     A contiguous (Bo..) array, or a single value, in which case the CTF is assigned to every output batch.
    /// \param options:
    ///     Spectrum options.
    ///     options.average is unused.
    template<typename Input, typename Output, typename InputCtf, typename OutputCtf>
        requires details::equiphase_rescale_and_averageable<Input, InputCtf, Output, OutputCtf, Empty>
    void equiphase_rescale_1d(
        Input&& input,
        const Linspace<f64>& input_fftfreq,
        InputCtf&& input_ctf,
        Output&& output,
        const Linspace<f64>& output_fftfreq,
        OutputCtf&& output_ctf,
        const PhaseSpectraOptions& options = {}
    ) {
        details::check_parameters_equiphase_rescale<false>(
            input, input_fftfreq, input_ctf, output, output_fftfreq, output_ctf
        );
        details::launch_equiphase_rescale<isize>(
            NOA_FWD(input), input_fftfreq, NOA_FWD(input_ctf),
            NOA_FWD(output), output_fftfreq, NOA_FWD(output_ctf),
            options
        );
    }
}
