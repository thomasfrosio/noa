#pragma once

#include "noa/runtime/core/Iwise.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/core/Atomic.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Ewise.hpp"
#include "noa/runtime/Iwise.hpp"
#include "noa/runtime/Factory.hpp"

#include "noa/fft/core/Frequency.hpp"
#include "noa/signal/core/CTF.hpp" // FIXME can be removed
#include "noa/xform/core/Interpolation.hpp"

namespace noa::xform::details {
    struct RotationalAverageUtils {
        template<typename T, typename U, typename C, typename I>
        NOA_FHD static void lerp_to_output(const T& op, const U& value, C fftfreq, I batch) noexcept {
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
                nd::atomic_add(op.m_output, value * static_cast<T::output_real_type>(fraction_low), batch, shell_low);
                if (op.m_weight)
                    nd::atomic_add(op.m_weight, static_cast<T::weight_value_type>(fraction_low), batch, shell_low);
            }

            if (shell_high >= 0 and shell_high <= op.m_max_shell_index) {
                nd::atomic_add(op.m_output, value * static_cast<T::output_real_type>(fraction_high), batch, shell_high);
                if (op.m_weight)
                    nd::atomic_add(op.m_weight, static_cast<T::weight_value_type>(fraction_high), batch, shell_high);
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
    template<nf::Layout REMAP,
             usize N,
             nt::real Coord,
             nt::sinteger Index,
             nt::readable_nd<N + 1> Input,
             nt::atomic_addable_nd<2> Output,
             nt::atomic_addable_nd_optional<2> Weight,
             nt::batch Ctf>
    class RotationalAverage {
    public:
        static_assert((N == 2 or N == 3) and REMAP.is_xx2h());
        static constexpr bool IS_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_RFFT = REMAP.is_hx2xx();

        using index_type = Index;
        using coord_type = Coord;
        using shape_type = Shape<index_type, N - IS_RFFT>;
        using coord_nd_type = Vec<coord_type, N>;
        using coord2_type = Vec<coord_type, 2>;
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
            coord_type max_input_fftfreq{-1};
            for (usize i{}; i < N; ++i) {
                const auto max_sample_size = input_shape[i] / 2 + 1;
                const auto fftfreq_end =
                    input_fftfreq.stop <= 0 ?
                    nf::highest_fftfreq<coord_type>(input_shape[i]) :
                    input_fftfreq.stop;
                max_input_fftfreq = max(max_input_fftfreq, fftfreq_end);
                m_input_fftfreq_step[i] = Linspace<coord_type>{
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
            m_fftfreq_cutoff[1] = output_fftfreq.stop + static_cast<coord_type>(m_max_shell_index) * output_fftfreq_step;
        }

        // 2d or 3d rotational average, with an optional anisotropic field correction.
        template<nt::same_as<index_type>... I> requires (N == sizeof...(I))
        NOA_HD void operator()(index_type batch, I... indices) const noexcept {
            // Input indices to fftfreq.
            const auto frequency = nf::index2frequency<IS_CENTERED, IS_RFFT>(Vec{indices...}, m_shape);
            const auto fftfreq_nd = coord_nd_type::from_vec(frequency) * m_input_fftfreq_step;

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
        coord2_type m_fftfreq_cutoff;
        coord_type m_output_fftfreq_start;
        coord_type m_output_fftfreq_span;
        index_type m_max_shell_index;
    };

    template<nt::real Coord,
             nt::sinteger Index,
             nt::readable_nd<2> Input,
             nt::atomic_addable_nd<2> Output,
             nt::atomic_addable_nd_optional<2> Weight,
             nt::batch InputCtf,
             nt::batch OutputCtf>
    class FuseSpectra {
    public:
        using index_type = Index;
        using coord_type = Coord;
        using coord_nd_type = Vec<coord_type, 1>;
        using coord2_type = Vec<coord_type, 2>;

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
        using batched_output_ctf_type = OutputCtf;
        using output_ctf_type = nt::mutable_value_type_t<batched_output_ctf_type>;
        static_assert(nt::ctf_isotropic<input_ctf_type, output_ctf_type>);

        friend RotationalAverageUtils;

    public:
        constexpr FuseSpectra(
            const input_type& input,
            const Linspace<coord_type>& input_fftfreq,
            const batched_input_ctf_type& input_ctf,
            index_type n_input_shells,
            const output_type& output,
            Linspace<coord_type> output_fftfreq,
            const batched_output_ctf_type& output_ctf,
            index_type n_output_shells,
            const weight_type& weight,
            index_type chunk_size
        ) :
            m_input(input),
            m_output(output),
            m_weight(weight),
            m_input_ctf(input_ctf),
            m_output_ctf(output_ctf),
            m_max_shell_index(n_output_shells - 1),
            m_chunk_size(chunk_size)
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
            m_fftfreq_cutoff[1] = output_fftfreq.stop + static_cast<coord_type>(m_max_shell_index) * output_fftfreq_step;
        }

        NOA_HD void operator()(index_type batch, index_type index) const noexcept {
            const auto output_batch = batch / m_chunk_size;
            const auto input_fftfreq = m_input_fftfreq_start + static_cast<coord_type>(index) * m_input_fftfreq_step;
            const auto input_phase = m_input_ctf[batch].phase_at(input_fftfreq);
            const auto fftfreq = static_cast<coord_type>(m_output_ctf[output_batch].fftfreq_at(input_phase));

            // Remove most out-of-bounds asap.
            if (fftfreq < m_fftfreq_cutoff[0] or fftfreq > m_fftfreq_cutoff[1])
                return;

            const auto value = cast_or_abs_squared<output_value_type>(m_input(batch, index));
            const auto weight = static_cast<output_real_type>(m_input_ctf[batch].scale());
            RotationalAverageUtils::lerp_to_output(*this, value * weight, fftfreq, output_batch);
        }

    private:
        input_type m_input;
        output_type m_output;
        weight_type m_weight;
        batched_input_ctf_type m_input_ctf;
        batched_output_ctf_type m_output_ctf;

        coord2_type m_fftfreq_cutoff;
        coord_type m_input_fftfreq_start;
        coord_type m_input_fftfreq_step;
        coord_type m_output_fftfreq_start;
        coord_type m_output_fftfreq_span;
        index_type m_max_shell_index;
        index_type m_chunk_size;
    };

    template<nt::real Coord,
             nt::sinteger Index,
             nt::interpolator_spectrum_nd<1> Input,
             nt::writable_nd<2> Output,
             nt::batch InputCtf,
             nt::batch OutputCtf>
    class PhaseSpectra {
    public:
        using index_type = Index;
        using coord_type = Coord;

        using input_type = Input;
        using output_type = Output;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::spectrum_types<nt::value_type_t<input_type>, output_value_type>);

        using batched_input_ctf_type = InputCtf;
        using batched_output_ctf_type = OutputCtf;
        using input_ctf_type = nt::mutable_value_type_t<batched_input_ctf_type>;
        using output_ctf_type = nt::mutable_value_type_t<batched_output_ctf_type>;
        static_assert(nt::ctf_isotropic<input_ctf_type, output_ctf_type>);

    public:
        constexpr PhaseSpectra(
            const input_type& input,
            const Linspace<coord_type>& input_fftfreq,
            const batched_input_ctf_type& input_ctf,
            index_type n_input_shells,
            const output_type& output,
            Linspace<coord_type> output_fftfreq,
            const batched_output_ctf_type& output_ctf,
            index_type n_output_shells
        ) :
            m_input{input},
            m_output{output},
            m_input_ctf{input_ctf},
            m_output_ctf{output_ctf},
            m_input_fftfreq_start{input_fftfreq.start},
            m_input_fftfreq_step{input_fftfreq.for_size(n_input_shells).step},
            m_output_fftfreq_start{output_fftfreq.start},
            m_output_fftfreq_step{output_fftfreq.for_size(n_output_shells).step}
        {}

        NOA_HD void operator()(index_type batch, index_type index) const noexcept {
            const auto output_fftfreq = m_output_fftfreq_start + static_cast<coord_type>(index) * m_output_fftfreq_step;
            const auto phase = m_output_ctf[batch].phase_at(output_fftfreq);
            const auto input_fftfreq = static_cast<coord_type>(m_input_ctf[batch].fftfreq_at(phase));

            const auto input_frequency = (input_fftfreq - m_input_fftfreq_start) / m_input_fftfreq_step;
            const auto interpolated_value = m_input.interpolate_spectrum_at(Vec{input_frequency}, batch);
            m_output(batch, index) = cast_or_abs_squared<output_value_type>(interpolated_value);
        }

    private:
        input_type m_input;
        output_type m_output;
        batched_input_ctf_type m_input_ctf;
        batched_output_ctf_type m_output_ctf;

        coord_type m_input_fftfreq_start;
        coord_type m_input_fftfreq_step;
        coord_type m_output_fftfreq_start;
        coord_type m_output_fftfreq_step;
    };

    template<typename T>
    auto check_parameters_ctf(
        const T& ctf, isize batch, Device device,
        const std::source_location& location = std::source_location::current()
    ) {
        if constexpr (nt::varray<T>) {
            check_at_location(
                location, is_contiguous_vector(ctf) and ctf.n_elements() == batch,
                "The CTFs, specified as a contiguous vector, should have the same size "
                "as the corresponding array batch size. Got ctf:strides={}, ctf:shape={}, batch={}",
                ctf.strides(), ctf.shape(), batch
            );
            check(ctf.device() == device,
                  "The input and output arrays must be on the same device, "
                  "but got ctf:device={} and output:device={}",
                  ctf.device(), device);
        }
    }

    template<nf::Layout REMAP, typename Input, typename Output, typename Weight, typename Ctf = Empty>
    auto check_parameters_rotational_average(
        const Input& input,
        const Shape4& shape,
        const Ctf& input_ctf,
        const Output& output,
        const Weight& weights,
        const Linspace<f64>& input_fftfreq
    ) -> isize {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        const bool weights_is_empty = weights.is_empty();

        check(input.shape() == (REMAP.is_hx2xx() ? shape.rfft() : shape),
              "The input array does not match the logical shape. Got input:shape={}, shape={}, remap={}",
              input.shape(), shape, REMAP);

        check(shape[0] == output.shape()[0] and
              (weights_is_empty or shape[0] == weights.shape()[0]),
              "The numbers of batches between arrays do not match. Got batch={}, output:batch={}{}",
              shape[0], output.shape()[0],
              weights_is_empty ? "" : fmt::format(" and weights:batch={}", weights.shape()[0]));

        check(is_contiguous_vector_batched_strided(output),
              "The output must be a (batch of) contiguous vector(s), but got output:shape={} and output:strides={}",
              output.shape(), output.strides());

        const isize n_shells = output.shape().pop_front().n_elements();
        if (not weights_is_empty) {
            check(is_contiguous_vector_batched_strided(weights),
                  "The weights must be a (batch of) contiguous vector(s), "
                  "but got weights:shape={} and weights:strides={}",
                  weights.shape(), weights.strides());

            const isize weights_n_shells = weights.shape().pop_front().n_elements();
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
        check_parameters_ctf(input_ctf, shape[0], output.device());

        check(allclose(input_fftfreq.start, 0.), "The starting fftfreq should be 0, but got {}", input_fftfreq.start);

        return n_shells;
    }

    template<bool REDUCE, typename Input, typename Output, typename Weight = Empty, typename InputCtf, typename OutputCtf>
    void check_parameters_fuse_spectra(
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
        const auto [ib, id, ih, iw] = input.shape();
        const auto [ob, od, oh, ow] = output.shape();
        check(is_contiguous_vector_batched_strided(input) and id == 1 and ih == 1,
              "The input must be a (batch of) contiguous row vector(s), but got input:shape={} and input:strides={}",
              input.shape(), input.strides());
        check(is_contiguous_vector_batched_strided(output) and od == 1 and oh == 1,
              "The output must be a (batch of) contiguous row vector(s), but got output:shape={} and output:strides={}",
              output.shape(), output.strides());
        if constexpr (REDUCE) {
            check(is_multiple_of(ib, ob), "Invalid reduction. input:batch={}, output:batch={}", ib, ob);
        } else {
            check(ib == 1 or ib == ob,
                  "Cannot broadcast an array with input:batch={} into an array with output:batch={}", ib, ob);
        }

        check(input.device() == output.device(),
              "The arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), output.device());

        if constexpr (not nt::empty<Weight>) {
            if (not weights.is_empty()) {
                const auto [wb, wd, wh, ww] = weights.shape();
                check(is_contiguous_vector_batched_strided(weights) and wd == 1 and wh == 1,
                      "The weights must be a contiguous row vector, but got weights:shape={} and weights:strides={}",
                      weights.shape(), weights.strides());
                check(ob == wb and ow == ww,
                      "The output and weights should have the same shape, but got output:shape={} and weights:shape={}",
                      output.shape(), weights.shape());
                check(input.device() == weights.device(),
                      "The arrays must be on the same device, but got input:device={}, weights:device={}",
                      input.device(), weights.device());
            }
        }

        check_parameters_ctf(input_ctf, ib, output.device());
        check_parameters_ctf(output_ctf, ob, output.device());
    }

    template<nf::Layout REMAP,
             typename Input, typename Index, typename Ctf,
             typename Output, typename Weight, typename Options>
    void launch_rotational_average(
        Input&& input, const Shape<Index, 4>& input_shape, Ctf&& input_ctf,
        Output&& output, Weight&& weight, isize n_shells, const Options& options
    ) {
        using input_value_t = nt::const_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using weight_value_t = nt::value_type_t<Weight>;
        using coord_t = nt::largest_type_t<f32, nt::value_type_t<output_value_t>>;

        // Output must be zeroed out.
        const auto output_view = output.view();
        if (not options.add_to_output)
            ewise({}, output_view, Zero{});

        // When computing the average, the weights must be valid.
        auto weight_view = weight.view();
        Array<weight_value_t> weight_buffer;
        if (options.average) {
            if (weight_view.is_empty()) {
                weight_buffer = zeros<weight_value_t>(output_view.shape(), ArrayOption{output.device(), Allocator::DEFAULT_ASYNC});
                weight_view = weight_buffer.view();
            } else if (not options.add_to_output) {
                ewise({}, weight_view, Zero{});
            }
        }

        using output_accessor_t = AccessorRestrictContiguous<output_value_t, 2, Index>;
        using weight_accessor_t = AccessorRestrictContiguous<weight_value_t, 2, Index>;
        auto output_accessor = output_accessor_t(output_view.get(), Strides<Index, 1>::from_value(output_view.strides()[0]));
        auto weight_accessor = weight_accessor_t(weight_view.get(), Strides<Index, 1>::from_value(weight_view.strides()[0]));

        const auto input_fftfreq = options.input_fftfreq.template as<coord_t>();
        const auto output_fftfreq = options.output_fftfreq.template as<coord_t>();
        const auto iwise_shape = input.shape().template as<Index>();
        const auto input_strides = input.strides().template as<Index>();

        if (input_shape.ndim() == 2) {
            auto ctf = nd::to_batch<true>(input_ctf);

            using input_accessor_t = AccessorRestrict<input_value_t, 3, Index>;
            auto op = RotationalAverage
                <REMAP, 2, coord_t, Index, input_accessor_t, output_accessor_t, weight_accessor_t, decltype(ctf)>(
                    input_accessor_t(input.get(), input_strides.filter(0, 2, 3)), input_shape.filter(2, 3),
                    ctf, output_accessor, weight_accessor, static_cast<Index>(n_shells),
                    input_fftfreq, output_fftfreq
                );
            iwise(
                iwise_shape.filter(0, 2, 3), output.device(), op,
                std::forward<Input>(input), output, weight, std::forward<Ctf>(input_ctf)
            );
        } else {
            using input_accessor_t = AccessorRestrict<input_value_t, 4, Index>;
            auto op = RotationalAverage
                <REMAP, 3, coord_t, Index, input_accessor_t, output_accessor_t, weight_accessor_t, nd::Batch<Empty>>(
                    input_accessor_t(input.get(), input_strides), input_shape.filter(1, 2, 3), {},
                    output_accessor, weight_accessor, static_cast<Index>(n_shells), input_fftfreq, output_fftfreq
                );
            iwise(iwise_shape, output.device(), op, std::forward<Input>(input), output, weight);
        }

        // Some shells can be 0, so use DivideSafe.
        if (options.average) {
            if (weight_buffer.is_empty()) {
                ewise(wrap(output_view, weight), std::forward<Output>(output), DivideSafe{});
            } else {
                ewise(wrap(output_view, std::move(weight_buffer)), std::forward<Output>(output), DivideSafe{});
            }
        }
    }

    template<typename Index,
             typename Input, typename Output, typename Weight,
             typename InputCtf, typename OutputCtf, typename Options>
    void launch_fuse_spectra(
        Input&& input, const Linspace<f64>& input_fftfreq, InputCtf&& input_ctf,
        Output&& output, const Linspace<f64>& output_fftfreq, OutputCtf&& output_ctf,
        Weight&& weight, const Options& options
    ) {
        using input_value_t = nt::const_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using weight_value_t = nt::value_type_t<Weight>;
        using coord_t = nt::largest_type_t<f32, nt::value_type_t<output_value_t>>;

        // Output must be zeroed out.
        const auto output_view = output.view();
        if (not options.add_to_output)
            ewise({}, output_view, Zero{});

        // When computing the average, the weights must be valid.
        auto weight_view = weight.view();
        Array<weight_value_t> weight_buffer;
        if (options.average) {
            if (weight_view.is_empty()) {
                weight_buffer = zeros<weight_value_t>(output_view.shape(), ArrayOption{output.device(), Allocator::DEFAULT_ASYNC});
                weight_view = weight_buffer.view();
            } else if (not options.add_to_output) {
                ewise({}, weight_view, Zero{});
            }
        }

        using input_accessor_t = AccessorRestrictContiguous<input_value_t, 2, Index>;
        using output_accessor_t = AccessorRestrictContiguous<output_value_t, 2, Index>;
        using weight_accessor_t = AccessorRestrictContiguous<weight_value_t, 2, Index>;
        auto input_accessor = input_accessor_t(input.get(), input.strides().filter(0).template as<Index>());
        auto output_accessor = output_accessor_t(output_view.get(), output_view.strides().filter(0).template as<Index>());
        auto weight_accessor = weight_accessor_t(weight_view.get(), weight_view.strides().filter(0).template as<Index>());

        const auto input_fftfreq_f = input_fftfreq.as<coord_t>();
        const auto output_fftfreq_f = output_fftfreq.as<coord_t>();
        const auto n_input_shells = static_cast<Index>(input.shape()[3]);
        const auto n_output_shells = static_cast<Index>(output.shape()[3]);
        const auto iwise_shape = Shape{static_cast<Index>(input.shape()[0]), n_input_shells};
        const auto chunk_size = static_cast<Index>(input.shape()[0]) / static_cast<Index>(output.shape()[0]);

        auto batched_input_ctf = nd::to_batch(input_ctf);
        auto batched_output_ctf = nd::to_batch(output_ctf);

        using op_t = FuseSpectra<
            coord_t, Index, input_accessor_t, output_accessor_t,
            weight_accessor_t, decltype(batched_input_ctf), decltype(batched_output_ctf)>;
        auto op = op_t(
            input_accessor, input_fftfreq_f, batched_input_ctf, n_input_shells,
            output_accessor, output_fftfreq_f, batched_output_ctf, n_output_shells, weight_accessor, chunk_size
        );
        iwise(
            iwise_shape, output.device(), op,
            std::forward<Input>(input), output, weight,
            std::forward<InputCtf>(input_ctf),
            std::forward<OutputCtf>(output_ctf)
        );

        // Some shells can be 0, so use DivideSafe.
        if (options.average) {
            if (weight_buffer.is_empty()) {
                ewise(wrap(output_view, weight), std::forward<Output>(output), DivideSafe{});
            } else {
                ewise(wrap(output_view, std::move(weight_buffer)), std::forward<Output>(output), DivideSafe{});
            }
        }
    }

    template<typename Index,
             typename Input, typename Output,
             typename InputCtf, typename OutputCtf, typename Options>
    void launch_phase_spectra(
        Input&& input, const Linspace<f64>& input_fftfreq, InputCtf&& input_ctf,
        Output&& output, const Linspace<f64>& output_fftfreq, OutputCtf&& output_ctf,
        const Options& options
    ) {
        using output_value_t = nt::value_type_t<Output>;
        using coord_t = nt::largest_type_t<f32, nt::value_type_t<output_value_t>>;

        const auto input_fftfreq_f = input_fftfreq.as<coord_t>();
        const auto output_fftfreq_f = output_fftfreq.as<coord_t>();
        const auto n_input_shells = static_cast<Index>(input.shape()[3]);
        const auto n_output_shells = static_cast<Index>(output.shape()[3]);
        const auto logical_shape = Shape<Index, 4>{1, 1, 1, (n_input_shells - 1) * 2};
        const auto iwise_shape = Shape{static_cast<Index>(output.shape()[0]), n_output_shells};

        using output_accessor_t = AccessorRestrictContiguous<output_value_t, 2, Index>;
        auto output_accessor = output_accessor_t(output.get(), output.strides().filter(0).template as<Index>());

        auto batched_input_ctf = nd::to_batch(input_ctf);
        auto batched_output_ctf = nd::to_batch(output_ctf);

        auto launch_iwise = [&](auto interp) {
            auto interpolator = to_interpolator_spectrum<1, "h2h", interp(), coord_t, false>(input, logical_shape);
            using op_t = PhaseSpectra<
                coord_t, Index, decltype(interpolator), output_accessor_t,
                decltype(batched_input_ctf), decltype(batched_output_ctf)>;
            auto op = op_t(
                interpolator, input_fftfreq_f, batched_input_ctf, n_input_shells,
                output_accessor, output_fftfreq_f, batched_output_ctf, n_output_shells
            );
            iwise(
                iwise_shape, output.device(), op,
                std::forward<Input>(input), output,
                std::forward<InputCtf>(input_ctf),
                std::forward<OutputCtf>(output_ctf)
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
        nf::Layout REMAP,
        nt::readable_varray_decay Input,
        nt::writable_varray_decay Output,
        nt::writable_varray_decay_of_any<nt::value_type_twice_t<Output>> Weight = View<nt::value_type_twice_t<Output>>>
    requires (REMAP.is_xx2h() and nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    NOA_NOINLINE void rotational_average(
        Input&& input,
        const Shape4& input_shape,
        Output&& output,
        Weight&& weights = {},
        RotationalAverageOptions options = {}
    ) {
        const auto n_shells = details::check_parameters_rotational_average<REMAP>(
            input, input_shape, Empty{}, output, weights, options.input_fftfreq);
        details::launch_rotational_average<REMAP>(
            std::forward<Input>(input), input_shape.as<isize>(), Empty{},
            std::forward<Output>(output),
            std::forward<Weight>(weights),
            n_shells, options
        );
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
    template<
        nf::Layout REMAP,
        nt::readable_varray_decay Input,
        nt::writable_varray_decay Output,
        details::rotational_average_anisotropic_ctf Ctf,
        nt::writable_varray_decay_of_any<nt::value_type_twice_t<Output>> Weight =
        View<nt::value_type_twice_t<Output>>>
    requires (REMAP.is_xx2h() and nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    NOA_NOINLINE void rotational_average_anisotropic(
        Input&& input,
        const Shape4& input_shape,
        Ctf&& input_ctf,
        Output&& output,
        Weight&& weights = {},
        RotationalAverageOptions options = {}
    ) {
        const auto n_shells = details::check_parameters_rotational_average<REMAP>(
            input, input_shape, input_ctf, output, weights, options.input_fftfreq);
        details::launch_rotational_average<REMAP>(
            std::forward<Input>(input), input_shape.as<isize>(),
            std::forward<Ctf>(input_ctf),
            std::forward<Output>(output),
            std::forward<Weight>(weights),
            n_shells, options
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
    /// \details The reduction is done in chunks:
    ///          B=C*N -> N: C*N input spectra are given, and they will be reduced to N outputs.
    ///          If N==1, the input spectra are fused into one output spectrum. Otherwise, the input spectra are
    ///          divided into N chunks of size C, and each chunk is fused into one spectrum.
    ///
    /// \param[in] input        Input 1d spectra to reduce. Can be real or complex.
    /// \param input_fftfreq    Frequency range of the input spectra.
    /// \param[in] input_ctf    Isotropic CTFs of the 1d input spectra. One per spectrum.
    ///                         The CTF scale is used to assign a weight to each input spectrum.
    /// \param[out] output      Output spectra. If real, and the input is complex, the power spectrum is computed.
    /// \param output_fftfreq   Frequency range of the output spectra.
    /// \param[in] output_ctf   Target isotropic CTFs. Inputs are rescaled to match the phases of these CTFs.
    /// \param[out] weights     Averaging weights.
    ///                         Can be empty, or be a contiguous vector with the same shape as the output.
    ///                         If valid, the output weights are saved in this array.
    ///                         If empty and options.average is true, a temporary vector is allocated.
    /// \param options          Spectrum and averaging options.
    ///
    /// \note While the C=1 case is supported (no reduction), in this case, one should use phase_spectra instead.
    template<
        nt::readable_varray_decay Input,
        nt::writable_varray_decay Output,
        details::rotational_average_isotropic_ctf InputCtf,
        details::rotational_average_isotropic_ctf OutputCtf,
        nt::writable_varray_decay_of_any<nt::value_type_twice_t<Output>> Weight = View<nt::value_type_twice_t<Output>>>
    requires (nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    NOA_NOINLINE void fuse_spectra(
        Input&& input,
        const Linspace<f64>& input_fftfreq,
        InputCtf&& input_ctf,
        Output&& output,
        const Linspace<f64>& output_fftfreq,
        OutputCtf&& output_ctf,
        Weight&& weights = {},
        FuseSpectraOptions options = {}
    ) {
        details::check_parameters_fuse_spectra<true>(
            input, input_fftfreq, input_ctf, output, output_fftfreq, output_ctf, weights
        );
        details::launch_fuse_spectra<isize>(
            std::forward<Input>(input), input_fftfreq, std::forward<InputCtf>(input_ctf),
            std::forward<Output>(output), output_fftfreq, std::forward<OutputCtf>(output_ctf),
            std::forward<Weight>(weights), options
        );
    }

    struct PhaseSpectraOptions {
        /// Interpolation used for the scaling.
        Interp interp{Interp::LINEAR};
    };

    /// Scale 1d rfft spectra so that their CTF phases match with the target CTF.
    /// \param[in] input        Input 1d spectra to scale. Can be real or complex.
    /// \param input_fftfreq    Frequency range of the input spectra.
    /// \param[in] input_ctf    Isotropic CTFs of the 1d input spectra. One per spectrum.
    /// \param[out] output      Output spectra. If real, and the input is complex, the power spectrum is computed.
    /// \param output_fftfreq   Frequency range of the output spectra.
    /// \param[in] output_ctf   Target isotropic CTFs. Inputs are rescaled to match the phases of these CTFs.
    /// \param options          Spectrum options.
    template<
        nt::readable_varray_decay Input,
        nt::writable_varray_decay Output,
        details::rotational_average_isotropic_ctf InputCtf,
        details::rotational_average_isotropic_ctf OutputCtf>
    requires (nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    NOA_NOINLINE void phase_spectra(
        Input&& input,
        const Linspace<f64>& input_fftfreq,
        InputCtf&& input_ctf,
        Output&& output,
        const Linspace<f64>& output_fftfreq,
        OutputCtf&& output_ctf,
        const PhaseSpectraOptions& options = {}
    ) {
        details::check_parameters_fuse_spectra<false>(
            input, input_fftfreq, input_ctf, output, output_fftfreq, output_ctf
        );
        details::launch_phase_spectra<isize>(
            std::forward<Input>(input), input_fftfreq, std::forward<InputCtf>(input_ctf),
            std::forward<Output>(output), output_fftfreq, std::forward<OutputCtf>(output_ctf),
            options
        );
    }
}
