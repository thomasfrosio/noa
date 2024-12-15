#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/Iwise.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/utils/Atomic.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Factory.hpp"
#include "noa/unified/geometry/PolarTransform.hpp"

namespace noa::geometry::guts {
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

    public:
        constexpr RotationalAverage(
            const input_type& input,
            const shape_nd_type& input_shape,
            const batched_ctf_type& input_ctf,
            const output_type& output,
            const weight_type& weight,
            index_type n_shells,
            coord2_type frequency_range,
            bool frequency_range_endpoint
        ) :
            m_input(input),
            m_output(output),
            m_weight(weight),
            m_ctf(input_ctf),
            m_shape(input_shape.template pop_back<IS_RFFT>()),
            m_max_shell_index(n_shells - 1),
            m_fftfreq_step(coord_type{1} / coord_nd_type::from_vec(input_shape.vec))
        {
            // Transform to inclusive range so that we only have to deal with one case.
            if (not frequency_range_endpoint) {
                auto step = Linspace{frequency_range[0], frequency_range[1], false}.for_size(n_shells).step;
                frequency_range[1] -= step;
            }
            if constexpr (nt::empty<ctf_type>)
                m_frequency_range_sqd = frequency_range * frequency_range;
            else
                m_frequency_range_sqd = frequency_range;
            m_frequency_range_start = frequency_range[0];
            m_frequency_range_span = frequency_range[1] - frequency_range[0];
        }

        template<nt::same_as<index_type>... I> requires (N == sizeof...(I))
        NOA_HD void operator()(index_type batch, I... indices) const noexcept {
            const auto frequency = noa::fft::index2frequency<IS_CENTERED, IS_RFFT>(Vec{indices...}, m_shape);
            const auto fftfreq_nd = coord_nd_type::from_vec(frequency) * m_fftfreq_step;

            coord_type fftfreq;
            if constexpr (nt::empty<ctf_type>) {
                fftfreq = dot(fftfreq_nd, fftfreq_nd);
            } else {
                // Correct for anisotropic pixel size and defocus.
                fftfreq = static_cast<coord_type>(m_ctf[batch].isotropic_fftfreq(fftfreq_nd));
            }

            if (fftfreq < m_frequency_range_sqd[0] or
                fftfreq > m_frequency_range_sqd[1])
                return;

            if constexpr (nt::empty<ctf_type>)
                fftfreq = sqrt(fftfreq);

            // Scale the normalized frequency back to the corresponding output shell.
            const coord_type scaled_fftfreq = (fftfreq - m_frequency_range_start) / m_frequency_range_span;
            const coord_type radius = scaled_fftfreq * static_cast<coord_type>(m_max_shell_index);
            const coord_type radius_floor = floor(radius);

            // Since by this point fftfreq has to be within the output frequency range,
            // "radius" is guaranteed to be within [0, m_max_shell_index].
            NOA_ASSERT(radius >= 0 and radius <= static_cast<coord_type>(m_max_shell_index));

            // Compute lerp weights.
            const index_type shell_low = static_cast<index_type>(radius_floor);
            const index_type shell_high = min(m_max_shell_index, shell_low + 1); // shell_low can be the last index
            const coord_type fraction_high = radius - radius_floor;
            const coord_type fraction_low = 1 - fraction_high;

            // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers?
            const auto value = cast_or_abs_squared<output_value_type>(m_input(batch, indices...));
            ng::atomic_add(m_output, value * static_cast<output_real_type>(fraction_low), batch, shell_low);
            ng::atomic_add(m_output, value * static_cast<output_real_type>(fraction_high), batch, shell_high);
            if (m_weight) {
                ng::atomic_add(m_weight, static_cast<weight_value_type>(fraction_low), batch, shell_low);
                ng::atomic_add(m_weight, static_cast<weight_value_type>(fraction_high), batch, shell_high);
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        weight_type m_weight;
        NOA_NO_UNIQUE_ADDRESS batched_ctf_type m_ctf;

        shape_type m_shape;
        index_type m_max_shell_index;
        coord_nd_type m_fftfreq_step;
        coord2_type m_frequency_range_sqd;
        coord_type m_frequency_range_start;
        coord_type m_frequency_range_span;
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

    template<Remap REMAP, bool IS_GPU = false,
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

        const auto fftfreq_range = options.fftfreq_range.template as<coord_t>();
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
                fftfreq_range, options.fftfreq_endpoint);

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
              fftfreq_range, options.fftfreq_endpoint);

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

    template<typename Ctf>
    concept rotational_average_ctf =
        nt::ctf_anisotropic<std::decay_t<Ctf>> or
        (nt::varray_decay<Ctf> and nt::ctf_anisotropic<nt::value_type_t<Ctf>>);

    inline void set_rotational_average_defaults(
        const Shape4<i64>& shape,
        Vec2<f64>& fftfreq_range
    ) {
        // Find highest fftfreq. If any dimension is even sized, this is 0.5.
        if (fftfreq_range[1] <= 0)
            fftfreq_range[1] = noa::max(noa::fft::highest_fftfreq<f64>(shape.pop_front()));
    }
}

// TODO Add rotation_average() for 2d only with frequency and angle range.
//      This should be able to take multiple angle ranges for the same input,
//      to "extract" multiple wedges efficiently.

namespace noa::geometry {
    struct RotationalAverageOptions {
        /// Output [start, end] fftfreq range. The output shells span over this range.
        /// A negative or zero end-frequency defaults the highest fftfreq along the cartesian axes.
        Vec2<f64> fftfreq_range{0, -1};

        /// Whether frequency_range's endpoint should be included in the range.
        bool fftfreq_endpoint{true};

        /// Whether the rotational average should be computed instead of the rotational sum.
        bool average{true};

        /// Whether the outputs (including the optional weights) are initialized.
        /// If so, the function can skip the extra zeroing.
        bool add_to_output{false};
    };

    /// Computes the rotational sum/average of a 2d or 3d DFT.
    /// \tparam REMAP       Should be either H2H, HC2H, F2H or FC2H. The output layout is "H" for no particularly good
    ///                     reasons other than the fact that the number of output shells is often (but not limited to)
    ///                     the half-dimension size, i.e. min(shape) // 2 + 1.
    /// \param[in] input    Input spectrum to reduce. Can be real or complex.
    /// \param input_shape  BDHW logical shape of input.
    /// \param[out] output  Rotational sum/average. Should be a (batch of) contiguous vector(s).
    ///                     If real and input is complex, the power spectrum is computed.
    /// \param[out] weights Rotational weights. Can be empty, or be a (batch of) contiguous vector(s) with the same
    ///                     shape as the output. If valid, the output weights are also saved in this array.
    /// \param options      Rotational average options.
    /// \note If weights is empty and options.average is true, a temporary vector like \p output is allocated.
    template<Remap REMAP,
             nt::readable_varray_decay Input,
             nt::writable_varray_decay Output,
             nt::writable_varray_decay_of_any<nt::value_type_twice_t<Output>> Weight =
                 View<nt::value_type_twice_t<Output>>>
    requires (REMAP.is_xx2h() and nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    void rotational_average(
        Input&& input,
        const Shape4<i64>& input_shape,
        Output&& output,
        Weight&& weights = {},
        RotationalAverageOptions options = {}
    ) {
        const auto n_shells = guts::check_parameters_rotational_average<REMAP>(
            input, input_shape, Empty{}, output, weights);
        guts::set_rotational_average_defaults(input_shape, options.fftfreq_range);

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
             guts::rotational_average_ctf Ctf,
             nt::writable_varray_decay_of_any<nt::value_type_twice_t<Output>> Weight =
                 View<nt::value_type_twice_t<Output>>>
    requires (REMAP.is_xx2h() and nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    void rotational_average_anisotropic(
        Input&& input,
        const Shape4<i64>& input_shape,
        Ctf&& input_ctf,
        Output&& output,
        Weight&& weights = {},
        RotationalAverageOptions options = {}
    ) {
        const auto n_shells = guts::check_parameters_rotational_average<REMAP>(
            input, input_shape, input_ctf, output, weights);
        guts::set_rotational_average_defaults(input_shape, options.fftfreq_range);

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
}
