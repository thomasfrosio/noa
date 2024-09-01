#pragma once

#include "noa/core/fft/Remap.hpp"
#include "noa/core/geometry/RotationalAverage.hpp"

#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Factory.hpp"
#include "noa/unified/geometry/PolarTransform.hpp"

namespace noa::geometry::guts {
    template<noa::fft::RemapInterface REMAP, typename Input, typename Ctf, typename Output, typename Weight>
    constexpr bool is_valid_rotational_average_v =
            REMAP.is_xx2h() and
            (nt::is_ctf_anisotropic_v<Ctf> or std::is_empty_v<Ctf> or
             (nt::is_varray_v<Ctf> and nt::is_ctf_anisotropic_v<nt::value_type_t<Ctf>>) and
             (nt::are_varray_of_real_v<Input, Output> or nt::are_varray_of_complex_v<Input, Output> or
              (nt::is_varray_of_complex_v<Input> and nt::is_varray_of_real_v<Output>)) and
             nt::are_varray_of_mutable_v<Output, Weight> and
             nt::is_varray_of_real_v<Weight>);

    template<noa::fft::RemapInterface REMAP, typename Input, typename Output, typename Weight, typename Ctf = Empty>
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

        const i64 n_shells = output.shape().pop_front().elements();
        if (not weights_is_empty) {
            check(ni::is_contiguous_vector_batched_strided(weights),
                  "The weights must be a (batch of) contiguous vector(s), but got weight:shape={} and weight:strides={}",
                  weights.shape(), weights.strides());

            const i64 weights_n_shells = weights.shape().pop_front().elements();
            check(n_shells == weights_n_shells,
                  "The number of shells does not match the input shape. Got output n_shells={} and weight n_shells={}",
                  n_shells, weights_n_shells);
        }

        check(input.device() == output.device() and
              (weights_is_empty or weights.device() == output.device()),
              "The arrays must be on the same device, but got input:device={}, output:device={}{}",
              input.device(), output.device(),
              weights_is_empty ? "" : fmt::format(" and weights:device={}", weights.device()));

        if constexpr (not std::is_empty_v<Ctf>) {
            check(shape.ndim() == 2,
                  "Only (batched) 2d arrays are supported with anisotropic CTFs, but got shape={}",
                  shape);
        }
        if constexpr (nt::is_varray_v<Ctf>) {
            check(ni::is_contiguous_vector(input_ctf) and input_ctf.ssize() == shape[0],
                  "The anisotropic input ctfs, specified as a contiguous vector, should have the same number of "
                  "elements as the batch size. Got ctf:strides={}, ctf:shape={}, input:batch={}",
                  input_ctf.strides(), input_ctf.shape(), shape[0]);
            check(input_ctf.device() == output.device(),
                  "The input and output arrays must be on the same device, but got ctf:device={} and output:device={}",
                  input_ctf.device(), output.device());
        }

        return n_shells;
    }

    template<noa::fft::RemapInterface REMAP,
             typename Input, typename Index, typename Ctf, typename Output, typename Weight, typename Options>
    void launch_rotational_average(
            const Input& input, const Shape4<Index>& input_shape, const Ctf& input_ctf,
            const Output& output, const Weight& weight, i64 n_shells, const Options& options
    ) {
        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using weight_value_t = nt::value_type_t<Weight>;
        using coord_t = nt::value_type_t<output_value_t>;

        const auto device = output.device();
        const auto fftfreq_range = options.fftfreq_range.template as<coord_t>();
        const auto iwise_shape = input.shape().template as<Index>();

        // Output must be zeroed out.
        const auto output_view = output.view();
        fill(output_view, 0);

        // When computing the average, the weights must be valid.
        auto weight_view = weight.view();
        Array<weight_value_t> weight_buffer;
        if (options.compute_average) {
            if (weight_view.is_empty()) {
                weight_buffer = fill(output_view.shape(), 0, ArrayOption{device, MemoryResource::DEFAULT_ASYNC});
                weight_view = weight_buffer.view();
            } else {
                fill(weight_view, 0);
            }
        }

        // Retrieve the CTF(s) and wrap into an accessor.
        auto ctf = [&] {
            if constexpr (nt::is_varray_v<Ctf>) {
                using ctf_t = nt::mutable_value_type_t<Ctf>;
                using ctf_accessor_t = AccessorRestrictContiguous<const ctf_t, 1, Index>;
                return ctf_accessor_t(input_ctf.get());
            } else if constexpr (std::is_empty_v<Ctf>) {
                return input_ctf;
            } else {
                using ctf_accessor_t = AccessorValue<Ctf, Index>;
                return ctf_accessor_t(input_ctf);
            }
        }();
        using ctf_t = decltype(ctf);

        using output_accessor_t = AccessorRestrictContiguous<output_value_t, 2, Index>;
        using weight_accessor_t = AccessorRestrictContiguous<weight_value_t, 2, Index>;
        const auto input_strides = input.strides().template as<Index>();
        const auto output_strides = Strides2<Index>::from_values(output_view.strides()[0], 1); // contiguous
        const auto weight_strides = Strides2<Index>::from_values(weight_view.strides()[0], 1); // contiguous
        auto output_accessor = output_accessor_t(output_view.get(), output_strides);
        auto weight_accessor = weight_accessor_t(weight_view.get(), weight_strides);

        if (input_shape.ndim() == 2) {
            using input_accessor_t = AccessorRestrict<const input_value_t, 3, Index>;
            auto op = RotationalAverage
                    <REMAP, 2, coord_t, Index, input_accessor_t, output_accessor_t, weight_accessor_t, ctf_t>(
                    input_accessor_t(input.get(), input_strides.filter(0, 2, 3)), input_shape.filter(2, 3),
                    ctf(), output_accessor, weight_accessor, n_shells,
                    fftfreq_range, options.fftfreq_endpoint);

            iwise(iwise_shape.filter(0, 2, 3), device, op,
                    /*attachments=*/ input, output, weight);
        } else {
            using input_accessor_t = AccessorRestrict<const input_value_t, 4, Index>;
            auto op = RotationalAverage
                    <REMAP, 3, coord_t, Index, input_accessor_t, output_accessor_t, weight_accessor_t, ctf_t>(
                    input_accessor_t(input.get(), input_strides), input_shape.filter(1, 2, 3),
                    ctf(), output_accessor, weight_accessor, n_shells,
                    fftfreq_range, options.fftfreq_endpoint);

            iwise(iwise_shape, device, op,
                    /*attachments=*/ input, output, weight);
        }

        // Some shells can be 0, so use DivideSafe.
        if (options.compute_average) {
            if (weight_buffer.is_empty()) {
                ewise(wrap(output, weight), output, DivideSafe{});
            } else {
                ewise(wrap(output, std::move(weight_buffer)), output, DivideSafe{});
            }
        }
    }
}

// TODO Add rotation_average() for 2d only with frequency and angle range.
//      This should be able to take multiple angle ranges for the same input,
//      to "extract" multiple wedges efficiently.

namespace noa::geometry {
    struct RotationalAverageOptions {
        /// Output fftfreq range. The output shells span over this range.
        /// Defaults to the full frequency range, i.e. [0, highest_fftfreq].
        Vec2<f64> fftfreq_range{};

        /// Whether frequency_range's endpoint should be included in the range.
        bool fftfreq_endpoint{true};

        /// Whether the rotational average should be computed instead of the rotational sum.
        bool compute_average{true};
    };

    /// Computes the rotational sum/average of a 2d or 3d dft.
    /// \tparam REMAP       Should be either H2H, HC2H, F2H or FC2H. The output layout is "H" for no particularly good
    ///                     reasons other than the fact that the number of output shells is often (but not limited to)
    ///                     the half dimension size, i.e. min(shape) // 2 + 1.
    /// \param[in] input    Input spectrum to reduce. Can be real or complex.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param[out] output  Rotational sum/average. Should be a (batch of) contiguous vector(s).
    ///                     If real and \p input is complex, the power spectrum computed.
    /// \param[out] weights Rotational weights. Can be empty, or be a (batch of) contiguous vector(s) with the same
    ///                     shape as the output. If valid, the output weights are also saved in this array.
    /// \param options      Rotational average options.
    /// \note If \p weights is empty and \p options.average is true, a temporary vector like \p output is allocated.
    template<noa::fft::RemapInterface REMAP,
            typename Input, typename Output,
            typename Weight = View<nt::value_type_twice_t<Output>>>
    requires guts::is_valid_rotational_average_v<REMAP, Input, Empty, Output, Weight>
    void rotational_average(
            const Input& input,
            const Shape4<i64>& input_shape,
            const Output& output,
            const Weight& weights = {},
            RotationalAverageOptions options = {}
    ) {
        auto n_shells = guts::check_parameters_rotational_average<REMAP>(input, input_shape, Empty{}, output, weights);
        guts::set_frequency_range_to_default(input_shape, options.fftfreq_range);

        if (output.device().is_gpu()) {
            check(ng::is_accessor_access_safe<i32>(input, input.shape()) and
                  ng::is_accessor_access_safe<i32>(output, output.shape()) and
                  ng::is_accessor_access_safe<i32>(weights, weights.shape()),
                  "GPU backend only instantiate i32-based accessor indexing, "
                  "which is unsafe for the given input and output arrays. "
                  "Please report this.");
            guts::launch_rotational_average<REMAP>(
                    input, input_shape.as<i32>(), Empty{}, output, weights, n_shells, options);
        } else {
            guts::launch_rotational_average<REMAP>(
                    input, input_shape.as<i64>(), Empty{}, output, weights, n_shells, options);
        }
    }

    /// Computes the rotational sum/average of a 2d dft, while correcting for the distortion from the anisotropic ctf.
    /// \tparam REMAP       Should be either H2H, HC2H, F2H or FC2H. The output layout is "H" for no particularly good
    ///                     reasons other than the fact that the number of output shells is often (but not limited to)
    ///                     the half dimension size, i.e. min(shape) // 2 + 1.
    /// \param[in] input    Input spectrum to reduce. Can be real or complex.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param input_ctf    Anisotropic CTF(s). The anisotropic spacing and astigmatic field of the defocus are
    ///                     accounted for, resulting in an isotropic rotational average(s). If an varray is passed,
    ///                     there should be one CTF per input batch. Otherwise, the same CTF is assigned to every batch.
    /// \param[out] output  Rotational sum/average. Should be a (batch of) contiguous vector(s).
    ///                     If real and \p input is complex, the power spectrum computed.
    /// \param[out] weights Rotational weights. Can be empty, or be a (batch of) contiguous vector(s) with the same
    ///                     shape as the output. If valid, the output weights are also saved in this array.
    /// \param options      Rotational average options.
    /// \note If \p weights is empty and \p options.average is true, a temporary vector like \p output is allocated.
    template<noa::fft::RemapInterface REMAP,
             typename Input, typename Ctf, typename Output,
             typename Weight = View<nt::value_type_twice_t<Output>>>
    requires guts::is_valid_rotational_average_v<REMAP, Input, Ctf, Output, Weight>
    void rotational_average_anisotropic(
            const Input& input,
            const Shape4<i64>& input_shape,
            const Ctf& input_ctf,
            const Output& output,
            const Weight& weights = {},
            RotationalAverageOptions options = {}
    ) {
        auto n_shells = guts::check_parameters_rotational_average<REMAP>(input, input_shape, input_ctf, output, weights);
        guts::set_frequency_range_to_default(input_shape, options.fftfreq_range);

        if (output.device().is_gpu()) {
            check(ng::is_accessor_access_safe<i32>(input, input.shape()) and
                  ng::is_accessor_access_safe<i32>(output, output.shape()) and
                  ng::is_accessor_access_safe<i32>(weights, weights.shape()),
                  "GPU backend only instantiate i32-based accessor indexing, "
                  "which is unsafe for the given input and output arrays. "
                  "Please report this.");
            guts::launch_rotational_average<REMAP>(
                    input, input_shape.as<i32>(), input_ctf, output, weights, n_shells, options);
        } else {
            guts::launch_rotational_average<REMAP>(
                    input, input_shape.as<i64>(), input_ctf, output, weights, n_shells, options);
        }
    }
}
