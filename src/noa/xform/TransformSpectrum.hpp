#pragma once

#include "noa/runtime/Array.hpp"
#include "noa/runtime/Utils.hpp"
#include "noa/runtime/Iwise.hpp"

#include "noa/fft/core/Frequency.hpp"
#include "noa/xform/Transform.hpp"
#include "noa/xform/Utils.hpp"

namespace noa::xform::details {
    template<usize B, usize R, nf::Layout REMAP,
             nt::integer Index,
             nt::readable_nd<B> Xform,
             nt::readable_nd_or_empty<B> PostShift,
             nt::interpolator_spectrum_nd<R> Interpolator,
             nt::readable_nd<B + R> Input,
             nt::writable_nd<B + R> Output>
    class TransformSpectrum {
    public:
        static constexpr bool IS_DST_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_DST_RFFT = REMAP.is_xx2hx();
        using index_type = Index;
        using shape_type = Shape<index_type, R>;

        using input_type = Input;
        using output_type = Output;
        using interpolator_type = Interpolator;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::spectrum_types<input_value_type, output_value_type>);

        // Xform:
        using batched_rotation_type = Xform;
        using rotation_type = nt::value_type_t<batched_rotation_type>;
        using coord_type = nt::value_type_t<rotation_type>;
        using vec_type = Vec<coord_type, R>;
        static_assert(nt::mat_of_shape<rotation_type, R, R> or (R == 3 and nt::quaternion<rotation_type>));

        // PostShift:
        using batched_postshift_type = PostShift;
        using postshift_type = nt::mutable_value_type_t<batched_postshift_type>;
        static_assert(nt::empty<postshift_type> or
                      (nt::vec_of_type<postshift_type, coord_type> and
                       nt::vec_of_size<postshift_type, R>));

    public:
        TransformSpectrum(
            const input_type& input,
            const output_type& output,
            const shape_type& output_shape,
            const Interpolator& interpolator,
            const batched_rotation_type& inverse_rotation,
            const batched_postshift_type& post_forward_shift,
            coord_type cutoff
        ) :
            m_input(input),
            m_output(output),
            m_inverse_rotation(inverse_rotation),
            m_f_shape(vec_type::from_vec(output_shape.vec)),
            m_post_forward_shift(post_forward_shift),
            m_interpolator(interpolator)
        {
            m_cutoff_fftfreq_sqd = clamp(cutoff, 0, 0.5);
            m_cutoff_fftfreq_sqd *= m_cutoff_fftfreq_sqd;
        }

        NOA_HD constexpr void operator()(const Vec<index_type, B + R>& batched_indices) const {
            const auto& [batches, indices] = batched_indices.template split<B>();

            // Given the output indices, compute the corresponding fftfreq.
            const auto frequency = nf::index2frequency<IS_DST_CENTERED, IS_DST_RFFT>(indices, m_interpolator.shape());
            const auto fftfreq = vec_type::from_vec(frequency) / m_f_shape;

            // Shortcut for the output cutoff.
            if (dot(fftfreq, fftfreq) > m_cutoff_fftfreq_sqd) {
                m_output[batched_indices] = 0;
                return;
            }

            vec_type rotated_fftfreq = transform_vector(m_inverse_rotation[batches], fftfreq);
            auto value = m_interpolator.get(m_input[batches], rotated_fftfreq * m_f_shape);

            // Phase-shift the interpolated value.
            // It is a post-shift, so we need to use the original fftfreq (the ones in the output reference frame)
            if constexpr (nt::complex<input_value_type> and not nt::empty<postshift_type>)
                value *= nf::phase_shift<input_value_type>(m_post_forward_shift[batches], fftfreq);

            m_output[batched_indices] = cast_or_abs_squared<output_value_type>(value);
        }

    private:
        input_type m_input;
        output_type m_output;
        batched_rotation_type m_inverse_rotation;
        vec_type m_f_shape;
        coord_type m_cutoff_fftfreq_sqd;
        NOA_NO_UNIQUE_ADDRESS batched_postshift_type m_post_forward_shift;
        NOA_NO_UNIQUE_ADDRESS interpolator_type m_interpolator;
    };

    template<nf::Layout REMAP, usize R, typename Input, typename Output, typename Matrix, typename Shift, usize N>
    void check_parameters_transform_spectrum_nd(
        const Input& input, const Output& output, const Shape<isize, N>& shape,
        const Matrix& inverse_rotation, const Shift& post_shifts
    ) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");

        // Check batch axes are compatible.
        constexpr usize B = N - R;
        auto input_shape_b = input.shape().template pop_back<R>();
        auto output_shape_b = output.shape().template pop_back<R>();
        auto shape_b = shape.template pop_back<R>();
        if constexpr (B >= 1) {
            for (usize i{}; i < B; ++i)
                input_shape_b[i] = input_shape_b[i] == 1 ? output_shape_b[i] : input_shape_b[i];
            check(input_shape_b == output_shape_b and output_shape_b == shape_b,
                  "The batch axes are not compatible, input:batches={}, output:batches={}, logical_shape:batches={}",
                  input_shape_b, output_shape_b, shape_b);
        }

        // Check the rank axes are compatible. No automatic broadcasting.
        auto input_shape_r = input.shape().template pop_front<B>();
        auto output_shape_r = output.shape().template pop_front<B>();
        auto shape_r = shape.template pop_front<B>();
        check((REMAP.is_hx2xx() ? shape_r.rfft() : shape_r) == input_shape_r,
              "The input shape doesn't match the logical shape. Got input:shape={} and logical_shape={}, rfft={}",
              input.shape(), shape, REMAP.is_hx2xx());
        check((REMAP.is_xx2hx() ? shape_r.rfft() : shape_r) == output_shape_r,
              "The input shape doesn't match the logical shape. Got input:shape={} and logical_shape={}, rfft={}",
              output.shape(), shape, REMAP.is_xx2hx());

        const Device device = output.device();
        if constexpr (nt::array<Matrix>) {
            check(inverse_rotation.is_contiguous() and inverse_rotation.shape() == output_shape_b,
                  "The transformations, specified as a contiguous array, should match the output shape, but got inverse_rotations:shape={}, inverse_rotations:strides={} and output:batches={}",
                  inverse_rotation.shape(), inverse_rotation.strides(), output_shape_b);
            check(device == inverse_rotation.device(),
                  "The transformations should be on the same device as the output, but got inverse_rotations:device={} and output:device={}",
                  inverse_rotation.device(), device);
        }
        if constexpr (nt::array<Shift>) {
            if (not post_shifts.is_empty()) {
                check(post_shifts.is_contiguous() and post_shifts.shape() == output_shape_b,
                  "The shifts, specified as a contiguous array, should match the output shape, but got post_shifts:shape={}, post_shifts:strides={} and output:batches={}",
                  post_shifts.shape(), post_shifts.strides(), output_shape_b);
                check(device == post_shifts.device(),
                      "The shifts should be on the same device as the output, but got post_shifts:device={} and output:device={}",
                      post_shifts.device(), device);
            }
        }

        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={} and output:device={}",
              input.device(), device);
        check(nd::are_elements_unique(output.strides(), output.shape()),
              "The elements in the output should not overlap in memory, otherwise a data-race might occur. Got output:strides={} and output:shape={}",
              output.strides(), output.shape());

        if constexpr (nt::array<Input>) {
            check(not are_overlapped(input, output), "The input and output arrays should not overlap");
        } else {
            check(input.device().is_gpu() or not are_overlapped(input.cpu(), output),
                  "The input and output arrays should not overlap");
            check(input.border() == Border::ZERO,
                  "The texture addressing should be {}, but got {}",
                  Border::ZERO, input.border());
        }
    }

    // nvcc struggles with C++20 template parameters in lambda, so use a worse C++17 syntax and create this type...
    template<bool VALUE>
    struct WrapNoShift {
        consteval auto operator()() const -> bool { return VALUE;}
    };

    template<nf::Layout REMAP, usize R, typename Index, bool IS_GPU = false,
             typename Input, typename Output, typename Matrix, typename Shift, usize N>
    void launch_transform_spectrum_nd(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        Matrix&& xforms,
        Shift&& shifts,
        auto options
    ) {
        constexpr usize B = N - R;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N, Index>;
        auto output_span = output.span().template as_index<Index>();
        auto output_accessor = output_accessor_t(output_span.get(), output_span.strides());
        auto logical_shape_r = shape.template pop_front<B>().template as<Index>();
        auto xform_accessor = xform_into_accessor(xforms);
        using xform_accessor_t = decltype(xform_accessor);

        if constexpr (nt::texture_decay<Input>)
            options.interp = input.interp();

        auto launch_iwise = [&](auto no_shift, auto interp) {
            auto postshift_accessor = xform_into_accessor<true, no_shift()>(shifts);
            using postshift_accessor_t = decltype(postshift_accessor);

            // Get the interpolator.
            using coord_t = nt::mutable_value_type_twice_t<Matrix>;
            auto result = prepare_interpolation_spectrum_inputs<R, REMAP, interp(), IS_GPU, coord_t, false>(input, logical_shape_r);
            using interpolator_t = decltype(result)::interpolator_type;
            using accessor_t = decltype(result)::accessor_type;

            using op_t = TransformSpectrum<
                B, R, REMAP, Index, xform_accessor_t, postshift_accessor_t,
                interpolator_t, accessor_t, output_accessor_t>;

            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(output_span.shape(), output.device(),
               op_t(result.accessor, output_accessor, logical_shape_r,
                    result.interpolator, xform_accessor, postshift_accessor,
                    static_cast<coord_t>(options.fftfreq_cutoff)),
               NOA_FWD(input), NOA_FWD(output), NOA_FWD(xforms), NOA_FWD(shifts));
        };

        auto launch_interp = [&](auto no_shift) {
            switch (options.interp) {
                case Interp::NEAREST:            return launch_iwise(no_shift, WrapInterp<Interp::NEAREST>{});
                case Interp::NEAREST_FAST:       return launch_iwise(no_shift, WrapInterp<Interp::NEAREST_FAST>{});
                case Interp::LINEAR:             return launch_iwise(no_shift, WrapInterp<Interp::LINEAR>{});
                case Interp::LINEAR_FAST:        return launch_iwise(no_shift, WrapInterp<Interp::LINEAR_FAST>{});
                case Interp::CUBIC:              return launch_iwise(no_shift, WrapInterp<Interp::CUBIC>{});
                case Interp::CUBIC_FAST:         return launch_iwise(no_shift, WrapInterp<Interp::CUBIC_FAST>{});
                case Interp::CUBIC_BSPLINE:      return launch_iwise(no_shift, WrapInterp<Interp::CUBIC_BSPLINE>{});
                case Interp::CUBIC_BSPLINE_FAST: return launch_iwise(no_shift, WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
                case Interp::LANCZOS4:           return launch_iwise(no_shift, WrapInterp<Interp::LANCZOS4>{});
                case Interp::LANCZOS6:           return launch_iwise(no_shift, WrapInterp<Interp::LANCZOS6>{});
                case Interp::LANCZOS8:           return launch_iwise(no_shift, WrapInterp<Interp::LANCZOS8>{});
                case Interp::LANCZOS4_FAST:      return launch_iwise(no_shift, WrapInterp<Interp::LANCZOS4_FAST>{});
                case Interp::LANCZOS6_FAST:      return launch_iwise(no_shift, WrapInterp<Interp::LANCZOS6_FAST>{});
                case Interp::LANCZOS8_FAST:      return launch_iwise(no_shift, WrapInterp<Interp::LANCZOS8_FAST>{});
            }
        };

        if constexpr (nt::complex<nt::value_type_t<Input>>) {
            using shift_t = std::decay_t<Shift>;
            bool has_shift{};
            if constexpr (nt::array<shift_t>)
                has_shift = not shifts.is_empty();
            else if constexpr (nt::vec<shift_t>)
                has_shift = shifts.any_ne(0);

            if (has_shift)
                launch_interp(WrapNoShift<false>{});
            else
                launch_interp(WrapNoShift<true>{});
        } else {
            launch_interp(WrapNoShift<false>{});
        }
    }

    template<usize R, usize N, typename Rotation, typename RotationValue = nt::value_type_t<Rotation>>
    concept transform_spectrum_nd_rotation =
        nt::mat_of_shape<std::decay_t<Rotation>, R, R> or
        (R == 3 and nt::quaternion<std::decay_t<Rotation>>) or
        (nt::array_decay_nd<Rotation, N - R> and (nt::mat_of_shape<RotationValue, R, R> or (R == 3 and nt::quaternion<RotationValue>)));

    template<usize R, usize N, typename Shift>
    concept transform_spectrum_nd_shift =
        nt::empty<std::decay_t<Shift>> or
        nt::vec_of_size<std::decay_t<Shift>, R> or
        (nt::array_decay_nd<Shift, N - R> and nt::vec_of_size<nt::value_type_t<Shift>, R>);

    template<usize R, typename Output, typename Rotation, typename Shift, usize N = nt::array_size_v<Output>>
    concept transform_spectrum_nd_rotation_shift =
        transform_spectrum_nd_rotation<R, N, Rotation> and
        transform_spectrum_nd_shift<R, N, Shift> and
        (nt::empty<std::decay_t<Shift>> or
         nt::almost_same_as<nt::value_type_twice_t<Rotation>, nt::value_type_twice_t<Shift>>);

    template<usize R, nf::Layout REMAP, typename Input, typename Output, typename Rotation, typename Shift, usize N>
    concept transformable_spectrum_nd =
        nt::readable_array_or_texture_rd_decay<Input, R> and
        nt::writable_array_decay<Output> and
        nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>> and
        nt::array_size_v<Input> == nt::array_size_v<Output> and nt::array_size_v<Output> == N and N >= R and
        transform_spectrum_nd_rotation_shift<R, Output, Rotation, Shift>;
}

namespace noa::xform {
    struct TransformSpectrumOptions {
        /// Interpolation method. All interpolation modes are supported.
        /// This is ignored if the input is a texture.
        Interp interp{Interp::LINEAR};

        /// Maximum output frequency to consider, in cycle/pix.
        /// Values are clamped from 0 (DC) to 0.5 (Nyquist).
        /// Frequencies higher than this value are set to 0.
        f64 fftfreq_cutoff{0.5};
    };

    /// Applies one or multiple 2D transforms (rotation and scaling, followed by translation) on 2D (r)FFTs.
    /// \tparam REMAP:
    ///     Remap operation. Every layout and remapping is supported.
    /// \tparam Rotation:
    ///     Mat22<Coord> or a array of that type.
    /// \tparam Shift:
    ///     Vec<Coord, 2>, a array of that type, or Empty.
    /// \param[in] input:
    ///     2D (r)FFT(s) to transform.
    ///     ((Bi..,)H,Wi) Input 2D array(s) or 2D texture(s), of type f16, f32, f64, c16, c32, c64.
    ///     The batch axes are broadcast to the output batch axes.
    /// \param[out] output:
    ///     2D transformed (r)FFT(s).
    ///     ((Bo..,)H,Wo) Output 2D array(s).
    /// \param shape:
    ///     Logical shape of input and output.
    /// \param[in] inverse_rotations:
    ///     2x2 inverse HW rotation/scaling matrix.
    ///     One matrix, or a contiguous (Bo..) array matching the output batches.
    ///     Sets the floating-point precision of the transformation and interpolation.
    /// \param[in] post_shifts:
    ///     2D real-space HW forward shift to apply (as phase shift) after the transformation.
    ///     One, or a contiguous (Bo..) array matching the output batches.
    ///     If Empty, a zero vector, an empty array or if the input is real, it is ignored.
    /// \param options:
    ///     Transformation options.
    ///
    /// \note For more details, see InterpolatorSpectrum.
    template<nf::Layout REMAP, typename Input, typename Output, typename Rotation, typename Shift = Empty, usize N>
        requires details::transformable_spectrum_nd<2, REMAP, Input, Output, Rotation, Shift, N>
    void transform_spectrum_2d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        Rotation&& inverse_rotations,
        Shift&& post_shifts = {},
        const TransformSpectrumOptions& options = {}
    ) {
        details::check_parameters_transform_spectrum_nd<REMAP, 2>(input, output, shape, inverse_rotations, post_shifts);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(nd::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
                      nd::is_accessor_access_safe<i32>(output.strides(), output.shape()),
                      "isize indexing not instantiated for GPU devices");
                details::launch_transform_spectrum_nd<REMAP, 2, i32, true>(
                    NOA_FWD(input), NOA_FWD(output), shape,
                    NOA_FWD(inverse_rotations), NOA_FWD(post_shifts),
                    options);
                return;
            }
            #else
            panic_no_gpu_backend();
            #endif
        }

        details::launch_transform_spectrum_nd<REMAP, 2, isize>(
            NOA_FWD(input), NOA_FWD(output), shape,
            NOA_FWD(inverse_rotations), NOA_FWD(post_shifts),
            options);
    }

    /// Applies one or multiple 3D transforms (rotation and scaling, followed by translation) on 3D (r)FFTs.
    /// \tparam REMAP:
    ///     Remap operation. Every layout and remapping is supported.
    /// \tparam Rotation:
    ///     Mat33<Coord>, Quaternion<Coord>, or a array of these types.
    /// \tparam Shift:
    ///     Vec<Coord, 3>, a array of that type, or Empty.
    /// \param[in] input:
    ///     3D (r)FFT(s) to transform.
    ///     ((Bi..,)D,H,Wi) Input 3D array(s) or 3D texture(s), of type f16, f32, f64, c16, c32, c64.
    ///     The batch axes are broadcast to the output batch axes.
    /// \param[out] output:
    ///     3D transformed (r)FFT(s).
    ///     ((Bo..,)D,H,Wo) Output 3D array(s).
    /// \param shape:
    ///     Logical shape of input and output.
    /// \param[in] inverse_rotations:
    ///     3x3 inverse DHW rotation/scaling matrix.
    ///     One matrix, or a contiguous (Bo..) array matching the output batches.
    ///     Sets the floating-point precision of the transformation and interpolation.
    /// \param[in] post_shifts:
    ///     3D real-space DHW forward shift to apply (as phase shift) after the transformation.
    ///     One, or a contiguous (Bo..) array matching the output batches.
    ///     If Empty, a zero vector, an empty array or if the input is real, it is ignored.
    /// \param options:
    ///     Transformation options.
    ///
    /// \note For more details, see InterpolatorSpectrum.
    template<nf::Layout REMAP, typename Input, typename Output, typename Rotation, typename Shift = Empty, usize N>
        requires details::transformable_spectrum_nd<3, REMAP, Input, Output, Rotation, Shift, N>
    void transform_spectrum_3d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        Rotation&& inverse_rotations,
        Shift&& post_shifts = {},
        const TransformSpectrumOptions& options = {}
    ) {
        details::check_parameters_transform_spectrum_nd<REMAP, 3>(input, output, shape, inverse_rotations, post_shifts);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                if (nd::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
                    nd::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
                    details::launch_transform_spectrum_nd<REMAP, 3, i32, true>(
                        NOA_FWD(input), NOA_FWD(output), shape,
                        NOA_FWD(inverse_rotations), NOA_FWD(post_shifts),
                        options);
                } else {
                    // For large volumes (>1290^3), isize indexing is required.
                    details::launch_transform_spectrum_nd<REMAP, 3, isize, true>(
                        NOA_FWD(input), NOA_FWD(output), shape,
                        NOA_FWD(inverse_rotations), NOA_FWD(post_shifts),
                        options);
                }
                return;
            }
            #else
            panic_no_gpu_backend();
            #endif
        }

        details::launch_transform_spectrum_nd<REMAP, 3, isize>(
            NOA_FWD(input), NOA_FWD(output), shape,
            NOA_FWD(inverse_rotations), NOA_FWD(post_shifts),
            options);
    }
}
