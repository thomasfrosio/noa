#pragma once

#include "noa/runtime/core/Enums.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Utilities.hpp"
#include "noa/runtime/Iwise.hpp"

#include "noa/fft/core/Frequency.hpp"
#include "noa/xform/core/Interpolation.hpp"
#include "noa/xform/Transform.hpp"

namespace noa::xform::details {
    /// 3d or 4d iwise operator to compute linear transformations (one rotation/scaling followed by one translation)
    /// of 2d or 3d array(s) by directly manipulating their 2d or 3d Fourier transforms(s).
    /// \tparam N                   2 or 3.
    /// \tparam REMAP               Every output layout is supported. The input controls the input layouts.
    /// \tparam Index               Signed integer.
    /// \tparam BatchedRotate       Batched of Mat<T, N, N> or, iff N=3, Quaternion<T>.
    /// \tparam BatchedPostShift    Batched of Empty or Vec<T, N>.
    /// \tparam Input               N-d spectrum interpolator.
    /// \tparam Output              (N + 1)-d writable.
    template<usize N, nf::Layout REMAP,
             nt::integer Index,
             nt::batch BatchedRotate,
             nt::batch BatchedPostShift,
             nt::interpolator_spectrum_nd<N> Input,
             nt::writable_nd<N + 1> Output>
    class TransformSpectrum {
    public:
        static constexpr bool IS_DST_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_DST_RFFT = REMAP.is_xx2hx();
        using index_type = Index;
        using shape_type = Shape<index_type, N - IS_DST_RFFT>;
        using indices_type = Vec<index_type, N>;

        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::spectrum_types<input_value_type, output_value_type>);

        // Rotation:
        using batched_rotation_type = BatchedRotate;
        using rotation_type = nt::value_type_t<batched_rotation_type>;
        using coord_type = nt::value_type_t<rotation_type>;
        using vec_type = Vec<coord_type, N>;
        static_assert(nt::mat_of_shape<rotation_type, N, N> or (N == 3 and nt::quaternion<rotation_type>));

        // PostShift:
        using batched_postshift_type = BatchedPostShift;
        using postshift_type = nt::value_type_t<batched_postshift_type>;
        static_assert(nt::empty<postshift_type> or
                      (nt::vec_of_type<postshift_type, coord_type> and
                       nt::vec_of_size<postshift_type, N>));

    public:
        TransformSpectrum(
            const input_type& input,
            const output_type& output,
            const Shape<index_type, N>& output_shape,
            const batched_rotation_type& inverse_rotation,
            const batched_postshift_type& post_forward_shift,
            coord_type cutoff
        ) :
            m_input(input),
            m_output(output),
            m_inverse_rotation(inverse_rotation),
            m_shape(output_shape.template pop_back<IS_DST_RFFT>()),
            m_f_shape(vec_type::from_vec(output_shape.vec)),
            m_post_forward_shift(post_forward_shift)
        {
            m_cutoff_fftfreq_sqd = clamp(cutoff, 0, 0.5);
            m_cutoff_fftfreq_sqd *= m_cutoff_fftfreq_sqd;
        }

        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        NOA_HD constexpr void operator()(index_type batch, I... indices) const {
            // Given the output indices, compute the corresponding fftfreq.
            const auto frequency = nf::index2frequency<IS_DST_CENTERED, IS_DST_RFFT>(
                indices_type{indices...}, m_shape);
            const auto fftfreq = vec_type::from_vec(frequency) / m_f_shape;

            // Shortcut for the output cutoff.
            if (dot(fftfreq, fftfreq) > m_cutoff_fftfreq_sqd) {
                m_output(batch, indices...) = 0;
                return;
            }

            vec_type rotated_fftfreq = transform_vector(m_inverse_rotation[batch], fftfreq);
            auto value = m_input.interpolate_spectrum_at(rotated_fftfreq * m_f_shape, batch);

            // Phase-shift the interpolated value.
            // It is a post-shift, so we need to use the original fftfreq (the ones in the output reference frame)
            if constexpr (nt::complex<input_value_type> and not nt::empty<postshift_type>)
                value *= nf::phase_shift<input_value_type>(m_post_forward_shift[batch], fftfreq);

            m_output(batch, indices...) = cast_or_abs_squared<output_value_type>(value);
        }

    private:
        input_type m_input;
        output_type m_output;
        batched_rotation_type m_inverse_rotation;
        shape_type m_shape;
        vec_type m_f_shape;
        coord_type m_cutoff_fftfreq_sqd;
        NOA_NO_UNIQUE_ADDRESS batched_postshift_type m_post_forward_shift;
    };

    template<usize N, typename Input, typename Output, typename Matrix, typename Shift>
    void check_parameters_transform_spectrum_nd(
        const Input& input, const Output& output, const Shape4& shape,
        const Matrix& inverse_rotation, const Shift& post_shifts
    ) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(shape[3] / 2 + 1 == input.shape()[3] and input.shape()[3] == output.shape()[3] and
              shape[2] == input.shape()[2] and input.shape()[2] == output.shape()[2] and
              shape[1] == input.shape()[1] and input.shape()[1] == output.shape()[1],
              "The rfft input and/or output shapes don't match the logical shape {}. "
              "Got input:shape={} and output:shape={}",
              shape, input.shape(), output.shape());
        check(N == 3 or (input.shape()[1] == 1 and output.shape()[1] == 1),
              "The input and output should be 2d arrays, but got shape input:shape={}, output:shape={}",
              input.shape(), output.shape());
        check(input.shape()[0] == 1 or input.shape()[0] == output.shape()[0],
              "The number of batches in the input ({}) is not compatible with the number of batches in the output ({})",
              input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, "
              "but got input:device={} and output:device={}", input.device(), device);

        if constexpr (nt::varray<Matrix>) {
            check(ni::is_contiguous_vector(inverse_rotation) and inverse_rotation.n_elements() == output.shape()[0],
                  "The number of rotations, specified as a contiguous vector, should be equal to the batch size "
                  "in the output, but got inverse_rotation:shape={}, inverse_rotation:strides={} and output:batch={}",
                  inverse_rotation.shape(), inverse_rotation.strides(), output.shape()[0]);
            check(device == inverse_rotation.device(),
                  "The input rotations and output arrays must be on the same device, "
                  "but got inverse_rotation:device={} and output:device={}", inverse_rotation.device(), device);
        }
        if constexpr (nt::varray<Shift>) {
            if (not post_shifts.is_empty()) {
                check(ni::is_contiguous_vector(post_shifts) and post_shifts.n_elements() == output.shape()[0],
                      "The number of shifts, specified as a contiguous vector, should be equal to the batch size"
                      "in the output, but got post_shifts:shape={}, post_shifts:strides={} and output:batch={}",
                      post_shifts.shape(), post_shifts.strides(), output.shape()[0]);
                check(device == post_shifts.device(),
                      "The input shifts and output arrays must be on the same device, "
                      "but got post_shifts:device={} and output:device={}", post_shifts.device(), device);
            }
        }

        check(device == input.device(),
              "The input and output arrays must be on the same device, "
              "but got input:device={} and output:device={}", input.device(), device);
        check(ni::are_elements_unique(output.strides(), output.shape()),
              "The elements in the output should not overlap in memory, "
              "otherwise a data-race might occur. Got output:strides={} and output:shape={}",
              output.strides(), output.shape());

        if constexpr (nt::varray<Input>) {
            check(not ni::are_overlapped(input, output),
                  "The input and output arrays should not overlap");
        } else {
            check(input.device().is_gpu() or not ni::are_overlapped(input.view(), output),
                  "The input and output arrays should not overlap");
            check(input.border() == Border::ZERO,
                  "The texture addressing should be {}, but got {}", Border::ZERO, input.border());
        }
    }

    // nvcc struggles with C++20 template parameters in lambda, so use a worse C++17 syntax and create this type...
    template<bool VALUE>
    struct WrapNoShift {
        consteval auto operator()() const -> bool { return VALUE;}
    };

    template<nf::Layout REMAP, usize N, typename Index, bool IS_GPU = false,
             typename Input, typename Output, typename Matrix, typename Shift>
    void launch_transform_spectrum_nd(
        Input&& input,
        Output&& output,
        const Shape4& shape,
        Matrix&& inverse_rotations,
        Shift&& post_shifts,
        const auto& options
    ) {
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N + 1, Index>;
        auto output_accessor = output_accessor_t(output.get(), output.strides().template filter_nd<N>().template as<Index>());
        auto logical_shape = shape.as<Index>();
        auto batched_inverse_rotations = nd::to_batched_transform(inverse_rotations);

        auto launch_iwise = [&](auto no_shift, auto interp) {
            auto batched_post_shifts = nd::to_batched_transform<true, no_shift()>(post_shifts);

            // Get the interpolator.
            using coord_t = nt::mutable_value_type_twice_t<Matrix>;
            auto interpolator = to_interpolator_spectrum<N, REMAP, interp(), coord_t, IS_GPU>(input, logical_shape);

            using op_t = TransformSpectrum<
                N, REMAP, Index, decltype(batched_inverse_rotations), decltype(batched_post_shifts),
                decltype(interpolator), output_accessor_t>;

            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(output.shape().template filter_nd<N>().template as<Index>(), output.device(),
               op_t(interpolator, output_accessor, logical_shape.template filter_nd<N>().pop_front(),
                    batched_inverse_rotations, batched_post_shifts,
                    static_cast<coord_t>(options.fftfreq_cutoff)),
               std::forward<Input>(input),
               std::forward<Output>(output),
               std::forward<Matrix>(inverse_rotations),
               std::forward<Shift>(post_shifts));
        };

        auto launch_interp = [&](auto no_shift) {
            Interp interp = options.interp;
            if constexpr (nt::is_texture_decay_v<Input>)
                interp = input.interp();

            using enum Interp::Method;
            switch (interp) {
                case NEAREST:            return launch_iwise(no_shift, WrapInterp<NEAREST>{});
                case NEAREST_FAST:       return launch_iwise(no_shift, WrapInterp<NEAREST_FAST>{});
                case LINEAR:             return launch_iwise(no_shift, WrapInterp<LINEAR>{});
                case LINEAR_FAST:        return launch_iwise(no_shift, WrapInterp<LINEAR_FAST>{});
                case CUBIC:              return launch_iwise(no_shift, WrapInterp<CUBIC>{});
                case CUBIC_FAST:         return launch_iwise(no_shift, WrapInterp<CUBIC_FAST>{});
                case CUBIC_BSPLINE:      return launch_iwise(no_shift, WrapInterp<CUBIC_BSPLINE>{});
                case CUBIC_BSPLINE_FAST: return launch_iwise(no_shift, WrapInterp<CUBIC_BSPLINE_FAST>{});
                case LANCZOS4:           return launch_iwise(no_shift, WrapInterp<LANCZOS4>{});
                case LANCZOS6:           return launch_iwise(no_shift, WrapInterp<LANCZOS6>{});
                case LANCZOS8:           return launch_iwise(no_shift, WrapInterp<LANCZOS8>{});
                case LANCZOS4_FAST:      return launch_iwise(no_shift, WrapInterp<LANCZOS4_FAST>{});
                case LANCZOS6_FAST:      return launch_iwise(no_shift, WrapInterp<LANCZOS6_FAST>{});
                case LANCZOS8_FAST:      return launch_iwise(no_shift, WrapInterp<LANCZOS8_FAST>{});
            }
        };

        using shift_t = std::decay_t<Shift>;
        bool has_shift{};
        if constexpr (nt::varray<shift_t>)
            has_shift = not post_shifts.is_empty();
        else if constexpr (nt::vec<shift_t>)
            has_shift = post_shifts.any_ne(0);

        if (nt::complex<nt::value_type_t<Input>> and has_shift)
            launch_interp(WrapNoShift<false>{});
        else
            launch_interp(WrapNoShift<true>{});
    }

    template<usize N, typename Rotation, typename RotationValue = nt::value_type_t<Rotation>>
    concept transform_spectrum_nd_rotation =
        nt::mat_of_shape<std::decay_t<Rotation>, N, N> or
        (N == 3 and nt::quaternion<std::decay_t<Rotation>>) or
        (nt::varray_decay<Rotation> and (nt::mat_of_shape<RotationValue, N, N> or (N == 3 and nt::quaternion<RotationValue>)));

    template<usize N, typename Shift>
    concept transform_spectrum_nd_shift =
        nt::empty<std::decay_t<Shift>> or
        nt::vec_of_size<std::decay_t<Shift>, N> or
        (nt::varray_decay<Shift> and nt::vec_of_size<nt::value_type_t<Shift>, N>);

    template<usize N, typename Rotation, typename Shift>
    concept transform_spectrum_nd_rotation_shift =
        transform_spectrum_nd_rotation<N, Rotation> and
        transform_spectrum_nd_shift<N, Shift> and
        (nt::empty<std::decay_t<Shift>> or
         nt::almost_same_as<nt::value_type_twice_t<Rotation>, nt::value_type_twice_t<Shift>>);
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

    /// Transforms 2d arrays (one rotation and/or scaling followed by one translation)
    /// by directly manipulating their 2d (r)FFTs.
    ///
    /// \details The input and output arrays should be 2d arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (N input -> N output). However, if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    ///
    /// \tparam REMAP                   Remap operation. Every layout is supported.
    /// \tparam Rotation                Mat22<Coord> or a varray of that type.
    /// \tparam Shift                   Vec<Coord, 2>, a varray of that type, or Empty.
    /// \param[in] input                2d (r)FFT(s) to transform, of type f32, f64, c32, c64.
    /// \param[out] output              Transformed 2d (r)FFT(s).
    /// \param shape                    BDHW logical shape of input and output.
    /// \param[in] inverse_rotations    2x2 inverse HW rotation/scaling matrix.
    ///                                 One, or if an array is provided, one per output batch.
    /// \param[in] post_shifts          2d real-space HW forward shift to apply (as phase shift) after the
    ///                                 transformation. One, or if an array is provided, one per output batch.
    ///                                 If an empty array is entered or if input is real, it is ignored.
    /// \param options                  Transformation options.
    ///
    /// \note For more details, see InterpolateSpectrum.
    template<nf::Layout REMAP,
             nt::varray_or_texture_decay Input,
             nt::writable_varray_decay Output,
             typename Rotation,
             typename Shift = Empty>
    requires (nt::varray_or_texture_decay_with_spectrum_types<Input, Output> and
              details::transform_spectrum_nd_rotation_shift<2, Rotation, Shift>)
    void transform_spectrum_2d(
        Input&& input,
        Output&& output,
        const Shape4& shape,
        Rotation&& inverse_rotations,
        Shift&& post_shifts = {},
        const TransformSpectrumOptions& options = {}
    ) {
        details::check_parameters_transform_spectrum_nd<2>(input, output, shape, inverse_rotations, post_shifts);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(nd::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
                      nd::is_accessor_access_safe<i32>(output.strides(), output.shape()),
                      "isize indexing not instantiated for GPU devices");
                details::launch_transform_spectrum_nd<REMAP, 2, i32, true>(
                    std::forward<Input>(input),
                    std::forward<Output>(output), shape,
                    std::forward<Rotation>(inverse_rotations),
                    std::forward<Shift>(post_shifts),
                    options);
                return;
            }
            #else
            panic_no_gpu_backend();
            #endif
        }

        details::launch_transform_spectrum_nd<REMAP, 2, isize>(
            std::forward<Input>(input),
            std::forward<Output>(output), shape,
            std::forward<Rotation>(inverse_rotations),
            std::forward<Shift>(post_shifts),
            options);
    }

    /// Transforms 3d arrays (one rotation and/or scaling followed by one translation)
    /// by directly manipulating their 3d (r)FFTs.
    ///
    /// \details The input and output arrays should be 3d arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (N input -> N output). However, if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    ///
    /// \tparam REMAP                   Remap operation. Every layout is supported.
    /// \tparam Rotation                Mat33<Coord> or a varray of that type.
    /// \tparam Shift                   Vec<Coord, 3>, Quaternion<Coord>, a varray of these types, or Empty.
    /// \param[in] input                3d (r)FFT(s) to transform, of type f32, f64, c32, c64.
    /// \param[out] output              Transformed 3d (r)FFT(s).
    /// \param shape                    BDHW logical shape of input and output.
    /// \param[in] inverse_rotations    3x3 inverse DHW rotation/scaling matrix.
    ///                                 One, or if an array is provided, one per output batch.
    /// \param[in] post_shifts          3d real-space DHW forward shift to apply (as phase shift) after the
    ///                                 transformation. One, or if an array is provided, one per output batch.
    ///                                 If an empty array is entered or if input is real, it is ignored.
    /// \param options                  Transformation options.
    ///
    /// \note For more details, see InterpolateSpectrum.
    template<nf::Layout REMAP,
             nt::varray_or_texture_decay Input,
             nt::writable_varray_decay Output,
             typename Rotation,
             typename Shift = Empty>
    requires (nt::varray_or_texture_decay_with_spectrum_types<Input, Output> and
              details::transform_spectrum_nd_rotation_shift<3, Rotation, Shift>)
    void transform_spectrum_3d(
        Input&& input,
        Output&& output,
        const Shape4& shape,
        Rotation&& inverse_rotations,
        Shift&& post_shifts = {},
        const TransformSpectrumOptions& options = {}
    ) {
        details::check_parameters_transform_spectrum_nd<3>(input, output, shape, inverse_rotations, post_shifts);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                if (nd::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
                    nd::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
                    details::launch_transform_spectrum_nd<REMAP, 3, i32, true>(
                        std::forward<Input>(input),
                        std::forward<Output>(output), shape,
                        std::forward<Rotation>(inverse_rotations),
                        std::forward<Shift>(post_shifts),
                        options);
                } else {
                    // For large volumes (>1290^3), isize indexing is required.
                    details::launch_transform_spectrum_nd<REMAP, 3, isize, true>(
                        std::forward<Input>(input),
                        std::forward<Output>(output), shape,
                        std::forward<Rotation>(inverse_rotations),
                        std::forward<Shift>(post_shifts),
                        options);
                }
                return;
            }
            #else
            panic_no_gpu_backend();
            #endif
        }

        details::launch_transform_spectrum_nd<REMAP, 3, isize>(
            std::forward<Input>(input),
            std::forward<Output>(output), shape,
            std::forward<Rotation>(inverse_rotations),
            std::forward<Shift>(post_shifts),
            options);
    }
}
