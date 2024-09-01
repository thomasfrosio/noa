#pragma once

#include "noa/core/fft/Remap.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/geometry/TransformSpectrum.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"
#include "noa/unified/Utilities.hpp"
#include "noa/unified/Iwise.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Interpolator.hpp"
#endif

namespace noa::geometry::guts {
    template<i32 NDIM, typename Input, typename Output, typename Matrix, typename Shift>
    void check_parameters_transform_spectrum_nd(
            const Input& input, const Output& output, const Shape4<i64>& shape,
            const Matrix& matrices, const Shift& shifts
    ) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(shape[3] / 2 + 1 == input.shape()[3] and input.shape()[3] == output.shape()[3] and
              shape[2] == input.shape()[2] and input.shape()[2] == output.shape()[2] and
              shape[1] == input.shape()[1] and input.shape()[1] == output.shape()[1],
              "The rfft input and/or output shapes don't match the logical shape {}. "
              "Got input:shape={} and output:shape={}",
              shape, input.shape(), output.shape());
        check(NDIM == 3 or (input.shape()[1] == 1 && output.shape()[1] == 1),
              "The input and output should be 2d arrays, but got shape input:shape={}, output:shape={}",
              input.shape(), output.shape());
        check(input.shape()[0] == 1 or input.shape()[0] == output.shape()[0],
              "The number of batches in the input ({}) is not compatible with the number of batches in the output ({})",
              input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, "
              "but got input:device={} and output:device={}", input.device(), device);

        if constexpr (not nt::is_mat_v<Matrix>) {
            check(ni::is_contiguous_vector(matrices) and matrices.elements() == output.shape()[0],
                  "The number of matrices, specified as a contiguous vector, should be equal to the number of "
                  "batches in the output, but got matrices:shape={}, matrices:strides={} and output:batch={}",
                  matrices.shape(), matrices.strides(), output.shape()[0]);
            check(device == matrices.device(),
                  "The input matrices and output arrays must be on the same device, "
                  "but got matrix:device={} and output:device={}", matrices.device(), device);
        }
        if constexpr (not nt::is_real_v<Shift>) {
            check(ni::is_contiguous_vector(shifts) and shifts.elements() == output.shape()[0],
                  "The number of shifts, specified as a contiguous vector, should be equal to the number of "
                  "batches in the output, but got shifts:shape={}, shifts:strides={} and output:batch={}",
                  shifts.shape(), shifts.strides(), output.shape()[0]);
            check(device == shifts.device(),
                  "The input shifts and output arrays must be on the same device, "
                  "but got shift:device={} and output:device={}", shifts.device(), device);
        }

        check(device == input.device(),
              "The input and output arrays must be on the same device, "
              "but got input:device={} and output:device={}", input.device(), device);
        check(ni::are_elements_unique(output.strides(), output.shape()),
              "The elements in the output should not overlap in memory, "
              "otherwise a data-race might occur. Got output strides:{} and shape:{}",
              output.strides(), output.shape());

        if constexpr (nt::is_varray_v<Input>) {
            check(not ni::are_overlapped(input, output),
                  "The input and output arrays should not overlap");
        } else {
            check(input.device().is_gpu() or not ni::are_overlapped(input.view(), output),
                  "The input and output arrays should not overlap");
            // TODO We could check the texture has its border_mode to Border::ZERO?
        }
    }

    template<noa::fft::Remap REMAP, size_t N, typename Index,
             typename Input, typename Output, typename Matrix, typename Shift>
    void launch_transform_spectrum_nd(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const Matrix& inverse_matrices,
            const Shift& post_shifts,
            Interp interp_mode,
            f64 fftfreq_cutoff
    ) {
        using input_value_t = nt::mutable_value_type_t<Input>;
        using input_accessor_t = AccessorRestrict<const input_value_t, N + 1, Index>;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N + 1, Index>;
        using matrix_t = decltype(extract_transform(inverse_matrices));
        using shift_t = decltype(extract_transform(post_shifts));

        auto get_strides_nd = [](const auto& array) {
            if constexpr (N == 2)
                return array.strides().filter(0, 2, 3).template as<Index>();
            else
                return array.strides().template as<Index>();
        };

        auto input_accessor = input_accessor_t(input.get(), get_strides_nd(input));
        auto output_accessor = output_accessor_t(output.get(), get_strides_nd(output));

        auto logical_shape = shape.as<Index>();
        auto input_shape_nd = [&input] {
            if constexpr (N == 2)
                return input.shape().filter(2, 3).template as<Index>();
            else
                return input.shape().filter(1, 2, 3).template as<Index>();
        }();
        auto output_shape = [&output] {
            if constexpr (N == 2)
                return output.shape().filter(0, 2, 3).template as<Index>();
            else
                return output.shape().template as<Index>();
        }();

        // Broadcast the input to every output batch.
        if (input.shape()[0] == 1)
            input_accessor.strides()[0] = 0;

        auto launch_for_each_border = [&]<bool HasShift>{
            using shift_or_empty_t = std::conditional_t<HasShift, shift_t, Empty>;
            shift_or_empty_t shifts{};
            if constexpr (HasShift)
                shifts = extract_transform(post_shifts);

            switch (interp_mode) {
                case Interp::NEAREST: {
                    using interpolator_t = InterpolatorNd<N, Border::ZERO, Interp::NEAREST, input_accessor_t>;
                    using op_t = TransformSpectrum<N, REMAP, Index, matrix_t, shift_or_empty_t, interpolator_t, output_accessor_t>;
                    auto interpolator = interpolator_t(input_accessor, input_shape_nd);
                    auto op = op_t(interpolator, output_accessor, logical_shape, extract_transform(inverse_matrices), shifts, fftfreq_cutoff);
                    return iwise(output_shape, output.device(), std::move(op), input, output, inverse_matrices, post_shifts);
                }
                case Interp::LINEAR:
                case Interp::LINEAR_FAST: {
                    using interpolator_t = InterpolatorNd<N, Border::ZERO, Interp::LINEAR, input_accessor_t>;
                    using op_t = TransformSpectrum<N, REMAP, Index, matrix_t, shift_or_empty_t, interpolator_t, output_accessor_t>;
                    auto interpolator = interpolator_t(input_accessor, input_shape_nd);
                    auto op = op_t(interpolator, output_accessor, logical_shape, extract_transform(inverse_matrices), shifts, fftfreq_cutoff);
                    return iwise(output_shape, output.device(), std::move(op), input, output, inverse_matrices, post_shifts);
                }
                case Interp::COSINE:
                case Interp::COSINE_FAST: {
                    using interpolator_t = InterpolatorNd<N, Border::ZERO, Interp::COSINE, input_accessor_t>;
                    using op_t = TransformSpectrum<N, REMAP, Index, matrix_t, shift_or_empty_t, interpolator_t, output_accessor_t>;
                    auto interpolator = interpolator_t(input_accessor, input_shape_nd);
                    auto op = op_t(interpolator, output_accessor, logical_shape, extract_transform(inverse_matrices), shifts, fftfreq_cutoff);
                    return iwise(output_shape, output.device(), std::move(op), input, output, inverse_matrices, post_shifts);
                }
                default:
                    panic("{} is currently not supported", interp_mode);
            }
        };

        bool has_shift;
        if constexpr (nt::is_varray_v<Shift>)
            has_shift = not post_shifts.is_empty();
        else
            has_shift = any(post_shifts != Shift{});
        if (nt::is_complex_v<input_value_t> and has_shift)
            launch_for_each_border.template operator()<true>();
        else
            launch_for_each_border.template operator()<false>();
    }

#ifdef NOA_ENABLE_CUDA
    template<noa::fft::Remap REMAP, size_t N, typename Index,
             typename Value, typename Output, typename Matrix, typename Shift>
    void launch_transform_spectrum_nd(
            const Texture<Value>& input,
            const Output& output,
            const Shape4<i64>& shape,
            const Matrix& inverse_matrices,
            const Shift& post_shifts,
            f64 fftfreq_cutoff
    ) {
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N + 1, Index>;
        using matrix_t = decltype(extract_transform(inverse_matrices));
        using shift_t = decltype(extract_transform(post_shifts));

        auto logical_shape = shape.as<Index>();
        auto output_shape = [&output] {
            if constexpr (N == 2)
                return output.shape().filter(0, 2, 3).template as<Index>();
            else
                return output.shape().template as<Index>();
        }();

        auto get_strides_nd = [](const auto& array) {
            if constexpr (N == 2)
                return array.strides().filter(0, 2, 3).template as<Index>();
            else
                return array.strides().template as<Index>();
        };
        const auto output_accessor = output_accessor_t(output.get(), get_strides_nd(output));

        using noa::cuda::geometry::InterpolatorNd;
        using coord_t = nt::value_type_twice_t<Matrix>;
        std::shared_ptr cuda_texture = input.cuda();
        cudaTextureObject_t texture = cuda_texture->texture;

        auto launch_for_each_interp = [&]<bool HasShift, bool IsLayered>{
            using shift_or_empty_t = std::conditional_t<HasShift, shift_t, Empty>;
            shift_or_empty_t shifts{};
            if constexpr (HasShift)
                shifts = extract_transform(post_shifts);

            switch (input.interp_mode()) {
                case Interp::NEAREST: {
                    using interpolator_t = InterpolatorNd<N, Interp::NEAREST, Value, false, IsLayered, coord_t>;
                    using op_t = TransformSpectrum<N, REMAP, Index, matrix_t, shift_or_empty_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, logical_shape, extract_transform(inverse_matrices), shifts, fftfreq_cutoff);
                    return iwise(output_shape, output.device(), std::move(op), input, output, inverse_matrices, post_shifts);
                }
                case Interp::LINEAR: {
                    using interpolator_t = InterpolatorNd<N, Interp::LINEAR, Value, false, IsLayered, coord_t>;
                    using op_t = TransformSpectrum<N, REMAP, Index, matrix_t, shift_or_empty_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, logical_shape, extract_transform(inverse_matrices), shifts, fftfreq_cutoff);
                    return iwise(output_shape, output.device(), std::move(op), input, output, inverse_matrices, post_shifts);
                }
                case Interp::LINEAR_FAST: {
                    using interpolator_t = InterpolatorNd<N, Interp::LINEAR_FAST, Value, false, IsLayered, coord_t>;
                    using op_t = TransformSpectrum<N, REMAP, Index, matrix_t, shift_or_empty_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, logical_shape, extract_transform(inverse_matrices), shifts, fftfreq_cutoff);
                    return iwise(output_shape, output.device(), std::move(op), input, output, inverse_matrices, post_shifts);
                }
                case Interp::COSINE: {
                    using interpolator_t = InterpolatorNd<N, Interp::COSINE, Value, false, IsLayered, coord_t>;
                    using op_t = TransformSpectrum<N, REMAP, Index, matrix_t, shift_or_empty_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, logical_shape, extract_transform(inverse_matrices), shifts, fftfreq_cutoff);
                    return iwise(output_shape, output.device(), std::move(op), input, output, inverse_matrices, post_shifts);
                }
                case Interp::COSINE_FAST: {
                    using interpolator_t = InterpolatorNd<N, Interp::COSINE_FAST, Value, false, IsLayered, coord_t>;
                    using op_t = TransformSpectrum<N, REMAP, Index, matrix_t, shift_or_empty_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, logical_shape, extract_transform(inverse_matrices), shifts, fftfreq_cutoff);
                    return iwise(output_shape, output.device(), std::move(op), input, output, inverse_matrices, post_shifts);
                }
                default:
                    panic("{} is currently not supported", input.interp_mode());
            }
        };

        auto launch_for_each_layered = [&]<bool IsLayered>{
            if (N == 2 and input.is_layered()) { // 3d textures cannot be layered
                launch_for_each_interp.template operator()<IsLayered, true>();
            } else {
                launch_for_each_interp.template operator()<IsLayered, false>();
            }
        };

        bool has_shift;
        if constexpr (nt::is_varray_v<Shift>)
            has_shift = not post_shifts.is_empty();
        else
            has_shift = any(post_shifts != Shift{});
        if (nt::is_complex_v<Value> and has_shift)
            launch_for_each_layered.template operator()<true>();
        else
            launch_for_each_layered.template operator()<false>();
    }
#endif

    template<typename Matrix>
    constexpr bool are_valid_transform_spectrum_2d_matrix_v =
            nt::is_mat22_v<Matrix> or (nt::is_varray_v<Matrix> and nt::is_mat22_v<nt::value_type_t<Matrix>>);

    template<typename Matrix>
    constexpr bool are_valid_transform_spectrum_3d_matrix_v =
            nt::is_mat33_v<Matrix> or nt::is_quaternion_v<Matrix> or
            (nt::is_varray_v<Matrix> and
             (nt::is_mat33_v<nt::value_type_t<Matrix>> or nt::is_quaternion_v<nt::value_type_t<Matrix>>));

    template<typename Shift, size_t N>
    constexpr bool are_valid_transform_spectrum_nd_shift_v =
            nt::is_vec_of_size_v<Shift, N> or
            (nt::is_varray_v<Shift> and nt::is_vec_of_size_v<nt::value_type_t<Shift>, N>);
}

namespace noa::geometry {
    struct TransformSpectrumOptions {
        /// Interpolation/filtering mode.
        /// Cubic modes are currently not supported.
        Interp interp_mode{Interp::LINEAR};

        /// Maximum output frequency to consider, in cycle/pix.
        /// Values are clamped from 0 (DC) to 0.5 (Nyquist).
        /// Frequencies higher than this value are set to 0.
        f64 fftfreq_cutoff{0.5};
    };

    /// Transforms a 2d (r)fft(s).
    /// \details The input and output arrays should be 2d arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    ///
    /// \tparam REMAP           Remap operation. Only HC2HC and HC2H are currently supported.
    /// \tparam Matrix          Mat22<Coord> or an varray of that type.
    /// \tparam Shift           Vec2<Coord> or an varray of that type.
    /// \param[in] input        2d (r)fft(s), of type f32, f64, c32, c64, to transform.
    /// \param[out] output      Transformed 2d (r)fft(s).
    /// \param shape            BDHW logical shape of \p input and \p output.
    /// \param[in] inv_matrices 2x2 inverse HW rotation/scaling matrix.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S
    ///                         in real space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] post_shifts  2d real-space HW forward shift to apply (as phase shift) after the transformation.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If an empty array is entered or if \p input is real, it is ignored.
    /// \param options          Transformation options.
    ///
    /// \note This function is optimized for rightmost arrays. Passing anything else will likely result in a
    ///       significant performance loss.
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<noa::fft::RemapInterface REMAP,
             typename Input, typename Output, typename Matrix,
             typename Shift = Vec2<nt::mutable_value_type_twice_t<Matrix>>>
    requires ((nt::are_varray_of_real_v<Input, Output> or nt::are_varray_of_complex_v<Input, Output>) and
              nt::is_varray_of_mutable_v<Output> and
              guts::are_valid_transform_spectrum_2d_matrix_v<Matrix> and
              guts::are_valid_transform_spectrum_nd_shift_v<Shift, 2> and
              nt::are_almost_all_same_v<nt::value_type_twice_t<Matrix>, nt::value_type_twice_t<Shift>>)
    void transform_spectrum_2d(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const Matrix& inverse_matrices,
            const Shift& post_shifts = {},
            const TransformSpectrumOptions& options = {}
    ) {
        guts::check_parameters_transform_spectrum_nd<2>(input, output, shape, inverse_matrices, post_shifts);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            return guts::launch_transform_spectrum_nd<REMAP.remap, 2, i32>(
                    input, output, shape, inverse_matrices, options.interp_mode, options.fftfreq_cutoff);
        }
        guts::launch_transform_spectrum_nd<REMAP.remap, 2, i64>(
                input, output, shape, inverse_matrices, options.interp_mode, options.fftfreq_cutoff);
    }

    /// Transforms a 2d (r)fft(s).
    /// \note This functions has the same features and limitations as the overload taking arrays.
    /// \note options.interp_mode is ignored, and input.interp_mode() is used instead.
    template<noa::fft::RemapInterface REMAP,
             typename Value, typename Output, typename Matrix,
             typename Shift = Vec2<nt::mutable_value_type_twice_t<Matrix>>>
    requires (nt::are_varray_of_real_or_complex_v<Output> and
              nt::is_varray_of_mutable_v<Output> and
              (nt::are_real_v<Value, nt::value_type_t<Output>> or
               nt::are_complex_v<Value, nt::value_type_t<Output>>) and
              guts::are_valid_transform_spectrum_2d_matrix_v<Matrix> and
              guts::are_valid_transform_spectrum_nd_shift_v<Shift, 2> and
              nt::are_almost_all_same_v<nt::value_type_twice_t<Matrix>, nt::value_type_twice_t<Shift>>)
    void transform_spectrum_2d(
            const Texture<Value>& input,
            const Output& output,
            const Shape4<i64>& shape,
            const Matrix& inverse_matrices,
            const Shift& post_shifts = {},
            const TransformSpectrumOptions& options = {}
    ) {
        guts::check_parameters_transform_spectrum_nd<2>(input, output, shape, inverse_matrices, post_shifts);

        const Device device = output.device();
        if (device.is_cpu()) {
            guts::launch_transform_spectrum_nd<REMAP.remap, 2, i64>(
                    input, output, shape, inverse_matrices, post_shifts, input.interp_mode(), options.fftfreq_cutoff);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (not nt::is_any_v<Value, f32, c32>) {
                panic("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                if (ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
                    guts::launch_transform_spectrum_nd<REMAP.remap, 2, i32>(
                            input, output, shape, inverse_matrices, post_shifts, options.fftfreq_cutoff);
                } else {
                    guts::launch_transform_spectrum_nd<REMAP.remap, 2, i64>(
                            input, output, shape, inverse_matrices, post_shifts, options.fftfreq_cutoff);
                }
            }
            #else
            panic("No GPU backend detected");
            #endif
        }
    }

    /// Transforms a 3d (r)fft(s).
    /// \details The input and output arrays should be 3d arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    ///
    /// \tparam REMAP           Remap operation. Only HC2HC and HC2H are currently supported.
    /// \tparam Matrix          Mat33<Coord>, Quaternion<Coord>, or an varray of either of these types.
    /// \tparam Shift           Vec3<Coord> or an varray of that type.
    /// \param[in] input        3d (r)fft(s), of type f32, f64, c32, c64, to transform.
    /// \param[out] output      Transformed 3d (r)fft(s).
    /// \param shape            BDHW logical shape of \p input and \p output.
    /// \param[in] inv_matrices 3x3 inverse DHW rotation/scaling matrix or quaternion.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S
    ///                         in real space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] post_shifts  3d real-space DHW forward shift to apply (as phase shift) after the transformation.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If an empty array is entered or if \p input is real, it is ignored.
    /// \param options          Transformation options.
    ///
    /// \note This function is optimized for rightmost arrays. Passing anything else will likely result in a
    ///       significant performance loss.
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<noa::fft::RemapInterface REMAP,
             typename Input, typename Output, typename Matrix,
             typename Shift = Vec3<nt::mutable_value_type_twice_t<Matrix>>>
    requires ((nt::are_varray_of_real_v<Input, Output> or nt::are_varray_of_complex_v<Input, Output>) and
              nt::is_varray_of_mutable_v<Output> and
              guts::are_valid_transform_spectrum_3d_matrix_v<Matrix> and
              guts::are_valid_transform_spectrum_nd_shift_v<Shift, 3> and
              nt::are_almost_all_same_v<nt::value_type_twice_t<Matrix>, nt::value_type_twice_t<Shift>>)
    void transform_spectrum_3d(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const Matrix& inverse_matrices,
            const Shift& post_shifts = {},
            const TransformSpectrumOptions& options = {}
    ) {
        guts::check_parameters_transform_spectrum_nd<3>(input, output, shape, inverse_matrices, post_shifts);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            return guts::launch_transform_spectrum_nd<REMAP.remap, 3, i32>(
                    input, output, shape, inverse_matrices, options.interp_mode, options.fftfreq_cutoff);
        }
        guts::launch_transform_spectrum_nd<REMAP.remap, 3, i64>(
                input, output, shape, inverse_matrices, options.interp_mode, options.fftfreq_cutoff);
    }

    /// Transforms a 3d (r)fft(s).
    /// \note This functions has the same features and limitations as the overload taking arrays.
    /// \note options.interp_mode is ignored, and input.interp_mode() is used instead.
    template<noa::fft::RemapInterface REMAP,
             typename Value, typename Output, typename Matrix,
             typename Shift = Vec2<nt::mutable_value_type_twice_t<Matrix>>>
    requires (nt::are_varray_of_real_or_complex_v<Output> and
              nt::is_varray_of_mutable_v<Output> and
              (nt::are_real_v<Value, nt::value_type_t<Output>> or
               nt::are_complex_v<Value, nt::value_type_t<Output>>) and
              guts::are_valid_transform_spectrum_3d_matrix_v<Matrix> and
              guts::are_valid_transform_spectrum_nd_shift_v<Shift, 3> and
              nt::are_almost_all_same_v<nt::value_type_twice_t<Matrix>, nt::value_type_twice_t<Shift>>)
    void transform_spectrum_3d(
            const Texture<Value>& input,
            const Output& output,
            const Shape4<i64>& shape,
            const Matrix& inverse_matrices,
            const Shift& post_shifts = {},
            const TransformSpectrumOptions& options = {}
    ) {
        guts::check_parameters_transform_spectrum_nd<3>(input, output, shape, inverse_matrices, post_shifts);

        const Device device = output.device();
        if (device.is_cpu()) {
            guts::launch_transform_spectrum_nd<REMAP.remap, 3, i64>(
                    input, output, shape, inverse_matrices, post_shifts, input.interp_mode(), options.fftfreq_cutoff);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (not nt::is_any_v<Value, f32, c32>) {
                panic("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                if (ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
                    guts::launch_transform_spectrum_nd<REMAP.remap, 3, i32>(
                            input, output, shape, inverse_matrices, post_shifts, options.fftfreq_cutoff);
                } else {
                    guts::launch_transform_spectrum_nd<REMAP.remap, 3, i64>(
                            input, output, shape, inverse_matrices, post_shifts, options.fftfreq_cutoff);
                }
            }
            #else
            panic("No GPU backend detected");
            #endif
        }
    }
}
