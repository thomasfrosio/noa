#pragma once

#include "noa/core/fft/Remap.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"
#include "noa/unified/Utilities.hpp"
#include "noa/unified/Iwise.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Interpolator.hpp"
#endif

namespace noa::geometry::guts {
    template<int32_t NDIM, typename Input, typename Output, typename Matrix>
    void check_parameters_transform_nd(const Input& input, const Output& output, const Matrix& matrix) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(NDIM == 3 or (input.shape()[1] == 1 and output.shape()[1] == 1),
              "The input and output arrays should be 2d, but got input:shape={}, output:shape={}",
              input.shape(), output.shape());
        check(input.shape()[0] == 1 or input.shape()[0] == output.shape()[0],
              "The number of batch in the input ({}) is not compatible with the number of batch in the output ({})",
              input.shape()[0], output.shape()[0]);

        const Device device = output.device();

        if constexpr (nt::is_varray_v<Matrix>) {
            check(ni::is_contiguous_vector(matrix) and matrix.elements() == output.shape()[0],
                  "The number of matrices, specified as a contiguous vector, should be equal to the batch size "
                  "of the output, but got matrix:shape={}, matrix:strides={} and output:batch={}",
                  matrix.shape(), matrix.strides(), output.shape()[0]);
            check(device == matrix.device(),
                  "The transformation matrices should be on the same device as the output, "
                  "but got matrices:device={} and output:device={}", matrix.device(), device);
        }

        check(input.device() == device,
              "The input array/texture and output array must be on the same device, "
              "but got input:device={} and output:device={}",
              input.device(), device);
        check(ni::are_elements_unique(output.strides(), output.shape()),
              "The elements in the output should not overlap in memory, otherwise a data-race might occur. "
              "Got output:strides={} and output:shape={}",
              output.strides(), output.shape());

        if constexpr (nt::is_varray_v<Input>) {
            check(not ni::are_overlapped(input, output),
                  "The input and output arrays should not overlap");
        } else {
            check(input.device().is_gpu() or not ni::are_overlapped(input.view(), output),
                  "The input and output arrays should not overlap");
        }
    }

    template<typename Xform>
    auto extract_transform(const Xform& xform) {
        using value_t = nt::value_type_t<Xform>;
        if constexpr (nt::is_mat_v<Xform> or nt::is_vec_v<Xform> or nt::is_quaternion_v<Xform>) {
            return xform;
        } else if constexpr (nt::is_varray_v<Xform> and
                             (nt::is_mat_v<value_t> or nt::is_vec_v<value_t> or nt::is_quaternion_v<value_t>)) {
            using ptr_t = const value_t*;
            return ptr_t(xform.get());
        } else {
            static_assert(nt::always_false_v<Xform>);
        }
    }

    template<size_t N, typename Index, typename Input, typename Output, typename Matrix, typename Value>
    void launch_transform_nd(
            const Input& input, const Output& output, const Matrix& inv_matrices,
            Interp interp_mode, Border border_mode, Value cvalue
    ) {
        using input_accessor_t = AccessorRestrict<const nt::mutable_value_type_t<Input>, N + 1, Index>;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N + 1, Index>;
        using matrix_t = decltype(extract_transform(inv_matrices));

        auto get_strides_nd = [](const auto& array) {
            if constexpr (N == 2)
                return array.strides().filter(0, 2, 3).template as<Index>();
            else
                return array.strides().template as<Index>();
        };

        auto input_accessor = input_accessor_t(input.get(), get_strides_nd(input));
        auto output_accessor = output_accessor_t(output.get(), get_strides_nd(output));

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

        auto launch_for_each_border = [&]<Interp interp>{
            switch (border_mode) {
                case Border::ZERO: {
                    using interpolator_t = InterpolatorNd<N, Border::ZERO, interp, input_accessor_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto interpolator = interpolator_t(input_accessor, input_shape_nd, cvalue);
                    auto op = op_t(interpolator, output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), input, output, inv_matrices);
                }
                case Border::VALUE: {
                    using interpolator_t = InterpolatorNd<N, Border::VALUE, interp, input_accessor_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto interpolator = interpolator_t(input_accessor, input_shape_nd, cvalue);
                    auto op = op_t(interpolator, output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), input, output, inv_matrices);
                }
                case Border::CLAMP: {
                    using interpolator_t = InterpolatorNd<N, Border::CLAMP, interp, input_accessor_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto interpolator = interpolator_t(input_accessor, input_shape_nd, cvalue);
                    auto op = op_t(interpolator, output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), input, output, inv_matrices);
                }
                case Border::PERIODIC: {
                    using interpolator_t = InterpolatorNd<N, Border::PERIODIC, interp, input_accessor_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto interpolator = interpolator_t(input_accessor, input_shape_nd, cvalue);
                    auto op = op_t(interpolator, output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), input, output, inv_matrices);
                }
                case Border::MIRROR: {
                    using interpolator_t = InterpolatorNd<N, Border::MIRROR, interp, input_accessor_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto interpolator = interpolator_t(input_accessor, input_shape_nd, cvalue);
                    auto op = op_t(interpolator, output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), input, output, inv_matrices);
                }
                case Border::REFLECT: {
                    using interpolator_t = InterpolatorNd<N, Border::REFLECT, interp, input_accessor_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto interpolator = interpolator_t(input_accessor, input_shape_nd, cvalue);
                    auto op = op_t(interpolator, output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), input, output, inv_matrices);
                }
                default:
                    panic("The border/addressing mode {} is not supported", border_mode);
            }
        };

        switch (interp_mode) {
            case Interp::NEAREST:
                return launch_for_each_border.template operator()<Interp::NEAREST>();
            case Interp::LINEAR:
            case Interp::LINEAR_FAST:
                return launch_for_each_border.template operator()<Interp::LINEAR>();
            case Interp::COSINE:
            case Interp::COSINE_FAST:
                return launch_for_each_border.template operator()<Interp::COSINE>();
            case Interp::CUBIC:
                return launch_for_each_border.template operator()<Interp::CUBIC>();
            case Interp::CUBIC_BSPLINE:
            case Interp::CUBIC_BSPLINE_FAST:
                return launch_for_each_border.template operator()<Interp::CUBIC_BSPLINE>();
        }
    }

#ifdef NOA_ENABLE_CUDA
    template<size_t N, typename Index, typename Input, typename Output, typename Matrix, typename Value>
    void launch_transform_nd(
            const Texture<Value>& input, const Output& output, const Matrix& inv_matrices
    ) {
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N + 1, Index>;
        using matrix_t = decltype(extract_transform(inv_matrices));

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
        constexpr bool is_layered = N == 2;

        if (input.border_mode() == Border::PERIODIC or input.border_mode() == Border::MIRROR) { // normalized coords
            const auto f_shape = [&input] {
                if constexpr (N == 2)
                    return input.shape().filter(2, 3).vec.template as<coord_t>();
                else
                    return input.shape().filter(1, 2, 3).vec.template as<coord_t>();
            }();
            if (input.interp_mode() == Interp::NEAREST) {
                using interpolator_t = InterpolatorNd<N, Interp::NEAREST, Value, true, is_layered, coord_t>;
                using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                auto interpolator = interpolator_t(texture, f_shape);
                auto op = op_t(interpolator, output_accessor, extract_transform(inv_matrices));
                return iwise(output_shape, output.device(), std::move(op), cuda_texture, output, inv_matrices);

            } else if (input.interp_mode() == Interp::LINEAR_FAST) {
                using interpolator_t = InterpolatorNd<N, Interp::LINEAR_FAST, Value, true, is_layered, coord_t>;
                using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                auto interpolator = interpolator_t(texture, f_shape);
                auto op = op_t(interpolator, output_accessor, extract_transform(inv_matrices));
                return iwise(output_shape, output.device(), std::move(op), cuda_texture, output, inv_matrices);

            } else {
                panic("{} is not supported with {}", input.interp_mode(), input.border_mode());
            }
        } else {
            switch (input.interp_mode()) {
                case Interp::NEAREST: {
                    using interpolator_t = InterpolatorNd<N, Interp::NEAREST, Value, false, is_layered, coord_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), cuda_texture, output, inv_matrices);
                }
                case Interp::LINEAR: {
                    using interpolator_t = InterpolatorNd<N, Interp::LINEAR, Value, false, is_layered, coord_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), cuda_texture, output, inv_matrices);
                }
                case Interp::COSINE: {
                    using interpolator_t = InterpolatorNd<N, Interp::COSINE, Value, false, is_layered, coord_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), cuda_texture, output, inv_matrices);
                }
                case Interp::CUBIC: {
                    using interpolator_t = InterpolatorNd<N, Interp::CUBIC, Value, false, is_layered, coord_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), cuda_texture, output, inv_matrices);
                }
                case Interp::CUBIC_BSPLINE: {
                    using interpolator_t = InterpolatorNd<N, Interp::CUBIC_BSPLINE, Value, false, is_layered, coord_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), cuda_texture, output, inv_matrices);
                }
                case Interp::LINEAR_FAST: {
                    using interpolator_t = InterpolatorNd<N, Interp::LINEAR_FAST, Value, false, is_layered, coord_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), cuda_texture, output, inv_matrices);
                }
                case Interp::COSINE_FAST: {
                    using interpolator_t = InterpolatorNd<N, Interp::COSINE_FAST, Value, false, is_layered, coord_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), cuda_texture, output, inv_matrices);
                }
                case Interp::CUBIC_BSPLINE_FAST: {
                    using interpolator_t = InterpolatorNd<N, Interp::CUBIC_BSPLINE_FAST, Value, false, is_layered, coord_t>;
                    using op_t = Transform<N, Index, matrix_t, interpolator_t, output_accessor_t>;
                    auto op = op_t(interpolator_t(texture), output_accessor, extract_transform(inv_matrices));
                    return iwise(output_shape, output.device(), std::move(op), cuda_texture, output, inv_matrices);
                }
            }
        }
    }
#endif

    template<typename Matrix>
    constexpr bool are_valid_transform_2d_matrix_v =
            nt::is_mat23_v<Matrix> or nt::is_mat33_v<Matrix> or
            (nt::is_varray_v<Matrix> and
             (nt::is_mat23_v<nt::value_type_t<Matrix>> or nt::is_mat33_v<nt::value_type_t<Matrix>>));

    template<typename Matrix>
    constexpr bool are_valid_transform_3d_matrix_v =
            nt::is_mat34_v<Matrix> or nt::is_mat44_v<Matrix> or
            (nt::is_varray_v<Matrix> and
             (nt::is_mat34_v<nt::value_type_t<Matrix>> or nt::is_mat44_v<nt::value_type_t<Matrix>>));
}

namespace noa::geometry {
    template<typename T>
    struct TransformOptions {
        Interp interp_mode = Interp::LINEAR;
        Border border_mode = Border::ZERO;

        /// Constant value to use for out-of-bounds coordinates.
        /// Only used if the border_mode is Border::VALUE.
        T value{};
    };

    /// Applies one or multiple 2d affine transforms.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 2d arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    ///
    /// \param[in] input            Input 2d array.
    /// \param[out] output          Output 2d array.
    /// \param[in] inverse_matrices 2x3 or 3x3 inverse HW affine matrices.
    ///                             One, or if an array is entered, one per output batch.
    /// \param options              Interpolation and border options.
    ///
    /// \note The floating-point precision of the transformation and interpolation is set by \p inverse_matrices.
    /// \note This function is optimized for rightmost arrays. Passing anything else will likely result in a
    ///       significant performance loss.
    /// \see The core operator performing the index-wise transformation is noa::geometry::Transform.
    template<typename Input, typename Output, typename Matrix, typename CValue = nt::value_type_t<Input>>
    requires ((nt::are_varray_of_real_v<Input, Output> or nt::are_varray_of_complex_v<Input, Output>) and
              nt::is_varray_of_mutable_v<Output> and guts::are_valid_transform_2d_matrix_v<Matrix>)
    void transform_2d(
            const Input& input,
            const Output& output,
            const Matrix& inverse_matrices,
            const TransformOptions<CValue>& options = {}
    ) {
        guts::check_parameters_transform_nd<2>(input, output, inverse_matrices);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            return guts::launch_transform_nd<2, i32>(
                    input, output, inverse_matrices,
                    options.interp_mode, options.border_mode, options.value);
        }
        guts::launch_transform_nd<2, i64>(
                input, output, inverse_matrices,
                options.interp_mode, options.border_mode, options.value);
    }

    /// Applies one or multiple 2d affine transforms.
    /// This overload has the same features and limitations as the overload taking Arrays.
    template<typename Value, typename Output, typename Matrix>
    requires (nt::are_varray_of_real_or_complex_v<Output> and
              nt::is_varray_of_mutable_v<Output> and
              (nt::are_real_v<Value, nt::value_type_t<Output>> or
               nt::are_complex_v<Value, nt::value_type_t<Output>>) and
              guts::are_valid_transform_2d_matrix_v<Matrix>)
    void transform_2d(const Texture<Value>& input, const Output& output, const Matrix& inverse_matrices) {
        guts::check_parameters_transform_nd<2>(input, output, inverse_matrices);

        const Device device = output.device();
        if (device.is_cpu()) {
            guts::launch_transform_nd<2, i64>(
                    input, output, inverse_matrices,
                    input.interp_mode(), input.border_mode(), input.cvalue());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (not nt::is_any_v<Value, f32, c32>) {
                panic("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                if (ng::is_accessor_access_safe<i32>(output.strides(), output.shape()))
                    guts::launch_transform_nd<2, i32>(input, output, inverse_matrices);
                else
                    guts::launch_transform_nd<2, i64>(input, output, inverse_matrices);
            }
            #else
            panic("No GPU backend detected");
            #endif
        }
    }

    /// Applies one or multiple 3d affine transforms.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 3d arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    ///
    /// \param[in] input            Input 3d array.
    /// \param[out] output          Output 3d array.
    /// \param[in] inverse_matrices 3x4 or 4x4 inverse DHW affine matrices.
    ///                             One, or if an array is entered, one per output batch.
    /// \param options              Interpolation and border options.
    ///
    /// \note The floating-point precision of the transformation and interpolation is set by \p inverse_matrices.
    /// \note This function is optimized for rightmost arrays. Passing anything else will likely result in a
    ///       significant performance loss.
    /// \see The core operator performing the index-wise transformation is noa::geometry::Transform.
    template<typename Input, typename Output, typename Matrix, typename CValue = nt::value_type_t<Input>>
    requires ((nt::are_varray_of_real_v<Input, Output> or nt::are_varray_of_complex_v<Input, Output>) and
              nt::is_varray_of_mutable_v<Output> and
              guts::are_valid_transform_3d_matrix_v<Matrix>)
    void transform_3d(
            const Input& input,
            const Output& output,
            const Matrix& inverse_matrices,
            const TransformOptions<CValue>& options = {}
    ) {
        guts::check_parameters_transform_nd<3>(input, output, inverse_matrices);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            return guts::launch_transform_nd<3, i32>(
                    input, output, inverse_matrices,
                    options.interp_mode, options.border_mode, options.value);
        }
        guts::launch_transform_nd<3, i64>(
                input, output, inverse_matrices,
                options.interp_mode, options.border_mode, options.value);
    }

    /// Applies one or multiple 3d affine transforms.
    /// This overload has the same features and limitations as the overload taking Arrays.
    template<typename Value, typename Output, typename Matrix>
    requires (nt::are_varray_of_real_or_complex_v<Output> and
              nt::is_varray_of_mutable_v<Output> and
              (nt::are_real_v<Value, nt::value_type_t<Output>> or
               nt::are_complex_v<Value, nt::value_type_t<Output>>) and
              guts::are_valid_transform_3d_matrix_v<Matrix>)
    void transform_3d(const Texture<Value>& input, const Output& output, const Matrix& inverse_matrices) {
        guts::check_parameters_transform_nd<3>(input, output, inverse_matrices);

        const Device device = output.device();
        if (device.is_cpu()) {
            guts::launch_transform_nd<3, i64>(
                    input, output, inverse_matrices,
                    input.interp_mode(), input.border_mode(), input.cvalue());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (not nt::is_any_v<Value, f32, c32>) {
                panic("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                if (ng::is_accessor_access_safe<i32>(output.strides(), output.shape()))
                    guts::launch_transform_nd<3, i32>(input, output, inverse_matrices);
                else
                    guts::launch_transform_nd<3, i64>(input, output, inverse_matrices);
            }
            #else
            panic("No GPU backend detected");
            #endif
        }
    }
}
