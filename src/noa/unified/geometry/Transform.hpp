#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Interpolation.hpp"
#include "noa/unified/Utilities.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::geometry::guts {
    template<size_t N, typename Input, typename Output, typename Matrix>
    void check_parameters_transform_nd(const Input& input, const Output& output, const Matrix& matrix) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(N == 3 or (input.shape()[1] == 1 and output.shape()[1] == 1),
              "The input and output arrays should be 2d, but got input:shape={}, output:shape={}",
              input.shape(), output.shape());
        check(input.shape()[0] == 1 or input.shape()[0] == output.shape()[0],
              "The batch size in the input ({}) is not compatible with the batch size in the output ({})",
              input.shape()[0], output.shape()[0]);

        const Device device = output.device();

        if constexpr (nt::varray<Matrix>) {
            check(ni::is_contiguous_vector(matrix) and matrix.n_elements() == output.shape()[0],
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

        if constexpr (nt::varray<Input>) {
            check(not ni::are_overlapped(input, output),
                  "The input and output arrays should not overlap");
        } else {
            check(input.device().is_gpu() or not ni::are_overlapped(input.view(), output),
                  "The input and output arrays should not overlap");
        }
    }

    // GPU path instantiates 42 kernels with arrays and 54 kernels with textures...
    template<size_t N, typename Index, bool IS_GPU = false, typename Input, typename Output, typename Matrix>
    void launch_transform_nd(Input&& input, Output&& output, Matrix&& inverse_matrices, auto options) {
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N + 1, Index>;
        auto output_accessor = output_accessor_t(output.get(), output.strides().template filter_nd<N>().template as<Index>());
        auto batched_inverse_matrices = ng::to_batched_transform(inverse_matrices);

        if constexpr (nt::texture_decay<Input>) {
            options.interp = input.interp();
            options.border = input.border();
            options.cvalue = input.cvalue();
        }

        auto launch_iwise = [&](auto interp, auto border) {
            using coord_t = nt::mutable_value_type_twice_t<Matrix>;
            auto interpolator = ng::to_interpolator<N, interp(), border(), Index, coord_t, IS_GPU>(input, options.cvalue);
            using op_t = Transform<N, Index, decltype(batched_inverse_matrices), decltype(interpolator), output_accessor_t>;

            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(output.shape().template filter_nd<N>().template as<Index>(), output.device(),
               op_t(interpolator, output_accessor, batched_inverse_matrices),
               std::forward<Input>(input),
               std::forward<Output>(output),
               std::forward<Matrix>(inverse_matrices));
        };

        auto launch_border = [&](auto interp) {
            switch (options.border) {
                case Border::ZERO:      return launch_iwise(interp, ng::WrapBorder<Border::ZERO>{});
                case Border::VALUE:     return launch_iwise(interp, ng::WrapBorder<Border::VALUE>{});
                case Border::CLAMP:     return launch_iwise(interp, ng::WrapBorder<Border::CLAMP>{});
                case Border::PERIODIC:  return launch_iwise(interp, ng::WrapBorder<Border::PERIODIC>{});
                case Border::MIRROR:    return launch_iwise(interp, ng::WrapBorder<Border::MIRROR>{});
                case Border::REFLECT:   return launch_iwise(interp, ng::WrapBorder<Border::REFLECT>{});
                case Border::NOTHING:   panic("The border mode {} is not supported", Border::NOTHING);
            }
        };

        switch (options.interp) {
            case Interp::NEAREST:            return launch_border(ng::WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_border(ng::WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_border(ng::WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_border(ng::WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_border(ng::WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_border(ng::WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_border(ng::WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_border(ng::WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            case Interp::LANCZOS4:           return launch_border(ng::WrapInterp<Interp::LANCZOS4>{});
            case Interp::LANCZOS6:           return launch_border(ng::WrapInterp<Interp::LANCZOS6>{});
            case Interp::LANCZOS8:           return launch_border(ng::WrapInterp<Interp::LANCZOS8>{});
            case Interp::LANCZOS4_FAST:      return launch_border(ng::WrapInterp<Interp::LANCZOS4_FAST>{});
            case Interp::LANCZOS6_FAST:      return launch_border(ng::WrapInterp<Interp::LANCZOS6_FAST>{});
            case Interp::LANCZOS8_FAST:      return launch_border(ng::WrapInterp<Interp::LANCZOS8_FAST>{});
        }
    }
}

namespace noa::traits {
    template<typename T, size_t N, typename U = std::remove_reference_t<T>, typename V = value_type_t<T>>
    concept transform_parameter_nd =
        mat_of_shape<U, N, N + 1> or
        mat_of_shape<U, N + 1, N + 1> or
        (nt::varray<U> and (mat_of_shape<V, N, N + 1> or mat_of_shape<V, N + 1, N + 1>));
}

namespace noa::geometry {
    template<typename T>
    struct TransformOptions {
        /// Interpolation method. All interpolation modes are supported.
        /// This is ignored if the input is a texture.
        Interp interp{Interp::LINEAR};

        /// Border method.
        Border border{Border::ZERO};

        /// Constant value to use for out-of-bounds coordinates.
        /// Only used if the border is Border::VALUE.
        T cvalue{};
    };

    /// Applies one or multiple 2d affine transforms.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          as the input window, so by entering a translation in \p inverse_matrices, one can move the center
    ///          of the output window relative to the input window.
    /// \details The input and output arrays should be 2d arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (N input -> N output). Furthermore, if the input is not batched, it is broadcasted
    ///          to all output batches (1 input -> N output).
    ///
    /// \param[in] input            Input 2d array(s).
    /// \param[out] output          Output 2d array(s).
    /// \param[in] inverse_matrices 2x3 or 3x3 inverse HW affine matrices.
    ///                             One, or if an array is entered, one per output batch.
    ///                             Sets the floating-point precision of the transformation and interpolation.
    /// \param options              Interpolation and border options.
    template<nt::varray_or_texture_decay_of_real_or_complex Input,
             nt::writable_varray_decay_of_any<nt::mutable_value_type_t<Input>> Output,
             nt::transform_parameter_nd<2> Matrix>
    void transform_2d(
        Input&& input,
        Output&& output,
        Matrix&& inverse_matrices,
        const TransformOptions<nt::mutable_value_type_t<Input>>& options = {}
    ) {
        guts::check_parameters_transform_nd<2>(input, output, inverse_matrices);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
                      ng::is_accessor_access_safe<i32>(output.strides(), output.shape()),
                      "i64 indexing not instantiated for GPU devices");

                guts::launch_transform_nd<2, i32, true>(
                    std::forward<Input>(input),
                    std::forward<Output>(output),
                    std::forward<Matrix>(inverse_matrices),
                    options);
            }
            return;
            #else
            panic_no_gpu_backend();
            #endif
        }
        guts::launch_transform_nd<2, i64>(
            std::forward<Input>(input),
            std::forward<Output>(output),
            std::forward<Matrix>(inverse_matrices),
            options);
    }

    /// Applies one or multiple 3d affine transforms.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          as the input window, so by entering a translation in \p inverse_matrices, one can move the center
    ///          of the output window relative to the input window.
    /// \details The input and output arrays should be 3d arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (N input -> N output). Furthermore, if the input is not batched, it is broadcasted
    ///          to all output batches (1 input -> N output).
    ///
    /// \param[in] input            Input 3d array(s).
    /// \param[out] output          Output 3d array(s).
    /// \param[in] inverse_matrices 3x4 or 4x4 inverse DHW affine matrices.
    ///                             One, or if an array is entered, one per output batch.
    ///                             Sets the floating-point precision of the transformation and interpolation.
    /// \param options              Interpolation and border options.
    template<nt::varray_or_texture_decay_of_real_or_complex Input,
             nt::writable_varray_decay_of_any<nt::mutable_value_type_t<Input>> Output,
             nt::transform_parameter_nd<3> Matrix>
    void transform_3d(
        Input&& input,
        Output&& output,
        Matrix&& inverse_matrices,
        const TransformOptions<nt::mutable_value_type_t<Input>>& options = {}
    ) {
        guts::check_parameters_transform_nd<3>(input, output, inverse_matrices);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                if (ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
                    ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
                    guts::launch_transform_nd<3, i32, true>(
                        std::forward<Input>(input),
                        std::forward<Output>(output),
                        std::forward<Matrix>(inverse_matrices),
                        options);
                } else {
                    // For large volumes (>1290^3), i64 indexing is required.
                    guts::launch_transform_nd<3, i64, true>(
                        std::forward<Input>(input),
                        std::forward<Output>(output),
                        std::forward<Matrix>(inverse_matrices),
                        options);
                }
            }
            return;
            #else
            panic_no_gpu_backend();
            #endif
        }

        guts::launch_transform_nd<3, i64>(
            std::forward<Input>(input),
            std::forward<Output>(output),
            std::forward<Matrix>(inverse_matrices),
            options);
    }
}
