#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/core/geometry/Project.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Interpolation.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::geometry::guts {
    enum class ProjectionType { BACKWARD, FORWARD, FUSED };

    template<ProjectionType TYPE,
             typename Input, typename Output,
             typename BackwardTransform = Empty,
             typename ForwardTransform = Empty>
    void check_projection_parameters(
        const Input& input, const Output& output,
        const BackwardTransform& backward_transforms,
        const ForwardTransform& forward_transforms
    ) {
        check(not output.is_empty(), "Empty array detected");
        const Device output_device = output.device();

        check(not input.is_empty(), "Empty array detected");
        if constexpr (nt::varray<Input>)
            check(not ni::are_overlapped(input, output), "Input and output arrays should not overlap");

        const Device device = input.device();
        check(device == output_device,
              "The arrays should be on the same device, but got input:device={} and output:device={}",
              device, output_device);

        constexpr std::string_view messages[]{
            "The input to backward project should be 2d images, but got input:shape={}",
            "The forward projected output should be 2d images, but got output:shape={}",
        };
        if constexpr (TYPE == ProjectionType::BACKWARD) {
            check(input.shape()[1] == 1, messages[0], input.shape());
            check(output.shape()[0] == 1, "A single 3d volume is expected, but got output:shape={}", output.shape());
        } else if constexpr (TYPE == ProjectionType::FORWARD) {
            check(input.shape()[0] == 1, "A single 3d volume is expected, but got input:shape={}", input.shape());
            check(output.shape()[1] == 1, messages[1], output.shape());
        } else {
            check(input.shape()[1] == 1, messages[0], input.shape());
            check(output.shape()[1] == 1, messages[1], output.shape());
        }

        auto check_transform = [&](const auto& transform, i64 required_size, std::string_view name) {
            check(not transform.is_empty(), "{} should not be empty", name);
            check(ni::is_contiguous_vector(transform) and transform.n_elements() == required_size,
                  "{} should be a contiguous vector with n_images={} elements, but got {}:shape={}, {}:strides={}",
                  name, required_size, name, transform.shape(), name, transform.strides());
            check(transform.device() == output_device, "{} should be on the compute device", name);
        };
        if constexpr (nt::varray<BackwardTransform>)
            check_transform(backward_transforms, input.shape()[0], "backward_projection_matrices");
        if constexpr (nt::varray<ForwardTransform>)
            check_transform(forward_transforms, output.shape()[0], "forward_projection_matrices");
    }

    template<typename Index, bool IS_GPU = false, typename Input, typename Output, typename Transform>
    void launch_backward_projection(Input&& input, Output&& output, Transform&& projection_matrices, auto& options) {
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 3, Index>;
        auto output_accessor = output_accessor_t(output.get(), output.strides().filter(1, 2, 3).template as<Index>());
        auto batched_projection_matrices = ng::to_batched_transform(projection_matrices);

        if constexpr (nt::texture_decay<Input>) {
            options.interp = input.interp();
            options.border = input.border();
            options.cvalue = input.cvalue();
        }

        auto launch_iwise = [&](auto interp, auto border) {
            using coord_t = nt::mutable_value_type_twice_t<Transform>;
            auto interpolator = ng::to_interpolator<2, interp(), border(), Index, coord_t, IS_GPU>(input, options.cvalue);
            using op_t = BackwardProject<Index, decltype(interpolator), output_accessor_t, decltype(batched_projection_matrices)>;
            auto op = op_t(interpolator, output_accessor, batched_projection_matrices,
                           static_cast<Index>(input.shape()[0]), options.add_to_output);

            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(output.shape().filter(1, 2, 3).template as<Index>(), output.device(), op,
               std::forward<Input>(input),
               std::forward<Output>(output),
               std::forward<Transform>(projection_matrices));
        };

        auto launch_border = [&](auto interp) {
            switch (options.border) {
                case Border::ZERO:      return launch_iwise(interp, ng::WrapBorder<Border::ZERO>{});
                case Border::VALUE:     return launch_iwise(interp, ng::WrapBorder<Border::VALUE>{});
                case Border::MIRROR:    return launch_iwise(interp, ng::WrapBorder<Border::MIRROR>{});
                case Border::REFLECT:   return launch_iwise(interp, ng::WrapBorder<Border::REFLECT>{});
                default:                panic("The border mode {} is not supported", options.border);
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
            default:                         panic("The interp mode {} is not supported", options.interp);
        }
    }

    template<typename Index, bool IS_GPU = false, typename Input, typename Output, typename Transform>
    void launch_forward_projection(Input&& input, Output&& output, Transform&& projection_matrices, auto& options) {
        if (options.add_to_output)
            fill(output, {}); // the operator adds to the output, so we need to initialize it in this case

        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 3, Index>;
        auto output_accessor = output_accessor_t(output.get(), output.strides().filter(0, 2, 3).template as<Index>());
        auto batched_projection_matrices = ng::to_batched_transform(projection_matrices);

        if constexpr (nt::texture_decay<Input>) {
            options.interp = input.interp();
            options.border = input.border();
            options.cvalue = input.cvalue();
        }

        auto launch_iwise = [&](auto interp, auto border) {
            using coord_t = nt::mutable_value_type_twice_t<Transform>;
            auto interpolator = ng::to_interpolator<3, interp(), border(), Index, coord_t, IS_GPU>(input, options.cvalue);
            using op_t = ForwardProject<Index, decltype(interpolator), output_accessor_t, decltype(batched_projection_matrices)>;
            auto op = op_t(interpolator, output_accessor, batched_projection_matrices);
            const auto longest_diagonal = op.set_volume_shape(input.shape().pop_front().template as<Index>());

            auto output_shape_3d = output.shape().filter(0, 2, 3).template as<Index>();
            auto iwise_shape = Shape{output_shape_3d[0], longest_diagonal, output_shape_3d[1], output_shape_3d[2]};

            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(iwise_shape, output.device(), op,
               std::forward<Input>(input),
               std::forward<Output>(output),
               std::forward<Transform>(projection_matrices));
        };

        auto launch_border = [&](auto interp) {
            switch (options.border) {
                case Border::ZERO:      return launch_iwise(interp, ng::WrapBorder<Border::ZERO>{});
                case Border::VALUE:     return launch_iwise(interp, ng::WrapBorder<Border::VALUE>{});
                case Border::MIRROR:    return launch_iwise(interp, ng::WrapBorder<Border::MIRROR>{});
                case Border::REFLECT:   return launch_iwise(interp, ng::WrapBorder<Border::REFLECT>{});
                default:                panic("The border mode {} is not supported", options.border);
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
            default:                         panic("The interp mode {} is not supported", options.interp);
        }
    }

    template<typename Index, bool IS_GPU = false, typename Input, typename Output,
             typename BackwardTransform, typename ForwardTransform>
    void launch_fused_projection(
        Input&& input, Output&& output, const Shape<i64, 3>& volume_shape,
        BackwardTransform&& backward_projection_matrices,
        ForwardTransform&& forward_projection_matrices,
        auto& options
    ) {
        if (options.add_to_output)
            fill(output, {}); // the operator adds to the output, so we need to initialize it in this case

        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 3, Index>;
        auto output_accessor = output_accessor_t(output.get(), output.strides().filter(0, 2, 3).template as<Index>());
        auto batched_backward_projection_matrices = ng::to_batched_transform(backward_projection_matrices);
        auto batched_forward_projection_matrices = ng::to_batched_transform(forward_projection_matrices);

        if constexpr (nt::texture_decay<Input>) {
            options.interp = input.interp();
            options.border = input.border();
            options.cvalue = input.cvalue();
        }

        auto launch_iwise = [&](auto interp, auto border) {
            using coord_t = nt::mutable_value_type_twice_t<BackwardTransform>;
            auto interpolator = ng::to_interpolator<2, interp(), border(), Index, coord_t, IS_GPU>(input, options.cvalue);

            using op_t = BackwardForwardProject<
                Index, decltype(interpolator), output_accessor_t,
                decltype(batched_backward_projection_matrices),
                decltype(batched_forward_projection_matrices)>;
            auto op = op_t(interpolator, output_accessor,
                           batched_backward_projection_matrices, batched_forward_projection_matrices,
                           input.shape()[0]);
            const auto longest_diagonal = op.set_volume_shape(volume_shape.as<Index>());

            auto output_shape_3d = output.shape().filter(0, 2, 3).template as<Index>();
            auto iwise_shape = Shape{output_shape_3d[0], longest_diagonal, output_shape_3d[1], output_shape_3d[2]};

            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(iwise_shape, output.device(), op,
               std::forward<Input>(input),
               std::forward<Output>(output),
               std::forward<Transform>(backward_projection_matrices),
               std::forward<Transform>(forward_projection_matrices));
        };

        auto launch_border = [&](auto interp) {
            switch (options.border) {
                case Border::ZERO:      return launch_iwise(interp, ng::WrapBorder<Border::ZERO>{});
                case Border::VALUE:     return launch_iwise(interp, ng::WrapBorder<Border::VALUE>{});
                case Border::MIRROR:    return launch_iwise(interp, ng::WrapBorder<Border::MIRROR>{});
                case Border::REFLECT:   return launch_iwise(interp, ng::WrapBorder<Border::REFLECT>{});
                default:                panic("The border mode {} is not supported", options.border);
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
            default:                         panic("The interp mode {} is not supported", options.interp);
        }
    }
}

namespace noa::geometry {
    template<typename T>
    struct ProjectionOptions {
        /// Interpolation method used to:
        /// - backward_project_3d: 2d interpolate backprojected images.
        /// - forward_project_3d: 3d interpolate the forward-projected volume.
        /// - backward_and_forward_project_3d: 2d interpolate the backprojected images making up the virtual volume.
        Interp interp{Interp::LINEAR};

        /// Border method.
        Border border{Border::ZERO};

        /// Constant value to use for out-of-bounds coordinates.
        /// Only used if the border is Border::VALUE.
        T cvalue{};

        /// Whether the (back)projection should be added to the output. This implies that the output is already set.
        /// Note: If false, (backward_and_)forward_project_3d need to zero-out the output first, so if the output
        ///       is already zeroed-out, this operation is redundant and this flag should be turned on.
        bool add_to_output{false};
    };

    /// Backward project 2d images into a 3d volume using real space backprojection.
    /// \tparam Transform               Mat44, Mat24, or a varray of these types.
    /// \param[in] input_images         Input images to backproject.
    /// \param[out] output_volume       Output volume.
    /// \param[in] projection_matrices  4x4 or 2x4 (y-x rows) matrices defining the transformation from
    ///                                 volume to image space. One or one per input image.
    /// \param options                  Additional options.
    /// \note Supporting affine matrices allows complete control on the projection center and axis.
    ///       Note that the input and output can have different dimension sizes, thus allowing to
    ///       only render small regions of the projected output.
    template<nt::varray_or_texture_decay_of_real_or_complex Input,
             nt::varray_decay_compatible_with<Input> Output,
             nt::transform_projection_nd<3> Transform>
    void backward_project_3d(
        Input&& input_images,
        Output&& output_volume,
        Transform&& projection_matrices,
        ProjectionOptions<nt::mutable_value_type_t<Input>> options = {}
    ) {
        guts::check_projection_parameters<guts::ProjectionType::BACKWARD>(
            input_images, output_volume, projection_matrices, {});

        if (output_volume.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(ng::is_accessor_access_safe<i32>(input_images.strides(), input_images.shape()) and
                      ng::is_accessor_access_safe<i32>(output_volume.strides(), output_volume.shape()),
                      "i64 indexing not instantiated for GPU devices");

                guts::launch_backward_projection<i32, true>(
                    std::forward<Input>(input_images),
                    std::forward<Output>(output_volume),
                    std::forward<Transform>(projection_matrices),
                    options);
            }
            return;
            #else
            panic_no_gpu_backend();
            #endif
        }
        guts::launch_backward_projection<i64>(
            std::forward<Input>(input_images),
            std::forward<Output>(output_volume),
            std::forward<Transform>(projection_matrices),
            options);
    }

    /// Forward project a 3d volume onto 2d images using real space backprojection.
    /// \tparam Transform               Mat44, Mat34, or a varray of these types.
    /// \param[in] input_volume         Input volume to forward-project.
    /// \param[out] output_images       Output projected images.
    /// \param[in] projection_matrices  4x4 or 3x4 (zyx rows) matrices defining the transformation from
    ///                                 image to volume space. One or one per input image.
    /// \param[in] options              Projection options.
    /// \note Supporting affine matrices allows complete control on the projection center and axis.
    ///       Note that the input and output can have different dimension sizes, thus allowing to
    ///       only render small regions of the projected output.
    template<nt::varray_or_texture_decay_of_real_or_complex Input,
             nt::varray_decay_compatible_with<Input> Output,
             nt::transform_affine_nd<3> Transform>
    void forward_project_3d(
        Input&& input_volume,
        Output&& output_images,
        Transform&& projection_matrices,
        const ProjectionOptions<nt::mutable_value_type_t<Input>>& options = {}
    ) {
        guts::check_projection_parameters<guts::ProjectionType::FORWARD>(
            input_volume, output_images, {}, projection_matrices);

        if (output_images.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(ng::is_accessor_access_safe<i32>(input_volume.strides(), input_volume.shape()) and
                      ng::is_accessor_access_safe<i32>(output_images.strides(), output_images.shape()),
                      "i64 indexing not instantiated for GPU devices");

                guts::launch_forward_projection<i32, true>(
                    std::forward<Input>(input_volume),
                    std::forward<Output>(output_images),
                    std::forward<Transform>(projection_matrices),
                    options);
            }
            return;
            #else
            panic_no_gpu_backend();
            #endif
        }
        guts::launch_forward_projection<i64>(
            std::forward<Input>(input_volume),
            std::forward<Output>(output_images),
            std::forward<Transform>(projection_matrices),
            options);
    }

    /// Backward project 2d images into a 3d virtual volume and immediately forward project
    /// this volume onto 2d images, using real space backprojection.
    ///
    /// \tparam InputTransform                  Mat44, Mat24, or a varray of these types.
    /// \tparam OutputTransform                 Mat44, Mat34, or a varray of these types.
    /// \param[in] input_images                 Input images to backproject.
    /// \param[out] output_images               Output projected images.
    /// \param[in] volume_shape                 Shape of the virtual volume (batch is ignored).
    ///                                         This is used to compute the size of the projection window,
    ///                                         i.e. the longest diagonal of the volume so that the forward projection
    ///                                         can traverse the entire volume in any direction.
    /// \param[in] backward_projection_matrices 4x4 or 2x4 (yx rows) matrices defining the transformation from
    ///                                         image to volume space. One or one per input image.
    /// \param[in] forward_projection_matrices  4x4 or 3x4 (zyx rows) matrices defining the transformation from
    ///                                         volume to image space. One or one per input image.
    /// \param[in] options                      Projection options.
    ///
    /// \note Supporting affine matrices allows complete control on the projection center and axis.
    ///       Note that the input and output can have different dimension sizes, thus allowing to
    ///       only render small regions of the projected output.
    template<nt::varray_or_texture_decay_of_real_or_complex Input,
             nt::varray_decay_compatible_with<Input> Output,
             nt::transform_projection_nd<3> InputTransform,
             nt::transform_affine_nd<3> OutputTransform>
    requires nt::almost_same_as<nt::value_type_twice_t<InputTransform>, nt::value_type_twice_t<OutputTransform>>
    void backward_and_forward_project_3d(
        Input&& input_images,
        Output&& output_images,
        const Shape<i64, 3>& volume_shape,
        InputTransform&& backward_projection_matrices,
        OutputTransform&& forward_projection_matrices,
        const ProjectionOptions<nt::mutable_value_type_t<Input>>& options = {}
    ) {
        guts::check_projection_parameters<guts::ProjectionType::FUSED>(
            input_images, output_images, backward_projection_matrices, forward_projection_matrices);

        if (output_images.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(ng::is_accessor_access_safe<i32>(input_images.strides(), input_images.shape()) and
                      ng::is_accessor_access_safe<i32>(output_images.strides(), output_images.shape()),
                      "i64 indexing not instantiated for GPU devices");

                guts::launch_fused_projection<i32, true>(
                    std::forward<Input>(input_images),
                    std::forward<Output>(output_images), volume_shape,
                    std::forward<InputTransform>(backward_projection_matrices),
                    std::forward<OutputTransform>(forward_projection_matrices),
                    options);
            }
            return;
            #else
            panic_no_gpu_backend();
            #endif
        }
        guts::launch_fused_projection<i64>(
            std::forward<Input>(input_images),
            std::forward<Output>(output_images), volume_shape,
            std::forward<InputTransform>(backward_projection_matrices),
            std::forward<OutputTransform>(forward_projection_matrices),
            options);
    }
}
