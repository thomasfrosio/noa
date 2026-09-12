#pragma once

#include "noa/base/Mat.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"
#include "noa/xform/core/Interpolation.hpp"
#include "noa/xform/Transform.hpp"
#include "noa/xform/Utils.hpp"

namespace noa::xform::details {
    template<nt::any_of<f32, f64> T, typename X>
    constexpr auto project_vector(
        const X& xform,
        const Vec<T, 4>& vector
    ) -> Vec<T, 2> {
        if constexpr (nt::mat_of_shape<X, 4, 4>) {
            return Vec{
                dot(xform[1], vector),
                dot(xform[2], vector),
            };
        } else if constexpr (nt::mat_of_shape<X, 2, 4>) {
            return xform * vector;
        } else {
            static_assert(nt::always_false<X>);
        }
    }

    template<typename T>
    constexpr auto forward_projection_transform_vector(
        const Vec<T, 3>& image_coordinates,
        const Vec<T, 3>& projection_window_center,
        const Mat<T, 3, 4>& projection_matrix
    ) -> Vec<T, 3> {
        const auto projection_axis = projection_matrix.col(0);
        const auto projection_matrix_no_z = projection_matrix.filter_columns(1, 2, 3);

        // Retrieve the base plane (z, y, or x) that was used to compute the projection window.
        i32 i{};
        {
            T distance{std::numeric_limits<T>::max()};
            for (i32 j{}; j < 3; ++j) {
                if (abs(projection_axis[j]) > 0) {
                    auto tmp = abs(projection_window_center[j] / projection_axis[j]);
                    if (tmp < distance) {
                        i = j;
                        distance = tmp;
                    }
                }
            }
        }

        // Transform from image space to volume space.
        // Since we need the transformed yz-plane, extract the 0yx transformed vector from the matrix-vector product.
        Vec<T, 3> plane_0yx = projection_matrix_no_z * image_coordinates.pop_front().push_back(1);
        auto volume_coordinates = plane_0yx + projection_axis * image_coordinates[0];

        // Compute and add the distance along the projection axis,
        // from the yx-plane to the selected plane centered on the window center.
        // This is from https://en.m.wikipedia.org/wiki/Line-plane_intersection
        const T distance = (projection_window_center[i] - plane_0yx[i]) / projection_axis[i];
        volume_coordinates += projection_axis * distance;

        return volume_coordinates;
    }

    template<usize B, nt::sinteger Index,
             nt::interpolator_nd<2> Interpolator,
             nt::readable_nd<B + 3> Input,
             nt::writable_nd<B + 3> Output,
             nt::readable_nd<B + 1> Matrix>
    class BackwardProject {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using interpolator_type = Interpolator;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;

        using batched_matrix_type = Matrix;
        using matrix_type = nt::mutable_value_type_t<batched_matrix_type>;
        using coord_type = nt::value_type_t<matrix_type>;
        using coord_4d_type = Vec<coord_type, 4>;

        static_assert(nt::any_of<matrix_type, Mat<coord_type, 4, 4>, Mat<coord_type, 2, 4>>);

    public:
        constexpr BackwardProject(
            const input_type& input,
            const interpolator_type& interpolator,
            const output_type& output,
            const batched_matrix_type& inverse_matrices,
            index_type n_inputs,
            bool add_to_output
        ) :
            m_input(input),
            m_output(output),
            m_interpolator(interpolator),
            m_inverse_matrices(inverse_matrices),
            m_n_inputs(n_inputs),
            m_add_to_output(add_to_output) {}

        constexpr void operator()(const Vec<index_type, B + 3>& batched_indices) const {
            const auto& [batches, indices] = batched_indices.template split<B>();
            const auto output_coordinates = coord_4d_type::from_values(indices[0], indices[1], indices[2], 1);

            input_value_type value{};
            auto inverse_matrices = m_inverse_matrices[batches];
            auto input = m_input[batches];
            for (index_type i{}; i < m_n_inputs; ++i) {
                const auto input_coordinates = project_vector(inverse_matrices[i], output_coordinates);
                value += static_cast<output_value_type>(m_interpolator.get(input[i], input_coordinates));
            }

            auto& output = m_output[batched_indices];
            output = m_add_to_output ? output + value : value;
        }

    private:
        input_type m_input;
        output_type m_output;
        interpolator_type m_interpolator;
        batched_matrix_type m_inverse_matrices;
        index_type m_n_inputs;
        bool m_add_to_output;
    };

    template<usize B, nt::sinteger Index,
             nt::interpolator_nd<3> Interpolator,
             nt::readable_nd<B + 3> Input,
             nt::atomic_addable_nd<B + 3> Output,
             nt::readable_nd<B + 1> Matrix>
    class ForwardProject {
    public:
        using output_value_type = nt::value_type_t<Output>;
        using matrix_type = nt::mutable_value_type_t<Matrix>;
        using coord_type = nt::value_type_t<matrix_type>;

        static_assert(nt::any_of<matrix_type, Mat<coord_type, 4, 4>, Mat<coord_type, 3, 4>>);
        static_assert(Interpolator::BORDER == Border::ZERO);

    public:
        constexpr ForwardProject(
            const Input& input,
            const Interpolator& interpolator,
            const Output& output,
            const Shape<Index, 3>& volume_shape,
            const Matrix& batched_forward_matrices,
            Index projection_window_size
        ) :
            m_input(input),
            m_output(output),
            m_interpolator(interpolator),
            m_batched_forward_matrices(batched_forward_matrices),
            m_volume_center((volume_shape.vec / 2).template as<coord_type>()),
            m_projection_window_radius(projection_window_size / 2) {}

        // indices: (b..n,z,y,x)
        // For every pixel (y,x) of the forward projected output image n.
        // z is the extra dimension for the projection window (the longest diagonal).
        constexpr void operator()(nt::compute_handle auto& ch, const Vec<Index, B + 4>& indices) const {
            const auto batches_n = indices.template pop_back<3>();
            const auto& z = indices[B + 1];
            const auto& y = indices[B + 2];
            const auto& x = indices[B + 3];

            const auto affine = m_batched_forward_matrices[batches_n].filter_rows(0, 1, 2); // truncated
            const auto image_coordinates = Vec<coord_type, 3>::from_values(z - m_projection_window_radius, y, x);
            const auto volume_coordinates = forward_projection_transform_vector(
                image_coordinates, m_volume_center, affine);

            // The interpolator handles OOB coordinates using Border::ZERO, so we could skip that.
            // However, we do expect a significant number of cases where the volume_coordinates are OOB,
            // so try to shortcut here directly.
            if (is_interpolation_window_outbound<Interpolator::INTERP>(m_interpolator.shape(), volume_coordinates))
                return;

            const auto value = static_cast<output_value_type>(
                m_interpolator.get(m_input[batches_n.pop_back()], volume_coordinates));
            ch.grid().atomic_add(value, m_output, batches_n.push_back(Vec{y, x})); // remove z, sum along z
        }

    private:
        Input m_input;
        Output m_output;
        Interpolator m_interpolator;
        Matrix m_batched_forward_matrices;
        Vec<coord_type, 3> m_volume_center{};
        Index m_projection_window_radius{};
    };

    template<usize B, nt::sinteger Index,
             nt::interpolator_nd<2> Interpolator,
             nt::readable_nd<B + 3> Input,
             nt::atomic_addable_nd<B + 3> Output,
             nt::readable_nd<B + 1> BatchedInputMatrix,
             nt::readable_nd<B + 1> BatchedOutputMatrix>
    class BackwardForwardProject {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using interpolator_type = Interpolator;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using output_real_type = nt::value_type_t<output_value_type>;

        using batched_input_matrix_type = BatchedInputMatrix;
        using batched_output_matrix_type = BatchedOutputMatrix;
        using input_matrix_type = nt::mutable_value_type_t<batched_input_matrix_type>;
        using output_matrix_type = nt::mutable_value_type_t<batched_output_matrix_type>;
        using coord_type = nt::value_type_t<input_matrix_type>;
        using coord_3d_type = Vec<coord_type, 3>;
        using shape_3d_type = Shape<index_type, 3>;

        static_assert(nt::any_of<input_matrix_type, Mat<coord_type, 4, 4>, Mat<coord_type, 2, 4>>);
        static_assert(nt::any_of<output_matrix_type, Mat<coord_type, 4, 4>, Mat<coord_type, 3, 4>>);

    public:
        constexpr BackwardForwardProject(
            const input_type& input,
            const interpolator_type& interpolator,
            const output_type& output,
            const shape_3d_type& volume_shape,
            const batched_input_matrix_type& batched_backward_matrices,
            const batched_output_matrix_type& batched_forward_matrices,
            index_type projection_window_size,
            index_type n_inputs
        ) :
            m_input(input),
            m_interpolator(interpolator),
            m_output(output),
            m_batched_backward_matrices(batched_backward_matrices),
            m_batched_forward_matrices(batched_forward_matrices),
            m_volume_shape(volume_shape.vec.template as<coord_type>()),
            m_volume_center((volume_shape.vec / 2).template as<coord_type>()),
            m_projection_window_radius(projection_window_size / 2),
            m_n_input_images(n_inputs) {}

    public:
        // indices: (b..,n,z,y,x)
        // For every pixel (y,x) of the forward projected output image n.
        // z is the extra dimension for the projection window (the longest diagonal).
        constexpr void operator()(nt::compute_handle auto& ch, const Vec<index_type, B + 4>& indices) const {
            const auto batches_n = indices.template pop_back<3>();
            const auto& z = indices[B + 1];
            const auto& y = indices[B + 2];
            const auto& x = indices[B + 3];

            const auto affine = m_batched_forward_matrices[batches_n].filter_rows(0, 1, 2);
            const auto image_coordinates = coord_3d_type::from_values(z - m_projection_window_radius, y, x);
            const auto volume_coordinates = forward_projection_transform_vector(
                image_coordinates, m_volume_center, affine);

            // The interpolator handles OOB coordinates on the 2d plane (yx) using Border::ZERO, so we could skip that.
            // However, along the z, we need to stop the virtual volume somewhere. We could stop exactly at the volume
            // edge, but that creates a pixel aliasing at the edges in certain orientations. While this is probably
            // fine, instead, do a linear antialiasing for the edges by sampling one more than necessary on each side
            // and do a lerp to smooth it out correctly.
            if (volume_coordinates[0] < -1 or volume_coordinates[0] > m_volume_shape[0] or
                volume_coordinates[1] < -1 or volume_coordinates[1] > m_volume_shape[1] or
                volume_coordinates[2] < -1 or volume_coordinates[2] > m_volume_shape[2])
                return;

            // Sample the virtual volume (backprojection).
            auto batches = batches_n.pop_back();
            auto batched_backward_matrices = m_batched_backward_matrices[batches];
            auto input = m_input[batches];
            input_value_type value{};
            for (index_type i{}; i < m_n_input_images; ++i) {
                const auto input_coordinates = project_vector(
                    batched_backward_matrices[i], volume_coordinates.push_back(1));
                value += static_cast<output_value_type>(m_interpolator.get(input[i], input_coordinates));
            }

            // Smooth the volume edges using a linear weighting.
            for (usize j{}; j < 3; ++j) {
                if (volume_coordinates[j] < 0) {
                    const auto fraction = volume_coordinates[j] + 1;
                    value = value * static_cast<output_real_type>(fraction);
                } else if (volume_coordinates[j] > m_volume_shape[j] - 1) {
                    const auto fraction = volume_coordinates[j] - m_volume_shape[j] + 1;
                    value = value * static_cast<output_real_type>(1 - fraction);
                }
            }

            ch.grid().atomic_add(value, m_output, batches_n.push_back(Vec{y, x})); // remove z, sum along z
        }

    private:
        input_type m_input;
        interpolator_type m_interpolator;
        output_type m_output;
        batched_input_matrix_type m_batched_backward_matrices;
        batched_output_matrix_type m_batched_forward_matrices;
        coord_3d_type m_volume_shape{};
        coord_3d_type m_volume_center{};
        index_type m_projection_window_radius{};
        index_type m_n_input_images;
    };

    template<typename Input, typename Output,
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
        if constexpr (nt::array<Input>)
            check(not are_overlapped(input, output), "Input and output arrays should not overlap");
        else
            check(input.device().is_gpu() or not are_overlapped(input.cpu(), output), "The input and output arrays should not overlap");

        const Device device = input.device();
        check(device == output_device,
              "The arrays should be on the same device, but got input:device={} and output:device={}",
              device, output_device);

        // Check batch axes are compatible.
        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize R = 3;
        constexpr usize B = N - R;
        auto input_shape_b = input.shape().template pop_back<R>();
        auto output_shape_b = output.shape().template pop_back<R>();
        if constexpr (B >= 1) {
            for (usize i{}; i < B; ++i)
                input_shape_b[i] = input_shape_b[i] == 1 ? output_shape_b[i] : input_shape_b[i];
            check(input_shape_b == output_shape_b,
                  "The batch axes are not compatible, input:batches={}, output:batches={}",
                  input_shape_b, output_shape_b);
        }

        auto check_xform = [&](const auto& transform, const Shape<isize, B + 1>& required_shape, std::string_view name) {
            check(not transform.is_empty(), "{} should not be empty", name);
            check(transform.is_contiguous() and transform.shape() == required_shape,
                  "{} should be a contiguous vector with shape={}, but got {}:shape={}, {}:strides={}",
                  name, required_shape, name, transform.shape(), name, transform.strides());
            check(transform.device() == output_device, "{} should be on the compute device", name);
        };
        if constexpr (nt::array<BackwardTransform>)
            check_xform(backward_transforms, input.shape().template pop_back<2>(), "backward_projection_matrices");
        if constexpr (nt::array<ForwardTransform>)
            check_xform(forward_transforms, output.shape().template pop_back<2>(), "forward_projection_matrices");
    }

    template<typename Index, bool IS_GPU = false, typename Input, typename Output, typename Transform>
    void launch_backward_projection(Input&& input, Output&& output, Transform&& xform, auto options) {
        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize R = 3;
        constexpr usize B = N - R;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N, Index>;
        auto output_span = output.span().template as_index<Index>();
        auto output_accessor = output_accessor_t(output_span.get(), output_span.strides());
        auto xform_accessor = xform_into_accessor(xform);
        using xform_accessor_t = decltype(xform_accessor);

        auto n_images = static_cast<Index>(input.shape()[N - 3]); // (b..,n,h,w)

        if constexpr (nt::texture_decay<Input>)
            options.interp = input.interp();

        auto launch_iwise = [&](auto interp) {
            using coord_t = nt::mutable_value_type_twice_t<Transform>;
            auto result = prepare_interpolation_inputs<2, interp(), Border::ZERO, IS_GPU, Index, coord_t, false>(input);
            using interpolator_t = decltype(result)::interpolator_type;
            using accessor_t = decltype(result)::accessor_type;
            using op_t = BackwardProject<B, Index, interpolator_t, accessor_t, output_accessor_t, xform_accessor_t>;
            auto op = op_t(result.accessor, result.interpolator, output_accessor, xform_accessor,
                           n_images, options.add_to_output);

            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(output_span.shape(), output.device(), op,
               std::forward<Input>(input),
               std::forward<Output>(output),
               std::forward<Transform>(xform));
        };

        switch (options.interp) {
            case Interp::NEAREST:            return launch_iwise(WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_iwise(WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_iwise(WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_iwise(WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_iwise(WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_iwise(WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_iwise(WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_iwise(WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            default:                         panic("The interp mode {} is not supported", options.interp);
        }
    }

    template<typename Index, bool IS_GPU = false, typename Input, typename Output, typename Transform>
    void launch_forward_projection(
        Input&& input, Output&& output, Transform&& xform,
        isize projection_window_size, auto options
    ) {
        if (not options.add_to_output)
            fill(output, {}); // the operator adds to the output, so we need to initialize it in this case

        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize R = 3;
        constexpr usize B = N - R;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N, Index>;
        auto output_span = output.span().template as_index<Index>();
        auto output_accessor = output_accessor_t(output_span.get(), output_span.strides());
        auto xform_accessor = xform_into_accessor(xform);
        using xform_accessor_t = decltype(xform_accessor);
        auto volume_shape = input.shape().filter(N - 3, N - 2, N - 1).template as<Index>();
        auto dhw = Vec<Index, 3>::from_values(
            projection_window_size,
            output_span.shape()[N - 2],
            output_span.shape()[N - 3]
        );

        if constexpr (nt::texture_decay<Input>)
            options.interp = input.interp();

        auto launch_iwise = [&](auto interp) {
            using coord_t = nt::mutable_value_type_twice_t<Transform>;
            auto result = prepare_interpolation_inputs<3, interp(), Border::ZERO, IS_GPU, Index, coord_t, false>(input);
            using interpolator_t = decltype(result)::interpolator_type;
            using accessor_t = decltype(result)::accessor_type;
            using op_t = ForwardProject<B, Index, interpolator_t, accessor_t, output_accessor_t, xform_accessor_t>;
            auto op = op_t(
                result.accessor, result.interpolator, output_accessor,
                volume_shape, xform_accessor, dhw[0]
            );

            auto iwise_shape = output_span.shape().template pop_back<2>().push_back(dhw); // (b..,n,h,w)->(b..,n,z,h,w)
            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(iwise_shape, output.device(), op,
               std::forward<Input>(input),
               std::forward<Output>(output),
               std::forward<Transform>(xform));
        };

        switch (options.interp) {
            case Interp::NEAREST:            return launch_iwise(WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_iwise(WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_iwise(WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_iwise(WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_iwise(WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_iwise(WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_iwise(WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_iwise(WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            default:                         panic("The interp mode {} is not supported", options.interp);
        }
    }

    template<typename Index, bool IS_GPU = false, typename Input, typename Output,
             typename BackwardTransform, typename ForwardTransform>
    void launch_fused_projection(
        Input&& input, Output&& output, const Shape<isize, 3>& volume_shape,
        BackwardTransform&& backward_xform,
        ForwardTransform&& forward_xform,
        isize projection_window_size, auto options
    ) {
        if (not options.add_to_output)
            fill(output, {}); // the operator adds to the output, so we need to initialize it in this case

        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize R = 3;
        constexpr usize B = N - R;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N, Index>;
        auto output_span = output.span().template as_index<Index>();
        auto output_accessor = output_accessor_t(output_span.get(), output_span.strides());
        auto bwd_xform_accessor = xform_into_accessor(backward_xform);
        auto fwd_xform_accessor = xform_into_accessor(forward_xform);
        using bwd_xform_accessor_t = decltype(bwd_xform_accessor);
        using fwd_xform_accessor_t = decltype(fwd_xform_accessor);

        auto n_input_images = static_cast<Index>(input.shape()[N - 3]); // (b..,n,h,w)
        auto dhw = Vec<Index, 3>::from_values(
            projection_window_size,
            output_span.shape()[N - 2],
            output_span.shape()[N - 3]
        );

        if constexpr (nt::texture_decay<Input>)
            options.interp = input.interp();

        auto launch_iwise = [&](auto interp) {
            using coord_t = nt::mutable_value_type_twice_t<BackwardTransform>;
            auto result = prepare_interpolation_inputs<2, interp(), Border::ZERO, IS_GPU, Index, coord_t, false>(input);
            using interpolator_t = decltype(result)::interpolator_type;
            using accessor_t = decltype(result)::accessor_type;

            using op_t = BackwardForwardProject<
                B, Index, interpolator_t, accessor_t, output_accessor_t,
                bwd_xform_accessor_t, fwd_xform_accessor_t>;
            auto op = op_t(
                result.accessor, result.interpolator, output_accessor, volume_shape.as<Index>(),
                bwd_xform_accessor, fwd_xform_accessor, dhw[0], n_input_images
            );

            auto iwise_shape = output_span.shape().template pop_back<2>().push_back(dhw); // (b..,n,h,w)->(b..,n,z,h,w)
            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(iwise_shape, output.device(), op,
               std::forward<Input>(input),
               std::forward<Output>(output),
               std::forward<BackwardTransform>(backward_xform),
               std::forward<ForwardTransform>(forward_xform));
        };

        switch (options.interp) {
            case Interp::NEAREST:            return launch_iwise(WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_iwise(WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_iwise(WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_iwise(WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_iwise(WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_iwise(WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_iwise(WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_iwise(WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            default:                         panic("The interp mode {} is not supported", options.interp);
        }
    }

    template<typename T, typename Output, usize R,
             typename U = std::remove_reference_t<T>,
             typename V = nt::value_type_t<T>,
             usize N = nt::array_size_v<Output>>
    concept transform_affine_nd =
        nt::mat_of_shape<U, R, R + 1> or
        nt::mat_of_shape<U, R + 1, R + 1> or
        (nt::array_nd<U, N - 2> and (nt::mat_of_shape<V, R, R + 1> or nt::mat_of_shape<V, R + 1, R + 1>));

    template<typename T, typename Output, usize R,
             typename U = std::remove_reference_t<T>,
             typename V = nt::value_type_t<T>,
             usize N = nt::array_size_v<Output>>
    concept transform_projection_nd =
        nt::mat_of_shape<U, R - 1, R + 1> or
        nt::mat_of_shape<U, R + 1, R + 1> or
        (nt::array_nd<U, N - 2> and (nt::mat_of_shape<V, R - 1, R + 1> or nt::mat_of_shape<V, R + 1, R + 1>));

    template<typename Input, typename Output>
    concept projectable_input_output =
        nt::readable_array_or_texture_rd_decay<Input, 3> and
        nt::writable_array_decay<Output> and
        nt::real_or_complex<nt::value_type_t<Input>> and
        nt::compatible_types<nt::value_type_t<Input>, nt::value_type_t<Output>> and
        nt::array_size_v<Input> == nt::array_size_v<Output> and nt::array_size_v<Input> >= 3;

    template<typename Input, typename Output, typename Xform>
    concept backward_projectable =
        projectable_input_output<Input, Output> and
        transform_projection_nd<Xform, Output, 3>;

    template<typename Input, typename Output, typename Xform>
    concept forward_projectable =
        projectable_input_output<Input, Output> and
        transform_affine_nd<Xform, Output, 3>;

    template<typename Input, typename Output, typename InputXform, typename OutputXform>
    concept backward_and_forward_projectable =
        projectable_input_output<Input, Output> and
        transform_projection_nd<InputXform, Output, 3> and
        transform_affine_nd<OutputXform, Output, 3> and
        nt::almost_same_as<nt::value_type_twice_t<InputXform>, nt::value_type_twice_t<OutputXform>>;
}

namespace noa::xform {
    /// Computes the projection window size of (backward_and_)forward_project_3d functions.
    /// \details In theory, the forward projection operators need to integrate the volume along the projection axis.
    ///          In practice, only a section of the projection axis is computed. This section, referred to as the
    ///          projection window, is the segment of the projection axis within the volume that goes through its
    ///          center. A larger section can be provided, but this would result in computing the forward projection
    ///          for segments that are outside the volume (and thus equal to zero), which is a waste of compute.
    ///          In other words, this function computes the minimal projection window size that will be required to
    ///          integrate the volume. If multiple projection matrices are to be used at once, one should take the
    ///          maximum window size to ensure the volume is correctly projected along any of the projection axes.
    ///
    /// \param[in] volume_shape         DHW shape of the volume to forward project.
    /// \param[in] projection_matrix    Matrices defining the transformation from image to volume space.
    template<typename T, usize R> requires (R == 3 or R == 4)
    constexpr auto forward_projection_window_size(
        const Shape<isize, 3>& volume_shape,
        const Mat<T, R, 4>& projection_matrix
    ) -> isize {
        const auto projection_axis = projection_matrix.col(0);
        const auto projection_window_center = (volume_shape.vec / 2).as<f64>();

        auto distance_to_volume_edge = Vec<f64, 3>::from_value(std::numeric_limits<f64>::max());
        for (auto i: irange(3))
            if (abs(projection_axis[i]) > 0) // not parallel to the ith-plane
                distance_to_volume_edge[i] = abs(projection_window_center[i] / projection_axis[i]);

        const auto index = argmin(distance_to_volume_edge);
        const auto projection_window_radius = static_cast<isize>(ceil(distance_to_volume_edge[index]));
        return (projection_window_radius + 1) * 2 + 1;
    }

    struct ProjectionOptions {
        /// Interpolation method used to:
        /// - backward_project_3d: 2d interpolate backprojected images.
        /// - forward_project_3d: 3d interpolate the forward-projected volume.
        /// - backward_and_forward_project_3d: 2d interpolate the backprojected images making up the virtual volume.
        Interp interp{Interp::LINEAR};

        /// Whether the projected values should be added to the output, implying that the output is already initialized.
        /// Note: If false, (backward_and_)forward_project_3d need to zero-out the output first, so if the output
        ///       is already zeroed-out, this flag should be turned on.
        bool add_to_output{false};
    };

    /// Backward project 2d images into a 3d volume using real space backprojection.
    /// \tparam Transform:
    ///     Mat44, Mat24, or a array of these types.
    /// \param[in] input_images:
    ///     ((B..,)N,Hi,Wi) Input images to backproject.
    ///     Batch axes are automatically broadcast to the output batches.
    /// \param[out] output_volume:
    ///     ((B..,)D,Ho,Wo) Output volume.
    ///     The input and output can have different dimension sizes,
    ///     thus allowing to only render small regions of the projected output.
    /// \param[in] projection_matrices:
    ///     ((B..,)N) array of 4x4 or 2x4 (HW rows) matrices defining the transformation from volume to image space.
    ///     A single matrix can also be passed, in which case all images are assigned to it.
    ///     Affine matrices allows complete control on the projection center and axis.
    /// \param options:
    ///     Projection options.
    template<typename Input, typename Output, typename Transform>
        requires details::backward_projectable<Input, Output, Transform>
    void backward_project_3d(
        Input&& input_images,
        Output&& output_volume,
        Transform&& projection_matrices,
        ProjectionOptions options = {}
    ) {
        details::check_projection_parameters(input_images, output_volume, projection_matrices, {});

        if (output_volume.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(nd::is_accessor_access_safe<i32>(input_images.strides(), input_images.shape()) and
                      nd::is_accessor_access_safe<i32>(output_volume.strides(), output_volume.shape()),
                      "isize indexing not instantiated for GPU devices");
                details::launch_backward_projection<i32, true>(
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
        details::launch_backward_projection<isize>(
            std::forward<Input>(input_images),
            std::forward<Output>(output_volume),
            std::forward<Transform>(projection_matrices),
            options);
    }

    /// Forward project a 3d volume onto 2d images using real space backprojection.
    /// \tparam Transform:
    ///     Mat44, Mat34, or a array of these types.
    /// \param[in] input_volume:
    ///     ((B..,)D,Hi,Wi) Input volume to forward-project.
    ///     Batch axes are automatically broadcast to the output batches.
    /// \param[out] output_images:
    ///     ((B..,)N,Ho,Wo) Output projected images.
    ///     The input and output can have different dimension sizes,
    ///     thus allowing to only render small regions of the projected output.
    /// \param[in] projection_matrices:
    ///     ((B..,)N) array of 4x4 or 3x4 (zyx rows) matrices defining the transformation from image to volume space.
    ///     A single matrix can also be passed, in which case all images are assigned to it.
    ///     Affine matrices allows complete control on the projection center and axis.
    /// \param projection_window_size:
    ///     Size of the projection window, as defined by forward_projection_window_size.
    /// \param[in] options:
    ///     Projection options.
    template<typename Input, typename Output, typename Transform>
        requires details::forward_projectable<Input, Output, Transform>
    void forward_project_3d(
        Input&& input_volume,
        Output&& output_images,
        Transform&& projection_matrices,
        isize projection_window_size,
        const ProjectionOptions& options = {}
    ) {
        details::check_projection_parameters(input_volume, output_images, {}, projection_matrices);

        if (output_images.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(nd::is_accessor_access_safe<i32>(input_volume.strides(), input_volume.shape()) and
                      nd::is_accessor_access_safe<i32>(output_images.strides(), output_images.shape()),
                      "isize indexing not instantiated for GPU devices");
                details::launch_forward_projection<i32, true>(
                    std::forward<Input>(input_volume),
                    std::forward<Output>(output_images),
                    std::forward<Transform>(projection_matrices),
                    projection_window_size, options);
            }
            return;
            #else
            panic_no_gpu_backend();
            #endif
        }
        details::launch_forward_projection<isize>(
            std::forward<Input>(input_volume),
            std::forward<Output>(output_images),
            std::forward<Transform>(projection_matrices),
            projection_window_size, options);
    }

    /// Backward project 2d images into a 3d virtual volume and immediately forward project
    /// this volume onto 2d images, using real space backprojection.
    ///
    /// \tparam InputTransform:
    ///     Mat44, Mat24, or a array of these types.
    /// \tparam OutputTransform:
    ///     Mat44, Mat34, or a array of these types.
    /// \param[in] input_images:
    ///     ((B..,)Ni,Hi,Wi) Input images to backproject.
    /// \param[out] output_images:
    ///     ((B..,)No,Ho,Wo) Output projected images.
    ///     Note that the input and output can have different dimension sizes, thus allowing to
    ///     only render small regions of the projected output.
    /// \param[in] volume_shape:
    ///     Shape of the virtual volume (batch axes are ignored).
    ///     This is used to compute the size of the projection window, i.e., the longest diagonal of the volume,
    ///     so that the forward projection can traverse the entire volume in any direction.
    /// \param[in] backward_projection_matrices:
    ///     ((B..,)Ni) array of 4x4 or 2x4 (HW rows) matrices defining the transformation from image to volume space.
    ///     A single matrix can also be passed, in which case all images are assigned to it.
    /// \param[in] forward_projection_matrices:
    ///     ((B..,)No) array of 4x4 or 3x4 (DHW rows) matrices defining the transformation from volume to image space.
    ///     A single matrix can also be passed, in which case all images are assigned to it.
    /// \param projection_window_size:
    ///     Size of the projection window, as defined by forward_projection_window_size.
    /// \param[in] options:
    ///     Projection options.
    ///
    /// \note The edges of the virtual volume are handled differently than the 3d interpolation of the physical volume
    ///       in forward_project_3d. Indeed, the 3d interpolator can handle the edges of the physical volume directly,
    ///       but this operator cannot because the volume is rendered from the 2d interpolation of the input
    ///       images, resulting in an infinite depth. To remedy this issue, the operator only renders elements
    ///       within the volume_shape, plus adds a linear antialiasing at the edges to remove sharp edges. As such,
    ///       while the elements within the volume_shape are unaffected, due to this divergence in handling elements at
    ///       the edges, these output images can be slightly different from the forward_project_3d output images.
    template<typename Input, typename Output, typename InputTransform, typename OutputTransform>
        requires details::backward_and_forward_projectable<Input, Output, InputTransform, OutputTransform>
    void backward_and_forward_project_3d(
        Input&& input_images,
        Output&& output_images,
        const Shape<isize, 3>& volume_shape,
        InputTransform&& backward_projection_matrices,
        OutputTransform&& forward_projection_matrices,
        isize projection_window_size,
        const ProjectionOptions& options = {}
    ) {
        details::check_projection_parameters(
            input_images, output_images,
            backward_projection_matrices, forward_projection_matrices
        );

        if (output_images.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(nd::is_accessor_access_safe<i32>(input_images.strides(), input_images.shape()) and
                      nd::is_accessor_access_safe<i32>(output_images.strides(), output_images.shape()),
                      "isize indexing not instantiated for GPU devices");
                details::launch_fused_projection<i32, true>(
                    std::forward<Input>(input_images),
                    std::forward<Output>(output_images), volume_shape,
                    std::forward<InputTransform>(backward_projection_matrices),
                    std::forward<OutputTransform>(forward_projection_matrices),
                    projection_window_size, options);
            }
            return;
            #else
            panic_no_gpu_backend();
            #endif
        }
        details::launch_fused_projection<isize>(
            std::forward<Input>(input_images),
            std::forward<Output>(output_images), volume_shape,
            std::forward<InputTransform>(backward_projection_matrices),
            std::forward<OutputTransform>(forward_projection_matrices),
            projection_window_size, options);
    }
}
