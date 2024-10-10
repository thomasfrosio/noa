#pragma once

#include "noa/core/geometry/Transform.hpp"
#include "noa/core/geometry/DrawShape.hpp"
#include "noa/core/utils/BatchedParameter.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/View.hpp"

namespace noa::geometry {
    template<size_t N>
    requires (N == 2 or N == 3)
    struct Ellipse {
        static constexpr size_t SIZE = N;

        /// (D)HW center of the ellipse.
        Vec<f64, N> center{};

        /// (D)HW radius of the ellipse.
        Vec<f64, N> radius;

        /// Size, in pixels, of the raised cosine smooth edge.
        /// This is the number pixels that will be used to compute the (1,0] taper range.
        f64 smoothness{0};

        /// Value of the pixels inside the object.
        /// Pixels outside the object are set to 0.
        f64 cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};
    };

    template<size_t N>
    requires (N == 2 or N == 3)
    struct Sphere {
        static constexpr size_t SIZE = N;

        /// (D)HW center of the sphere/circle.
        Vec<f64, N> center{};

        /// Radius of the sphere/circle.
        f64 radius;

        /// Size, in pixels, of the raised cosine smooth edge.
        /// This is the number pixels that will be used to compute the (1,0] taper range.
        f64 smoothness{0};

        /// Value of the pixels inside the object.
        /// Pixels outside the object are set to 0.
        f64 cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};
    };

    template<size_t N>
    requires (N == 2 or N == 3)
    struct Rectangle {
        static constexpr size_t SIZE = N;

        /// (D)HW center of the rectangle.
        Vec<f64, N> center{};

        /// (D)HW radius of the rectangle.
        Vec<f64, N> radius;

        /// Size, in pixels, of the raised cosine smooth edge.
        /// This is the number pixels that will be used to compute the (1,0] taper range.
        f64 smoothness{0};

        /// Value of the pixels inside the object.
        /// Pixels outside the object are set to 0.
        f64 cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};
    };

    struct Cylinder {
        static constexpr size_t SIZE = 3;

        /// DHW center of the cylinder.
        Vec3<f64> center{};

        /// Radius of the cylinder.
        f64 radius;

        /// Length of the cylinder.
        f64 length;

        /// Size, in pixels, of the raised cosine smooth edge.
        /// This is the number pixels that will be used to compute the (1,0] taper range.
        f64 smoothness{0};

        /// Value of the pixels inside the object.
        /// Pixels outside the object are set to 0.
        f64 cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};
    };
}

namespace noa::geometry::guts {
    template<size_t N, typename Input, typename Output, typename Transform>
    auto check_draw_shape_parameters(
        const Input& input,
        const Output& output,
        const Transform& transform
    ) {
        const Device device = output.device();
        check(output.shape().ndim() <= static_cast<i64>(N),
              "3d arrays are not supported with 2d geometric shapes. Use 3d overload to support 2d and 3d arrays");

        if (input.is_empty()) {
            check(not output.is_empty(), "Empty array detected");
        } else {
            check(not output.is_empty() and not input.is_empty(), "Empty array detected");
            check(N == 3 or (input.shape()[1] == 1 and output.shape()[1] == 1),
                  "The input and output arrays should be 2d, but got input:shape={}, output:shape={}",
                  input.shape(), output.shape());
            check(input.shape()[0] == 1 or input.shape()[0] == output.shape()[0],
                  "The number of batch in the input ({}) is not compatible with the number of batch in the output ({})",
                  input.shape()[0], output.shape()[0]);
            check(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);
        }

        if constexpr (nt::varray<Transform>) {
            check(not transform.is_empty(), "Empty array detected");
            check(transform.device() == device,
                  "The input and output arrays must be on the same device, but got transform:device={}, output:device={}",
                  transform.device(), device);
            check(ni::is_contiguous_vector(transform) and transform.n_elements() == output.shape()[0],
                  "The transforms (matrices or quaternions) should be specified as a contiguous vector of size {}, "
                  "but got transforms:shape={} and transforms:strides={}",
                  output.shape()[0], transform.shape(), transform.strides());
        }
    }

    template<typename T, size_t N, typename Coord>
    concept drawable_transform =
            std::same_as<nt::value_type_t<T>, Coord> and
            (nt::mat_of_shape<T, N, N> or
             nt::mat_of_shape<T, N, N + 1> or
             nt::mat_of_shape<T, N + 1, N + 1>
             or (N == 3 and nt::quaternion<T>));

    template<typename T>
    concept drawable_shape = nt::any_of<T, Ellipse<T::SIZE>, Sphere<T::SIZE>, Rectangle<T::SIZE>, Cylinder>;

    template<typename Output, typename Input, typename Draw, typename BinaryOp>
    concept drawable_binary_op = (nt::empty<BinaryOp> and nt::compatible_or_spectrum_types<Input, Output>) or
        requires (const BinaryOp op, Input lhs, Draw rhs) {
            static_cast<nt::value_type_t<Output>>(op(lhs, rhs));
        };

    template<typename Input, typename Transform, typename InputValue = nt::mutable_value_type_twice_t<Input>>
    using drawable_value_type_t =
        std::conditional_t<not nt::empty<std::decay_t<Transform>>, nt::value_type_twice_t<Transform>,
        std::conditional_t<nt::real_or_complex<InputValue>, InputValue, f64>>;

    template<typename Input, typename Output, typename Shape, typename Transform, typename BinaryOp,
             typename InputValue = nt::mutable_value_type_t<Input>,
             typename InputScalar = nt::mutable_value_type_twice_t<Input>,
             typename OutputValue = nt::value_type_t<Output>,
             typename DrawValue = drawable_value_type_t<Input, Transform>,
             typename TransformValue = nt::value_type_t<Transform>>
    concept drawable =
            nt::readable_varray_decay_of_numeric<Input> and
            nt::writable_varray_decay_of_numeric<Output> and
            drawable_shape<Shape> and
            drawable_binary_op<OutputValue, InputValue, DrawValue, BinaryOp> and
            (nt::empty<std::decay_t<Transform>> or
             drawable_transform<std::decay_t<Transform>, Shape::SIZE, DrawValue> or
             (nt::varray_decay<Transform> and drawable_transform<TransformValue, Shape::SIZE, DrawValue>));

    template<IwiseOptions OPTIONS,
             typename Index, typename Input, typename Output,
             typename Shape, typename Transform, typename BinaryOp>
    void launch_draw_shape(
        Input&& input,
        Output&& output,
        const Shape& geometric_shape,
        Transform&& inverse_transform,
        const BinaryOp& binary_op
    ) {
        constexpr size_t N = Shape::SIZE;

        // Prepare the input/output accessors.
        // TODO Reorder to rightmost? Only possible with zero or one transform.
        using input_accessor_t = Accessor<nt::const_value_type_t<Input>, N + 1, Index>;
        using output_accessor_t = Accessor<nt::value_type_t<Output>, N + 1, Index>;
        auto input_accessor = input_accessor_t(input.get(), input.strides().template filter_nd<N>().template as<Index>());
        auto output_accessor = output_accessor_t(output.get(), output.strides().template filter_nd<N>().template as<Index>());

        // Broadcast the input to every output batch.
        if (input.shape()[0] == 1)
            input_accessor.strides()[0] = 0;

        // Skip the transform if possible.
        using xform_t = std::decay_t<Transform>;
        bool has_transform{}; // default: Empty
        if constexpr (nt::varray<xform_t>)
            has_transform = not inverse_transform.is_empty();
        else if constexpr (nt::quaternion<xform_t>)
            has_transform = inverse_transform != xform_t{.z=0, .y=0, .x=0, .w=1};
        else if constexpr (nt::mat<xform_t>)
            has_transform = inverse_transform != xform_t::eye(1);
        else if constexpr (not nt::empty<xform_t>)
            static_assert(nt::always_false<>);

        // Get the drawing operator. Use the fp precision of the input,
        // fall back to double if input is not real or complex.
        using coord_t = drawable_value_type_t<Input, xform_t>;
        auto extract_drawing_operator = [&]<bool is_smooth>() {
            auto cvalue = static_cast<coord_t>(geometric_shape.cvalue);
            auto center = geometric_shape.center.template as<coord_t>();
            auto smoothness = static_cast<coord_t>(geometric_shape.smoothness);
            if constexpr (nt::is_same_v<Shape, Ellipse<N>>) {
                auto radius = geometric_shape.radius.template as<coord_t>();
                using ellipse_t = DrawEllipse<N, coord_t, is_smooth>;
                return ellipse_t(center, radius, cvalue, geometric_shape.invert, smoothness);
            } else if constexpr (nt::is_same_v<Shape, Sphere<N>>) {
                auto radius = static_cast<coord_t>(geometric_shape.radius);
                using sphere_t = DrawSphere<N, coord_t, is_smooth>;
                return sphere_t(center, radius, cvalue, geometric_shape.invert, smoothness);
            } else if constexpr (nt::is_same_v<Shape, Rectangle<N>>) {
                auto radius = geometric_shape.radius.template as<coord_t>();
                using rectangle_t = DrawRectangle<N, coord_t, is_smooth>;
                return rectangle_t(center, radius, cvalue, geometric_shape.invert, smoothness);
            } else if constexpr (nt::is_same_v<Shape, Cylinder>) {
                auto radius_length = Vec2<coord_t>::from_values(geometric_shape.radius, geometric_shape.length);
                using cylinder_t = DrawCylinder<coord_t, is_smooth>;
                return cylinder_t(center, radius_length, cvalue, geometric_shape.invert, smoothness);
            } else {
                static_assert(nt::always_false<>);
            }
        };

        // Launch, with or without transformation.
        auto launch = [&]<typename T>(T draw_op) {
            // Loop through every element of the output.
            auto iwise_shape = output.shape().template filter_nd<N>().template as<Index>();

            // Wrap the transform (only called if there is a transform).
            auto extract_transform = [&] {
                if constexpr (nt::mat_of_shape<xform_t, N + 1, N + 1>) {
                    return BatchedParameter{affine2truncated(inverse_transform)};
                } else if constexpr (nt::varray<xform_t>) {
                    using accessor_t = AccessorRestrictContiguousI64<nt::const_value_type_t<xform_t>, 1>;
                    return BatchedParameter{accessor_t(inverse_transform.get())}; // inverse_transform is contiguous
                } else if constexpr (nt::mat<xform_t> or nt::quaternion<xform_t> or nt::empty<xform_t>) {
                    return BatchedParameter{inverse_transform};
                } else {
                    static_assert(nt::always_false<>);
                }
            };

            if (has_transform) {
                using transform_t = decltype(extract_transform());
                using op_t = DrawShape<N, Index, T, BinaryOp, transform_t, input_accessor_t, output_accessor_t>;
                auto op = op_t(input_accessor, output_accessor, draw_op, extract_transform(), binary_op);
                return iwise<OPTIONS>(
                    iwise_shape, output.device(), std::move(op),
                    std::forward<Input>(input),
                    std::forward<Output>(output),
                    std::forward<Transform>(inverse_transform));
            } else {
                using op_t = DrawShape<N, Index, T, BinaryOp, BatchedParameter<Empty>, input_accessor_t, output_accessor_t>;
                auto op = op_t(input_accessor, output_accessor, draw_op, BatchedParameter<Empty>{}, binary_op);
                return iwise<OPTIONS>(
                    iwise_shape, output.device(), std::move(op),
                    std::forward<Input>(input),
                    std::forward<Output>(output),
                    std::forward<Transform>(inverse_transform));
            }
        };

        if (geometric_shape.smoothness > 1e-8) {
            launch(extract_drawing_operator.template operator()<true>());
        } else {
            launch(extract_drawing_operator.template operator()<false>());
        }
    }
}

namespace noa::geometry {
    /// Returns or draws a geometric shape (ellipse, sphere, cylinder or rectangle).
    /// \details The shape can be directly saved in \p output or applied (\p see binary_op) to \p input and
    ///          then saved in \p output. The same transformation can be applied to every batch or there can be
    ///          one transformation per batch (\p see inverse_transforms).
    ///
    /// \tparam Transform           2d case: Mat22, Mat23, Mat33 or an varray of either one of these types.
    ///                             3d case: Mat33, Mat34, Mat44, Quaternion or an varray of either one of these types.
    /// \param[in] input            2d or 3d array(s). If empty, write directly in \p output.
    /// \param[out] output          2d or 3d array(s). Can be equal to \p input for in-place drawing.
    /// \param geometric_shape      Geometric shape to draw.
    /// \param inverse_transforms   Optional inverse (D)HW (affine) matrix/matrices or quaternion(s) to apply to the
    ///                             shape to draw. For non-affine matrices and quaternion(s), the rotation center is
    ///                             the center of the shape. Note that passing Empty (the default), the identity
    ///                             transformation, or an empty array is valid and equivalent to no transformation.
    /// \param binary_op            Optional binary operator used to draw the shape onto the input values:
    ///                             (input_value, drawn_value) -> value ("value" is then casted to the actual output type).
    ///                             The default operator (i.e. passing Empty) multiplies both values; if the input is
    ///                             an integer, drawn_value if first rounded to the nearest integer; if the input is
    ///                             complex and the output is real, abs(input_value)^2 (the power-spectrum) is first
    ///                             computed. This is ignored if \p input is empty.
    ///
    /// \note The floating-point precision of the computed geometric shape is set by the inverse transform.
    ///       If no transforms are provided (Empty), the precision is the one of the input value type (which
    ///       defaults to the output if no input is provided). If the input value type is not a (complex)
    ///       floating-point, f64 is used.
    template<typename Output, typename Shape,
             typename Input = View<nt::value_type_t<Output>>,
             typename Transform = Empty,
             typename BinaryOp = Empty>
    requires guts::drawable<Input, Output, Shape, Transform, BinaryOp>
    void draw_shape(
        Input&& input,
        Output&& output,
        const Shape& geometric_shape,
        Transform&& inverse_transforms = {},
        BinaryOp binary_op = {}
    ) {
        constexpr size_t N = Shape::SIZE;
        guts::check_draw_shape_parameters<N>(input, output, inverse_transforms);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            #ifdef NOA_ENABLE_CUDA
            return guts::launch_draw_shape<IwiseOptions{.generate_cpu = false}, i32>(
                std::forward<Input>(input),
                std::forward<Output>(output),
                geometric_shape,
                std::forward<Transform>(inverse_transforms),
                binary_op);
            #else
            std::terminate(); // unreachable
            #endif
        }
        return guts::launch_draw_shape<IwiseOptions{}, i64>(
            std::forward<Input>(input),
            std::forward<Output>(output),
            geometric_shape,
            std::forward<Transform>(inverse_transforms),
            binary_op);
    }
}
