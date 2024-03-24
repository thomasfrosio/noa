#pragma once

#include "noa/core/geometry/Transform.hpp"
#include "noa/core/geometry/DrawShape.hpp"
#include "noa/core/Ewise.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/View.hpp"

namespace noa::geometry {
    template<size_t N>
    requires (N == 2 or N == 3)
    struct Ellipse {
        static constexpr size_t SIZE = N;

        /// (D)HW center of the ellipse.
        Vec<f64, N> center;

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
        Vec<f64, N> center;

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
        Vec<f64, N> center;

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
        static constexpr size_t SIZE = 2;

        /// DHW center of the cylinder.
        Vec3<f64> center;

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
    template<typename T> struct proclaim_is_ellipse : std::false_type {};
    template<typename T> struct proclaim_is_sphere : std::false_type {};
    template<typename T> struct proclaim_is_rectangle : std::false_type {};
    template<typename T> struct proclaim_is_cylinder : std::false_type {};
    template<size_t N> struct proclaim_is_ellipse<Ellipse<N>> : std::true_type {};
    template<size_t N> struct proclaim_is_sphere<Sphere<N>> : std::true_type {};
    template<size_t N> struct proclaim_is_rectangle<Rectangle<N>> : std::true_type {};
    template<> struct proclaim_is_cylinder<Cylinder> : std::true_type {};

    template<typename T> constexpr bool is_ellipse_v = proclaim_is_ellipse<std::decay_t<T>>::value;
    template<typename T> constexpr bool is_sphere_v = proclaim_is_sphere<std::decay_t<T>>::value;
    template<typename T> constexpr bool is_rectangle_v = proclaim_is_rectangle<std::decay_t<T>>::value;
    template<typename T> constexpr bool is_cylinder_v = proclaim_is_cylinder<std::decay_t<T>>::value;

    template<size_t N, typename Input, typename Output, typename Transform>
    auto check_draw_shape_parameters(
            const Input& input,
            const Output& output,
            const Transform transform
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

        if constexpr (nt::is_varray_v<Transform>) {
            check(not transform.is_empty(), "Empty array detected");
            check(transform.device() == device,
                  "The input and output arrays must be on the same device, but got transform:device={}, output:device={}",
                  transform.device(), device);
            check(ni::is_contiguous_vector(transform) and transform.elements() == output.shape()[0],
                  "The transforms (matrices or quaternions) should be specified as a contiguous vector of size {}, "
                  "but got transforms:shape={} and transforms:strides={}",
                  output.shape()[0], transform.shape(), transform.strides());
        }
    }

    template<typename Index, typename Input, typename Output, typename Shape, typename Transform, typename BinaryOp>
    void launch_draw_shape(
            const Input& input, const Output& output, const Shape& shape,
            const Transform& inverse_transform, const BinaryOp& binary_op

    ) {
        constexpr size_t N = Shape::SIZE;
        using input_value_t = nt::mutable_value_type_t<Input>;
        using input_real_t = nt::value_type_t<input_value_t>;
        using output_value_t = nt::value_type_t<Output>;

        // Prepare the input/output accessors.
        using input_accessor_t = Accessor<const input_value_t, N + 1, Index>;
        using output_accessor_t = Accessor<output_value_t, N + 1, Index>;
        auto get_strides = [](const auto& array) {
            if constexpr (N == 2)
                return array.strides().filter(0, 2, 3).template as<Index>();
            else
                return array.strides().template as<Index>();
        };
        auto input_accessor = input_accessor_t(input.get(), get_strides(input));
        auto output_accessor = output_accessor_t(output.get(), get_strides(output));

        // Broadcast the input to every output batch.
        if (input.shape()[0] == 1)
            input_accessor.strides()[0] = 0;

        // Skip the transform if possible.
        bool has_transform{}; // default: Empty
        if constexpr (nt::is_varray_v<Transform>)
            has_transform = not inverse_transform.is_empty();
        else if constexpr (nt::is_quaternion_v<Transform>)
            has_transform = inverse_transform != Transform{.z=0, .y=0, .x=0, .w=1};
        else if constexpr (nt::is_mat_v<Transform>)
            has_transform = inverse_transform != Transform::eye(1);

        // Wrap the transform if necessary.
        auto extract_transform = [&] {
            if constexpr ((N == 2 and nt::is_mat33_v<Transform>) or
                          (N == 3 and nt::is_mat44_v<Transform>)) {
                return AccessorValue(affine2truncated(inverse_transform));
            } else if constexpr (nt::is_varray_v<Transform>) {
                using value_t = Transform::mutable_value_type;
                using accessor_t = AccessorRestrictContiguousI64<const value_t, 1>;
                return accessor_t(inverse_transform.get());
            } else if constexpr (nt::is_mat_v<Transform> or nt::is_quaternion_v<Transform>) {
                return AccessorValue(inverse_transform);
            } else { // Empty
                return inverse_transform;
            }
        };

        // Get the drawing operator. Use the fp precision of the input,
        // fall back to double if input is not real or complex.
        using coord_t = std::conditional_t<nt::is_any_v<input_real_t, f32, f64>, input_real_t, f64>;
        auto extract_drawing_operator = [&]<bool is_smooth>() {
            auto cvalue = static_cast<coord_t>(shape.cvalue);
            auto center = shape.center.template as<coord_t>();
            auto smoothness = static_cast<coord_t>(shape.smoothness);
            if constexpr (is_ellipse_v<Shape>) {
                auto radius = shape.radius.template as<coord_t>();
                using ellipse_t = DrawEllipse<N, coord_t, is_smooth>;
                return ellipse_t(center, radius, cvalue, shape.invert, smoothness);
            } else if constexpr (is_sphere_v<Shape>) {
                auto radius = static_cast<coord_t>(shape.radius);
                using sphere_t = DrawSphere<N, coord_t, is_smooth>;
                return sphere_t(center, radius, cvalue, shape.invert, smoothness);
            } else if constexpr (is_rectangle_v<Shape>) {
                auto radius = shape.radius.template as<coord_t>();
                using rectangle_t = DrawRectangle<N, coord_t, is_smooth>;
                return rectangle_t(center, radius, cvalue, shape.invert, smoothness);
            } else if constexpr (is_cylinder_v<Shape>) {
                auto radius_length = Vec2<coord_t>::from_values(shape.radius, shape.length);
                using cylinder_t = DrawCylinder<coord_t, is_smooth>;
                return cylinder_t(center, radius_length, cvalue, shape.invert, smoothness);
            } else {
                static_assert(nt::always_false_v<Shape>);
            }
        };

        // Loop through every element of the output.
        auto iwise_shape = [&]() {
            if constexpr (N == 2)
                return output.shape().filter(0, 2, 3).template as<Index>();
            else
                return output.shape().template as<Index>();
        }();

        // Launch, with or without transformation.
        auto launch = [&]<typename T>(T draw_op) {
            if (has_transform) {
                using transform_t = decltype(extract_transform());
                using op_t = DrawShape<N, Index, T, transform_t, BinaryOp, input_accessor_t, output_accessor_t>;
                auto op = op_t(input_accessor, output_accessor, draw_op, extract_transform(), binary_op);
                return iwise(iwise_shape, output.device(), std::move(op));
            } else {
                using op_t = DrawShape<N, Index, T, Empty, BinaryOp, input_accessor_t, output_accessor_t>;
                auto op = op_t(input_accessor, output_accessor, draw_op, Empty{}, binary_op);
                return iwise(iwise_shape, output.device(), std::move(op));
            }
        };

        if (shape.smoothness > 1e-8) {
            launch(extract_drawing_operator.template operator()<true>());
        } else {
            launch(extract_drawing_operator.template operator()<false>());
        }
    }

    template<typename T, size_t N, typename Coord>
    constexpr bool is_valid_draw_transform_v =
            std::is_same_v<nt::value_type_t<T>, Coord> and
            ((N == 2 and (nt::is_mat22_v<T> or nt::is_mat23_v<T> or nt::is_mat33_v<T>)) or
             (N == 3 and (nt::is_mat33_v<T> or nt::is_mat34_v<T> or nt::is_mat44_v<T> or nt::is_quaternion_v<T>)));

    template<typename T>
    constexpr bool is_valid_draw_shape_v =
            is_ellipse_v<T> or is_sphere_v<T> or is_rectangle_v<T> or is_rectangle_v<T>;

    template<typename Output, typename Lhs, typename Rhs, typename BinaryOp>
    constexpr bool is_valid_draw_binary_op_v = std::is_empty_v<BinaryOp> or requires {
        static_cast<nt::value_type_t<Output>>(std::declval<BinaryOp>()(std::declval<Lhs>(), std::declval<Rhs>()));
    };

    template<typename Input, typename Output, typename Shape, typename Transform, typename BinaryOp,
             typename InputValue = nt::mutable_value_type_t<Input>,
             typename InputScalar = nt::mutable_value_type_twice_t<Input>,
             typename Coord = std::conditional_t<nt::is_real_v<InputScalar>, InputScalar, f64>,
             typename TransformValue = nt::value_type_t<Transform>>
    concept ValidDrawShape =
            nt::are_varray_of_numeric_v<Input, Output> and
            nt::is_varray_of_mutable_v<Output> and
            is_valid_draw_shape_v<Shape> and
            (std::is_empty_v<Transform> or is_valid_draw_transform_v<Transform, Shape::SIZE, Coord> or
             (nt::is_varray_v<Transform> and is_valid_draw_transform_v<TransformValue, Shape::SIZE, Coord>)) and
            is_valid_draw_binary_op_v<Output, InputValue, Coord, BinaryOp>;
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
    /// \param shape                Geometric shape to draw.
    /// \param inverse_transforms   Inverse (D)HW (affine) matrix/matrices or quaternion(s) to apply on the shape.
    ///                             For non-affine matrices and quaternion(s), the rotation center is the center of
    ///                             the shape. Note that the identity transformation (the default), as well as an
    ///                             empty array is explicitly checked and not passed to the backends.
    /// \param binary_op            Binary operator used to draw the shape onto the input values:
    ///                             (input_value, drawn_value) -> value ("value" is then casted to the actual output type).
    ///                             The default operator multiplies both values; if the input is an integer, drawn_value
    ///                             if first rounded to the nearest integer; if the input is complex and the output is
    ///                             real, abs(input_value)^2 (the power-spectrum) is first computed.
    ///                             This is ignored if \p input is empty.
    ///
    /// \note The floating-point precision of the computed geometric shape is set by the input value type (which
    ///       defaults to the output if no input is provided). If the input value type is not a (complex)
    ///       floating-point, f64 is used. If a transformation is provided, it should have the same floating-point type.
    ///
    /// \note This function is optimized for rightmost arrays.
    ///       Passing anything else will likely result in a significant performance loss.
    template<typename Output, typename Shape,
             typename Input = View<nt::value_type_t<Output>>,
             typename Transform = Empty,
             typename BinaryOp = Empty>
    requires guts::ValidDrawShape<Input, Output, Shape, Transform, BinaryOp>
    void draw_shape(
            const Input& input,
            const Output& output,
            const Shape& shape,
            const Transform& inverse_transforms = {},
            BinaryOp binary_op = {}
    ) {
        constexpr size_t N = Shape::SIZE;
        guts::check_draw_shape_parameters<N>(input, output, inverse_transforms);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            return guts::launch_draw_shape<i32>(input, output, shape, inverse_transforms, binary_op);
        }
        return guts::launch_draw_shape<i64>(input, output, shape, inverse_transforms, binary_op);
    }
}
