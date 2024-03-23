#pragma once

#include "noa/core/geometry/Transform.hpp"
#include "noa/core/geometry/Shape.hpp"
#include "noa/core/Ewise.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/View.hpp"

namespace noa::geometry::guts {
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

    template<size_t N, typename Index, typename Coord,
             typename DrawOp, typename DrawOpSmooth,
             typename Input, typename Output, typename Transform, typename BinaryOp, typename Radius, typename CValue>
    void launch_draw_shape(
            const Input& input, const Output& output, const Transform& inverse_transform, const BinaryOp& binary_op,
            const Vec<Coord, N>& center, const Radius& radius, Coord smoothness, CValue cvalue, bool invert
    ) {
        using input_accessor_t = Accessor<const nt::mutable_value_type_t<Input>, N + 1, Index>;
        using output_accessor_t = Accessor<nt::value_type_t<Output>, N + 1, Index>;

        auto get_strides = [](const auto& array) {
            if constexpr (N == 2)
                return array.strides().filter(0, 2, 3).template as<Index>();
            else
                return array.strides().template as<Index>();
        };

        auto input_accessor = input_accessor_t(input, get_strides(input));
        auto output_accessor = output_accessor_t(output, get_strides(output));
        auto shape = [&]() {
            if constexpr (N == 2)
                return output.shape().filter(0, 2, 3).template as<Index>();
            else
                return output.shape().template as<Index>();
        };

        // Broadcast the input to every output batch.
        if (input.shape()[0] == 1)
            input_accessor.strides()[0] = 0;

        bool has_transform{};
        if constexpr (nt::is_varray_v<Transform>)
            has_transform = not inverse_transform.is_empty();
        else if constexpr (nt::is_quaternion_v<Transform>)
            has_transform = inverse_transform != Transform{.z=0, .y=0, .x=0, .w=1};
        else
            has_transform = inverse_transform != Transform::eye(1);

        auto extract_transform = [&] {
            if constexpr ((N == 2 and nt::is_mat33_v<Transform>) or
                          (N == 3 and nt::is_mat44_v<Transform>)) {
                return affine2truncated(inverse_transform);
            } else if constexpr (nt::is_varray_v<Transform>) {
                using const_ptr_t = const Transform::mutable_value_type*;
                return const_ptr_t(inverse_transform.get());
            } else {
                return inverse_transform;
            }
        };

        auto launch_for_each_matrix = [&](auto draw_op) {
            if (has_transform) {
                using transform_t = decltype(extract_transform(inverse_transform));
                using op_t = DrawShape<2, Index, Coord, decltype(draw_op), transform_t, BinaryOp, input_accessor_t, output_accessor_t>;
                auto op = op_t(input_accessor, output_accessor, draw_op, extract_transform(inverse_transform), binary_op);
                return iwise(shape, output.device(), std::move(op));
            } else {
                using op_t = DrawShape<2, Index, Coord, decltype(draw_op), Empty, BinaryOp, input_accessor_t, output_accessor_t>;
                auto op = op_t(input_accessor, output_accessor, draw_op, Empty{}, binary_op);
                return iwise(shape, output.device(), std::move(op));
            }
        };

        if (smoothness > static_cast<Coord>(1e-8)) {
            launch_for_each_matrix(DrawOpSmooth(center, radius, smoothness, cvalue, invert));
        } else {
            launch_for_each_matrix(DrawOp(center, radius, cvalue, invert));
        }
    }

    template<typename T, size_t N>
    constexpr bool is_valid_draw_transform_v =
            (N == 2 and (nt::is_mat22_v<T> or nt::is_mat23_v<T> or nt::is_mat33_v<T>)) or
            (N == 3 and (nt::is_mat33_v<T> or nt::is_mat34_v<T> or nt::is_mat44_v<T> or nt::is_quaternion_v<T>));

    template<typename Output, typename Input, typename BinaryOp>
    constexpr bool is_valid_draw_binary_op_v = requires {
        static_cast<nt::value_type_t<Output>>(std::declval<BinaryOp>()(
                std::declval<nt::value_type_t<Input>>(),
                std::declval<nt::value_type_t<Input>>()
        ));
    };

    template<size_t N, typename Output, typename Input, typename CValue, typename Matrix, typename BinaryOp>
    concept ValidDrawShape =
            (nt::are_varray_of_scalar_v<Input, Output> or nt::are_varray_of_complex_v<Input, Output>) and
            nt::is_varray_of_mutable_v<Output> and
            std::is_same_v<CValue, nt::mutable_value_type_twice_t<Input>> and
            (is_valid_draw_transform_v<Matrix, 2> or
             (nt::is_varray_v<Matrix> and is_valid_draw_transform_v<nt::value_type_t<Matrix>, 2>)) and
            is_valid_draw_binary_op_v<Output, Input, BinaryOp>;
}

namespace noa::geometry {
    template<typename T, size_t N>
    requires (N == 2 or N == 3)
    struct Ellipse {
        /// (D)HW center of the ellipse.
        Vec<f64, N> center;

        /// (D)HW radius of the ellipse.
        Vec<f64, N> radius;

        /// Size, in pixels, of the raised cosine smooth edge.
        /// This is the number pixels that will be used to compute the (1,0] taper range.
        f64 smoothness{0};

        /// Value of the pixels inside the object.
        /// Pixels outside the object are set to 0.
        T cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};
    };

    /// Returns or draws an ellipse.
    /// \details The mask can be directly saved in \p output or applied (\p see binary_op) to \p input and
    ///          save in \p output. The same transformation can be applied to every batch or there can be
    ///          one transformation per batch (\p see inverse_matrices).
    ///
    /// \tparam Transform           2d case: Mat22, Mat23, Mat33 or an varray of either one of these types.
    ///                             3d case: Mat33, Mat34, Mat44, Quaternion or an varray of either one of these types.
    /// \param[in] input            2d or 3d array(s). If empty, write directly in \p output.
    /// \param[out] output          2d or 3d array(s). Can be equal to \p input for in-place drawing.
    /// \param ellipse              Ellipse to draw, with the \p input value type.
    /// \param inverse_transforms   Inverse (D)HW (affine) matrix/matrices or quaternion(s) to apply on the ellipse.
    ///                             For non-affine matrices and quaternion(s), the rotation center is the center of
    ///                             the ellipse. Note that the identity transformation (the default), as well as an
    ///                             empty array is explicitly checked and not passed to the backends.
    /// \param binary_op            Binary operator used to draw the ellipse onto the input values:
    ///                             (input_value, ellipse_value) -> value ("value" is casted to the actual output type).
    ///                             This is ignored if \p input is empty.
    ///
    /// \note The floating-point precision of the drawing operator is set by the transformation floating-point type,
    ///       which defaults to double precision.
    /// \note This function is optimized for rightmost arrays. Passing anything else will likely result in a
    ///       significant performance loss.
    template<size_t N,
             typename Output,
             typename Input = View<nt::value_type_t<Output>>,
             typename CValue = nt::mutable_value_type_twice_t<Input>,
             typename Transform = std::conditional_t<N == 2, Mat22<f64>, Mat33<f64>>,
             typename BinaryOp = Multiply>
    requires guts::ValidDrawShape<N, Output, Input, CValue, Transform, BinaryOp>
    void draw_ellipse(
            const Input& input,
            const Output& output,
            const Ellipse<CValue, N>& ellipse,
            const Transform& inverse_transforms = {},
            BinaryOp binary_op = {}
    ) {
        guts::check_draw_shape_parameters<N>(input, output, inverse_transforms);

        using input_value_t = nt::mutable_value_type_t<Input>;
        using coord_t = nt::mutable_value_type_twice_t<Transform>;
        using smooth_ellipse_t = DrawEllipse<N, input_value_t, coord_t, true>;
        using ellipse_t = DrawEllipse<N, input_value_t, coord_t, false>;
        auto center = ellipse.center.template as<coord_t>();
        auto radius = ellipse.radius.template as<coord_t>();
        auto smoothness = static_cast<coord_t>(ellipse.smoothness);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            return guts::launch_draw_shape<N, i32, coord_t, ellipse_t, smooth_ellipse_t>(
                    input, output, inverse_transforms, binary_op,
                    center, radius, smoothness, ellipse.cvalue, ellipse.invert);
        }
        return guts::launch_draw_shape<N, i64, coord_t, ellipse_t, smooth_ellipse_t>(
                input, output, inverse_transforms, binary_op,
                center, radius, smoothness, ellipse.cvalue, ellipse.invert);
    }

    template<typename T, size_t N>
    requires (N == 2 or N == 3)
    struct Sphere {
        /// (D)HW center of the sphere/circle.
        Vec<f64, N> center;

        /// Radius of the sphere/circle.
        f64 radius;

        /// Size, in pixels, of the raised cosine smooth edge.
        /// This is the number pixels that will be used to compute the (1,0] taper range.
        f64 smoothness{0};

        /// Value of the pixels inside the object.
        /// Pixels outside the object are set to 0.
        T cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};
    };

    /// Returns or draws a sphere.
    /// \details The mask can be directly saved in \p output or applied (\p see binary_op) to \p input and
    ///          save in \p output. The same transformation can be applied to every batch or there can be
    ///          one transformation per batch (\p see inverse_matrices).
    ///
    /// \tparam Transform           2d case: Mat22, Mat23, Mat33 or an varray of either one of these types.
    ///                             3d case: Mat33, Mat34, Mat44, Quaternion or an varray of either one of these types.
    /// \param[in] input            2d or 3d array(s). If empty, write directly in \p output.
    /// \param[out] output          2d or 3d array(s). Can be equal to \p input for in-place drawing.
    /// \param sphere               Sphere to draw, with the \p input value type.
    /// \param inverse_transforms   Inverse (D)HW (affine) matrix/matrices or quaternion(s) to apply on the sphere.
    ///                             For non-affine matrices and quaternion(s), the rotation center is the center of
    ///                             the sphere. Note that the identity transformation (the default), as well as an
    ///                             empty array is explicitly checked and not passed to the backends.
    /// \param binary_op            Binary operator used to draw the sphere onto the input values:
    ///                             (input_value, sphere_value) -> value ("value" is casted to the actual output type).
    ///                             This is ignored if \p input is empty.
    ///
    /// \note The floating-point precision of the drawing operator is set by the transformation floating-point type,
    ///       which defaults to double precision.
    /// \note This function is optimized for rightmost arrays. Passing anything else will likely result in a
    ///       significant performance loss.
    template<size_t N,
             typename Output,
             typename Input = View<nt::value_type_t<Output>>,
             typename CValue = nt::mutable_value_type_twice_t<Input>,
             typename Transform = std::conditional_t<N == 2, Mat22<f64>, Mat33<f64>>,
             typename BinaryOp = Multiply>
    requires guts::ValidDrawShape<N, Output, Input, CValue, Transform, BinaryOp>
    void draw_sphere(
            const Input& input,
            const Output& output,
            const Sphere<CValue, N>& sphere,
            const Transform& inverse_transforms = {},
            BinaryOp binary_op = {}
    ) {
        guts::check_draw_shape_parameters<N>(input, output, inverse_transforms);

        using input_value_t = nt::mutable_value_type_t<Input>;
        using coord_t = nt::mutable_value_type_twice_t<Transform>;
        using smooth_sphere_t = DrawSphere<N, input_value_t, coord_t, true>;
        using sphere_t = DrawSphere<N, input_value_t, coord_t, false>;
        auto center = sphere.center.template as<coord_t>();
        auto radius = static_cast<coord_t>(sphere.radius);
        auto smoothness = static_cast<coord_t>(sphere.smoothness);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            return guts::launch_draw_shape<N, i32, coord_t, sphere_t, smooth_sphere_t>(
                    input, output, inverse_transforms, binary_op,
                    center, radius, smoothness, sphere.cvalue, sphere.invert);
        }
        return guts::launch_draw_shape<N, i64, coord_t, sphere_t, smooth_sphere_t>(
                input, output, inverse_transforms, binary_op,
                center, radius, smoothness, sphere.cvalue, sphere.invert);
    }

    template<typename T, size_t N>
    requires (N == 2 or N == 3)
    struct Rectangle {
        /// (D)HW center of the rectangle.
        Vec<f64, N> center;

        /// (D)HW radius of the rectangle.
        Vec<f64, N> radius;

        /// Size, in pixels, of the raised cosine smooth edge.
        /// This is the number pixels that will be used to compute the (1,0] taper range.
        f64 smoothness{0};

        /// Value of the pixels inside the object.
        /// Pixels outside the object are set to 0.
        T cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};
    };

    /// Returns or draws a rectangle.
    /// \details The mask can be directly saved in \p output or applied (\p see binary_op) to \p input and
    ///          save in \p output. The same transformation can be applied to every batch or there can be
    ///          one transformation per batch (\p see inverse_matrices).
    ///
    /// \tparam Transform           2d case: Mat22, Mat23, Mat33 or an varray of either one of these types.
    ///                             3d case: Mat33, Mat34, Mat44, Quaternion or an varray of either one of these types.
    /// \param[in] input            2d or 3d array(s). If empty, write directly in \p output.
    /// \param[out] output          2d or 3d array(s). Can be equal to \p input for in-place drawing.
    /// \param rectangle            Rectangle to draw, with the \p input value type.
    /// \param inverse_transforms   Inverse (D)HW (affine) matrix/matrices or quaternion(s) to apply on the rectangle.
    ///                             For non-affine matrices and quaternion(s), the rotation center is the center of
    ///                             the rectangle. Note that the identity transformation (the default), as well as an
    ///                             empty array is explicitly checked and not passed to the backends.
    /// \param binary_op            Binary operator used to draw the rectangle onto the input values:
    ///                             (input_value, rectangle_value) -> value ("value" is casted to the actual output type).
    ///                             This is ignored if \p input is empty.
    ///
    /// \note The floating-point precision of the drawing operator is set by the transformation floating-point type,
    ///       which defaults to double precision.
    /// \note This function is optimized for rightmost arrays. Passing anything else will likely result in a
    ///       significant performance loss.
    template<size_t N,
             typename Output,
             typename Input = View<nt::value_type_t<Output>>,
             typename CValue = nt::mutable_value_type_twice_t<Input>,
             typename Transform = std::conditional_t<N == 2, Mat22<f64>, Mat33<f64>>,
             typename BinaryOp = Multiply>
    requires guts::ValidDrawShape<N, Output, Input, CValue, Transform, BinaryOp>
    void draw_rectangle(
            const Input& input,
            const Output& output,
            const Rectangle<CValue, N>& rectangle,
            const Transform& inverse_transforms = {},
            BinaryOp binary_op = {}
    ) {
        guts::check_draw_shape_parameters<N>(input, output, inverse_transforms);

        using input_value_t = nt::mutable_value_type_t<Input>;
        using coord_t = nt::mutable_value_type_twice_t<Transform>;
        using smooth_rectangle_t = DrawRectangle<N, input_value_t, coord_t, true>;
        using rectangle_t = DrawRectangle<N, input_value_t, coord_t, false>;
        auto center = rectangle.center.template as<coord_t>();
        auto radius = rectangle.radius.template as<coord_t>();
        auto smoothness = static_cast<coord_t>(rectangle.smoothness);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            return guts::launch_draw_shape<N, i32, rectangle_t, smooth_rectangle_t>(
                    input, output, inverse_transforms, binary_op,
                    center, radius, smoothness, rectangle.cvalue, rectangle.invert);
        }
        return guts::launch_draw_shape<N, i64, rectangle_t, smooth_rectangle_t>(
                input, output, inverse_transforms, binary_op,
                center, radius, smoothness, rectangle.cvalue, rectangle.invert);
    }

    template<typename T>
    struct Cylinder {
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
        T cvalue{1};

        /// Whether the object should be inverted, i.e. elements inside the object are set to 0,
        /// and elements outside the object are set to cvalue.
        bool invert{false};
    };

    /// Returns or draws a cylinder.
    /// \details The mask can be directly saved in \p output or applied (\p see binary_op) to \p input and
    ///          save in \p output. The same transformation can be applied to every batch or there can be
    ///          one transformation per batch (\p see inverse_matrices).
    ///
    /// \tparam Transform           Mat33, Mat34, Mat44, Quaternion or an varray of either one of these types.
    /// \param[in] input            3d array(s). If empty, write directly in \p output.
    /// \param[out] output          3d array(s). Can be equal to \p input for in-place drawing.
    /// \param cylinder             Cylinder to draw, with the \p input value type.
    /// \param inverse_transforms   Inverse DHW (affine) matrix/matrices or quaternion(s) to apply on the cylinder.
    ///                             For non-affine matrices and quaternion(s), the rotation center is the center of
    ///                             the cylinder. Note that the identity transformation (the default), as well as an
    ///                             empty array is explicitly checked and not passed to the backends.
    /// \param binary_op            Binary operator used to draw the cylinder onto the input values:
    ///                             (input_value, cylinder_value) -> value ("value" is casted to the actual output type).
    ///                             This is ignored if \p input is empty.
    ///
    /// \note The floating-point precision of the drawing operator is set by the transformation floating-point type,
    ///       which defaults to double precision.
    /// \note This function is optimized for rightmost arrays. Passing anything else will likely result in a
    ///       significant performance loss.
    template<typename Output,
             typename Input = View<nt::value_type_t<Output>>,
             typename CValue = nt::mutable_value_type_twice_t<Input>,
             typename Transform = Mat33<f64>,
             typename BinaryOp = Multiply>
    requires guts::ValidDrawShape<3, Output, Input, CValue, Transform, BinaryOp>
    void draw_cylinder(
            const Input& input,
            const Output& output,
            const Cylinder<CValue>& cylinder,
            const Transform& inverse_transforms = {},
            BinaryOp binary_op = {}
    ) {
        guts::check_draw_shape_parameters<3>(input, output, inverse_transforms);

        using input_value_t = nt::mutable_value_type_t<Input>;
        using coord_t = nt::mutable_value_type_twice_t<Transform>;
        using smooth_cylinder_t = DrawCylinder<input_value_t, coord_t, true>;
        using cylinder_t = DrawCylinder<input_value_t, coord_t, false>;

        auto center = cylinder.center.template as<coord_t>();
        auto radius = Vec2<coord_t>::from_values(cylinder.radius, cylinder.length);
        auto smoothness = static_cast<coord_t>(cylinder.smoothness);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            return guts::launch_draw_shape<3, i32, cylinder_t, smooth_cylinder_t>(
                    input, output, inverse_transforms, binary_op,
                    center, radius, smoothness, cylinder.cvalue, cylinder.invert);
        }
        return guts::launch_draw_shape<3, i64, cylinder_t, smooth_cylinder_t>(
                input, output, inverse_transforms, binary_op,
                center, radius, smoothness, cylinder.cvalue, cylinder.invert);
    }
}
