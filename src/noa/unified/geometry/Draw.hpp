#pragma once

#include "noa/core/geometry/Transform.hpp"
#include "noa/core/geometry/Draw.hpp"
#include "noa/core/utils/BatchedParameter.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/View.hpp"

namespace noa::geometry::guts {
    template<
        size_t N,
        nt::sinteger Index,
        typename DrawOp,
        typename BinaryOp,
        nt::batched_parameter Xform,
        nt::readable_nd_optional<N + 1> Input,
        nt::writable_nd<N + 1> Output>
    requires (N == 2 or N == 3)
    class Draw {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using draw_op_type = DrawOp;
        using binary_op_type = BinaryOp;
        using xform_type = Xform;

        using output_value_type = nt::value_type_t<output_type>;
        using draw_vec_type = Vec<nt::value_type_t<draw_op_type>, N>;

    public:
        constexpr Draw(
            const input_type& input,
            const output_type& output,
            const draw_op_type& drawable,
            const xform_type& inverse_xform,
            const binary_op_type& binary_op
        ) :
            m_input(input),
            m_output(output),
            m_drawable(drawable),
            m_inverse_xform(inverse_xform),
            m_binary_op(binary_op) {}

        template<nt::same_as<index_type>... I> requires (N == sizeof...(I))
        NOA_HD constexpr void operator()(index_type batch, I... indices) const {
            draw_(batch, indices...);
        }

    private:
        template<typename T>
        NOA_HD constexpr auto default_op_(const T& input, auto shape) const {
            using scalar_t = nt::value_type_t<T>;
            if constexpr (nt::complex<T> and nt::real<output_value_type>) {
                return abs_squared(input) * static_cast<scalar_t>(shape);
            } else if constexpr (nt::real_or_complex<T>) {
                return input * static_cast<scalar_t>(shape);
            } else if constexpr (nt::integer<T>) {
                return input * static_cast<scalar_t>(round(shape));
            } else {
                static_assert(nt::always_false<>);
            }
        }

        NOA_HD constexpr auto draw_(index_type batch, auto... indices) const {
            const auto shape = m_drawable.draw_at(draw_vec_type::from_values(indices...), m_inverse_xform[batch]);
            if (m_input) {
                auto input = m_input(batch, indices...);
                if constexpr (nt::same_as<BinaryOp, Empty>) {
                    m_output(batch, indices...) = static_cast<output_value_type>(default_op_(input, shape));
                } else {
                    m_output(batch, indices...) = static_cast<output_value_type>(m_binary_op(input, shape));
                }
            } else {
                m_output(batch, indices...) = static_cast<nt::value_type_t<output_value_type>>(shape);
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        draw_op_type m_drawable;
        NOA_NO_UNIQUE_ADDRESS xform_type m_inverse_xform;
        NOA_NO_UNIQUE_ADDRESS binary_op_type m_binary_op;
    };

    template<size_t N, typename Input, typename Output, typename Transform>
    auto check_draw_parameters(
        const Input& input,
        const Output& output,
        const Transform& transform
    ) {
        const Device device = output.device();
        check(output.shape().ndim() <= static_cast<i64>(N), "3d arrays are not supported with 2d drawing operators");

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

    template<typename Output, typename Input, typename Draw, typename BinaryOp>
    concept drawable_binary_op = (nt::empty<BinaryOp> and nt::compatible_or_spectrum_types<Input, Output>) or
        requires (const BinaryOp op, Input lhs, Draw rhs) {
        static_cast<Output>(op(lhs, rhs));
        };

    template<typename Input, typename Output, typename Drawable, typename Transform, typename BinaryOp,
             typename InputValue = nt::mutable_value_type_t<Input>,
             typename OutputValue = nt::value_type_t<Output>,
             typename DrawValue = nt::value_type_t<Drawable>,
             typename TransformValue = nt::value_type_t<Transform>>
    concept drawable =
        nt::readable_varray_decay_of_numeric<Input> and
        nt::writable_varray_decay_of_numeric<Output> and
        drawable_binary_op<OutputValue, InputValue, DrawValue, BinaryOp> and
        (nt::drawable<std::decay_t<Drawable>, std::decay_t<Drawable>::SIZE, DrawValue, std::decay_t<Transform>> or
         (nt::varray_decay<std::decay_t<Transform>> and
          nt::drawable<std::decay_t<Drawable>, std::decay_t<Drawable>::SIZE, DrawValue, nt::value_type_t<Transform>>));

    template<IwiseOptions OPTIONS,
             typename Index, typename Input, typename Output,
             typename Drawable, typename Transform, typename BinaryOp>
    void launch_draw(
        Input&& input,
        Output&& output,
        Drawable&& drawable,
        Transform&& inverse_transform,
        const BinaryOp& binary_op
    ) {
        using drawable_t = std::decay_t<Drawable>;
        constexpr size_t N = drawable_t::SIZE;

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

        // Loop through every element of the output.
        auto iwise_shape = output.shape().template filter_nd<N>().template as<Index>();

        if (has_transform) {
            auto extract_transform = [&] {
                if constexpr (nt::is_mat_of_shape_v<xform_t, N + 1, N + 1>) {
                    return BatchedParameter{affine2truncated(inverse_transform)};
                } else if constexpr (nt::is_varray_v<xform_t>) {
                    using accessor_t = AccessorRestrictContiguousI64<nt::const_value_type_t<xform_t>, 1>;
                    return BatchedParameter{accessor_t(inverse_transform.get())}; // inverse_transform is contiguous
                } else if constexpr (nt::is_mat_v<xform_t> or nt::is_quaternion_v<xform_t> or std::is_empty_v<xform_t>) {
                    return BatchedParameter{inverse_transform};
                } else {
                    static_assert(nt::always_false<>);
                }
            };
            using transform_t = decltype(extract_transform());
            using op_t = Draw<N, Index, drawable_t, BinaryOp, transform_t, input_accessor_t, output_accessor_t>;
            auto op = op_t(input_accessor, output_accessor, drawable, extract_transform(), binary_op);
            return iwise<OPTIONS>(
                iwise_shape, output.device(), std::move(op),
                std::forward<Input>(input),
                std::forward<Output>(output),
                std::forward<Transform>(inverse_transform)
            );
        } else {
            using op_t = Draw<N, Index, drawable_t, BinaryOp, BatchedParameter<Empty>, input_accessor_t, output_accessor_t>;
            auto op = op_t(input_accessor, output_accessor, drawable, BatchedParameter<Empty>{}, binary_op);
            return iwise<OPTIONS>(
                iwise_shape, output.device(), std::move(op),
                std::forward<Input>(input),
                std::forward<Output>(output),
                std::forward<Transform>(inverse_transform)
            );
        }
    }
}

namespace noa::geometry {
    /// Draws onto an array.
    /// \details The drawing can be directly saved in the output, or applied to the input (see binary_op) and
    ///          then saved in the output. The same transformation can be applied to every batch, or there can be
    ///          one transformation per batch (see inverse_transforms). This was originally intended to draw simple
    ///          geometric shapes (see Sphere, Ellipse, Rectangle, Cylinder), but any nt::drawable should work.
    ///
    /// \tparam Transform         2d case: Mat22, Mat23, Mat33, a varray of these types, or Empty.
    ///                           3d case: Mat33, Mat34, Mat44, Quaternion, a varray of these types, or Empty.
    /// \param[in] input          2d or 3d array(s). If empty, write directly in the output.
    /// \param[out] output        2d or 3d array(s). Can be equal to the input for in-place drawing.
    /// \param drawable           Drawing operator.
    /// \param inverse_transforms Inverse (D)HW (affine) matrix/matrices or quaternion(s) to apply to the coordinates
    ///                           that will be subsequently given to the drawing operator. For non-affine matrices and
    ///                           quaternion(s), the rotation center is the center returned by the drawing operator.
    ///                           Note that passing Empty (the default), the identity transform, or an empty array,
    ///                           is valid and equivalent to no transformation. Otherwise, the floating-point precision
    ///                           of the transform should match the precision of the drawing operator.
    /// \param binary_op          Optional binary operator used to apply the drawing onto the input values:
    ///                           (input_value, drawn_value) -> value ("value" is then casted to the actual output type).
    ///                           The default operator (i.e. passing Empty) multiplies both values; if the input is an
    ///                           integer, drawn_value if first rounded to the nearest integer; if the input is complex
    ///                           and the output is real, abs(input_value)^2 (the power-spectrum) is first computed.
    ///                           This is ignored if the input is empty.
    ///
    /// \example
    /// \code
    /// // Draw a sphere onto the output.
    /// ng::draw({}, output_2d, ng::Sphere{.center=Vec{64.,64.}, .radius=10.}.draw());
    ///
    /// // Add an ellipse rotated by 10 degrees onto the input.
    /// const auto ellipse = ng::Ellipse{.center=Vec{64.,64.}, .radius=Vec{10.,20.};
    /// const auto ellipse_rotation = ng::rotate(noa::deg2rad(10.));
    /// ng::draw(input_2d, output_2d, ellipse.draw(), ellipse_rotation.inverse(), noa::Plus{});
    /// \endcode
    template<typename Output, typename Drawable,
             typename Input = View<nt::value_type_t<Output>>,
             typename Transform = Empty,
             typename BinaryOp = Empty>
    requires guts::drawable<Input, Output, Drawable, Transform, BinaryOp>
    void draw(
        Input&& input,
        Output&& output,
        Drawable&& drawable,
        Transform&& inverse_transforms = {},
        BinaryOp binary_op = {}
    ) {
        constexpr size_t N = Drawable::SIZE;
        guts::check_draw_parameters<N>(input, output, inverse_transforms);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            #ifdef NOA_ENABLE_CUDA
            return guts::launch_draw<IwiseOptions{.generate_cpu = false}, i32>(
                std::forward<Input>(input),
                std::forward<Output>(output),
                std::forward<Drawable>(drawable),
                std::forward<Transform>(inverse_transforms),
                binary_op);
            #else
            std::terminate(); // unreachable
            #endif
        }
        return guts::launch_draw<IwiseOptions{}, i64>(
            std::forward<Input>(input),
            std::forward<Output>(output),
            std::forward<Drawable>(drawable),
            std::forward<Transform>(inverse_transforms),
            binary_op
        );
    }
}
