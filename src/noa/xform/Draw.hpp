#pragma once

#include "noa/runtime/Iwise.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/xform/core/Draw.hpp"

namespace noa::xform::details {
    template<
        usize B, usize R,
        nt::sinteger Index,
        nt::readable_nd<B> DrawOp,
        typename BinaryOp,
        nt::readable_nd<B> Xform,
        nt::readable_nd_optional<B + R> Input,
        nt::writable_nd<B + R> Output>
    requires (R == 2 or R == 3)
    class Draw {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using draw_op_type = DrawOp;
        using binary_op_type = BinaryOp;
        using xform_type = Xform;

        using output_value_type = nt::value_type_t<output_type>;
        using draw_vec_type = Vec<nt::value_type_twice_t<draw_op_type>, R>;

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

        NOA_HD constexpr void operator()(const Vec<index_type, B + R>& batched_indices) const {
            draw_(batched_indices);
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
                static_assert(nt::always_false<T>);
            }
        }

        NOA_HD constexpr auto draw_(const Vec<index_type, B + R>& batched_indices) const {
            const auto& [batches, indices] = batched_indices.template split<B>();
            const auto shape = m_drawable[batches].draw_at(draw_vec_type::from_vec(indices), m_inverse_xform[batches]);
            if (m_input) {
                auto input = m_input[batched_indices];
                if constexpr (nt::same_as<BinaryOp, Empty>) {
                    m_output[batched_indices] = static_cast<output_value_type>(default_op_(input, shape));
                } else {
                    m_output[batched_indices] = static_cast<output_value_type>(m_binary_op(input, shape));
                }
            } else {
                m_output[batched_indices] = static_cast<nt::value_type_t<output_value_type>>(shape);
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        draw_op_type m_drawable;
        NOA_NO_UNIQUE_ADDRESS xform_type m_inverse_xform;
        NOA_NO_UNIQUE_ADDRESS binary_op_type m_binary_op;
    };

    template<usize R, typename Input, typename Output, typename DrawingOp, typename Transform>
    auto check_draw_parameters(
        const Input& input,
        const Output& output,
        const DrawingOp& drawing_op,
        const Transform& transform
    ) {
        const Device device = output.device();

        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize B = N - R;
        const auto [output_shape_b, output_shape_r] = output.shape().template split<B>();

        if (input.is_empty()) {
            check(not output.is_empty(), "Empty array detected");
        } else {
            check(not output.is_empty() and not input.is_empty(), "Empty array detected");
            check(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

            auto [input_shape_b, input_shape_r] = input.shape().template split<B>();
            if constexpr (B >= 1) {
                for (usize i{}; i < B; ++i)
                    input_shape_b[i] = input_shape_b[i] == 1 ? output_shape_b[i] : input_shape_b[i];
                check(input_shape_b == output_shape_b,
                      "The batch axes are not compatible, input:batches={}, output:batches={}",
                      input_shape_b, output_shape_b);
            }
            check(input_shape_r == output_shape_r,
                  "The rank axes are not compatible, input:ranks={}, output:ranks={}",
                  input_shape_r, output_shape_r);
        }

        if constexpr (nt::array<DrawingOp>) {
            check(not drawing_op.is_empty(), "Empty array detected");
            check(drawing_op.device() == device,
                  "The input and output arrays must be on the same device, but got drawing_op:device={}, output:device={}",
                  drawing_op.device(), device);
            check(drawing_op.is_contiguous() and drawing_op.shape() == output_shape_b,
                  "The drawing operators should be specified as a contiguous vector of shape {}, but got drawing_ops:shape={} and drawing_ops:strides={}",
                  output_shape_b, drawing_op.shape(), drawing_op.strides());
        }
        if constexpr (nt::array<Transform>) {
            check(not transform.is_empty(), "Empty array detected");
            check(transform.device() == device,
                  "The input and output arrays must be on the same device, but got transform:device={}, output:device={}",
                  transform.device(), device);
            check(transform.is_contiguous() and transform.shape() == output_shape_b,
                  "The transforms (matrices or quaternions) should be specified as a contiguous vector of shape {}, but got transforms:shape={} and transforms:strides={}",
                  output_shape_b, transform.shape(), transform.strides());
        }
    }

    template<typename Output, typename Input, typename Draw, typename BinaryOp>
    concept drawable_binary_op = (nt::empty<BinaryOp> and nt::compatible_or_spectrum_types<Input, Output>) or
        requires (const BinaryOp op, Input lhs, Draw rhs) {
            static_cast<Output>(op(lhs, rhs));
        };

    template<usize RANK, typename Input, typename Output, typename Drawable, typename Transform, typename BinaryOp,
             typename DrawableDecay = std::decay_t<Drawable>,
             typename TransformDecay = std::decay_t<Transform>,
             typename DrawValue = std::conditional_t<nt::array<DrawableDecay>, nt::value_type_twice_t<Drawable>, nt::value_type_t<Drawable>>,
             usize N = nt::array_size_v<Output>,
             usize B = N - RANK>
    concept drawable_nd =
        N >= RANK and
        nt::readable_array_decay<Input> and nt::array_size_v<Input> == N and
        nt::writable_array_decay<Output> and
        drawable_binary_op<nt::mutable_value_type_t<Input>, nt::value_type_t<Output>, DrawValue, BinaryOp> and
        ((nt::array_nd<DrawableDecay, B> and nt::array_nd<TransformDecay, B> and nt::drawable<nt::value_type_t<Drawable>, RANK, DrawValue, nt::value_type_t<Transform>>) or
         (nt::array_nd<DrawableDecay, B> and nt::drawable<nt::value_type_t<Drawable>, RANK, DrawValue, TransformDecay>) or
         (nt::array_nd<TransformDecay, B> and nt::drawable<DrawableDecay, RANK, DrawValue, nt::value_type_t<Transform>>) or
         (nt::drawable<DrawableDecay, RANK, DrawValue, TransformDecay>));

    template<usize RANK, IwiseOptions OPTIONS,
             typename Index, typename Input, typename Output,
             typename Drawable, typename Transform, typename BinaryOp>
    void launch_draw(
        Input&& input,
        Output&& output,
        Drawable&& drawable,
        Transform&& inverse_transform,
        const BinaryOp& binary_op
    ) {
        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize B = N - RANK;

        // TODO Reorder to rightmost? Only possible with zero or one transform.
        // TODO Broadcast input batches.

        // Prepare the input/output accessors.
        using input_accessor_t = Accessor<nt::const_value_type_t<Input>, N, Index>;
        using output_accessor_t = Accessor<nt::value_type_t<Output>, N, Index>;
        auto input_accessor = input_accessor_t(input.get(), input.strides().template as<Index>());
        auto output_accessor = output_accessor_t(output.get(), output.strides().template as<Index>());

        // Broadcast input batches.
        if constexpr (B > 0)
            for (usize i = 0; i < B; ++i)
                if (input.shape()[i] == 1)
                    input_accessor.strides()[i] = 0;

        auto drawable_accessor = [&] {
            if constexpr (nt::is_array_v<std::decay_t<Drawable>>) {
                using accessor_t = AccessorRestrictContiguous<nt::const_value_type_t<Drawable>, B, isize>;
                return accessor_t(drawable.get(), drawable.strides());
            } else {
                return AccessorValue{drawable};
            }
        }();
        using drawable_t = decltype(drawable_accessor);

        // Skip the transform if possible.
        using xform_t = std::decay_t<Transform>;
        bool has_transform{}; // default: Empty
        if constexpr (nt::array<xform_t>)
            has_transform = not inverse_transform.is_empty();
        else if constexpr (nt::quaternion<xform_t>)
            has_transform = inverse_transform != xform_t{.z=0, .y=0, .x=0, .w=1};
        else if constexpr (nt::mat<xform_t>)
            has_transform = inverse_transform != xform_t::eye(1);
        else if constexpr (not nt::empty<xform_t>)
            static_assert(nt::always_false<xform_t>);

        if (has_transform) {
            auto inverse_transform_accessor = [&] {
                if constexpr (nt::is_mat_of_shape_v<xform_t, N + 1, N + 1>) {
                    return AccessorValue{inverse_transform.pop_back()};
                } else if constexpr (nt::is_mat_v<xform_t> or nt::is_quaternion_v<xform_t> or std::is_empty_v<xform_t>) {
                    return AccessorValue{inverse_transform};
                } else if constexpr (nt::is_array_v<xform_t>) {
                    using accessor_t = AccessorRestrictContiguous<nt::const_value_type_t<xform_t>, B, isize>;
                    return accessor_t(inverse_transform.get(), inverse_transform.strides());
                } else {
                    static_assert(nt::always_false<xform_t>);
                }
            }();
            using transform_t = decltype(inverse_transform_accessor);
            using op_t = Draw<B, RANK, Index, drawable_t, BinaryOp, transform_t, input_accessor_t, output_accessor_t>;
            auto op = op_t(input_accessor, output_accessor, drawable_accessor, inverse_transform_accessor, binary_op);
            return iwise<OPTIONS>(
                output.shape().template as<Index>(), output.device(), std::move(op),
                std::forward<Input>(input),
                std::forward<Output>(output),
                std::forward<Transform>(inverse_transform)
            );
        } else {
            using op_t = Draw<B, RANK, Index, drawable_t, BinaryOp, AccessorValue<Empty>, input_accessor_t, output_accessor_t>;
            auto op = op_t(input_accessor, output_accessor, drawable_accessor, {}, binary_op);
            return iwise<OPTIONS>(
                output.shape().template as<Index>(), output.device(), std::move(op),
                std::forward<Input>(input),
                std::forward<Output>(output),
                std::forward<Transform>(inverse_transform)
            );
        }
    }
}

namespace noa::xform {
    /// Draws onto an array.
    /// \details The drawing can be directly saved in the output, or applied to the input (see binary_op) and
    ///          then saved in the output. The same drawing operator or transformation can be applied to every batch,
    ///          or there can be one per batch. This was originally intended to draw simple geometric shapes (see
    ///          Sphere, Ellipse, Rectangle, Cylinder), but any type matching the nt::drawable concept should work.
    ///
    /// \tparam RANK:
    ///     Rank of the arrays.
    /// \tparam Transform:
    ///     2D case: Mat22, Mat23, Mat33, a array of these types, or Empty.
    ///     3D case: Mat33, Mat34, Mat44, Quaternion, a array of these types, or Empty.
    /// \param[in] input:
    ///     1D: ((B..,)W), 2D: ((B..,)H,W), or 3D: ((B..,)D,H,W) arrays.
    ///     If empty array, write directly in the output.
    ///     The batch axes are broadcasted to the output batches.
    /// \param[out] output:
    ///     Output array(s) matching the input ranked axes.
    ///     Can be equal to the input for in-place drawing.
    /// \param drawing_op:
    ///     Drawing operator(s).
    ///     A (B..) array of drawing operators or a single operator.
    /// \param inverse_transforms:
    ///     A (B..) array of inverse ((D)H)W (affine) matrices or quaternions (if 3D) to apply to the coordinates that
    ///     will be subsequently given to the corresponding drawing operator. For non-affine matrices and quaternion(s),
    ///     the rotation center is the center returned by the drawing operator.
    ///     Passing a single transform is allowed, in which case all batches are assigned to it.
    ///     Passing Empty (the default), the identity transform, or an empty array, is valid and equivalent to no
    ///     transformation. Otherwise, the floating-point precision of the transform should match the precision
    ///     of the drawing operator.
    /// \param binary_op:
    ///     Optional binary operator used to apply the drawing onto the input values:
    ///     (input_value, drawn_value) -> value ("value" is then cast to the actual output type).
    ///     The default operator (i.e. passing Empty) multiplies both values; if the input is an integer, drawn_value
    ///     if first rounded to the nearest integer; if the input is complex and the output is real, abs(input_value)^2
    ///     (the power-spectrum) is first computed. This is ignored if the input is empty.
    ///
    /// \example
    /// \code
    /// // Draw a sphere into the output.
    /// nd::draw({}, output_2d, nd::Sphere{.center=Vec{64.,64.}, .radius=10.}.get());
    ///
    /// // Add an ellipse rotated by 10 degrees onto the input.
    /// const auto ellipse = nd::Ellipse{.center=Vec{64.,64.}, .radius=Vec{10.,20.};
    /// const auto ellipse_rotation = nd::rotate(noa::deg2rad(10.));
    /// nd::draw(input_2d, output_2d, ellipse.get(), ellipse_rotation.inverse(), noa::Plus{});
    /// \endcode
    template<usize RANK, typename Output, typename Drawable,
             typename Input = Array<nt::value_type_t<Output>, nt::array_size_v<Output>, ArrayOwnership::VIEW>,
             typename Transform = Empty,
             typename BinaryOp = Empty>
        requires details::drawable_nd<RANK, Input, Output, Drawable, Transform, BinaryOp>
    void draw(
        Input&& input,
        Output&& output,
        Drawable&& drawing_op,
        Transform&& inverse_transforms = {},
        BinaryOp binary_op = {}
    ) {
        details::check_draw_parameters<RANK>(input, output, drawing_op, inverse_transforms);

        if (output.device().is_gpu() and
            nd::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            nd::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            #ifdef NOA_ENABLE_CUDA
            return details::launch_draw<RANK, IwiseOptions{.generate_cpu = false}, i32>(
                NOA_FWD(input), NOA_FWD(output), NOA_FWD(drawing_op), NOA_FWD(inverse_transforms),
                binary_op);
            #else
            std::terminate(); // unreachable
            #endif
        }
        return details::launch_draw<RANK, IwiseOptions{}, isize>(
            NOA_FWD(input), NOA_FWD(output), NOA_FWD(drawing_op), NOA_FWD(inverse_transforms),
            binary_op
        );
    }
    template<typename Output, typename Drawable,
             typename Input = Array<nt::value_type_t<Output>, nt::array_size_v<Output>, ArrayOwnership::VIEW>,
             typename Transform = Empty,
             typename BinaryOp = Empty>
    void draw_2d(
        Input&& input, Output&& output, Drawable&& drawable,
        Transform&& inverse_transforms = {},
        BinaryOp binary_op = {}
    ) {
        draw<2>(NOA_FWD(input), NOA_FWD(output), NOA_FWD(drawable), NOA_FWD(inverse_transforms), NOA_FWD(binary_op));
    }
    template<typename Output, typename Drawable,
             typename Input = Array<nt::value_type_t<Output>, nt::array_size_v<Output>, ArrayOwnership::VIEW>,
             typename Transform = Empty,
             typename BinaryOp = Empty>
    void draw_3d(
        Input&& input, Output&& output, Drawable&& drawable,
        Transform&& inverse_transforms = {},
        BinaryOp binary_op = {}
    ) {
        draw<3>(NOA_FWD(input), NOA_FWD(output), NOA_FWD(drawable), NOA_FWD(inverse_transforms), NOA_FWD(binary_op));
    }
}
