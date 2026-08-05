#pragma once

#include "noa/runtime/Array.hpp"
#include "noa/runtime/Utils.hpp"
#include "noa/runtime/Iwise.hpp"
#include "noa/xform/core/Interpolation.hpp"
#include "noa/xform/core/Transform.hpp"
#include "noa/xform/Utils.hpp"

namespace noa::xform::details {
    /// Iwise operator computing 2d or 3d affine transformations:
    ///  * Works on real and complex arrays.
    ///  * The interpolated value is static_cast to Output::value_type.
    ///  * The floating-point precision of the transformation and interpolation is set by Xform.
    ///  * Use (truncated) affine matrices to transform coordinates.
    ///    A single matrix can be used for every output batch. Otherwise, an array of matrices
    ///    is expected. In this case, there should be as many matrices as they are output batches.
    ///  * The matrices should be inverted since the inverse transformation is performed.
    ///  * Multiple batches can be processed. The operator expects a "batch" index, which is passed
    ///    to the interpolator. It is up to the interpolator to decide what to do with this index.
    ///    The interpolator can ignore that batch index, effectively broadcasting the input to
    ///    every dimension of the output.
    template<usize B, usize R,
             nt::integer Index,
             nt::readable_nd<B> Xform,
             nt::interpolator_nd<R> Interpolator,
             nt::readable_nd<B + R> Input,
             nt::writable_nd<B + R> Output>
    requires (R == 2 or R == 3)
    class Transform {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using output_value_type = nt::value_type_t<output_type>;

        using xform_parameter_type = Xform;
        using xform_type = nt::value_type_t<xform_parameter_type>;
        using coord_type = nt::value_type_t<xform_type>;
        static_assert(nt::mat_of_shape<xform_type, R + 0, R + 1> or
                      nt::mat_of_shape<xform_type, R + 1, R + 1>);

    public:
        Transform(
            const input_type& input,
            const output_type& output,
            const xform_parameter_type& inverse_xform,
            const Interpolator& interpolator
        ) :
            m_input(input),
            m_output(output),
            m_inverse_xform(inverse_xform),
            m_interpolator(interpolator) {}

        NOA_HD constexpr void operator()(const Vec<index_type, B + R>& batched_indices) const {
            const auto& [batches, indices] = batched_indices.template split<B>();
            auto coordinates = indices.template as<coord_type>();
            coordinates = transform_vector(m_inverse_xform[batches], coordinates);
            m_output(batched_indices) = static_cast<output_value_type>(m_interpolator(m_input[batches], coordinates));
        }

    private:
        input_type m_input;
        output_type m_output;
        xform_parameter_type m_inverse_xform;
        Interpolator m_interpolator;
    };

    template<bool ALLOW_EMPTY = false, bool ENFORCE_EMPTY = false, typename Xform>
    auto xform_into_accessor(const Xform& xform) {
        using value_t = nt::const_value_type_t<Xform>;
        if constexpr (nt::mat<Xform> or nt::vec<Xform> or nt::quaternion<Xform> or (ALLOW_EMPTY and nt::empty<Xform>)) {
            if constexpr (ENFORCE_EMPTY)
                return AccessorValue<Empty>{};
            else
                return AccessorValue{xform};
        } else if constexpr (nt::array<Xform> and (nt::mat<value_t> or nt::vec<value_t> or nt::quaternion<value_t>)) {
            if constexpr (ENFORCE_EMPTY)
                return AccessorValue<Empty>{};
            else {
                NOA_ASSERT(xform.is_contiguous());
                return AccessorRestrictContiguous<value_t, 1>{xform.get()};
            }
        } else {
            static_assert(nt::always_false<Xform>);
        }
    }

    template<usize R, typename Input, typename Output, typename Matrix>
    auto check_parameters_transform_nd(const Input& input, const Output& output, const Matrix& matrix) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");

        // Check batch axes are compatible.
        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize B = N - R;
        auto input_shape_b = input.shape().template pop_back<R>();
        auto output_shape_b = output.shape().template pop_back<R>();
        for (usize i{}; i < B; ++i)
            input_shape_b[i] = input_shape_b[i] == 1 ? output_shape_b[i] : input_shape_b[i];
        check(input_shape_b == output_shape_b,
              "The batch axes are not compatible, input:batches={}, output:batches={}",
              input_shape_b, output_shape_b);

        const Device device = output.device();
        if constexpr (nt::array<Matrix>) {
            check(matrix.is_contiguous() and matrix.shape() == output_shape_b,
                  "The matrices, specified as a contiguous array, should match the output shape, but got matrices:shape={}, matrices:strides={} and output:batches={}",
                  matrix.shape(), matrix.strides(), output_shape_b);
            check(device == matrix.device(),
                  "The transformation matrices should be on the same device as the output, but got matrices:device={} and output:device={}",
                  matrix.device(), device);
        }

        check(input.device() == device,
              "The input and output must be on the same device, but got input:device={} and output:device={}",
              input.device(), device);
        check(nd::are_elements_unique(output.strides(), output.shape()),
              "The elements in the output should not overlap in memory, otherwise a data-race might occur. Got output:strides={} and output:shape={}",
              output.strides(), output.shape());

        if constexpr (nt::array<Input>) {
            check(not are_overlapped(input, output), "The input and output arrays should not overlap");
        } else {
            check(input.device().is_gpu() or not are_overlapped(input.cpu(), output), "The input and output arrays should not overlap");
        }
    }

    // GPU path instantiates 42 kernels with arrays and 54 kernels with textures...
    template<usize R, typename Index, bool IS_GPU = false, typename Input, typename Output, typename Matrix>
    void launch_transform_nd(Input&& input, Output&& output, Matrix&& xform, auto options) {
        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize B = N - R;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N, Index>;
        auto output_span = output.span().template as_index<Index>();
        auto output_accessor = output_accessor_t(output_span.get(), output_span.strides());
        auto xform_accessor = xform_into_accessor(xform);
        using xform_accessor_t = decltype(xform_accessor);

        if constexpr (nt::texture_decay<Input>) {
            options.interp = input.interp();
            options.border = input.border();
            options.cvalue = input.cvalue();
        }

        auto launch_iwise = [&](auto interp, auto border) {
            using coord_t = nt::mutable_value_type_twice_t<Matrix>;
            auto result = prepare_interpolation_inputs<R, interp(), border(), IS_GPU, Index, coord_t, false>(input, options.cvalue);
            using interpolator_t = decltype(result)::interpolator_type;
            using input_accessor_t = decltype(result)::input_accessor_type;
            using op_t = Transform<B, R, Index, xform_accessor_t, interpolator_t, input_accessor_t, output_accessor_t>;

            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(output_span.shape(), output.device(),
               op_t(result.input_accessor, output_accessor, xform_accessor, result.interpolator),
               std::forward<Input>(input),
               std::forward<Output>(output),
               std::forward<Matrix>(xform));
        };

        auto launch_border = [&](auto interp) {
            switch (options.border) {
                case Border::ZERO:      return launch_iwise(interp, WrapBorder<Border::ZERO>{});
                case Border::VALUE:     return launch_iwise(interp, WrapBorder<Border::VALUE>{});
                case Border::CLAMP:     return launch_iwise(interp, WrapBorder<Border::CLAMP>{});
                case Border::PERIODIC:  return launch_iwise(interp, WrapBorder<Border::PERIODIC>{});
                case Border::MIRROR:    return launch_iwise(interp, WrapBorder<Border::MIRROR>{});
                case Border::REFLECT:   return launch_iwise(interp, WrapBorder<Border::REFLECT>{});
                case Border::NOTHING:   panic("The border mode {} is not supported", Border::NOTHING);
            }
        };

        switch (options.interp) {
            case Interp::NEAREST:            return launch_border(WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_border(WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_border(WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_border(WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_border(WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_border(WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_border(WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_border(WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            case Interp::LANCZOS4:           return launch_border(WrapInterp<Interp::LANCZOS4>{});
            case Interp::LANCZOS6:           return launch_border(WrapInterp<Interp::LANCZOS6>{});
            case Interp::LANCZOS8:           return launch_border(WrapInterp<Interp::LANCZOS8>{});
            case Interp::LANCZOS4_FAST:      return launch_border(WrapInterp<Interp::LANCZOS4_FAST>{});
            case Interp::LANCZOS6_FAST:      return launch_border(WrapInterp<Interp::LANCZOS6_FAST>{});
            case Interp::LANCZOS8_FAST:      return launch_border(WrapInterp<Interp::LANCZOS8_FAST>{});
        }
    }
}

namespace noa::traits {
    template<typename Matrix, usize R>
    concept transform_matrix_nd = mat_of_shape<Matrix, R, R + 1> or mat_of_shape<Matrix, R + 1, R + 1>;

    template<typename T, usize R, typename Output,
             typename U = std::remove_reference_t<T>,
             typename V = value_type_t<T>,
             usize N = nt::array_size_v<Output>>
    concept transform_matrices_nd =
        transform_matrix_nd<U, R> or (nt::array_nd<U, (N > R ? N - R : 0)> and transform_matrix_nd<V, R>);
}

namespace noa::xform {
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

    /// Applies one or multiple 2D affine transforms.
    /// \param[in] input:
    ///     ((Bi..,)Hi,Wi) Input 2D array(s) or 2D texture(s).
    ///     The batch axes are broadcast to the output batch axes.
    /// \param[out] output:
    ///     ((Bo..,)Ho,Wo) Output 2D array(s).
    /// \param[in] inverse_matrices:
    ///     2x3 or 3x3 inverse HW affine matrices.
    ///     One matrix, or a contiguous (Bo..) array matching the output batches.
    ///     Sets the floating-point precision of the transformation and interpolation.
    /// \param options:
    ///     Interpolation and border options.
    /// \note
    ///     The input and output array can have different shapes ((Hi,Wi) vs (Ho,Wo)). The output window starts at
    ///     the same index as the input window, so by entering a translation in inverse_matrices, one can move the
    ///     center of the output window relative to the input window, e.g., to render only a specific subregion.
    template<nt::array_or_texture_decay_of_real_or_complex Input,
             nt::writable_array_decay_of_any<nt::mutable_value_type_t<Input>> Output,
             nt::transform_matrices_nd<2, Output> Matrix>
        requires (nt::array_size_v<Input> == nt::array_size_v<Output> and nt::array_size_v<Output> >= 2)
    void transform_2d(
        Input&& input,
        Output&& output,
        Matrix&& inverse_matrices,
        const TransformOptions<nt::mutable_value_type_t<Input>>& options = {}
    ) {
        details::check_parameters_transform_nd<2>(input, output, inverse_matrices);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                panic(); // unreachable
            } else {
                check(nd::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
                      nd::is_accessor_access_safe<i32>(output.strides(), output.shape()),
                      "isize indexing not instantiated for GPU devices");

                details::launch_transform_nd<2, i32, true>(
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
        details::launch_transform_nd<2, isize>(
            std::forward<Input>(input),
            std::forward<Output>(output),
            std::forward<Matrix>(inverse_matrices),
            options);
    }

    /// Applies one or multiple 3D affine transforms.
    /// \param[in] input:
    ///     ((Bi..,)Di,Hi,Wi) Input 3D array(s) or 3D texture(s).
    ///     The batch axes are broadcast to the output batch axes.
    /// \param[out] output:
    ///     ((Bo..,)Do,Ho,Wo) Output 3D array(s).
    /// \param[in] inverse_matrices:
    ///     3x4 or 4x4 inverse HW affine matrices.
    ///     One matrix, or a contiguous (Bo..) array matching the output batches.
    ///     Sets the floating-point precision of the transformation and interpolation.
    /// \param options:
    ///     Interpolation and border options.
    /// \note
    ///     The input and output array can have different shapes ((Di,Hi,Wi) vs (Do,Ho,Wo)). The output window starts at
    ///     the same index as the input window, so by entering a translation in inverse_matrices, one can move the
    ///     center of the output window relative to the input window, e.g., to render only a specific subregion.
    template<nt::array_or_texture_decay_of_real_or_complex Input,
             nt::writable_array_decay_of_any<nt::mutable_value_type_t<Input>> Output,
             nt::transform_matrices_nd<3, Output> Matrix>
        requires (nt::array_size_v<Input> == nt::array_size_v<Output> and nt::array_size_v<Output> >= 3)
    void transform_3d(
        Input&& input,
        Output&& output,
        Matrix&& inverse_matrices,
        const TransformOptions<nt::mutable_value_type_t<Input>>& options = {}
    ) {
        details::check_parameters_transform_nd<3>(input, output, inverse_matrices);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                if (nd::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
                    nd::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
                    details::launch_transform_nd<3, i32, true>(
                        std::forward<Input>(input),
                        std::forward<Output>(output),
                        std::forward<Matrix>(inverse_matrices),
                        options);
                } else {
                    // For large volumes (>1290^3), isize indexing is required.
                    details::launch_transform_nd<3, isize, true>(
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

        details::launch_transform_nd<3, isize>(
            std::forward<Input>(input),
            std::forward<Output>(output),
            std::forward<Matrix>(inverse_matrices),
            options);
    }
}
