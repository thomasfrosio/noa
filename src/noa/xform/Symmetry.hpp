#pragma once

#include <optional>

#include "noa/base/Mat.hpp"
#include "noa/base/Strings.hpp"
#include "noa/base/Zip.hpp"
#include "noa/runtime/core/Span.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"
#include "noa/runtime/Utils.hpp"
#include "noa/xform/core/Symmetry.hpp"
#include "noa/xform/Texture.hpp"
#include "noa/xform/Transform.hpp"
#include "noa/xform/Utils.hpp"

namespace noa::xform {
    /// Symmetry utility class, storing and managing symmetry rotation matrices.
    /// \note By convention, the identity matrix is not stored as
    ///       it is implicitly applied by the "symmetrize" functions.
    /// \note Supported symmetries:
    ///     - CX, with X being a non-zero positive number.
    /// TODO Add quaternions and more symmetries...
    template<typename T, usize N, ArrayOwnership O = ArrayOwnership::RC>
    class Symmetry {
    public:
        static_assert(nt::real<T>);
        static constexpr bool IS_VIEW = O == ArrayOwnership::VIEW;
        static constexpr usize SIZE = N;
        static constexpr usize SSIZE = N;

        using value_type = T;
        using mutable_value_type = std::remove_const_t<T>;
        using mutable_matrix_type = Mat<mutable_value_type, N, N>;
        using matrix_type = std::conditional_t<std::is_const_v<value_type>, const mutable_matrix_type, mutable_matrix_type>;
        using array_type = Array<matrix_type, 1, O>;
        using shared_type = array_type::shared_type;

        /// Construct an empty object.
        constexpr Symmetry() = default;

        /// Allocates and initializes the symmetry matrices on the device.
        constexpr explicit Symmetry(const SymmetryCode& code, const ArrayOption& options = {}) requires (not IS_VIEW) :
            m_code(code)
        {
            validate_and_set_buffer_(options);
        }

        constexpr explicit Symmetry(std::string_view code, const ArrayOption& options = {}) requires (not IS_VIEW) {
            const auto parsed_code = SymmetryCode::from_string(code);
            check(parsed_code.has_value(), "Failed to parse \"{}\" to a valid symmetry", code);
            m_code = parsed_code.value();
            validate_and_set_buffer_(options);
        }

        /// Imports existing matrices. No validation is done.
        /// The identity matrix should not be included as it is implicitly applied by the "symmetrize" functions.
        constexpr explicit Symmetry(Array<T, N, O> matrices, const SymmetryCode& code) :
            m_array(std::move(matrices)),
            m_code(code)
        {
            check(m_array.is_contiguous(),
                  "The symmetry matrices should be saved in a contiguous vector, but got matrices:shape={} and matrices:strides={}",
                  m_array.shape(), m_array.strides());
        }

    public:
        [[nodiscard]] auto code() const -> SymmetryCode { return m_code; }
        [[nodiscard]] auto device() const { return m_array.device(); }
        [[nodiscard]] auto options() const { return m_array.options(); }
        [[nodiscard]] auto is_empty() const { return m_array.is_empty(); }

        [[nodiscard]] auto span() const -> SpanContiguous<matrix_type> {
            return m_array.span_1d();
        }

        [[nodiscard]] auto share() const& -> const shared_type& { return m_array.share(); }
        [[nodiscard]] auto share() && -> shared_type&& { return std::move(m_array).share(); }

        [[nodiscard]] auto array() const& -> const array_type& { return m_array; }
        [[nodiscard]] auto array() && -> array_type&& { return std::move(m_array); }

        [[nodiscard]] auto to(ArrayOption option) const& -> Symmetry {
            array_type out(array().shape(), option);
            array().to(out);
            return Symmetry(std::move(out), code());
        }
        [[nodiscard]] auto to(ArrayOption option) && -> Symmetry {
            array_type out(array().shape(), option);
            array().to(out);
            return Symmetry(std::move(out), code());
        }

    private:
        void validate_and_set_buffer_(const ArrayOption& options) {
            check(m_code.type == 'C' and m_code.order > 0, "{} symmetry is not supported", m_code.to_string());

            isize n_matrices = m_code.order - 1; // -1 to remove the identity from the matrices
            if (options.device.is_cpu()) {
                m_array = array_type(n_matrices, options);
                details::set_cx_symmetry_matrices(m_array.span_1d());
            } else {
                // Create a new sync stream so that the final copy doesn't sync the default cpu stream of the user.
                const auto guard = StreamGuard(Device{}, Stream::DEFAULT);
                array_type cpu_matrices(n_matrices);
                details::set_cx_symmetry_matrices(cpu_matrices.span_1d());

                // Copy to gpu.
                m_array = array_type(n_matrices, options);
                copy(std::move(cpu_matrices), m_array);
            }
        }

    private:
        array_type m_array;
        SymmetryCode m_code;
    };
}

namespace noa::xform::details {
    /// 3d or 4d iwise operator used to symmetrize 2d or 3d array(s).
    ///  * Can apply a per batch affine transformation before and after the symmetry.
    ///  * The symmetry is applied around a specified center.
    template<usize B, usize R,
            nt::integer Index,
            nt::span_contiguous_nd<1> SymmetryMatrices,
            nt::interpolator_nd<R> Interpolator,
            nt::readable_nd<B + R> Input,
            nt::writable_nd<B + R> Output,
            nt::readable_nd<B> PreInvAffine,
            nt::readable_nd<B> PostInvAffine>
    requires (R == 2 or R == 3)
    class Symmetrize {
    public:
        using index_type = Index;
        using symmetry_matrices_type = SymmetryMatrices;
        using interpolator_type = Interpolator;
        using input_type = Input;
        using output_type = Output;
        using batched_pre_inverse_affine_type = PreInvAffine;
        using batched_post_inverse_affine_type = PostInvAffine;

        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;

        using symmetry_matrix_type = nt::value_type_t<symmetry_matrices_type>;
        static_assert(nt::mat_of_shape<symmetry_matrix_type, R, R>);
        using coord_type = nt::value_type_t<symmetry_matrix_type>;
        using vec_type = Vec<coord_type, R>;

        // Expect the (truncated) affine with the same precision as symmetry matrices.
        using pre_inverse_affine_type = nt::value_type_t<batched_pre_inverse_affine_type>;
        using post_inverse_affine_type = nt::value_type_t<batched_post_inverse_affine_type>;
        static_assert(nt::empty<pre_inverse_affine_type> or
                      (nt::same_as<coord_type, nt::value_type_t<pre_inverse_affine_type>> and
                       (nt::mat_of_shape<pre_inverse_affine_type, R, R + 1> or
                        nt::mat_of_shape<pre_inverse_affine_type, R + 1, R + 1>)));
        static_assert(nt::empty<post_inverse_affine_type> or
                      (nt::same_as<coord_type, nt::value_type_t<post_inverse_affine_type>> and
                       (nt::mat_of_shape<post_inverse_affine_type, R, R + 1> or
                        nt::mat_of_shape<post_inverse_affine_type, R + 1, R + 1>)));

    public:
        constexpr Symmetrize(
            const input_type& input,
            const output_type& output,
            const interpolator_type& interpolator,
            symmetry_matrices_type symmetry_inverse_rotation_matrices,
            const vec_type& symmetry_center,
            input_real_type symmetry_scaling,
            const batched_pre_inverse_affine_type& pre_inverse_affine_matrices,
            const batched_post_inverse_affine_type& post_inverse_affine_matrices
        ) noexcept :
            m_input(input), m_output(output), m_interpolator(interpolator),
            m_symmetry_matrices(symmetry_inverse_rotation_matrices),
            m_symmetry_center(symmetry_center),
            m_symmetry_scaling(symmetry_scaling),
            m_pre_inverse_affine_matrices(pre_inverse_affine_matrices),
            m_post_inverse_affine_matrices(post_inverse_affine_matrices) {}

        NOA_HD constexpr void operator()(const Vec<index_type, B + R>& batched_indices) const {
            const auto& [batches, indices] = batched_indices.template split<B>();
            auto coordinates = indices.template as<coord_type>();

            input_value_type value;
            if constexpr (nt::empty<post_inverse_affine_type> and nt::empty<pre_inverse_affine_type>) {
                value = m_input[batched_indices]; // skip interpolation if possible
            } else {
                coordinates = transform_vector(m_post_inverse_affine_matrices[batches], coordinates);

                auto i_coord = coordinates;
                if constexpr (not nt::empty<pre_inverse_affine_type>)
                    i_coord = transform_vector(m_pre_inverse_affine_matrices[batches], i_coord);
                value = m_interpolator.get(m_input[batches], i_coord);
            }

            coordinates -= m_symmetry_center;
            for (const auto& symmetry_matrix: m_symmetry_matrices) {
                auto i_coord = symmetry_matrix * coordinates + m_symmetry_center;
                if constexpr (not nt::empty<pre_inverse_affine_type>)
                    i_coord = transform_vector(m_pre_inverse_affine_matrices[batches], i_coord);
                value += m_interpolator.get(m_input, i_coord);
            }
            value *= m_symmetry_scaling;

            m_output[batched_indices] = static_cast<output_value_type>(value);
        }

    private:
        input_type m_input;
        output_type m_output;
        interpolator_type m_interpolator;
        symmetry_matrices_type m_symmetry_matrices;
        vec_type m_symmetry_center;
        input_real_type m_symmetry_scaling;
        NOA_NO_UNIQUE_ADDRESS batched_pre_inverse_affine_type m_pre_inverse_affine_matrices;
        NOA_NO_UNIQUE_ADDRESS batched_post_inverse_affine_type m_post_inverse_affine_matrices;
    };

    template<typename>
    struct is_symmetry : std::false_type {};
    template<typename T, usize N, ArrayOwnership O>
    struct is_symmetry<Symmetry<T, N, O>> : std::true_type {};

    template<typename T, usize N>
    concept symmetry_nd = is_symmetry<std::decay_t<T>>::value and std::decay_t<T>::SIZE == N;

    template<typename Input, typename Output, typename Coord, usize R>
    void check_parameters_symmetrize_nd(const Input& input, const Output& output, const Symmetry<Coord, R>& symmetry) {
        check(not input.is_empty() and not output.is_empty() and not symmetry.is_empty(), "Empty array detected");

        // Check batch axes are compatible.
        constexpr usize N = nt::array_size_v<Output>;
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

        const Device device = output.device();
        check(input.device() == device and symmetry.device() == device,
              "The input array/texture, output array and symmetry matrices must be on the same device, but got input:device={} and output:device={}, symmetry:device={}",
              input.device(), device, symmetry.device());
        check(nd::are_elements_unique(output.strides(), output.shape()),
              "The elements in the output should not overlap in memory, otherwise a data-race might occur. Got output:strides={} and output:shape={}",
              output.strides(), output.shape());

        if constexpr (nt::array<Input>) {
            check(not are_overlapped(input, output),
                  "The input and output arrays should not overlap");
        } else {
            check(input.device().is_gpu() or not are_overlapped(input.cpu(), output),
                  "The input and output arrays should not overlap");
            check(input.border() == Border::ZERO,
                  "Texture border mode is expected to be {}, but got {}",
                  Border::ZERO, input.border());
        }
    }

    template<usize R, typename Index, bool IS_GPU = false,
             typename Input, typename Output, typename Symmetry, typename PreMatrix, typename PostMatrix>
    void launch_symmetrize_nd(
        Input&& input, Output&& output, Symmetry&& symmetry,
        PreMatrix&& pre_inverse_matrices,
        PostMatrix&& post_inverse_matrices,
        auto options
    ) {
        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize B = N - R;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N, Index>;
        auto output_span = output.span().template as_index<Index>();
        auto output_accessor = output_accessor_t(output_span.get(), output_span.strides());

        // Batch the optional pre/post matrices.
        auto pre_inverse_matrices_accessor = xform_into_accessor<true>(pre_inverse_matrices);
        auto post_inverse_matrices_accessor = xform_into_accessor<true>(post_inverse_matrices);
        using pre_inverse_matrices_accessor_t = decltype(pre_inverse_matrices_accessor);
        using post_inverse_matrices_accessor_t = decltype(post_inverse_matrices_accessor);

        // Prepare the symmetry.
        using real_t = nt::mutable_value_type_twice_t<Input>;
        auto symmetry_matrices = symmetry.span();
        auto symmetry_scaling = options.normalize ? 1 / static_cast<real_t>(symmetry_matrices.size() + 1) : real_t{1};
        using symmetry_matrices_t = decltype(symmetry_matrices);

        // Set the default symmetry center to the input center (assuming center is n//2).
        using coord_t = nt::value_type_t<Symmetry>;
        auto input_shape_r = input.shape().template pop_front<B>().template as<Index>();
        for (auto&& [center, size]: noa::zip(options.symmetry_center, input_shape_r))
            if (center == std::numeric_limits<f64>::max())
                center = static_cast<f64>(size / 2);

        auto launch_iwise = [&](auto interp) {
            auto result = prepare_interpolation_inputs<R, interp(), Border::ZERO, IS_GPU, Index, coord_t, false>(input);
            using interpolator_t = decltype(result)::interpolator_type;
            using accessor_t = decltype(result)::accessor_type;

            using op_t = Symmetrize<
                B, R, Index, symmetry_matrices_t, interpolator_t, accessor_t, output_accessor_t,
                pre_inverse_matrices_accessor_t, post_inverse_matrices_accessor_t>;

            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(output_span.shape(), output.device(),
               op_t(result.accessor, output_accessor, result.interpolator,
                    symmetry_matrices, options.symmetry_center.template as<coord_t>(), symmetry_scaling,
                    pre_inverse_matrices_accessor, post_inverse_matrices_accessor),
               std::forward<Input>(input),
               std::forward<Output>(output),
               std::forward<Symmetry>(symmetry),
               std::forward<PreMatrix>(pre_inverse_matrices),
               std::forward<PostMatrix>(post_inverse_matrices));
        };

        if constexpr (nt::texture_decay<Input>)
            options.interp = input.interp();
        switch (options.interp) {
            case Interp::NEAREST:            return launch_iwise(WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_iwise(WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_iwise(WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_iwise(WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_iwise(WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_iwise(WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_iwise(WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_iwise(WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            case Interp::LANCZOS4:           return launch_iwise(WrapInterp<Interp::LANCZOS4>{});
            case Interp::LANCZOS6:           return launch_iwise(WrapInterp<Interp::LANCZOS6>{});
            case Interp::LANCZOS8:           return launch_iwise(WrapInterp<Interp::LANCZOS8>{});
            case Interp::LANCZOS4_FAST:      return launch_iwise(WrapInterp<Interp::LANCZOS4_FAST>{});
            case Interp::LANCZOS6_FAST:      return launch_iwise(WrapInterp<Interp::LANCZOS6_FAST>{});
            case Interp::LANCZOS8_FAST:      return launch_iwise(WrapInterp<Interp::LANCZOS8_FAST>{});
        }
    }

    template<typename T, typename Symmetry, typename Output, usize R,
             typename U = std::remove_reference_t<T>,
             typename Matrix = nt::value_type_t<T>,
             typename MatrixValue = nt::value_type_t<Matrix>,
             typename Coord = nt::value_type_t<Symmetry>,
             usize N = nt::array_size_v<Output>>
    concept symmetry_matrix_parameter_nd =
        nt::empty<T> or
        (nt::same_as<Coord, MatrixValue> and (
            nt::mat_of_shape<U, N, N + 1> or
            nt::mat_of_shape<U, N + 1, N + 1> or
            (nt::array_nd<U, N - R> and
             (nt::mat_of_shape<Matrix, N, N + 1> or nt::mat_of_shape<Matrix, N + 1, N + 1>))
         ));

    template<usize R, typename Input, typename Output, typename Symmetry, typename PreMatrix, typename PostMatrix>
    concept symmetrizable_nd =
        nt::readable_array_or_texture_rd_decay<Input, R> and
        nt::writable_array_decay<Output> and
        nt::real_or_complex<nt::value_type_t<Input>> and
        nt::compatible_types<nt::value_type_t<Input>, nt::value_type_t<Output>> and
        nt::array_size_v<Input> == nt::array_size_v<Output> and nt::array_size_v<Output> >= R and
        symmetry_nd<Symmetry, R> and
        symmetry_matrix_parameter_nd<PreMatrix, Symmetry, Output, R> and
        symmetry_matrix_parameter_nd<PostMatrix, Symmetry, Output, R>;
}

namespace noa::xform {
    template<usize N>
    struct SymmetrizeOptions {
        /// (D)HW coordinates of the symmetry center.
        /// By default, use the input center, defined as `shape//2` (integer division).
        Vec<f64, N> symmetry_center = Vec<f64, N>::from_value(std::numeric_limits<f64>::max());

        /// Interpolation method. All interpolation modes are supported.
        /// This is ignored if the input is a texture.
        Interp interp{Interp::LINEAR};

        /// Whether the symmetrized output should be normalized to have the same value range as the input.
        /// If false, output values end up being scaled by the symmetry count.
        bool normalize{true};
    };

    /// Symmetrizes 2D array(s).
    /// \param[in] input:
    ///     ((Bi..,)Hi,Wi) Input 2D array(s) or 2D texture(s).
    ///     The batch axes are broadcast to the output batch axes.
    /// \param[out] output:
    ///     ((Bo..,)Ho,Wo) Output 2D array(s).
    /// \param[in] symmetry:
    ///     Symmetry operator.
    /// \param[in] options:
    ///     Symmetry and interpolation options.
    ///     During transformation, out-of-bound elements are set to 0, i.e., Border::ZERO is used.
    /// \param[in] pre_inverse_matrices:
    ///     Mat23, Mat33, a contiguous (Bo..) array of these types matching the output batches array, or Empty.
    ///     HW inverse affine matrices to apply before the symmetry.
    ///     This is used to align the input with the symmetry axis/center.
    ///     This needs to be applied for each symmetry count as opposed to the post-transformation which
    ///     is applied once per pixel, so it may be more efficient to apply it separately using transform_2d.
    /// \param[in] post_inverse_matrices:
    ///     Mat23, Mat33, a contiguous (Bo..) array of these types matching the output batches array, or Empty.
    ///     HW inverse affine matrix to apply after the symmetry.
    ///     This is often used to move the symmetrized output to the original input location,
    ///     as if the symmetry was applied in-place.
    /// \note
    ///     The input and output array can have different shapes ((Hi,Wi) vs (Ho,Wo)). The output window starts at
    ///     the same index as the input window, so by entering a translation in pre_/post_inverse_matrices, one can
    ///     move the center of the output window relative to the input window, e.g., to render only a specific subregion.
    template<typename Input, typename Output, typename Symmetry, typename PreMatrix = Empty, typename PostMatrix = Empty>
        requires details::symmetrizable_nd<2, Input, Output, Symmetry, PreMatrix, PostMatrix>
    void symmetrize_2d(
        Input&& input, Output&& output, Symmetry&& symmetry,
        const SymmetrizeOptions<2>& options = {},
        PreMatrix&& pre_inverse_matrices = {},
        PostMatrix&& post_inverse_matrices = {}
    ) {
        details::check_parameters_symmetrize_nd(input, output, symmetry);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::mutable_value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(nd::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
                      nd::is_accessor_access_safe<i32>(output.strides(), output.shape()),
                      "isize indexing not instantiated for GPU devices");
                details::launch_symmetrize_nd<2, i32, true>(
                    std::forward<Input>(input), std::forward<Output>(output),
                    std::forward<Symmetry>(symmetry),
                    std::forward<PreMatrix>(pre_inverse_matrices),
                    std::forward<PostMatrix>(post_inverse_matrices),
                    options);
                return;
            }
            #else
            panic_no_gpu_backend();
            #endif
        }
        details::launch_symmetrize_nd<2, isize>(
            std::forward<Input>(input), std::forward<Output>(output),
            std::forward<Symmetry>(symmetry),
            std::forward<PreMatrix>(pre_inverse_matrices),
            std::forward<PostMatrix>(post_inverse_matrices),
            options);
    }

    /// Symmetrizes 3D array(s).
    /// \param[in] input:
    ///     ((Bi..,)Di,Hi,Wi) Input 3D array(s) or 3D texture(s).
    ///     The batch axes are broadcast to the output batch axes.
    /// \param[out] output:
    ///     ((Bo..,)Do,Ho,Wo) Output 3D array(s).
    /// \param[in] symmetry:
    ///     Symmetry operator.
    /// \param[in] options:
    ///     Symmetry and interpolation options.
    ///     During transformation, out-of-bound elements are set to 0, i.e., Border::ZERO is used.
    /// \param[in] pre_inverse_matrices:
    ///     Mat34, Mat44, a contiguous (Bo..) array of these types matching the output batches array, or Empty.
    ///     DHW inverse affine matrices to apply before the symmetry.
    ///     This is used to align the input with the symmetry axis/center.
    ///     This needs to be applied for each symmetry count as opposed to the post-transformation which
    ///     is applied once per pixel, so it may be more efficient to apply it separately using transform_2d.
    /// \param[in] post_inverse_matrices:
    ///     Mat34, Mat44, a contiguous (Bo..) array of these types matching the output batches array, or Empty.
    ///     DHW inverse affine matrix to apply after the symmetry.
    ///     This is often used to move the symmetrized output to the original input location,
    ///     as if the symmetry was applied in-place.
    /// \note
    ///     The input and output array can have different shapes ((Di,Hi,Wi) vs (Do,Ho,Wo)). The output window starts at
    ///     the same index as the input window, so by entering a translation in pre_/post_inverse_matrices, one can
    ///     move the center of the output window relative to the input window, e.g., to render only a specific subregion.
    template<typename Input, typename Output, typename Symmetry, typename PreMatrix = Empty, typename PostMatrix = Empty>
        requires details::symmetrizable_nd<3, Input, Output, Symmetry, PreMatrix, PostMatrix>
    void symmetrize_3d(
        Input&& input, Output&& output, Symmetry&& symmetry,
        const SymmetrizeOptions<3>& options = {},
        PreMatrix&& pre_inverse_matrices = {},
        PostMatrix&& post_inverse_matrices = {}
    ) {
        details::check_parameters_symmetrize_nd(input, output, symmetry);

        if (output.device().is_gpu()) {
            #ifdef NOA_ENABLE_GPU
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::mutable_value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(nd::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
                      nd::is_accessor_access_safe<i32>(output.strides(), output.shape()),
                      "isize indexing not instantiated for GPU devices");
                details::launch_symmetrize_nd<3, i32, true>(
                    std::forward<Input>(input), std::forward<Output>(output),
                    std::forward<Symmetry>(symmetry),
                    std::forward<PreMatrix>(pre_inverse_matrices),
                    std::forward<PostMatrix>(post_inverse_matrices),
                    options);
                return;
            }
            #else
            panic_no_gpu_backend();
            #endif
        }
        details::launch_symmetrize_nd<3, isize>(
            std::forward<Input>(input), std::forward<Output>(output),
            std::forward<Symmetry>(symmetry),
            std::forward<PreMatrix>(pre_inverse_matrices),
            std::forward<PostMatrix>(post_inverse_matrices),
            options);
    }
}
