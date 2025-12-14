#pragma once

#include "noa/core/Interpolation.hpp"
#include "noa/core/geometry/Symmetry.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"
#include "noa/unified/Utilities.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Interpolation.hpp"

namespace noa::geometry {
    /// Symmetry utility class, storing and managing symmetry rotation matrices.
    /// \note By convention, the identity matrix is not stored as
    ///       it is implicitly applied by the "symmetrize" functions.
    /// \note Supported symmetries:
    ///     - CX, with X being a non-zero positive number.
    /// TODO Add quaternions and more symmetries...
    template<typename Real, usize N>
    class Symmetry {
    public:
        using value_type = Real;
        using matrix_type = Mat<value_type, N, N>;
        using array_type = Array<matrix_type>;
        using shared_type = array_type::shared_type;

        /// Construct an empty object.
        constexpr Symmetry() = default;

        /// Allocates and initializes the symmetry matrices on the device.
        constexpr explicit Symmetry(const SymmetryCode& code, const ArrayOption& options = {}) : m_code(code) {
            validate_and_set_buffer_(options);
        }

        constexpr explicit Symmetry(std::string_view code, const ArrayOption& options = {}) {
            const auto parsed_code = SymmetryCode::from_string(code);
            check(parsed_code.has_value(), "Failed to parse \"{}\" to a valid symmetry", code);
            m_code = parsed_code.value();
            validate_and_set_buffer_(options);
        }

        /// Imports existing matrices. No validation is done.
        /// The identity matrix should not be included as it is implicitly applied by the "symmetrize" functions.
        template<nt::almost_any_of<array_type> A>
        constexpr explicit Symmetry(A&& matrices, const SymmetryCode& code) :
            m_buffer(std::forward<A>(matrices)),
            m_code(code)
        {
            check(ni::is_contiguous_vector(m_buffer),
                  "The symmetry matrices should be saved in a contiguous vector, "
                  "but got matrices:shape={} and matrices:strides={}",
                  m_buffer.shape(), m_buffer.strides());
        }

    public:
        [[nodiscard]] auto code() const -> SymmetryCode { return m_code; }
        [[nodiscard]] auto device() const { return m_buffer.device(); }
        [[nodiscard]] auto options() const { return m_buffer.options(); }
        [[nodiscard]] auto is_empty() const { return m_buffer.is_empty(); }

        [[nodiscard]] auto span() const -> Span<const matrix_type> {
            return m_buffer.template span<const matrix_type>();
        }

        [[nodiscard]] auto share() const& -> const shared_type& { return m_buffer.share(); }
        [[nodiscard]] auto share() && -> shared_type&& { return std::move(m_buffer).share(); }

        [[nodiscard]] auto array() const& -> const array_type& { return m_buffer; }
        [[nodiscard]] auto array() && -> array_type&& { return std::move(m_buffer); }

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
                m_buffer = array_type(n_matrices, options);
                details::set_cx_symmetry_matrices(m_buffer.span_1d_contiguous());
            } else {
                // Create a new sync stream so that the final copy doesn't sync the default cpu stream of the user.
                const auto guard = StreamGuard(Device{}, Stream::DEFAULT);
                array_type cpu_matrices(n_matrices);
                details::set_cx_symmetry_matrices(cpu_matrices.span_1d_contiguous());

                // Copy to gpu.
                m_buffer = array_type(n_matrices, options);
                copy(std::move(cpu_matrices), m_buffer);
            }
        }

    private:
        array_type m_buffer;
        SymmetryCode m_code;
    };
}

namespace noa::geometry::details {
    /// 3d or 4d iwise operator used to symmetrize 2d or 3d array(s).
    ///  * Can apply a per batch affine transformation before and after the symmetry.
    ///  * The symmetry is applied around a specified center.
    template<usize N,
            nt::integer Index,
            nt::span_contiguous_nd<1> SymmetryMatrices,
            nt::interpolator_nd<N> Input,
            nt::writable_nd<N + 1> Output,
            nt::batched_parameter PreInvAffine,
            nt::batched_parameter PostInvAffine>
    requires (N == 2 or N == 3)
    class Symmetrize {
    public:
        using index_type = Index;
        using symmetry_matrices_type = SymmetryMatrices;
        using input_type = Input;
        using output_type = Output;
        using batched_pre_inverse_affine_type = PreInvAffine;
        using batched_post_inverse_affine_type = PostInvAffine;

        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;

        using symmetry_matrix_type = nt::value_type_t<symmetry_matrices_type>;
        static_assert(nt::mat_of_shape<symmetry_matrix_type, N, N>);
        using coord_type = nt::value_type_t<symmetry_matrix_type>;
        using vec_type = Vec<coord_type, N>;

        // Expect the (truncated) affine with the same precision as symmetry matrices.
        using pre_inverse_affine_type = nt::value_type_t<batched_pre_inverse_affine_type>;
        using post_inverse_affine_type = nt::value_type_t<batched_post_inverse_affine_type>;
        static_assert(nt::empty<pre_inverse_affine_type> or
                      (nt::same_as<coord_type, nt::value_type_t<pre_inverse_affine_type>> and
                       (nt::mat_of_shape<pre_inverse_affine_type, N, N + 1> or
                        nt::mat_of_shape<pre_inverse_affine_type, N + 1, N + 1>)));
        static_assert(nt::empty<post_inverse_affine_type> or
                      (nt::same_as<coord_type, nt::value_type_t<post_inverse_affine_type>> and
                       (nt::mat_of_shape<post_inverse_affine_type, N, N + 1> or
                        nt::mat_of_shape<post_inverse_affine_type, N + 1, N + 1>)));

    public:
        constexpr Symmetrize(
            const input_type& input,
            const output_type& output,
            symmetry_matrices_type symmetry_inverse_rotation_matrices,
            const vec_type& symmetry_center,
            input_real_type symmetry_scaling,
            const batched_pre_inverse_affine_type& pre_inverse_affine_matrices,
            const batched_post_inverse_affine_type& post_inverse_affine_matrices
        ) noexcept :
            m_input(input), m_output(output),
            m_symmetry_matrices(symmetry_inverse_rotation_matrices),
            m_symmetry_center(symmetry_center),
            m_symmetry_scaling(symmetry_scaling),
            m_pre_inverse_affine_matrices(pre_inverse_affine_matrices),
            m_post_inverse_affine_matrices(post_inverse_affine_matrices) {}

        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        NOA_HD constexpr void operator()(index_type batch, I... indices) const {
            auto coordinates = vec_type::from_values(indices...);

            input_value_type value;
            if constexpr (nt::empty<post_inverse_affine_type> and
                          nt::empty<pre_inverse_affine_type> and
                          nt::readable_nd<input_type, N + 1>) {
                value = m_input(batch, indices...); // skip interpolation if possible
            } else {
                coordinates = transform_vector(m_post_inverse_affine_matrices[batch], coordinates);

                auto i_coord = coordinates;
                if constexpr (not nt::empty<pre_inverse_affine_type>)
                    i_coord = transform_vector(m_pre_inverse_affine_matrices[batch], i_coord);
                value = m_input.interpolate_at(i_coord, batch);
            }

            coordinates -= m_symmetry_center;
            for (const auto& symmetry_matrix: m_symmetry_matrices) {
                auto i_coord = symmetry_matrix * coordinates + m_symmetry_center;
                if constexpr (not nt::empty<pre_inverse_affine_type>)
                    i_coord = transform_vector(m_pre_inverse_affine_matrices[batch], i_coord);
                value += m_input.interpolate_at(i_coord, batch);
            }
            value *= m_symmetry_scaling;

            m_output(batch, indices...) = static_cast<output_value_type>(value);
        }

    private:
        input_type m_input;
        output_type m_output;
        symmetry_matrices_type m_symmetry_matrices;
        vec_type m_symmetry_center;
        input_real_type m_symmetry_scaling;
        NOA_NO_UNIQUE_ADDRESS batched_pre_inverse_affine_type m_pre_inverse_affine_matrices;
        NOA_NO_UNIQUE_ADDRESS batched_post_inverse_affine_type m_post_inverse_affine_matrices;
    };

    template<typename T, usize N>
    concept symmetry_nd = nt::any_of<std::decay_t<T>, Symmetry<f32, N>, Symmetry<f64, N>>;

    template<typename T, typename Coord, usize N,
             typename U = std::remove_reference_t<T>,
             typename V = nt::value_type_t<T>,
             typename C = nt::value_type_t<V>>
    concept symmetry_matrix_parameter_nd =
        nt::empty<T> or
        (nt::same_as<Coord, C> and (
            nt::mat_of_shape<U, N, N + 1> or
            nt::mat_of_shape<U, N + 1, N + 1> or
            (nt::varray<U> and (nt::mat_of_shape<V, N, N + 1> or nt::mat_of_shape<V, N + 1, N + 1>))
         ));

    template<typename Input, typename Output, typename Coord, usize N>
    void check_parameters_symmetrize_nd(const Input& input, const Output& output, const Symmetry<Coord, N>& symmetry) {
        check(not input.is_empty() and not output.is_empty() and not symmetry.is_empty(), "Empty array detected");
        check(N == 3 or (input.shape()[1] == 1 and output.shape()[1] == 1),
              "The input and output arrays should be 2d, but got input:shape={}, output:shape={}",
              input.shape(), output.shape());
        check(input.shape() == output.shape(),
              "The input and output shapes are not compatible, got input:shape={}, output:shape={}",
              input.shape(), output.shape());

        const Device device = output.device();

        check(input.device() == device and symmetry.device() == device,
              "The input array/texture, output array and symmetry matrices must be on the same device, "
              "but got input:device={} and output:device={}, symmetry:device={}",
              input.device(), device, symmetry.device());
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
            check(input.border() == Border::ZERO, "Texture border mode is expected to be {}, but got {}",
                  Border::ZERO, input.border());
        }
    }

    template<usize N, typename Index, bool IS_GPU = false,
             typename Input, typename Output, typename Symmetry, typename PreMatrix, typename PostMatrix>
    void launch_symmetrize_nd(
        Input&& input, Output&& output, Symmetry&& symmetry,
        PreMatrix&& pre_inverse_matrices,
        PostMatrix&& post_inverse_matrices,
        auto options
    ) {
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N + 1, Index>;
        auto output_accessor = output_accessor_t(
            output.get(), output.strides().template filter_nd<N>().template as<Index>());

        // Batch the optional pre/post matrices.
        auto batched_pre_inverse_matrices = nd::to_batched_transform<true>(pre_inverse_matrices);
        auto batched_post_inverse_matrices = nd::to_batched_transform<true>(post_inverse_matrices);

        // Prepare the symmetry.
        using real_t = nt::mutable_value_type_twice_t<Input>;
        auto symmetry_matrices = symmetry.array().span_1d_contiguous();
        auto symmetry_scaling =
            options.normalize ? 1 / static_cast<real_t>(symmetry_matrices.size() + 1) : real_t{1};

        // Set the default symmetry center to the input center (assuming center is n//2).
        using coord_t = nt::value_type_t<Symmetry>;
        auto input_shape_nd = input.shape().template filter_nd<N>().pop_front().template as<Index>();
        for (usize i{}; auto& center: options.symmetry_center)
            if (center == std::numeric_limits<f64>::max())
                center = static_cast<f64>(input_shape_nd[i++] / 2);

        auto launch_iwise = [&](auto interp) {
            auto interpolator = nd::to_interpolator<N, interp(), Border::ZERO, Index, coord_t, IS_GPU>(input);
            using op_t = details::Symmetrize<
                N, Index, decltype(symmetry_matrices), decltype(interpolator), output_accessor_t,
                decltype(batched_pre_inverse_matrices), decltype(batched_post_inverse_matrices)>;

            iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(output.shape().template filter_nd<N>().template as<Index>(), output.device(),
               op_t(interpolator, output_accessor,
                    symmetry_matrices, options.symmetry_center.template as<coord_t>(), symmetry_scaling,
                    batched_pre_inverse_matrices, batched_post_inverse_matrices),
               std::forward<Input>(input),
               std::forward<Output>(output),
               std::forward<Symmetry>(symmetry),
               std::forward<PreMatrix>(pre_inverse_matrices),
               std::forward<PostMatrix>(post_inverse_matrices));
        };

        if constexpr (nt::texture_decay<Input>)
            options.interp = input.interp();
        switch (options.interp) {
            case Interp::NEAREST:            return launch_iwise(nd::WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_iwise(nd::WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_iwise(nd::WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_iwise(nd::WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_iwise(nd::WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_iwise(nd::WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_iwise(nd::WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_iwise(nd::WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            case Interp::LANCZOS4:           return launch_iwise(nd::WrapInterp<Interp::LANCZOS4>{});
            case Interp::LANCZOS6:           return launch_iwise(nd::WrapInterp<Interp::LANCZOS6>{});
            case Interp::LANCZOS8:           return launch_iwise(nd::WrapInterp<Interp::LANCZOS8>{});
            case Interp::LANCZOS4_FAST:      return launch_iwise(nd::WrapInterp<Interp::LANCZOS4_FAST>{});
            case Interp::LANCZOS6_FAST:      return launch_iwise(nd::WrapInterp<Interp::LANCZOS6_FAST>{});
            case Interp::LANCZOS8_FAST:      return launch_iwise(nd::WrapInterp<Interp::LANCZOS8_FAST>{});
        }
    }
}

namespace noa::geometry {
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

    /// Symmetrizes 2d array(s).
    /// \tparam PreMatrix, PostMatrix       Mat23, Mat33, a varray of these types, or Empty.
    /// \param[in] input                    Input 2d array(s).
    /// \param[out] output                  Output 2d array(s).
    /// \param[in] symmetry                 Symmetry operator.
    /// \param[in] options                  Symmetry and interpolation options.
    /// \param[in] pre_inverse_matrices     HW inverse affine matrices to apply before the symmetry.
    ///                                     This is used to align the input with the symmetry axis/center.
    ///                                     In practice, this needs to be applied for each symmetry count
    ///                                     as opposed to the post-transformation which is applied once per pixel,
    ///                                     so it may be more efficient to apply it separately using transform_2d.
    /// \param[in] post_inverse_matrices    HW inverse affine matrix to apply after the symmetry.
    ///                                     This is often used to move the symmetrized output to the original input
    ///                                     location, as if the symmetry was applied in-place.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. Border::ZERO is used.
    /// \note The input and output array can have different shapes. The output window starts at the same index
    ///       as the input window, so by entering a translation in pre-/post-matrices, one can move the center
    ///       of the output window relative to the input window.
    template<nt::varray_or_texture_decay_of_real_or_complex Input,
             nt::writable_varray_decay Output,
             details::symmetry_nd<2> Symmetry,
             details::symmetry_matrix_parameter_nd<nt::value_type_t<Symmetry>, 2> PreMatrix = Empty,
             details::symmetry_matrix_parameter_nd<nt::value_type_t<Symmetry>, 2> PostMatrix = Empty>
    requires nt::compatible_types<nt::value_type_t<Input>, nt::value_type_t<Output>>
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

    /// Symmetrizes 3d array(s).
    /// \tparam PreMatrix, PostMatrix       Mat34, Mat44, a varray of these types, or Empty.
    /// \param[in] input                    Input 3d array(s).
    /// \param[out] output                  Output 3d array(s).
    /// \param[in] symmetry                 Symmetry operator.
    /// \param[in] options                  Symmetry and interpolation options.
    /// \param[in] pre_inverse_matrices     HW inverse affine matrices to apply before the symmetry.
    ///                                     This is used to align the input with the symmetry axis/center.
    ///                                     In practice, this needs to be applied for each symmetry count
    ///                                     as opposed to the post-transformation which is applied once per pixel,
    ///                                     so it may be more efficient to apply it separately using transform_3d.
    /// \param[in] post_inverse_matrices    HW inverse affine matrix to apply after the symmetry.
    ///                                     This is often used to move the symmetrized output to the original input
    ///                                     location, as if the symmetry was applied in-place.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. Border::ZERO is used.
    /// \note The input and output array can have different shapes. The output window starts at the same index
    ///       as the input window, so by entering a translation in pre-/post-matrices, one can move the center
    ///       of the output window relative to the input window.
    template<nt::varray_or_texture_decay_of_real_or_complex Input,
             nt::writable_varray_decay Output,
             details::symmetry_nd<3> Symmetry,
             details::symmetry_matrix_parameter_nd<nt::value_type_t<Symmetry>, 3> PreMatrix = Empty,
             details::symmetry_matrix_parameter_nd<nt::value_type_t<Symmetry>, 3> PostMatrix = Empty>
    requires nt::compatible_types<nt::value_type_t<Input>, nt::value_type_t<Output>>
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
