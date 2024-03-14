#pragma once

#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/geometry/Symmetry.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"
#include "noa/unified/Utilities.hpp"
#include "noa/unified/Iwise.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Interpolator.hpp"
#endif

namespace noa::geometry {
    /// Symmetry utility class, storing and managing symmetry rotation matrices.
    /// \note By convention, the identity matrix is not stored as
    ///       it is implicitly applied by the "symmetrize" functions.
    /// \note Supported symmetries:
    ///     - CX, with X being a non-zero positive number.
    ///     - TODO WIP...
    template<typename Real, size_t N>
    class Symmetry {
    public:
        using value_type = Real;
        using matrix_type = std::conditional_t<N == 2, Mat22<value_type>, Mat33<value_type>>;
        using array_type = Array<matrix_type>;
        using shared_type = array_type::shared_type;

        /// Construct an empty object.
        constexpr Symmetry() = default;

        /// Allocates and initializes the symmetry matrices on the device.
        constexpr explicit Symmetry(const SymmetryCode& code, const ArrayOption& options = {}) : m_code(code) {
            validate_and_set_buffer_(options);
        }

        constexpr explicit Symmetry(std::string_view code, const ArrayOption& options = {}) {
            auto parsed_code = SymmetryCode::from_string(code);
            check(parsed_code.has_value(), "Failed to parse \"{}\" to a valid symmetry", code);
            m_code = parsed_code.value();
            validate_and_set_buffer_(options);
        }

        /// Imports existing matrices. No validation is done.
        /// The identity matrix should not be included as it is implicitly applied by the "symmetrize" functions.
        template<typename A>
        requires std::is_same_v<std::decay_t<A>, array_type>
        constexpr explicit Symmetry(A&& matrices, const SymmetryCode& code)
                : m_buffer(std::forward<A>(matrices)), m_code(code) {
            check(ni::is_contiguous_vector(m_buffer),
                  "The symmetry matrices should be saved in a contiguous vector, "
                  "but got matrices:shape={} and matrices:strides={}",
                  m_buffer.shape(), m_buffer.strides());
        }

    public:
        [[nodiscard]] auto code() const -> SymmetryCode { return m_code; }
        [[nodiscard]] auto span() const -> Span<const matrix_type> { return m_buffer.template span<const matrix_type>(); }
        [[nodiscard]] auto share() const& -> const shared_type& { return m_buffer.share(); }
        [[nodiscard]] auto share() && -> shared_type { return std::move(m_buffer).share(); }

        [[nodiscard]] auto array() const -> const array_type& { return m_buffer; }
        [[nodiscard]] auto device() const { return m_buffer.device(); }
        [[nodiscard]] auto options() const { return m_buffer.options(); }
        [[nodiscard]] auto is_empty() const { return m_buffer.is_empty(); }

    private:
        void validate_and_set_buffer_(const ArrayOption& options) const {
            check(m_code.type == 'C' and m_code.order > 0, "{} symmetry is not supported", m_code.to_string());

            i64 n_matrices = m_code.order - 1; // -1 to remove the identity from the matrices
            if (options.device.is_cpu()) {
                m_buffer = Array<matrix_type>(n_matrices, options);
                set_cx_symmetry_matrices(m_buffer.span(), m_code.order);
            } else {
                // Create a new sync stream so that the final copy doesn't sync the default cpu stream of the user.
                const StreamGuard guard(Device{}, StreamMode::DEFAULT);
                Array<matrix_type> cpu_matrices(n_matrices);
                set_cx_symmetry_matrices(cpu_matrices.span(), m_code.order);

                // Copy to gpu.
                m_buffer = Array<matrix_type>(n_matrices);
                copy(std::move(cpu_matrices), m_buffer);
            }
        }

    private:
        Array<matrix_type> m_buffer;
        SymmetryCode m_code;
    };
}

namespace noa::geometry::guts {
    template<typename Input, typename Output, typename Coord, size_t N>
    void check_parameters_symmetrize_nd(const Input& input, const Output& output, const Symmetry<Coord, N>& symmetry) {
        check(not input.is_empty() and not output.is_empty() and not symmetry.is_empty(), "Empty array detected");
        check(N == 3 or (input.shape()[1] == 1 and output.shape()[1] == 1),
              "The input and output arrays should be 2d, but got input:shape={}, output:shape={}",
              input.shape(), output.shape());
        check(all(input.shape() == output.shape()),
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

        if constexpr (nt::is_varray_v<Input>) {
            check(not ni::are_overlapped(input, output),
                  "The input and output arrays should not overlap");
        } else {
            check(input.device().is_gpu() or not ni::are_overlapped(input.view(), output),
                  "The input and output arrays should not overlap");
            check(input.border_mode() == Border::ZERO, "Texture border mode is expected to be {}, but got {}",
                  Border::ZERO, input.border_mode());
        }
    }

    template<typename Index, typename Input, typename Output,
             typename Coord, size_t N, typename PreMatrix, typename PostMatrix>
    void launch_symmetrize_nd(
            const Input& input, const Output& output, const Symmetry<Coord, N>& symmetry,
            const Vec<Coord, N>& symmetry_center, Interp interp_mode, bool normalize,
            const PreMatrix& pre_matrix, const PostMatrix& post_matrix
    ) {
        using input_value_t = nt::mutable_value_type_t<Input>;
        using input_accessor_t = AccessorRestrict<const input_value_t, N + 1, Index>;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N + 1, Index>;

        auto get_strides_nd = [](const auto& array) {
            if constexpr (N == 2)
                return array.strides().filter(0, 2, 3).template as<Index>();
            else
                return array.strides().template as<Index>();
        };
        auto input_accessor = input_accessor_t(input.get(), get_strides_nd(input));
        auto output_accessor = output_accessor_t(output.get(), get_strides_nd(output));

        auto input_shape_nd = [&input] {
            if constexpr (N == 2)
                return input.shape().filter(2, 3).template as<Index>();
            else
                return input.shape().filter(1, 2, 3).template as<Index>();
        }();
        auto output_shape = [&output] {
            if constexpr (N == 2)
                return output.shape().filter(0, 2, 3).template as<Index>();
            else
                return output.shape().template as<Index>();
        }();

        // Set the default symmetry center to the input center (integer division).
        if (any(symmetry_center == std::numeric_limits<Coord>::max()))
            symmetry_center = (input_shape_nd.vec / 2).template as<Coord>();

        using real_t = nt::value_type_t<input_value_t>;
        using symmetry_span_t = decltype(symmetry.span());
        const i64 symmetry_count = symmetry.count();
        const auto symmetry_scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : real_t{1};
        const auto symmetry_matrices = symmetry.span();

        switch (interp_mode) {
            case Interp::NEAREST: {
                using interpolator_t = InterpolatorNd<N, Border::ZERO, Interp::NEAREST, input_accessor_t>;
                using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                auto interpolator = interpolator_t(input_accessor, input_shape_nd);
                auto op = op_t(interpolator, output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
            }
            case Interp::LINEAR:
            case Interp::LINEAR_FAST: {
                using interpolator_t = InterpolatorNd<N, Border::ZERO, Interp::LINEAR, input_accessor_t>;
                using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                auto interpolator = interpolator_t(input_accessor, input_shape_nd);
                auto op = op_t(interpolator, output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
            }
            case Interp::COSINE:
            case Interp::COSINE_FAST: {
                using interpolator_t = InterpolatorNd<N, Border::ZERO, Interp::COSINE, input_accessor_t>;
                using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                auto interpolator = interpolator_t(input_accessor, input_shape_nd);
                auto op = op_t(interpolator, output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
            }
            case Interp::CUBIC: {
                using interpolator_t = InterpolatorNd<N, Border::ZERO, Interp::CUBIC, input_accessor_t>;
                using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                auto interpolator = interpolator_t(input_accessor, input_shape_nd);
                auto op = op_t(interpolator, output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
            }
            case Interp::CUBIC_BSPLINE:
            case Interp::CUBIC_BSPLINE_FAST: {
                using interpolator_t = InterpolatorNd<N, Border::ZERO, Interp::CUBIC_BSPLINE, input_accessor_t>;
                using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                auto interpolator = interpolator_t(input_accessor, input_shape_nd);
                auto op = op_t(interpolator, output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
            }
        }
    }

#ifdef NOA_ENABLE_CUDA
    template<typename Index, typename Value, typename Output,
             typename Coord, size_t N, typename PreMatrix, typename PostMatrix>
    void launch_symmetrize_nd(
            const Texture<Value>& input, const Output& output, const Symmetry<Coord, N>& symmetry,
            const Vec<Coord, N>& symmetry_center, bool normalize,
            const PreMatrix& pre_matrix, const PostMatrix& post_matrix
    ) {
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N + 1, Index>;
        auto output_accessor = output_accessor_t(output.get(), [&output] {
            if constexpr (N == 2)
                return output.strides().filter(0, 2, 3).template as<Index>();
            else
                return output.strides().template as<Index>();
        }());

        auto input_shape_nd = [&input] {
            if constexpr (N == 2)
                return input.shape().filter(2, 3).template as<Index>();
            else
                return input.shape().filter(1, 2, 3).template as<Index>();
        }();
        auto output_shape = [&output] {
            if constexpr (N == 2)
                return output.shape().filter(0, 2, 3).template as<Index>();
            else
                return output.shape().template as<Index>();
        }();

        // Set the default symmetry center to the input center (integer division).
        if (any(symmetry_center == std::numeric_limits<Coord>::max()))
            symmetry_center = (input_shape_nd.vec / 2).template as<Coord>();

        using real_t = nt::value_type_t<Value>;
        using symmetry_span_t = decltype(symmetry.span());
        const i64 symmetry_count = symmetry.count();
        const auto symmetry_scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : real_t{1};
        const auto symmetry_matrices = symmetry.span();

        using noa::cuda::geometry::InterpolatorNd;
        using coord_t = nt::value_type_twice_t<PreMatrix>;
        auto cuda_texture = input.cuda();
        cudaTextureObject_t texture = input.cuda()->texture;

        auto launch_for_each_interp = [&]<bool IsLayered> {
            switch (input.interp_mode()) {
                case Interp::NEAREST: {
                    using interpolator_t = InterpolatorNd<N, Interp::NEAREST, Value, false, IsLayered, coord_t>;
                    using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                    auto op = op_t(interpolator_t(texture), output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                    return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
                }
                case Interp::LINEAR: {
                    using interpolator_t = InterpolatorNd<N, Interp::LINEAR, Value, false, IsLayered, coord_t>;
                    using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                    auto op = op_t(interpolator_t(texture), output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                    return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
                }
                case Interp::COSINE: {
                    using interpolator_t = InterpolatorNd<N, Interp::COSINE, Value, false, IsLayered, coord_t>;
                    using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                    auto op = op_t(interpolator_t(texture), output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                    return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
                }
                case Interp::CUBIC: {
                    using interpolator_t = InterpolatorNd<N, Interp::CUBIC, Value, false, IsLayered, coord_t>;
                    using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                    auto op = op_t(interpolator_t(texture), output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                    return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
                }
                case Interp::CUBIC_BSPLINE: {
                    using interpolator_t = InterpolatorNd<N, Interp::CUBIC_BSPLINE, Value, false, IsLayered, coord_t>;
                    using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                    auto op = op_t(interpolator_t(texture), output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                    return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
                }
                case Interp::LINEAR_FAST: {
                    using interpolator_t = InterpolatorNd<N, Interp::LINEAR_FAST, Value, false, IsLayered, coord_t>;
                    using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                    auto op = op_t(interpolator_t(texture), output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                    return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
                }
                case Interp::COSINE_FAST: {
                    using interpolator_t = InterpolatorNd<N, Interp::COSINE_FAST, Value, false, IsLayered, coord_t>;
                    using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                    auto op = op_t(interpolator_t(texture), output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                    return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
                }
                case Interp::CUBIC_BSPLINE_FAST: {
                    using interpolator_t = InterpolatorNd<N, Interp::CUBIC_BSPLINE_FAST, Value, false, IsLayered, coord_t>;
                    using op_t = Symmetrize<N, Index, symmetry_span_t, interpolator_t, output_accessor_t, PreMatrix, PostMatrix>;
                    auto op = op_t(interpolator_t(texture), output_accessor, symmetry_matrices, symmetry_center, symmetry_scaling, pre_matrix, post_matrix);
                    return iwise(output_shape, output.device(), std::move(op), input, output, symmetry);
                }
            }
        };
        if (N == 2 and input.is_layered()) { // 3d texture cannot be layered
            launch_for_each_interp.template operator()<true>();
        } else {
            launch_for_each_interp.template operator()<false>();
        }
    }
    #endif
}

namespace noa::geometry {
    template<typename Coord, size_t N>
    struct SymmetrizeOptions {
        using matrix_t = std::conditional_t<N == 2, Mat23<Coord>, Mat34<Coord>>;

        /// (D)HW coordinates of the center of symmetry.
        /// By default, use the input center, defined as `shape//2` (integer division).
        Vec<Coord, N> symmetry_center{Vec<Coord, N>::from_value(std::numeric_limits<Coord>::max())};

        /// Interpolation/filter method. All interpolation modes are supported.
        Interp interp_mode{Interp::LINEAR};

        /// Whether the symmetrized output should be normalized to have the same value range as input.
        /// If false, output values end up being scaled by the symmetry count.
        bool normalize{true};
    };

    /// Symmetrizes 2d array(s).
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in pre/post matrices, one can move the center
    ///          of the output window relative to the input window.
    ///
    /// \param[in] input    Input 2d array(s).
    /// \param[out] output  Output 2d array(s).
    /// \param[in] symmetry Symmetry operator.
    /// \param[in] options  Symmetry and interpolation options.
    /// \param[in] pre_inverse_matrix   HW inverse truncated affine matrix to apply before the symmetry.
    ///                                 In practice, this needs to be applied for each symmetry count as opposed to
    ///                                 the post transformation which is applied once per pixel/voxel.
    /// \param[in] post_inverse_matrix  HW inverse truncated affine matrix to apply after the symmetry.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. Border::ZERO is used.
    /// \note This function is optimized for rightmost arrays. Passing anything else will likely result in a
    ///       significant performance loss.
    template<typename Input, typename Output, typename Coord,
             typename PreMatrix = Empty, typename PostMatrix = Empty>
    requires ((nt::are_varray_of_real_v<Input, Output> or nt::are_varray_of_complex_v<Input, Output>) and
              nt::is_varray_of_mutable_v<Output> and
              nt::is_any_v<PreMatrix, Empty, Mat23<Coord>> and
              nt::is_any_v<PostMatrix, Empty, Mat23<Coord>>)
    void symmetrize_2d(
            const Input& input, const Output& output,
            const Symmetry<Coord, 2>& symmetry,
            const SymmetrizeOptions<Coord, 2>& options = {},
            const PreMatrix& pre_inverse_matrix = {},
            const PostMatrix& post_inverse_matrix = {}
    ) {
        guts::check_parameters_symmetrize_nd(input, output, symmetry);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            return guts::launch_symmetrize_nd<i32>(
                    input, output, symmetry,
                    options.symmetry_center, options.interp_mode, options.normalize,
                    pre_inverse_matrix, post_inverse_matrix);
        }
        guts::launch_symmetrize_nd<i64>(
                input, output, symmetry,
                options.symmetry_center, options.interp_mode, options.normalize,
                pre_inverse_matrix, post_inverse_matrix);
    }

    /// Symmetrizes 2d array(s).
    /// \note options.interp_mode is ignored, input.interp_mode() is used instead.
    /// \note This overload has the same features and limitations as the overload taking Arrays.
    template<typename Value, typename Output, typename Coord,
             typename PreMatrix = Empty, typename PostMatrix = Empty>
    requires (nt::are_varray_of_real_or_complex_v<Output> and
              nt::is_varray_of_mutable_v<Output> and
              (nt::are_real_v<Value, nt::value_type_t<Output>> or
               nt::are_complex_v<Value, nt::value_type_t<Output>>) and
              nt::is_any_v<PreMatrix, Empty, Mat23<Coord>> and
              nt::is_any_v<PostMatrix, Empty, Mat23<Coord>>)
    void symmetrize_2d(
            const Texture<Value>& input, const Output& output,
            const Symmetry<Coord, 2>& symmetry,
            const SymmetrizeOptions<Coord, 2>& options = {},
            const PreMatrix& pre_inverse_matrix = {},
            const PostMatrix& post_inverse_matrix = {}
    ) {
        guts::check_parameters_symmetrize_nd(input, output, symmetry);

        const Device device = output.device();
        if (device.is_cpu()) {
            guts::launch_symmetrize_nd<i64>(
                    input, output, symmetry,
                    options.symmetry_center, input.interp_mode(), options.normalize,
                    pre_inverse_matrix, post_inverse_matrix);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (not nt::is_any_v<Value, f32, c32>) {
                panic("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                if (ng::is_accessor_access_safe<i32>(output.strides(), output.shape()))
                    guts::launch_symmetrize_nd<i32>(
                            input, output, symmetry,
                            options.symmetry_center, options.normalize,
                            pre_inverse_matrix, post_inverse_matrix);
                else
                    guts::launch_symmetrize_nd<i64>(
                            input, output, symmetry,
                            options.symmetry_center, options.normalize,
                            pre_inverse_matrix, post_inverse_matrix);
            }
            #else
            panic("No GPU backend detected");
            #endif
        }
    }

    /// Symmetrizes 3d array(s).
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in pre/post matrices, one can move the center
    ///          of the output window relative to the input window.
    ///
    /// \param[in] input    Input 3d array(s).
    /// \param[out] output  Output 3d array(s).
    /// \param[in] symmetry Symmetry operator.
    /// \param[in] options  Symmetry and interpolation options.
    /// \param[in] pre_inverse_matrix   DHW inverse truncated affine matrix to apply before the symmetry.
    ///                                 In practice, this needs to be applied for each symmetry count as opposed to
    ///                                 the post transformation which is applied once per pixel/voxel.
    /// \param[in] post_inverse_matrix  DHW inverse truncated affine matrix to apply after the symmetry.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. Border::ZERO is used.
    /// \note This function is optimized for rightmost arrays. Passing anything else will likely result in a
    ///       significant performance loss.
    template<typename Input, typename Output, typename Coord,
             typename PreMatrix = Empty, typename PostMatrix = Empty>
    requires ((nt::are_varray_of_real_v<Input, Output> or nt::are_varray_of_complex_v<Input, Output>) and
              nt::is_varray_of_mutable_v<Output> and
              nt::is_any_v<PreMatrix, Empty, Mat34<Coord>> and
              nt::is_any_v<PostMatrix, Empty, Mat34<Coord>>)
    void symmetrize_3d(
            const Input& input, const Output& output,
            const Symmetry<Coord, 3>& symmetry,
            const SymmetrizeOptions<Coord, 3>& options = {},
            const PreMatrix& pre_inverse_matrix = {},
            const PostMatrix& post_inverse_matrix = {}
    ) {
        guts::check_parameters_symmetrize_nd(input, output, symmetry);

        if (output.device().is_gpu() and
            ng::is_accessor_access_safe<i32>(input.strides(), input.shape()) and
            ng::is_accessor_access_safe<i32>(output.strides(), output.shape())) {
            return guts::launch_symmetrize_nd<i32>(
                    input, output, symmetry,
                    options.symmetry_center, options.interp_mode, options.normalize,
                    pre_inverse_matrix, post_inverse_matrix);
        }
        guts::launch_symmetrize_nd<i64>(
                input, output, symmetry,
                options.symmetry_center, options.interp_mode, options.normalize,
                pre_inverse_matrix, post_inverse_matrix);
    }

    /// Symmetrizes 3d array(s).
    /// \note options.interp_mode is ignored, input.interp_mode() is used instead.
    /// \note This overload has the same features and limitations as the overload taking Arrays.
    template<typename Value, typename Output, typename Coord,
             typename PreMatrix = Empty, typename PostMatrix = Empty>
    requires (nt::are_varray_of_real_or_complex_v<Output> and
              nt::is_varray_of_mutable_v<Output> and
              (nt::are_real_v<Value, nt::value_type_t<Output>> or
               nt::are_complex_v<Value, nt::value_type_t<Output>>) and
              nt::is_any_v<PreMatrix, Empty, Mat34<Coord>> and
              nt::is_any_v<PostMatrix, Empty, Mat34<Coord>>)
    void symmetrize_3d(
            const Texture<Value>& input, const Output& output,
            const Symmetry<Coord, 3>& symmetry,
            const SymmetrizeOptions<Coord, 3>& options = {},
            const PreMatrix& pre_inverse_matrix = {},
            const PostMatrix& post_inverse_matrix = {}
    ) {
        guts::check_parameters_symmetrize_nd(input, output, symmetry);

        const Device device = output.device();
        if (device.is_cpu()) {
            guts::launch_symmetrize_nd<i64>(
                    input, output, symmetry,
                    options.symmetry_center, input.interp_mode(), options.normalize,
                    pre_inverse_matrix, post_inverse_matrix);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (not nt::is_any_v<Value, f32, c32>) {
                panic("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                if (ng::is_accessor_access_safe<i32>(output.strides(), output.shape()))
                    guts::launch_symmetrize_nd<i32>(
                            input, output, symmetry,
                            options.symmetry_center, options.normalize,
                            pre_inverse_matrix, post_inverse_matrix);
                else
                    guts::launch_symmetrize_nd<i64>(
                            input, output, symmetry,
                            options.symmetry_center, options.normalize,
                            pre_inverse_matrix, post_inverse_matrix);
            }
            #else
            panic("No GPU backend detected");
            #endif
        }
    }
}
