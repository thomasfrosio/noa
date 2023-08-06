#pragma once

#include "noa/cpu/geometry/Transform.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Transform.hpp"
#endif

#include "noa/core/geometry/Symmetry.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"

namespace noa::geometry::details {
    template<typename Value, typename Matrix>
    constexpr bool is_valid_transform_2d_v =
            noa::traits::is_any_v<Value, f32, f64, c32, c64> &&
            (noa::traits::is_almost_any_v<Matrix, Float23, Float33> ||
             noa::traits::is_varray_of_almost_any_v<Matrix, Float23, Float33>);

    template<typename Value, typename Matrix>
    constexpr bool is_valid_transform_3d_v =
            noa::traits::is_any_v<Value, f32, f64, c32, c64> &&
            (noa::traits::is_almost_any_v<Matrix, Float34, Float44> ||
             noa::traits::is_varray_of_almost_any_v<Matrix, Float34, Float44>);


    template<int32_t NDIM, bool SYMMETRY = false, typename Input, typename Output, typename Matrix>
    void transform_nd_check_parameters(const Input& input, const Output& output, const Matrix& matrix) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(NDIM == 3 || (input.shape()[1] == 1 && output.shape()[1] == 1),
                  "The input and output arrays should be 2D, but got shape input:{}, output:{}",
                  input.shape(), output.shape());
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        if constexpr (SYMMETRY) {
            NOA_CHECK(input.shape()[3] == output.shape()[3] &&
                      input.shape()[2] == output.shape()[2] &&
                      input.shape()[1] == output.shape()[1],
                      "The input {} and output {} shapes don't match",
                      input.shape(), output.shape());
        }

        const Device device = output.device();

        if constexpr (!traits::is_matXX_v<Matrix>) {
            NOA_CHECK(noa::indexing::is_contiguous_vector(matrix) && matrix.elements() == output.shape()[0],
                      "The number of matrices, specified as a contiguous vector, should be equal to the number of "
                      "batches in the output, but got matrix shape:{}, strides:{} and {} output batches",
                      matrix.shape(), matrix.strides(), output.shape()[0]);
            NOA_CHECK(device == matrix.device(),
                      "The transformation matrices should be on the same device as the output, "
                      "but got matrices:{} and output:{}", matrix.device(), device);
        }

        if constexpr (noa::traits::is_varray_v<Input>) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(!noa::indexing::are_overlapped(input, output),
                      "The input and output arrays should not overlap");
            NOA_CHECK(noa::indexing::are_elements_unique(output.strides(), output.shape()),
                      "The elements in the output should not overlap in memory, "
                      "otherwise a data-race might occur. Got output strides:{} and shape:{}",
                      output.strides(), output.shape());
        } else {
            NOA_CHECK(input.device() == output.device(),
                      "The input texture and output array must be on the same device, "
                      "but got input:{} and output:{}", input.device(), output.device());
        }
    }

    template<typename Matrix>
    auto extract_matrix(const Matrix& matrix) {
        if constexpr (traits::is_matXX_v<Matrix>) {
            return matrix;
        } else {
            using ptr_t = const noa::traits::value_type_t<Matrix>*;
            return ptr_t(matrix.get());
        }
    }
}

// -- Affine transformations -- //
namespace noa::geometry {
    /// Applies one or multiple 2D affine transforms.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    ///
    /// \tparam Value           f32, f64, c32, c64.
    /// \tparam Matrix          Float23, Float33, or an varray of these types.
    /// \param[in] input        Input 2D array.
    /// \param[out] output      Output 2D array.
    /// \param[in] inv_matrices 2x3 or 3x3 inverse HW affine matrices.
    ///                         One, or if an array is entered, one per output batch.
    /// \param interp_mode      Filter mode. See InterpMode.
    /// \param border_mode      Address mode. See BorderMode.
    /// \param value            Constant value to use for out-of-bounds coordinates.
    ///                         Only used if \p border_mode is BorderMode::VALUE.
    template<typename Input, typename Output, typename Matrix,
             typename Value = noa::traits::value_type_t<Output>, typename = std::enable_if_t<
             noa::traits::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_transform_2d_v<Value, Matrix>>>
    void transform_2d(const Input& input, const Output& output, const Matrix& inv_matrices,
                      InterpMode interp_mode = InterpMode::LINEAR,
                      BorderMode border_mode = BorderMode::ZERO,
                      Value value = Value{0}) {
        details::transform_nd_check_parameters<2>(input, output, inv_matrices);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::transform_2d(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        details::extract_matrix(inv_matrices),
                        interp_mode, border_mode, value, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::transform_2d(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(), output.shape(),
                    details::extract_matrix(inv_matrices),
                    interp_mode, border_mode, value, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_varray_v<Matrix>)
                cuda_stream.enqueue_attach(inv_matrices.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Applies one or multiple 2D affine transforms.
    /// This overload has the same features and limitations as the overload taking Arrays.
    template<typename Value, typename Output, typename Matrix, typename = std::enable_if_t<
             noa::traits::is_varray_of_any_v<Output, Value> &&
             details::is_valid_transform_2d_v<Value, Matrix>>>
    void transform_2d(const Texture<Value>& input, const Output& output, const Matrix& inv_matrices) {
        details::transform_nd_check_parameters<2>(input, output, inv_matrices);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                const cpu::Texture<Value>& texture = input.cpu();
                cpu::geometry::transform_2d(
                        texture.ptr.get(), texture.strides, input.shape(),
                        output.get(), output.strides(), output.shape(),
                        details::extract_matrix(inv_matrices),
                        input.interp_mode(), input.border_mode(),
                        texture.cvalue, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, f32, c32>) {
                NOA_THROW("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::transform_2d(
                        texture.array.get(), *texture.texture,
                        input.shape(), input.interp_mode(), input.border_mode(),
                        output.get(), output.strides(), output.shape(),
                        details::extract_matrix(inv_matrices), cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, output.share());
                if constexpr (noa::traits::is_varray_v<Matrix>)
                    cuda_stream.enqueue_attach(inv_matrices.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Applies one or multiple 3D affine transforms.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    ///
    /// \tparam Value           f32, f64, c32, c64.
    /// \tparam Matrix          Float34, Float44, or an varray of these types.
    /// \param[in] input        Input 3D array.
    /// \param[out] output      Output 3D array.
    /// \param[in] inv_matrices 3x4 or 4x4 inverse DHW affine matrix/matrices.
    ///                         One, or if an array is entered, one per output batch.
    /// \param interp_mode      Filter mode. See InterpMode.
    /// \param border_mode      Address mode. See BorderMode.
    /// \param value            Constant value to use for out-of-bounds coordinates.
    ///                         Only used if \p border_mode is BorderMode::VALUE.
    template<typename Input, typename Output, typename Matrix,
             typename Value = noa::traits::value_type_t<Output>, typename = std::enable_if_t<
                    noa::traits::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
                    noa::traits::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
                    noa::traits::are_almost_same_value_type_v<Input, Output> &&
                    details::is_valid_transform_3d_v<Value, Matrix>>>
    void transform_3d(const Input& input, const Output& output, const Matrix& inv_matrices,
                      InterpMode interp_mode = InterpMode::LINEAR,
                      BorderMode border_mode = BorderMode::ZERO,
                      Value value = Value{0}) {
        details::transform_nd_check_parameters<3>(input, output, inv_matrices);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::transform_3d(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        details::extract_matrix(inv_matrices),
                        interp_mode, border_mode, value, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::transform_3d(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(), output.shape(),
                    details::extract_matrix(inv_matrices),
                    interp_mode, border_mode, value, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_varray_v<Matrix>)
                cuda_stream.enqueue_attach(inv_matrices.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Applies one or multiple 3D affine transforms.
    /// This overload has the same features and limitations as the overload taking Arrays.
    template<typename Value, typename Output, typename Matrix, typename = std::enable_if_t<
             noa::traits::is_varray_of_any_v<Output, Value> &&
             details::is_valid_transform_3d_v<Value, Matrix>>>
    void transform_3d(const Texture<Value>& input, const Output& output, const Matrix& inv_matrices) {
        details::transform_nd_check_parameters<3>(input, output, inv_matrices);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                const cpu::Texture<Value>& texture = input.cpu();
                cpu::geometry::transform_3d(
                        texture.ptr.get(), texture.strides, input.shape(),
                        output.get(), output.strides(), output.shape(),
                        details::extract_matrix(inv_matrices),
                        input.interp_mode(), input.border_mode(),
                        texture.cvalue, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!noa::traits::is_any_v<Value, f32, c32>) {
                NOA_THROW("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::transform_3d(
                        texture.array.get(), *texture.texture,
                        input.shape(), input.interp_mode(), input.border_mode(),
                        output.get(), output.strides(), output.shape(),
                        details::extract_matrix(inv_matrices), cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, output.share());
                if constexpr (noa::traits::is_varray_v<Matrix>)
                    cuda_stream.enqueue_attach(inv_matrices.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

// -- With symmetry -- //
namespace noa::geometry {
    /// Shifts, then rotates/scales and applies the symmetry on the 2D input array.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    ///
    /// \param[in] input    Input 2D array of f32, f64, c32, or c64.
    /// \param[out] output  Output 2D array.
    /// \param shift        HW forward shift to apply before the other transformations.
    ///                     Positive shifts translate the object to the right.
    /// \param inv_matrices HW inverse rotation/scaling to apply after the shift.
    /// \param[in] symmetry Symmetry operator to apply after the rotation/scaling.
    /// \param center       HW index of the transformation center.
    ///                     Both \p matrix and \p symmetry operates around this center.
    /// \param interp_mode  Interpolation/filter method. All interpolation modes are supported.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BorderMode::ZERO is used.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void transform_and_symmetrize_2d(
            const Input& input, const Output& output,
            const Vec2<f32>& shift, const Float22& inv_matrix,
            const Symmetry& symmetry, const Vec2<f32>& center,
            InterpMode interp_mode = InterpMode::LINEAR,
            bool normalize = true) {
        details::transform_nd_check_parameters<2>(input, output, Float22{});

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::transform_and_symmetrize_2d(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        shift, inv_matrix, symmetry, center, interp_mode,
                        normalize, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::transform_and_symmetrize_2d(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(), output.shape(),
                    shift, inv_matrix, symmetry, center, interp_mode,
                    normalize, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Shifts, then rotates/scales and applies the symmetry on the 2D input array.
    /// This overload has the same features and limitations as the overload taking Arrays.
    template<typename Value, typename Output, typename = std::enable_if_t<
             noa::traits::is_varray_of_any_v<Output, Value> &&
             noa::traits::is_any_v<Value, f32, f64, c32, c64>>>
    void transform_and_symmetrize_2d(
            const Texture<Value>& input, const Output& output,
            const Vec2<f32>& shift, const Float22& inv_matrix,
            const Symmetry& symmetry, const Vec2<f32>& center,
            bool normalize = true) {
        details::transform_nd_check_parameters<2>(input, output, Float22{});

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                const cpu::Texture<Value>& texture = input.cpu();
                cpu::geometry::transform_and_symmetrize_2d(
                        texture.ptr.get(), texture.strides, input.shape(),
                        output.get(), output.strides(), output.shape(),
                        shift, inv_matrix, symmetry, center, input.interp_mode(),
                        normalize, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!noa::traits::is_any_v<Value, f32, c32>) {
                NOA_THROW("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::transform_and_symmetrize_2d(
                        texture.array.get(), *texture.texture,
                        input.interp_mode(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        shift, inv_matrix, symmetry, center,
                        normalize, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, output.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Shifts, then rotates/scales and applies the symmetry on the 3D input array.
    /// \details The input and output array can have different shapes. The output window starts at the same index
    ///          than the input window, so by entering a translation in \p matrices, one can move the center of the
    ///          output window relative to the input window.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix will
    ///          be applied to each batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast to all
    ///          output batches (1 input -> N output).
    ///
    /// \param[in] input    Input 3D array of f32, f64, c32, or c64.
    /// \param[out] output  Output 3D array.
    /// \param shift        DHW forward shift to apply before the other transformations.
    ///                     Positive shifts translate the object to the right.
    /// \param inv_matrices DHW inverse rotation/scaling to apply after the shift.
    /// \param[in] symmetry Symmetry operator to apply after the rotation/scaling.
    /// \param center       DHW index of the transformation center.
    ///                     Both \p matrix and \p symmetry operates around this center.
    /// \param interp_mode  Interpolation/filter mode. All interpolation modes are supported.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BorderMode::ZERO is used.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void transform_and_symmetrize_3d(
            const Input& input, const Output& output,
            const Vec3<f32>& shift, const Float33& inv_matrix,
            const Symmetry& symmetry, const Vec3<f32>& center,
            InterpMode interp_mode = InterpMode::LINEAR,
            bool normalize = true) {
        details::transform_nd_check_parameters<3>(input, output, Float33{});

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::transform_and_symmetrize_3d(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        shift, inv_matrix, symmetry, center, interp_mode,
                        normalize, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::transform_and_symmetrize_3d(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(), output.shape(),
                    shift, inv_matrix, symmetry, center, interp_mode,
                    normalize, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Shifts, then rotates/scales and applies the symmetry on the 3D input array.
    /// \details This overload has the same features and limitations as the overload taking Arrays.
    ///          This is mostly for the GPU, since "CPU textures" are simple Arrays.
    template<typename Value, typename Output, typename = std::enable_if_t<
             noa::traits::is_varray_of_any_v<Output, Value> &&
             noa::traits::is_any_v<Value, f32, f64, c32, c64>>>
    void transform_and_symmetrize_3d(
            const Texture<Value>& input, const Output& output,
            const Vec3<f32>& shift, const Float33& inv_matrix,
            const Symmetry& symmetry, const Vec3<f32>& center,
            bool normalize = true) {
        details::transform_nd_check_parameters<3>(input, output, Float33{});

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                const cpu::Texture<Value>& texture = input.cpu();
                cpu::geometry::transform_and_symmetrize_3d(
                        texture.ptr.get(), texture.strides, input.shape(),
                        output.get(), output.strides(), output.shape(),
                        shift, inv_matrix, symmetry, center, input.interp_mode(),
                        normalize, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!noa::traits::is_any_v<Value, f32, c32>) {
                NOA_THROW("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::transform_and_symmetrize_3d(
                        texture.array.get(), *texture.texture,
                        input.interp_mode(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        shift, inv_matrix, symmetry, center,
                        normalize, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, output.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Symmetrizes the 2D (batched) input array.
    /// \param[in] input    Input 2D array of f32, f64, c32, c64.
    /// \param[out] output  Output 2D array.
    /// \param[in] symmetry Symmetry operator.
    /// \param center       HW center of the symmetry.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BorderMode::ZERO is used.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void symmetrize_2d(const Input& input, const Output& output,
                       const Symmetry& symmetry, const Vec2<f32>& center,
                       InterpMode interp_mode = InterpMode::LINEAR,
                       bool normalize = true) {
        details::transform_nd_check_parameters<2, true>(input, output, Float22{});
        auto input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::symmetrize_2d(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        symmetry, center, interp_mode, normalize, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::symmetrize_2d(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    symmetry, center, interp_mode, normalize, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Symmetrizes the 2D (batched) input array.
    /// This overload has the same features and limitations as the overload taking Arrays.
    template<typename Value, typename Output, typename = std::enable_if_t<
             noa::traits::is_varray_of_any_v<Output, Value> &&
             noa::traits::is_any_v<Value, f32, f64, c32, c64>>>
    void symmetrize_2d(const Texture<Value>& input, const Output& output,
                       const Symmetry& symmetry, const Vec2<f32>& center,
                       bool normalize = true) {
        details::transform_nd_check_parameters<2, true>(input, output, Float22{});

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                const cpu::Texture<Value>& texture = input.cpu();
                cpu::geometry::symmetrize_2d(
                        texture.ptr.get(), texture.strides,
                        output.get(), output.strides(), output.shape(),
                        symmetry, center, input.interp_mode(),
                        normalize, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!noa::traits::is_any_v<Value, f32, c32>) {
                NOA_THROW("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::symmetrize_2d(
                        texture.array.get(), *texture.texture,
                        input.interp_mode(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        symmetry, center, normalize, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, output.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Symmetrizes the 3D (batched) input array.
    /// \param[in] input    Input 3D array of f32, f64, c32, c64.
    /// \param[out] output  Output 3D array.
    /// \param[in] symmetry Symmetry operator.
    /// \param center       DHW center of the symmetry.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BorderMode::ZERO is used.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void symmetrize_3d(const Input& input, const Output& output,
                       const Symmetry& symmetry, const Vec3<f32>& center,
                       InterpMode interp_mode = InterpMode::LINEAR,
                       bool normalize = true) {
        details::transform_nd_check_parameters<3, true>(input, output, Float33{});
        auto input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::symmetrize_3d(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        symmetry, center, interp_mode, normalize, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::symmetrize_3d(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    symmetry, center, interp_mode, normalize, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Symmetrizes the 2D (batched) input array.
    /// This overload has the same features and limitations as the overload taking Arrays.
    template<typename Value, typename Output, typename = std::enable_if_t<
            noa::traits::is_varray_of_any_v<Output, Value> &&
            noa::traits::is_any_v<Value, f32, f64, c32, c64>>>
    void symmetrize_3d(const Texture<Value>& input, const Output& output,
                       const Symmetry& symmetry, const Vec3<f32>& center,
                       bool normalize = true) {
        details::transform_nd_check_parameters<3, true>(input, output, Float33{});

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                const cpu::Texture<Value>& texture = input.cpu();
                cpu::geometry::symmetrize_3d(
                        texture.ptr.get(), texture.strides,
                        output.get(), output.strides(), output.shape(),
                        symmetry, center, input.interp_mode(),
                        normalize, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!noa::traits::is_any_v<Value, f32, c32>) {
                NOA_THROW("In the CUDA backend, textures don't support double-precision floating-points");
            } else {
                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::symmetrize_3d(
                        texture.array.get(), *texture.texture,
                        input.interp_mode(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        symmetry, center, normalize, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, output.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
