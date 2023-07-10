#pragma once

#include "noa/core/geometry/Euler.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/core/geometry/Symmetry.hpp"

#include "noa/cpu/geometry/fft/Transform.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Transform.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"

namespace noa::geometry::fft::details {
    using namespace ::noa::fft;

    template<Remap REMAP, typename Value, typename Matrix, typename Shift>
    constexpr bool is_valid_transform_2d_v =
            noa::traits::is_any_v<Value, f32, f64, c32, c64> && (REMAP == HC2HC || REMAP == HC2H) &&
            (noa::traits::is_any_v<Matrix, Float22> || noa::traits::is_array_or_view_of_almost_any_v<Matrix, Float22>) &&
            (noa::traits::is_any_v<Shift, Vec2<f32>> || noa::traits::is_array_or_view_of_almost_any_v<Shift, Vec2<f32>>);

    template<Remap REMAP, typename Value, typename Matrix, typename Shift>
    constexpr bool is_valid_transform_3d_v =
            noa::traits::is_any_v<Value, f32, f64, c32, c64> && (REMAP == HC2HC || REMAP == HC2H) &&
            (noa::traits::is_any_v<Matrix, Float33> || noa::traits::is_array_or_view_of_almost_any_v<Matrix, Float33>) &&
            (noa::traits::is_any_v<Shift, Vec3<f32>> || noa::traits::is_array_or_view_of_almost_any_v<Shift, Vec3<f32>>);

    template<Remap REMAP, typename Value>
    constexpr bool is_valid_transform_sym_v =
            noa::traits::is_any_v<Value, f32, f64, c32, c64> &&
            (REMAP == HC2HC || REMAP == HC2H);


    template<i32 NDIM, typename Input, typename Output, typename Matrix, typename Shift>
    void transform_nd_check_parameters(
            const Input& input, const Output& output, const Shape4<i64>& shape,
            const Matrix& inv_matrices, const Shift& post_shifts) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(shape[3] / 2 + 1 == input.shape()[3] && input.shape()[3] == output.shape()[3] &&
                  shape[2] == input.shape()[2] && input.shape()[2] == output.shape()[2] &&
                  shape[1] == input.shape()[1] && input.shape()[1] == output.shape()[1],
                  "The non-redundant input {} and/or output {} shapes don't match the logical shape {}",
                  input.shape(), output.shape(), shape);
        NOA_CHECK(NDIM == 3 || (input.shape()[1] == 1 && output.shape()[1] == 1),
                  "The input and output arrays should be 2D, but got shape input:{}, output:{}",
                  input.shape(), output.shape());
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                       "The input and output arrays must be on the same device, "
                       "but got input:{} and output:{}", input.device(), device);

        if constexpr (!traits::is_matXX_v<Matrix>) {
            NOA_CHECK(noa::indexing::is_contiguous_vector(inv_matrices) && inv_matrices.elements() == output.shape()[0],
                      "The number of matrices, specified as a contiguous vector, should be equal to the number of "
                      "batches in the output, but got matrix shape:{}, strides:{} and {} output batches",
                      inv_matrices.shape(), inv_matrices.strides(), output.shape()[0]);
            NOA_CHECK(device == inv_matrices.device(),
                      "The input matrices and output arrays must be on the same device, "
                      "but got matrix:{} and output:{}", inv_matrices.device(), device);
        }
        if constexpr (!traits::is_realX_v<Shift>) {
            NOA_CHECK(noa::indexing::is_contiguous_vector(post_shifts) &&  post_shifts.elements() == output.shape()[0],
                      "The number of shifts, specified as a contiguous vector, should be equal to the number of "
                      "batches in the output, but got shift shape:{}, strides:{} and {} output batches",
                      post_shifts.shape(), post_shifts.strides(), output.shape()[0]);
            NOA_CHECK(device == post_shifts.device(),
                      "The input shifts and output arrays must be on the same device, "
                      "but got shift:{} and output:{}", post_shifts.device(), device);
        }

        if constexpr (noa::traits::is_array_or_view_v<Input>) {
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

    template<typename MatrixOrShift>
    auto extract_matrix_or_shift(const MatrixOrShift& matrix_or_shift) {
        if constexpr (traits::is_matXX_v<MatrixOrShift> || traits::is_realX_v<MatrixOrShift>) {
            return matrix_or_shift;
        } else {
            using ptr_t = const typename MatrixOrShift::value_type*;
            return ptr_t(matrix_or_shift.get());
        }
    }
}

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam Matrix          Float22 or an array/view of that type.
    /// \tparam Shift           Vec2<f32> or an array/view of that type.
    /// \param[in] input        Non-redundant 2D FFT, of type f32, f64, c32, c64, to transform.
    /// \param[out] output      Non-redundant transformed 2D FFT.
    /// \param shape            BDHW logical shape of \p input and \p output.
    /// \param[in] inv_matrices 2x2 inverse HW rotation/scaling matrix.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S
    ///                         in real space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] post_shifts  2D real-space HW forward shift to apply (as phase shift) after the transformation.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If an empty array is entered or if \p input is real, it is ignored.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename Input, typename Output, typename Matrix,
             typename Shift = Vec2<f32>, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_transform_2d_v<REMAP, noa::traits::value_type_t<Output>, Matrix, Shift>>>
    void transform_2d(const Input& input, const Output& output, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& post_shifts = {},
                      f32 cutoff = 0.5f, InterpMode interp_mode = InterpMode::LINEAR) {
        details::transform_nd_check_parameters<2>(input, output, shape, inv_matrices, post_shifts);
        auto input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::transform_2d<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), shape,
                        details::extract_matrix_or_shift(inv_matrices),
                        details::extract_matrix_or_shift(post_shifts),
                        cutoff, interp_mode, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::transform_2d<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), shape,
                    details::extract_matrix_or_shift(inv_matrices),
                    details::extract_matrix_or_shift(post_shifts),
                    cutoff, interp_mode, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_array_or_view_v<Matrix>)
                cuda_stream.enqueue_attach(inv_matrices.share());
            if constexpr (noa::traits::is_array_or_view_v<Shift>)
                cuda_stream.enqueue_attach(post_shifts.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// This functions has the same features and limitations as the overload taking arrays.
    template<Remap REMAP, typename Value, typename Output,
             typename Matrix, typename Shift = Vec2<f32>, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_any_v<Output, Value> &&
             details::is_valid_transform_2d_v<REMAP, Value, Matrix, Shift>>>
    void transform_2d(const Texture<Value>& input, const Output& output, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& post_shifts = {},
                      f32 cutoff = 0.5f) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);

        if (input.device().is_cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            const Array<Value> input_array(texture.ptr, input.shape(), texture.strides, input.options());
            transform_2d<REMAP>(input_array, output, shape, inv_matrices, post_shifts, cutoff, input.interp_mode());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(noa::traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                details::transform_nd_check_parameters<2>(input, output, shape, inv_matrices, post_shifts);
                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::fft::transform_2d<REMAP>(
                        texture.array.get(), *texture.texture, input.interp_mode(),
                        output.get(), output.strides(), shape,
                        details::extract_matrix_or_shift(inv_matrices),
                        details::extract_matrix_or_shift(post_shifts),
                        cutoff, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, output.share());
                if constexpr (noa::traits::is_array_or_view_v<Matrix>)
                    cuda_stream.enqueue_attach(inv_matrices.share());
                if constexpr (noa::traits::is_array_or_view_v<Shift>)
                    cuda_stream.enqueue_attach(post_shifts.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam Matrix          Float33 or an array/view of that type.
    /// \tparam Shift           Vec3<f32> or an array/view of that type.
    /// \param[in] input        Non-redundant 3D FFT, of type f32, f64, c32, c64, to transform.
    /// \param[out] output      Non-redundant transformed 3D FFT.
    /// \param shape            BDHW logical shape of \p input and \p output.
    /// \param[in] inv_matrices 3x3 inverse DHW rotation/scaling matrix.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space. One per output batch.
    /// \param[in] post_shifts  DHW 3D real-space forward shift to apply (as phase shift) after the transformation.
    ///                         One, or if an array is provided, one per output batch.
    ///                         If an empty array is entered or if \p input is real, it is ignored.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename Input, typename Output, typename Matrix,
             typename Shift = Vec3<f32>, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_transform_3d_v<REMAP, noa::traits::value_type_t<Output>, Matrix, Shift>>>
    void transform_3d(const Input& input, const Output& output, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& post_shifts = {},
                      f32 cutoff = 0.5f, InterpMode interp_mode = InterpMode::LINEAR) {
        details::transform_nd_check_parameters<3>(input, output, shape, inv_matrices, post_shifts);
        auto input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::transform_3d<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), shape,
                        details::extract_matrix_or_shift(inv_matrices),
                        details::extract_matrix_or_shift(post_shifts),
                        cutoff, interp_mode, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::transform_3d<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), shape,
                    details::extract_matrix_or_shift(inv_matrices),
                    details::extract_matrix_or_shift(post_shifts),
                    cutoff, interp_mode, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_array_or_view_v<Matrix>)
                cuda_stream.enqueue_attach(inv_matrices.share());
            if constexpr (noa::traits::is_array_or_view_v<Shift>)
                cuda_stream.enqueue_attach(post_shifts.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// This functions has the same features and limitations as the overload taking arrays.
    template<Remap REMAP, typename Value, typename Output,
             typename Matrix, typename Shift = Vec3<f32>, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_any_v<Output, Value> &&
             details::is_valid_transform_3d_v<REMAP, Value, Matrix, Shift>>>
    void transform_3d(const Texture<Value>& input, const Output& output, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& post_shifts = {},
                      f32 cutoff = 0.5f) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);

        if (input.device().is_cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            const Array<Value> input_array(texture.ptr, input.shape(), texture.strides, input.options());
            transform_3d<REMAP>(input_array, output, shape, inv_matrices, post_shifts, cutoff, input.interp_mode());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(noa::traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                details::transform_nd_check_parameters<3>(input, output, shape, inv_matrices, post_shifts);
                auto& cuda_stream = stream.cuda();
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::fft::transform_3d<REMAP>(
                        texture.array.get(), *texture.texture, input.interp_mode(),
                        output.get(), output.strides(), shape,
                        details::extract_matrix_or_shift(inv_matrices),
                        details::extract_matrix_or_shift(post_shifts),
                        cutoff, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, output.share());
                if constexpr (noa::traits::is_array_or_view_v<Matrix>)
                    cuda_stream.enqueue_attach(inv_matrices.share());
                if constexpr (noa::traits::is_array_or_view_v<Shift>)
                    cuda_stream.enqueue_attach(post_shifts.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    using Symmetry = ::noa::geometry::Symmetry;

    /// Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \param[in] input        Non-redundant 2D FFT, of f32, f64, c32, or c64, to transform.
    /// \param[out] output      Non-redundant transformed 2D FFT.
    /// \param shape            BDHW logical shape of \p input and \p output.
    /// \param[in] inv_matrix   2x2 inverse HW rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] post_shift   HW 2D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             (REMAP == Remap::HC2HC || REMAP == Remap::HC2H)>>
    void transform_and_symmetrize_2d(
            const Input& input, const Output& output, const Shape4<i64>& shape,
            const Float22& inv_matrix, const Symmetry& symmetry, const Vec2<f32>& post_shift = {},
            f32 cutoff = 0.5f, InterpMode interp_mode = InterpMode::LINEAR, bool normalize = true) {
        details::transform_nd_check_parameters<2>(input, output, shape, inv_matrix, post_shift);
        auto input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::transform_and_symmetrize_2d<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), shape,
                        inv_matrix, symmetry, post_shift, cutoff,
                        interp_mode, normalize, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::transform_and_symmetrize_2d<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), shape,
                    inv_matrix, symmetry, post_shift, cutoff,
                    interp_mode, normalize, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    /// This functions has the same features and limitations as the overload taking arrays.
    template<Remap REMAP, typename Value, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_any_v<Output, Value> &&
             details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform_and_symmetrize_2d(
            const Texture<Value>& input, const Output& output, const Shape4<i64>& shape,
            const Float22& inv_matrix, const Symmetry& symmetry, const Vec2<f32>& post_shift = {},
            f32 cutoff = 0.5f, bool normalize = true) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            const Array<Value> input_array(texture.ptr, input.shape(), texture.strides, input.options());
            transform_and_symmetrize_2d<REMAP>(
                    input_array, output, shape, inv_matrix,
                    symmetry, post_shift, cutoff, input.interp_mode(), normalize);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                details::transform_nd_check_parameters<2>(input, output, shape, inv_matrix, post_shift);
                const cuda::Texture<Value>& texture = input.cuda();
                auto& cuda_stream = stream.cuda();
                cuda::geometry::fft::transform_and_symmetrize_2d<REMAP>(
                        texture.array.get(), *texture.texture, input.interp_mode(),
                        output.get(), output.strides(), shape,
                        inv_matrix, symmetry, post_shift, cutoff, normalize, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, output.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \param[in] input        Non-redundant 3D FFT, of f32, f64, c32, or c64, to transform.
    /// \param[out] output      Non-redundant transformed 3D FFT.
    /// \param shape            BDHW logical shape of \p input and \p output.
    /// \param[in] inv_matrix   3x3 inverse DHW rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] post_shift   DHW 3D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             (REMAP == Remap::HC2HC || REMAP == Remap::HC2H)>>
    void transform_and_symmetrize_3d(
            const Input& input, const Output& output, const Shape4<i64>& shape,
            const Float33& inv_matrix, const Symmetry& symmetry, const Vec3<f32>& post_shift = {},
            f32 cutoff = 0.5f, InterpMode interp_mode = InterpMode::LINEAR, bool normalize = true) {
        details::transform_nd_check_parameters<3>(input, output, shape, inv_matrix, post_shift);
        auto input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::geometry::fft::transform_and_symmetrize_3d<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), shape,
                        inv_matrix, symmetry, post_shift, cutoff,
                        interp_mode, normalize, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::geometry::fft::transform_and_symmetrize_3d<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), shape,
                    inv_matrix, symmetry, post_shift, cutoff,
                    interp_mode, normalize, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    /// This functions has the same features and limitations as the overload taking arrays.
    template<Remap REMAP, typename Value, typename Output, typename = std::enable_if_t<
            noa::traits::is_array_or_view_of_any_v<Output, Value> &&
            details::is_valid_transform_sym_v<REMAP, Value>>>
    void transform_and_symmetrize_3d(
            const Texture<Value>& input, const Output& output, const Shape4<i64>& shape,
            const Float33& inv_matrix, const Symmetry& symmetry, const Vec3<f32>& post_shift = {},
            f32 cutoff = 0.5f, bool normalize = true) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            const Array<Value> input_array(texture.ptr, input.shape(), texture.strides, input.options());
            transform_and_symmetrize_3d<REMAP>(
                    input_array, output, shape, inv_matrix,
                    symmetry, post_shift, cutoff, input.interp_mode(), normalize);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                details::transform_nd_check_parameters<3>(input, output, shape, inv_matrix, post_shift);
                const cuda::Texture<Value>& texture = input.cuda();
                auto& cuda_stream = stream.cuda();
                cuda::geometry::fft::transform_and_symmetrize_3d<REMAP>(
                        texture.array.get(), *texture.texture, input.interp_mode(),
                        output.get(), output.strides(), shape,
                        inv_matrix, symmetry, post_shift, cutoff, normalize, cuda_stream);
                cuda_stream.enqueue_attach(texture.array, texture.texture, output.share());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Symmetrizes a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \param[in] input        Non-redundant 2D FFT, of f32, f64, c32, or c64, to transform.
    /// \param[out] output      Non-redundant transformed 2D FFT.
    /// \param shape            Rightmost logical shape of \p input and \p output.
    /// \param[in] symmetry     Symmetry operator to apply.
    /// \param[in] post_shift   Rightmost 2D real-space forward shift to apply (as phase shift) after the symmetry.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    template<Remap REMAP, typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             (REMAP == Remap::HC2HC || REMAP == Remap::HC2H)>>
    void symmetrize_2d(const Input& input, const Output& output, const Shape4<i64>& shape,
                      const Symmetry& symmetry, const Vec2<f32>& post_shift = {},
                      f32 cutoff = 0.5f, InterpMode interp_mode = InterpMode::LINEAR, bool normalize = true) {
        transform_and_symmetrize_2d<REMAP>(
                input, output, shape, Float22{}, symmetry, post_shift,
                cutoff, interp_mode, normalize);
    }

    /// Symmetrizes a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \param[in] input        Non-redundant 3D FFT, of f32, f64, c32, or c64, to transform.
    /// \param[out] output      Non-redundant transformed 3D FFT.
    /// \param shape            Rightmost logical shape of \p input and \p output.
    /// \param[in] symmetry     Symmetry operator to apply.
    /// \param[in] post_shift   Rightmost 3D real-space forward shift to apply (as phase shift) after the symmetry.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    template<Remap REMAP, typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             (REMAP == Remap::HC2HC || REMAP == Remap::HC2H)>>
    void symmetrize_3d(const Input& input, const Output& output, const Shape4<i64>& shape,
                       const Symmetry& symmetry, const Vec3<f32>& post_shift = {},
                       f32 cutoff = 0.5f, InterpMode interp_mode = InterpMode::LINEAR, bool normalize = true) {
        transform_and_symmetrize_3d<REMAP>(
                input, output, shape, Float33{}, symmetry, post_shift,
                cutoff, interp_mode, normalize);
    }
}
