#pragma once

#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"

#include "noa/cpu/geometry/fft/Transform.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Transform.h"
#endif

#include "noa/unified/Array.h"

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix and
    ///          shift will be applied to each batch. In this case, the input can be batched as well, resulting in a
    ///          fully batched operation (1 input -> 1 output). However if the input is not batched, it is broadcast
    ///          to all output batches (1 input -> N output).
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 2D FFT to transform.
    /// \param[out] output      Non-redundant transformed 2D FFT.
    /// \param[in] matrices     2x2 inverse rightmost rotation/scaling matrix. One per output batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       Rightmost 2D real-space forward shift to apply (as phase shift) after the transformation.
    ///                         One per output batch. If empty or if \p T is real, it is ignored.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    ///
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input, \p matrices and \p shifts can be on any device, including the CPU.\n
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     const Array<float22_t>& matrices, const Array<float2_t>& shifts,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR) {
        NOA_CHECK(input.shape()[2] == input.shape()[2] &&
                  input.shape()[1] == input.shape()[1],
                  "The input {} and output {} shapes don't match", input.shape(), output.shape());
        size4_t input_stride = input.stride();
        if (input.shape()[0] == 1) {
            input_stride[0] = 0;
        } else if (input.shape()[0] == output.shape()[0]) {
            NOA_THROW("The number of batches in the input ({}) is not compatible with the number of "
                      "batches in the output ({})", input.shape()[0], output.shape()[0]);
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");
            NOA_CHECK(matrices.dereferencable() && shifts.dereferencable(),
                      "The rotation parameters should be accessible to the host");

            if (matrices.device().gpu())
                Stream::current(matrices.device()).synchronize();
            if (shifts.device().gpu() && matrices.device() != shifts.device())
                Stream::current(shifts.device()).synchronize();

            cpu::geometry::fft::transform2D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrices.share(), shifts.share(), cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(input_stride[3] == 1,
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          input.shape(), input_stride);

                if (input.device().cpu() || matrices.device().cpu() || shifts.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform2D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrices.share(), shifts.share(), cutoff, interp_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     float22_t matrix, float2_t shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR) {
        NOA_CHECK(input.shape()[2] == input.shape()[2] &&
                  input.shape()[1] == input.shape()[1],
                  "The input {} and output {} shapes don't match", input.shape(), output.shape());
        size4_t input_stride = input.stride();
        if (input.shape()[0] == 1) {
            input_stride[0] = 0;
        } else if (input.shape()[0] == output.shape()[0]) {
            NOA_THROW("The number of batches in the input ({}) is not compatible with the number of "
                      "batches in the output ({})", input.shape()[0], output.shape()[0]);
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::fft::transform2D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrix, shift, cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(input_stride[3] == 1,
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          input.shape(), input_stride);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform2D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrix, shift, cutoff, interp_mode, stream.cuda());
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
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 3D FFT to transform.
    /// \param[out] output      Non-redundant transformed 3D FFT.
    /// \param[in] matrices     3x3 inverse rightmost rotation/scaling matrix. One per output batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       Rightmost 3D real-space forward shift to apply (as phase shift) after the transformation.
    ///                         One per output batch. If empty or if \p T is real, it is ignored.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    ///
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input, \p matrices and \p shifts can be on any device, including the CPU.\n
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     const Array<float33_t>& matrices, const Array<float3_t>& shifts,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR) {
        NOA_CHECK(input.shape()[3] == input.shape()[3] &&
                  input.shape()[2] == input.shape()[2] &&
                  input.shape()[1] == input.shape()[1],
                  "The input {} and output {} shapes don't match", input.shape(), output.shape());
        size4_t input_stride = input.stride();
        if (input.shape()[0] == 1) {
            input_stride[0] = 0;
        } else if (input.shape()[0] == output.shape()[0]) {
            NOA_THROW("The number of batches in the input ({}) is not compatible with the number of "
                      "batches in the output ({})", input.shape()[0], output.shape()[0]);
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");
            NOA_CHECK(matrices.dereferencable() && shifts.dereferencable(),
                      "The rotation parameters should be accessible to the host");

            if (matrices.device().gpu())
                Stream::current(matrices.device()).synchronize();
            if (shifts.device().gpu() && matrices.device() != shifts.device())
                Stream::current(shifts.device()).synchronize();

            cpu::geometry::fft::transform3D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrices.share(), shifts.share(), cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isContiguous(input_stride, input.shape())[1] && input_stride[3] == 1,
                          "The third-most and innermost dimension of the input should be contiguous, but got shape {} "
                          "and stride {}", input.shape(), input_stride);

                if (input.device().cpu() || matrices.device().cpu() || shifts.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform3D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrices.share(), shifts.share(), cutoff, interp_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     float33_t matrix, float3_t shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR) {
        NOA_CHECK(input.shape()[3] == input.shape()[3] &&
                  input.shape()[2] == input.shape()[2] &&
                  input.shape()[1] == input.shape()[1],
                  "The input {} and output {} shapes don't match", input.shape(), output.shape());
        size4_t input_stride = input.stride();
        if (input.shape()[0] == 1) {
            input_stride[0] = 0;
        } else if (input.shape()[0] == output.shape()[0]) {
            NOA_THROW("The number of batches in the input ({}) is not compatible with the number of "
                      "batches in the output ({})", input.shape()[0], output.shape()[0]);
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::fft::transform3D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrix, shift, cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isContiguous(input_stride, input.shape())[1] && input_stride[3] == 1,
                          "The third-most and innermost dimension of the input should be contiguous, but got shape {} "
                          "and stride {}", input.shape(), input_stride);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform3D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrix, shift, cutoff, interp_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

namespace noa::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    /// Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 2D FFT to transform.
    /// \param[out] output      Non-redundant transformed 2D FFT.
    /// \param[in] matrix       2x2 inverse rightmost rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] shift        Rightmost 2D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    ///
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input can be on any device, including the CPU.\n
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void transform2D(const Array<T>& input, const Array<T>& output,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true) {
        NOA_CHECK(input.shape()[2] == input.shape()[2] &&
                  input.shape()[1] == input.shape()[1],
                  "The input {} and output {} shapes don't match", input.shape(), output.shape());
        size4_t input_stride = input.stride();
        if (input.shape()[0] == 1) {
            input_stride[0] = 0;
        } else if (input.shape()[0] == output.shape()[0]) {
            NOA_THROW("The number of batches in the input ({}) is not compatible with the number of "
                      "batches in the output ({})", input.shape()[0], output.shape()[0]);
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::fft::transform2D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(input_stride[3] == 1,
                          "The innermost dimension of the input should be contiguous, but got shape {} and stride {}",
                          input.shape(), input_stride);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform2D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 3D FFT to transform.
    /// \param[out] output      Non-redundant transformed 3D FFT.
    /// \param[in] matrix       3x3 inverse rightmost rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] shift        Rightmost 3D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    ///
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input can be on any device, including the CPU.\n
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void transform3D(const Array<T>& input, const Array<T>& output,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true) {
        NOA_CHECK(input.shape()[3] == input.shape()[3] &&
                  input.shape()[2] == input.shape()[2] &&
                  input.shape()[1] == input.shape()[1],
                  "The input {} and output {} shapes don't match", input.shape(), output.shape());
        size4_t input_stride = input.stride();
        if (input.shape()[0] == 1) {
            input_stride[0] = 0;
        } else if (input.shape()[0] == output.shape()[0]) {
            NOA_THROW("The number of batches in the input ({}) is not compatible with the number of "
                      "batches in the output ({})", input.shape()[0], output.shape()[0]);
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(input.get != output.get(), "In-place transformations are not supported");

            cpu::geometry::fft::transform3D<REMAP>(
                    input.share(), input_stride,
                    output.share(), output.stride(), output.shape(),
                    matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isContiguous(input_stride, input.shape())[1] && input_stride[3] == 1,
                          "The third-most and innermost dimension of the input should be contiguous, but got shape {} "
                          "and stride {}", input.shape(), input_stride);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::fft::transform3D<REMAP>(
                        input.share(), input.stride(),
                        output.share(), output.stride(), output.shape(),
                        matrix, symmetry, shift, cutoff, interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
