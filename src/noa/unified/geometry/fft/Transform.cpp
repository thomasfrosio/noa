#include "noa/unified/geometry/fft/Transform.h"

#include "noa/cpu/geometry/fft/Transform.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Transform.h"
#endif

namespace {
    using namespace ::noa;

    template<int32_t NDIM, typename ArrayOrTexture, typename Value, typename Matrix, typename Shift>
    void transformNDCheckParameters_(const ArrayOrTexture& input, const Array<Value>& output, dim4_t shape,
                                     const Matrix& inv_matrices, const Shift& post_shifts) {
        const char* FUNC_NAME = NDIM == 2 ? "transform2D" : "transform3D";

        NOA_CHECK_FUNC(FUNC_NAME, !input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK_FUNC(FUNC_NAME,
                       shape[3] / 2 + 1 == input.shape()[3] && input.shape()[3] == output.shape()[3] &&
                       shape[2] == input.shape()[2] && input.shape()[2] == output.shape()[2] &&
                       shape[1] == input.shape()[1] && input.shape()[1] == output.shape()[1],
                       "The non-redundant input {} and/or output {} shapes don't match the logical shape {}",
                       input.shape(), output.shape(), shape);
        NOA_CHECK_FUNC(FUNC_NAME, NDIM == 3 || (input.shape()[1] == 1 && output.shape()[1] == 1),
                       "The input and output arrays should be 2D, but got shape input:{}, output:{}",
                       input.shape(), output.shape());
        NOA_CHECK_FUNC(FUNC_NAME, input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                       "The number of batches in the input ({}) is not compatible with the number of "
                       "batches in the output ({})", input.shape()[0], output.shape()[0]);

        if constexpr (!traits::is_floatXX_v<Matrix>) {
            NOA_CHECK_FUNC(FUNC_NAME,
                           indexing::isVector(inv_matrices.shape()) &&
                           inv_matrices.elements() == output.shape()[0] && inv_matrices.contiguous(),
                           "The number of matrices, specified as a contiguous vector, should be equal "
                           "to the number of batches in the output, got {} matrices and {} output batches",
                           inv_matrices.elements(), output.shape()[0]);
        }
        if constexpr (!traits::is_floatX_v<Shift>) {
            NOA_CHECK_FUNC(FUNC_NAME,
                           indexing::isVector(post_shifts.shape()) &&
                           post_shifts.elements() == output.shape()[0] && post_shifts.contiguous(),
                           "The number of shifts, specified as a contiguous vector, should be equal "
                           "to the number of batches in the output, got {} shifts and {} output batches",
                           post_shifts.elements(), output.shape()[0]);
        }

        const Device device = output.device();
        if (device.cpu()) {
            NOA_CHECK_FUNC(FUNC_NAME, device == input.device(),
                           "The input and output arrays must be on the same device, "
                           "but got input:{} and output:{}", input.device(), device);

            if constexpr (!traits::is_floatXX_v<Matrix>) {
                NOA_CHECK_FUNC(FUNC_NAME, inv_matrices.dereferenceable(),
                               "The matrices should be accessible to the CPU");
                if (inv_matrices.device().gpu())
                    Stream::current(inv_matrices.device()).synchronize();
                if constexpr (!traits::is_floatX_v<Shift>) {
                    NOA_CHECK_FUNC(FUNC_NAME, post_shifts.empty() || post_shifts.dereferenceable(),
                                   "The shifts should be accessible to the CPU");
                    if (!post_shifts.empty() && post_shifts.device().gpu() &&
                        inv_matrices.device() != post_shifts.device())
                        Stream::current(post_shifts.device()).synchronize();
                }
            } else if constexpr (!traits::is_floatX_v<Shift>) {
                NOA_CHECK_FUNC(FUNC_NAME, post_shifts.empty() || post_shifts.dereferenceable(),
                               "The shifts should be accessible to the CPU");
                if (!post_shifts.empty() && post_shifts.device().gpu())
                    Stream::current(post_shifts.device()).synchronize();
            }

            if constexpr (std::is_same_v<ArrayOrTexture, Array<Value>>) {
                NOA_CHECK_FUNC(FUNC_NAME, !indexing::isOverlap(input, output),
                               "The input and output arrays should not overlap");
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            bool sync_cpu = false;
            if constexpr (!traits::is_floatXX_v<Matrix>) {
                if (inv_matrices.device().cpu())
                    sync_cpu = true;
                else if (inv_matrices.device().id() != device.id())
                    Stream::current(inv_matrices.device()).synchronize();
            }
            if constexpr (!traits::is_floatX_v<Shift>) {
                if (!post_shifts.empty()) {
                    if (post_shifts.device().cpu())
                        sync_cpu = true;
                    else if (post_shifts.device().id() != device.id())
                        Stream::current(post_shifts.device()).synchronize();
                }
            }

            if constexpr (std::is_same_v<ArrayOrTexture, Array<Value>>) {
                if (input.device().cpu())
                    sync_cpu = true;
            } else {
                NOA_CHECK_FUNC(FUNC_NAME, input.device() == output.device(),
                               "The input texture and output array must be on the same device, "
                               "but got input:{} and output:{}", input.device(), output.device());
            }
            if (sync_cpu)
                Stream::current(Device(Device::CPU)).synchronize();
            #endif
        }
    }

    // Matrix or Shift: floatXX_t, floatX_t, Array<floatXX_t> or Array<floatX_t>.
    // Returns a reference to either the matrix/shift or the shared pointer of the array.
    template<typename MatrixOrShift>
    auto extractMatrixOrShift_(const MatrixOrShift& matrix_or_shift) {
        using shared_matrix_or_shift_t = const traits::shared_type_t<MatrixOrShift>&;
        if constexpr (traits::is_floatXX_v<MatrixOrShift> || traits::is_floatX_v<MatrixOrShift>)
            return shared_matrix_or_shift_t(matrix_or_shift);
        else
            return shared_matrix_or_shift_t(matrix_or_shift.share());
    }
}

namespace noa::geometry::fft {
    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform2D(const Array<Value>& input, const Array<Value>& output, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& post_shifts,
                     float cutoff, InterpMode interp_mode) {
        transformNDCheckParameters_<2>(input, output, shape, inv_matrices, post_shifts);
        dim4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::transform2D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    extractMatrixOrShift_(inv_matrices),
                    extractMatrixOrShift_(post_shifts),
                    cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                cuda::geometry::fft::transform2D<REMAP>(
                        input.share(), input_strides,
                        output.share(), output.strides(), shape,
                        extractMatrixOrShift_(inv_matrices),
                        extractMatrixOrShift_(post_shifts),
                        cutoff, interp_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform2D(const Texture<Value>& input, const Array<Value>& output, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& post_shifts,
                     float cutoff) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);

        if (input.device().cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            const Array<Value> tmp(texture.ptr, input.shape(), texture.strides, input.options());
            transform2D<REMAP>(tmp, output, shape, inv_matrices, post_shifts, cutoff, input.interp());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                transformNDCheckParameters_<2>(input, output, shape, inv_matrices, post_shifts);
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::fft::transform2D<REMAP>(
                        texture.array, texture.texture, input.interp(),
                        output.share(), output.strides(), output.shape(),
                        extractMatrixOrShift_(inv_matrices),
                        extractMatrixOrShift_(post_shifts),
                        cutoff, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform3D(const Array<Value>& input, const Array<Value>& output, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& post_shifts,
                     float cutoff, InterpMode interp_mode) {
        transformNDCheckParameters_<3>(input, output, shape, inv_matrices, post_shifts);
        dim4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::transform3D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    extractMatrixOrShift_(inv_matrices),
                    extractMatrixOrShift_(post_shifts),
                    cutoff, interp_mode, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                cuda::geometry::fft::transform3D<REMAP>(
                        input.share(), input_strides,
                        output.share(), output.strides(), shape,
                        extractMatrixOrShift_(inv_matrices),
                        extractMatrixOrShift_(post_shifts),
                        cutoff, interp_mode, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform3D(const Texture<Value>& input, const Array<Value>& output, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& post_shifts,
                     float cutoff) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            const Array<Value> tmp(texture.ptr, input.shape(), texture.strides, input.options());
            transform3D<REMAP>(tmp, output, shape, inv_matrices, post_shifts, cutoff, input.interp());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                transformNDCheckParameters_<3>(input, output, shape, inv_matrices, post_shifts);
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::fft::transform3D<REMAP>(
                        texture.array, texture.texture, input.interp(),
                        output.share(), output.strides(), output.shape(),
                        extractMatrixOrShift_(inv_matrices),
                        extractMatrixOrShift_(post_shifts),
                        cutoff, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_(T, R, M, S)                                                              \
    template void transform2D<R, T, M, S, void>(const Array<T>&, const Array<T>&, dim4_t, const M&, const S&, float, InterpMode);   \
    template void transform2D<R, T, M, S, void>(const Texture<T>&, const Array<T>&, dim4_t, const M&, const S&, float)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_(T, R, M, S)                                                              \
    template void transform3D<R, T, M, S, void>(const Array<T>&, const Array<T>&, dim4_t, const M&, const S&, float, InterpMode);   \
    template void transform3D<R, T, M, S, void>(const Texture<T>&, const Array<T>&, dim4_t, const M&, const S&, float)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_REMAP_(T, M, S)   \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_(T, Remap::HC2HC, M, S);  \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_(T, Remap::HC2H, M, S)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_REMAP_(T, M, S)   \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_(T, Remap::HC2HC, M, S);  \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_(T, Remap::HC2H, M, S)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_ALL(T)                            \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_REMAP_(T, float22_t, float2_t);           \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_REMAP_(T, float22_t, Array<float2_t>);    \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_REMAP_(T, Array<float22_t>, float2_t);    \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_REMAP_(T, Array<float22_t>, Array<float2_t>)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_ALL(T)                            \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_REMAP_(T, float33_t, float3_t);           \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_REMAP_(T, float33_t, Array<float3_t>);    \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_REMAP_(T, Array<float33_t>, float3_t);    \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_REMAP_(T, Array<float33_t>, Array<float3_t>)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_ALL(T)   \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_ALL(T);       \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_ALL(T)

    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_ALL(float);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_ALL(double);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_ALL(cfloat_t);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_ALL(cdouble_t);
}

namespace noa::geometry::fft {
    template<Remap REMAP, typename Value, typename>
    void transform2D(const Array<Value>& input, const Array<Value>& output, dim4_t shape,
                     float22_t inv_matrix, const Symmetry& symmetry, float2_t post_shift,
                     float cutoff, InterpMode interp_mode, bool normalize) {
        transformNDCheckParameters_<2>(input, output, shape, inv_matrix, post_shift);
        dim4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::transform2D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    inv_matrix, symmetry, post_shift, cutoff,
                    interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                cuda::geometry::fft::transform2D<REMAP>(
                        input.share(), input_strides,
                        output.share(), output.strides(), shape,
                        inv_matrix, symmetry, post_shift, cutoff,
                        interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename>
    void transform2D(const Texture<Value>& input, const Array<Value>& output, dim4_t shape,
                     float22_t inv_matrix, const Symmetry& symmetry, float2_t post_shift,
                     float cutoff, bool normalize) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            const Array<Value> tmp(texture.ptr, input.shape(), texture.strides, input.options());
            transform2D<REMAP>(tmp, output, shape, inv_matrix, symmetry, post_shift, cutoff, input.interp(), normalize);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                transformNDCheckParameters_<2>(input, output, shape, inv_matrix, post_shift);
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::fft::transform2D<REMAP>(
                        texture.array, texture.texture, input.interp(),
                        output.share(), output.strides(), output.shape(),
                        inv_matrix, symmetry, post_shift, cutoff, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename>
    void transform3D(const Array<Value>& input, const Array<Value>& output, dim4_t shape,
                     float33_t inv_matrix, const Symmetry& symmetry, float3_t post_shift,
                     float cutoff, InterpMode interp_mode, bool normalize) {
        transformNDCheckParameters_<3>(input, output, shape, inv_matrix, post_shift);
        dim4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::transform3D<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(), shape,
                    inv_matrix, symmetry, post_shift, cutoff,
                    interp_mode, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                cuda::geometry::fft::transform3D<REMAP>(
                        input.share(), input_strides,
                        output.share(), output.strides(), shape,
                        inv_matrix, symmetry, post_shift, cutoff,
                        interp_mode, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename>
    void transform3D(const Texture<Value>& input, const Array<Value>& output, dim4_t shape,
                     float33_t inv_matrix, const Symmetry& symmetry, float3_t post_shift,
                     float cutoff, bool normalize) {
        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            const Array<Value> tmp(texture.ptr, input.shape(), texture.strides, input.options());
            transform3D<REMAP>(tmp, output, shape, inv_matrix, symmetry, post_shift,
                               cutoff, input.interp(), normalize);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                transformNDCheckParameters_<3>(input, output, shape, inv_matrix, post_shift);
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::fft::transform3D<REMAP>(
                        texture.array, texture.texture, input.interp(),
                        output.share(), output.strides(), output.shape(),
                        inv_matrix, symmetry, post_shift, cutoff, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_2D_(T, R)                                                                           \
    template void transform2D<R, T, void>(const Array<T>&, const Array<T>&, dim4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool); \
    template void transform2D<R, T, void>(const Texture<T>&, const Array<T>&, dim4_t, float22_t, const Symmetry&, float2_t, float, bool)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_3D_(T, R)                                                                           \
    template void transform3D<R, T, void>(const Array<T>&, const Array<T>&, dim4_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool); \
    template void transform3D<R, T, void>(const Texture<T>&, const Array<T>&, dim4_t, float33_t, const Symmetry&, float3_t, float, bool)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_2D_REMAP_(T)    \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_2D_(T, Remap::HC2HC);   \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_2D_(T, Remap::HC2H)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_3D_REMAP_(T)    \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_3D_(T, Remap::HC2HC);   \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_3D_(T, Remap::HC2H)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_ALL(T)  \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_2D_REMAP_(T);   \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_3D_REMAP_(T)

    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_ALL(float);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_ALL(double);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_ALL(cfloat_t);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_ALL(cdouble_t);
}
