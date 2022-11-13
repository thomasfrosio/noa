#include "noa/unified/geometry/Transform.h"

#include "noa/cpu/geometry/Transform.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Transform.h"
#endif

namespace {
    using namespace ::noa;

    template<int32_t NDIM, bool SYMMETRY = false, typename ArrayOrTexture, typename Value, typename Matrix>
    void transformNDCheckParameters_(const ArrayOrTexture& input, const Array<Value>& output, const Matrix& matrix,
                                     InterpMode interp_mode = INTERP_LINEAR, bool prefilter = false) {
        const char* FUNC_NAME = SYMMETRY ?
                                NDIM == 2 ? "symmetry2D" : "symmetry3D" :
                                NDIM == 2 ? "transform2D" : "transform3D";

        NOA_CHECK_FUNC(FUNC_NAME, !input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK_FUNC(FUNC_NAME, NDIM == 3 || (input.shape()[1] == 1 && output.shape()[1] == 1),
                       "The input and output arrays should be 2D, but got shape input:{}, output:{}",
                       input.shape(), output.shape());
        NOA_CHECK_FUNC(FUNC_NAME, input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                       "The number of batches in the input ({}) is not compatible with the number of "
                       "batches in the output ({})", input.shape()[0], output.shape()[0]);

        if constexpr (SYMMETRY) {
            NOA_CHECK_FUNC(FUNC_NAME,
                           input.shape()[3] == output.shape()[3] &&
                           input.shape()[2] == output.shape()[2] &&
                           input.shape()[1] == output.shape()[1],
                           "The input {} and output {} shapes don't match",
                           input.shape(), output.shape());
        }

        if constexpr (!traits::is_floatXX_v<Matrix>) {
            NOA_CHECK_FUNC(FUNC_NAME,
                           indexing::isVector(matrix.shape()) &&
                           matrix.elements() == output.shape()[0] && matrix.contiguous(),
                           "The number of matrices, specified as a contiguous vector, should be equal "
                           "to the number of batches in the output, got {} matrices and {} output batches",
                           matrix.elements(), output.shape()[0]);
        }

        const Device device = output.device();
        if (device.cpu()) {
            NOA_CHECK_FUNC(FUNC_NAME, device == input.device(),
                           "The input and output arrays must be on the same device, "
                           "but got input:{} and output:{}", input.device(), device);

            if constexpr (!traits::is_floatXX_v<Matrix>) {
                NOA_CHECK_FUNC(FUNC_NAME, matrix.dereferenceable(),
                               "The matrices should be accessible to the host");
                if (matrix.device().gpu())
                    Stream::current(matrix.device()).synchronize();
            }

            if constexpr (std::is_same_v<ArrayOrTexture, Array<Value>>) {
                NOA_CHECK_FUNC(FUNC_NAME, !indexing::isOverlap(input, output),
                               "The input and output arrays should not overlap");
            }
        } else {
            #ifdef NOA_ENABLE_CUDA
            bool sync_cpu{false};
            if constexpr (!traits::is_floatXX_v<Matrix>) {
                if (matrix.device().cpu())
                    sync_cpu = true;
                else if (matrix.device().id() != device.id())
                    Stream::current(matrix.device()).synchronize();
            }

            if constexpr (std::is_same_v<ArrayOrTexture, Array<Value>>) {
                if (input.device().cpu())
                    sync_cpu = true;
                const bool is_bspline = interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST;
                NOA_CHECK_FUNC(FUNC_NAME, device == input.device() || (!prefilter || !is_bspline),
                               "The input is about to be prefiltered for cubic B-spline interpolation. "
                               "In this case, the input and output arrays must be on the same device. "
                               "Got device input:{} and output:{}", input.device(), device);
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

    // Matrix: floatXX_t or Array<floatXX_t>.
    // Returns a reference to either the matrix or the shared pointer of the array.
    template<typename Matrix>
    auto extractMatrix_(const Matrix& matrix) {
        using shared_matrix_t = const traits::shared_type_t<Matrix>&;
        if constexpr (traits::is_floatXX_v<Matrix>)
            return shared_matrix_t(matrix);
        else
            return shared_matrix_t(matrix.share());
    }
}

namespace noa::geometry {
    template<typename Value, typename Matrix, typename>
    void transform2D(const Array<Value>& input, const Array<Value>& output, const Matrix& inv_matrices,
                     InterpMode interp_mode, BorderMode border_mode, Value value, bool prefilter) {
        transformNDCheckParameters_<2>(input, output, inv_matrices, interp_mode, prefilter);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::transform2D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    extractMatrix_(inv_matrices), interp_mode, border_mode, value,
                    prefilter, stream.cpu());

        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                cuda::geometry::transform2D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        extractMatrix_(inv_matrices), interp_mode, border_mode,
                        prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename Matrix, typename>
    void transform2D(const Texture<Value>& input, const Array<Value>& output, const Matrix& inv_matrices) {
        transformNDCheckParameters_<2>(input, output, inv_matrices);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            cpu::geometry::transform2D(
                    texture.ptr, texture.strides, input.shape(),
                    output.share(), output.strides(), output.shape(),
                    extractMatrix_(inv_matrices), input.interp(), input.border(),
                    texture.cvalue, false, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::transform2D(
                        texture.array, texture.texture, input.shape(),
                        input.interp(), input.border(),
                        output.share(), output.strides(), output.shape(),
                        extractMatrix_(inv_matrices), stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename Matrix, typename>
    void transform3D(const Array<Value>& input, const Array<Value>& output, const Matrix& inv_matrices,
                     InterpMode interp_mode, BorderMode border_mode, Value value, bool prefilter) {
        transformNDCheckParameters_<3>(input, output, inv_matrices, interp_mode, prefilter);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::transform3D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    extractMatrix_(inv_matrices), interp_mode, border_mode,
                    value, prefilter, stream.cpu());

        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                cuda::geometry::transform3D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        extractMatrix_(inv_matrices), interp_mode, border_mode,
                        prefilter, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename Matrix, typename>
    void transform3D(const Texture<Value>& input, const Array<Value>& output, const Matrix& inv_matrices) {
        transformNDCheckParameters_<3>(input, output, inv_matrices);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            cpu::geometry::transform3D(
                    texture.ptr, texture.strides, input.shape(),
                    output.share(), output.strides(), output.shape(),
                    extractMatrix_(inv_matrices), input.interp(), input.border(),
                    texture.cvalue, false, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::transform3D(
                        texture.array, texture.texture, input.shape(),
                        input.interp(), input.border(),
                        output.share(), output.strides(), output.shape(),
                        extractMatrix_(inv_matrices), stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_(T, M)                                                        \
    template void transform2D<T, M, void>(const Array<T>&, const Array<T>&, const M&, InterpMode, BorderMode, T, bool); \
    template void transform2D<T, M, void>(const Texture<T>&, const Array<T>&, const M&)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_(T, M)                                                        \
    template void transform3D<T, M, void>(const Array<T>&, const Array<T>&, const M&, InterpMode, BorderMode, T, bool); \
    template void transform3D<T, M, void>(const Texture<T>&, const Array<T>&, const M&)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_ALL(T)           \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_(T, float23_t);          \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_(T, float33_t);          \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_(T, Array<float23_t>);   \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_(T, Array<float33_t>)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_ALL(T)           \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_(T, float34_t);          \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_(T, float44_t);          \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_(T, Array<float34_t>);   \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_(T, Array<float44_t>)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_ALL(T)   \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_2D_ALL(T);       \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_3D_ALL(T)

    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_ALL(float);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_ALL(double);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_ALL(cfloat_t);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_ALL(cdouble_t);
}

namespace noa::geometry {
    template<typename Value, typename>
    void transform2D(const Array<Value>& input, const Array<Value>& output,
                     float2_t shift, float22_t inv_matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize) {
        transformNDCheckParameters_<2>(input, output, float22_t{}, interp_mode, prefilter);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::transform2D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    shift, inv_matrix, symmetry, center, interp_mode,
                    prefilter, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                cuda::geometry::transform2D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        shift, inv_matrix, symmetry, center, interp_mode,
                        prefilter, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename>
    void transform2D(const Texture<Value>& input, const Array<Value>& output,
                     float2_t shift, float22_t inv_matrix, const Symmetry& symmetry, float2_t center,
                     bool normalize) {
        transformNDCheckParameters_<2>(input, output, float22_t{});

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            cpu::geometry::transform2D(
                    texture.ptr, texture.strides, input.shape(),
                    output.share(), output.strides(), output.shape(),
                    shift, inv_matrix, symmetry, center, input.interp(),
                    false, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::transform2D(
                        texture.array, texture.texture,
                        input.interp(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        shift, inv_matrix, symmetry, center,
                        normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename>
    void transform3D(const Array<Value>& input, const Array<Value>& output,
                     float3_t shift, float33_t inv_matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize) {
        transformNDCheckParameters_<3>(input, output, float33_t{});

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::transform3D(
                    input.share(), input.strides(), input.shape(),
                    output.share(), output.strides(), output.shape(),
                    shift, inv_matrix, symmetry, center, interp_mode,
                    prefilter, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                cuda::geometry::transform3D(
                        input.share(), input.strides(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        shift, inv_matrix, symmetry, center, interp_mode,
                        prefilter, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename>
    void transform3D(const Texture<Value>& input, const Array<Value>& output,
                     float3_t shift, float33_t inv_matrix, const Symmetry& symmetry, float3_t center,
                     bool normalize) {
        transformNDCheckParameters_<3>(input, output, float33_t{});

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            cpu::geometry::transform3D(
                    texture.ptr, texture.strides, input.shape(),
                    output.share(), output.strides(), output.shape(),
                    shift, inv_matrix, symmetry, center, input.interp(),
                    false, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::transform3D(
                        texture.array, texture.texture,
                        input.interp(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        shift, inv_matrix, symmetry, center,
                        normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename>
    void symmetrize2D(const Array<Value>& input, const Array<Value>& output,
                      const Symmetry& symmetry, float2_t center,
                      InterpMode interp_mode, bool prefilter, bool normalize) {
        transformNDCheckParameters_<2, true>(input, output, float22_t{}, interp_mode, prefilter);
        dim4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::symmetrize2D(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    symmetry, center, interp_mode, prefilter, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                cuda::geometry::symmetrize2D(
                        input.share(), input_strides,
                        output.share(), output.strides(), output.shape(),
                        symmetry, center, interp_mode, prefilter, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename>
    void symmetrize2D(const Texture<Value>& input, const Array<Value>& output,
                      const Symmetry& symmetry, float2_t center,
                      bool normalize) {
        transformNDCheckParameters_<2, true>(input, output, float22_t{}, INTERP_LINEAR, false);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            cpu::geometry::symmetrize2D(
                    texture.ptr, texture.strides,
                    output.share(), output.strides(), output.shape(),
                    symmetry, center, input.interp(),
                    false, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::symmetrize2D(
                        texture.array, texture.texture, input.interp(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        symmetry, center, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename>
    void symmetrize3D(const Array<Value>& input, const Array<Value>& output,
                      const Symmetry& symmetry, float3_t center,
                      InterpMode interp_mode, bool prefilter, bool normalize) {
        transformNDCheckParameters_<2, true>(input, output, float22_t{}, interp_mode, prefilter);
        dim4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::symmetrize3D(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    symmetry, center, interp_mode, prefilter, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<Value>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                cuda::geometry::symmetrize3D(
                        input.share(), input_strides,
                        output.share(), output.strides(), output.shape(),
                        symmetry, center, interp_mode, prefilter, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename>
    void symmetrize3D(const Texture<Value>& input, const Array<Value>& output,
                      const Symmetry& symmetry, float3_t center,
                      bool normalize) {
        transformNDCheckParameters_<3, true>(input, output, float33_t{}, INTERP_LINEAR, false);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = input.cpu();
            cpu::geometry::symmetrize3D(
                    texture.ptr, texture.strides,
                    output.share(), output.strides(), output.shape(),
                    symmetry, center, input.interp(),
                    false, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                const cuda::Texture<Value>& texture = input.cuda();
                cuda::geometry::symmetrize3D(
                        texture.array, texture.texture, input.interp(), input.shape(),
                        output.share(), output.strides(), output.shape(),
                        symmetry, center, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_2D_ALL(T)                                                                           \
    template void transform2D<T, void>(const Array<T>&, const Array<T>&, float2_t, float22_t, const Symmetry&, float2_t, InterpMode, bool, bool);   \
    template void transform2D<T, void>(const Texture<T>&, const Array<T>&, float2_t, float22_t, const Symmetry&, float2_t, bool);                   \
    template void symmetrize2D<T, void>(const Array<T>&, const Array<T>&, const Symmetry&, float2_t, InterpMode, bool, bool);                       \
    template void symmetrize2D<T, void>(const Texture<T>&, const Array<T>&, const Symmetry&, float2_t, bool)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_3D_ALL(T)                                                                           \
    template void transform3D<T, void>(const Array<T>&, const Array<T>&, float3_t, float33_t, const Symmetry&, float3_t, InterpMode, bool, bool);   \
    template void transform3D<T, void>(const Texture<T>&, const Array<T>&, float3_t, float33_t, const Symmetry&, float3_t, bool);                   \
    template void symmetrize3D<T, void>(const Array<T>&, const Array<T>&, const Symmetry&, float3_t, InterpMode, bool, bool);                       \
    template void symmetrize3D<T, void>(const Texture<T>&, const Array<T>&, const Symmetry&, float3_t, bool)

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_ALL(T)   \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_2D_ALL(T);       \
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_3D_ALL(T)

    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_ALL(float);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_ALL(double);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_ALL(cfloat_t);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_TRANSFORM_SYMMETRY_ALL(cdouble_t);
}
