#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/geometry/Interpolator.h"
#include "noa/common/geometry/details/LinearTransform2DFourier.h"
#include "noa/common/geometry/details/LinearTransform3DFourier.h"

#include "noa/cpu/geometry/fft/Transform.h"
#include "noa/cpu/utils/Iwise.h"

namespace {
    using namespace ::noa;

    template<typename MatrixOrShift>
    auto matrixOrShiftOrRawConstPtr(const MatrixOrShift& wrapper) {
        using clean_t = traits::remove_ref_cv_t<MatrixOrShift>;
        if constexpr (traits::is_floatXX_v<MatrixOrShift> || traits::is_floatX_v<MatrixOrShift>) {
            return clean_t(wrapper);
        } else {
            NOA_ASSERT(wrapper != nullptr);
            using raw_const_ptr_t = const typename clean_t::element_type*;
            return static_cast<raw_const_ptr_t>(wrapper.get());
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename ShiftOrEmpty>
    void linearTransform_(const AccessorRestrict<const Value, 3, int64_t>& input,
                          const AccessorRestrict<Value, 3, int64_t>& output, dim4_t shape,
                          Matrix matrices, ShiftOrEmpty shift, float cutoff,
                          InterpMode interp_mode, dim_t threads) {
        NOA_ASSERT(shape[1] == 1);
        const auto input_shape_2d = long2_t(shape.get(2)).fft();
        const auto iwise_shape = safe_cast<long3_t>(dim3_t{shape[0], shape[2], shape[3]}).fft();

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int64_t>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise3D(iwise_shape, kernel, threads);
            }
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int64_t>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise3D(iwise_shape, kernel, threads);
            }
            case INTERP_COSINE_FAST:
            case INTERP_COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int64_t>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise3D(iwise_shape, kernel, threads);
            }
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename ShiftOrEmpty>
    void linearTransform_(const AccessorRestrict<const Value, 4, int64_t>& input,
                          const AccessorRestrict<Value, 4, int64_t>& output, dim4_t shape,
                          Matrix matrices, ShiftOrEmpty shift, float cutoff,
                          InterpMode interp_mode, dim_t threads) {
        const auto input_shape_3d = long3_t(shape.get(1)).fft();
        const auto iwise_shape = safe_cast<long4_t>(shape).fft();

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_NEAREST>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transform3D<REMAP, int64_t>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise4D(iwise_shape, kernel, threads);
            }
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_LINEAR>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transform3D<REMAP, int64_t>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise4D(iwise_shape, kernel, threads);
            }
            case INTERP_COSINE_FAST:
            case INTERP_COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_COSINE>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transform3D<REMAP, int64_t>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise4D(iwise_shape, kernel, threads);
            }
            default:
                NOA_THROW_FUNC("transform3D", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, int32_t NDIM, typename Value, typename Matrix, typename Shift>
    void launchLinearTransform_(const AccessorRestrict<const Value, NDIM + 1, int64_t>& input,
                                const AccessorRestrict<Value, NDIM + 1, int64_t>& output, dim4_t shape,
                                Matrix matrices, Shift shift, float cutoff,
                                InterpMode interp_mode, dim_t threads) {
        const bool do_shift = noa::any(shift != Shift{});
        if (do_shift)
            linearTransform_<REMAP>(input, output, shape, matrices, shift, cutoff, interp_mode, threads);
        else
            linearTransform_<REMAP>(input, output, shape, matrices, empty_t{}, cutoff, interp_mode, threads);
    }

    template<fft::Remap REMAP, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void linearTransformSymmetry_(const AccessorRestrict<const Value, 3, int64_t>& input,
                                  const AccessorRestrict<Value, 3, int64_t>& output, dim4_t shape,
                                  MatrixOrEmpty matrix, const geometry::Symmetry& symmetry,
                                  ShiftOrEmpty shift, float cutoff, bool normalize,
                                  InterpMode interp_mode, dim_t threads) {
        NOA_ASSERT(shape[1] == 1);
        const auto input_shape_2d = long2_t(shape.get(2)).fft();
        const auto iwise_shape = safe_cast<long3_t>(dim3_t{shape[0], shape[2], shape[3]}).fft();

        const auto symmetry_count = static_cast<int64_t>(symmetry.count());
        const float33_t* symmetry_matrices = symmetry.get();
        using real_t = traits::value_type_t<Value>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return cpu::utils::iwise3D(iwise_shape, kernel, threads);
            }
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return cpu::utils::iwise3D(iwise_shape, kernel, threads);
            }
            case INTERP_COSINE_FAST:
            case INTERP_COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return cpu::utils::iwise3D(iwise_shape, kernel, threads);
            }
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void linearTransformSymmetry_(const AccessorRestrict<const Value, 4, int64_t>& input,
                                  const AccessorRestrict<Value, 4, int64_t>& output, dim4_t shape,
                                  MatrixOrEmpty matrix, const geometry::Symmetry& symmetry,
                                  ShiftOrEmpty shift, float cutoff, bool normalize,
                                  InterpMode interp_mode, dim_t threads) {
        NOA_ASSERT(shape[1] == 1);
        const auto input_shape_3d = long3_t(shape.get(1)).fft();
        const auto iwise_shape = safe_cast<long4_t>(shape).fft();

        const auto symmetry_count = static_cast<int64_t>(symmetry.count());
        const float33_t* symmetry_matrices = symmetry.get();
        using real_t = traits::value_type_t<Value>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_NEAREST>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry3D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return cpu::utils::iwise4D(iwise_shape, kernel, threads);
            }
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_LINEAR>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry3D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return cpu::utils::iwise4D(iwise_shape, kernel, threads);
            }
            case INTERP_COSINE_FAST:
            case INTERP_COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_COSINE>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry3D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return cpu::utils::iwise4D(iwise_shape, kernel, threads);
            }
            default:
                NOA_THROW_FUNC("transform3D", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, int32_t NDIM, typename Value, typename Matrix, typename Shift>
    void launchLinearTransformSymmetry_(const AccessorRestrict<const Value, NDIM + 1, int64_t>& input,
                                        const AccessorRestrict<Value, NDIM + 1, int64_t>& output, dim4_t shape,
                                        Matrix matrix, const geometry::Symmetry& symmetry, Shift shift,
                                        float cutoff, bool normalize, InterpMode interp_mode, dim_t threads) {
        const bool apply_shift = noa::any(shift != Shift{});
        const bool apply_matrix = matrix != Matrix{};

        if (apply_shift && apply_matrix) {
            linearTransformSymmetry_<REMAP>(
                    input, output, shape,
                    matrix, symmetry, shift, cutoff,
                    normalize, interp_mode, threads);
        } else if (apply_shift) {
            linearTransformSymmetry_<REMAP>(
                    input, output, shape,
                    empty_t{}, symmetry, shift, cutoff,
                    normalize, interp_mode, threads);
        } else if (apply_matrix) {
            linearTransformSymmetry_<REMAP>(
                    input, output, shape,
                    matrix, symmetry, empty_t{}, cutoff,
                    normalize, interp_mode, threads);
        } else {
            linearTransformSymmetry_<REMAP>(
                    input, output, shape,
                    empty_t{}, symmetry, empty_t{}, cutoff,
                    normalize, interp_mode, threads);
        }
    }
}

namespace noa::cpu::geometry::fft {
    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform2D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(shape > 0));
        NOA_ASSERT(!indexing::isOverlap(input.get(), input_strides, output.get(), output_strides, shape.fft()));

        const auto threads = stream.threads();
        const auto input_strides_2d = safe_cast<long3_t>(dim3_t{input_strides[0], input_strides[2], input_strides[3]});
        const auto output_strides_2d = safe_cast<long3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});

        stream.enqueue([=]() {
            const AccessorRestrict<const Value, 3, int64_t> input_accessor(input.get(), input_strides_2d);
            const AccessorRestrict<Value, 3, int64_t> output_accessor(output.get(), output_strides_2d);
            launchLinearTransform_<REMAP, 2>(
                    input_accessor, output_accessor, shape,
                    matrixOrShiftOrRawConstPtr(inv_matrices), matrixOrShiftOrRawConstPtr(shifts),
                    cutoff, interp_mode, threads);
        });
    }

    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform3D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(shape > 0));
        NOA_ASSERT(!indexing::isOverlap(input.get(), input_strides, output.get(), output_strides, shape.fft()));

        const auto threads = stream.threads();
        const auto input_strides_3d = safe_cast<long4_t>(input_strides);
        const auto output_strides_3d = safe_cast<long4_t>(output_strides);

        stream.enqueue([=]() {
            const AccessorRestrict<const Value, 4, int64_t> input_accessor(input.get(), input_strides_3d);
            const AccessorRestrict<Value, 4, int64_t> output_accessor(output.get(), output_strides_3d);
            launchLinearTransform_<REMAP, 3>(
                    input_accessor, output_accessor, shape,
                    matrixOrShiftOrRawConstPtr(inv_matrices), matrixOrShiftOrRawConstPtr(shifts),
                    cutoff, interp_mode, threads);
        });
    }

    template<Remap REMAP, typename Value, typename>
    void transform2D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     float22_t inv_matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(shape > 0));
        NOA_ASSERT(!indexing::isOverlap(input.get(), input_strides, output.get(), output_strides, shape.fft()));

        if (!symmetry.count()) {
            return transform2D<REMAP>(
                    input, input_strides, output, output_strides,
                    shape, inv_matrix, shift, cutoff, interp_mode, stream);
        }

        const auto threads = stream.threads();
        const auto input_strides_2d = safe_cast<long3_t>(dim3_t{input_strides[0], input_strides[2], input_strides[3]});
        const auto output_strides_2d = safe_cast<long3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});

        stream.enqueue([=]() {
            const AccessorRestrict<const Value, 3, int64_t> input_accessor(input.get(), input_strides_2d);
            const AccessorRestrict<Value, 3, int64_t> output_accessor(output.get(), output_strides_2d);
            launchLinearTransformSymmetry_<REMAP, 2>(
                    input_accessor, output_accessor, shape,
                    inv_matrix, symmetry, shift, cutoff, normalize, interp_mode, threads);
        });
    }

    template<Remap REMAP, typename Value, typename>
    void transform3D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     float33_t inv_matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(shape > 0));
        NOA_ASSERT(!indexing::isOverlap(input.get(), input_strides, output.get(), output_strides, shape.fft()));

        if (!symmetry.count()) {
            return transform3D<REMAP>(
                    input, input_strides, output, output_strides,
                    shape, inv_matrix, shift, cutoff, interp_mode, stream);
        }

        const auto threads = stream.threads();
        const auto input_strides_3d = safe_cast<long4_t>(input_strides);
        const auto output_strides_3d = safe_cast<long4_t>(output_strides);

        stream.enqueue([=]() {
            const AccessorRestrict<const Value, 4, int64_t> input_accessor(input.get(), input_strides_3d);
            const AccessorRestrict<Value, 4, int64_t> output_accessor(output.get(), output_strides_3d);
            launchLinearTransformSymmetry_<REMAP, 3>(
                    input_accessor, output_accessor, shape,
                    inv_matrix, symmetry, shift, cutoff, normalize, interp_mode, threads);
        });
    }

    #define NOA_INSTANTIATE_TRANSFORM_SYMMETRY(T)                                                                                                                                                   \
    template void transform2D<Remap::HC2HC, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&l);  \
    template void transform2D<Remap::HC2H, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&);    \
    template void transform3D<Remap::HC2HC, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&);   \
    template void transform3D<Remap::HC2H, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_(T, M, S) \
    template void transform2D<Remap::HC2H,  T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&); \
    template void transform2D<Remap::HC2HC, T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM3D_(T, M, S) \
    template void transform3D<Remap::HC2H,  T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&); \
    template void transform3D<Remap::HC2HC, T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_ALL_(T)                         \
    NOA_INSTANTIATE_TRANSFORM2D_(T, float22_t, float2_t);               \
    NOA_INSTANTIATE_TRANSFORM2D_(T, shared_t<float22_t[]>, float2_t);   \
    NOA_INSTANTIATE_TRANSFORM2D_(T, float22_t, shared_t<float2_t[]>);   \
    NOA_INSTANTIATE_TRANSFORM2D_(T, shared_t<float22_t[]>, shared_t<float2_t[]>)

    #define NOA_INSTANTIATE_TRANSFORM3D_ALL_(T)                         \
    NOA_INSTANTIATE_TRANSFORM3D_(T, float33_t, float3_t);               \
    NOA_INSTANTIATE_TRANSFORM3D_(T, shared_t<float33_t[]>, float3_t);   \
    NOA_INSTANTIATE_TRANSFORM3D_(T, float33_t, shared_t<float3_t[]>);   \
    NOA_INSTANTIATE_TRANSFORM3D_(T, shared_t<float33_t[]>, shared_t<float3_t[]>)

    #define NOA_INSTANTIATE_TRANSFORM_ALL_(T)   \
    NOA_INSTANTIATE_TRANSFORM2D_ALL_(T);        \
    NOA_INSTANTIATE_TRANSFORM3D_ALL_(T);        \
    NOA_INSTANTIATE_TRANSFORM_SYMMETRY(T)

    NOA_INSTANTIATE_TRANSFORM_ALL_(float);
    NOA_INSTANTIATE_TRANSFORM_ALL_(double);
    NOA_INSTANTIATE_TRANSFORM_ALL_(cfloat_t);
    NOA_INSTANTIATE_TRANSFORM_ALL_(cdouble_t);
}
