#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/common/Exception.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Interpolator.h"
#include "noa/cpu/geometry/Prefilter.h"
#include "noa/cpu/geometry/Transform.h"

namespace {
    using namespace ::noa;

    template<InterpMode INTERP, typename T>
    void transformWithSymmetry2D_(AccessorRestrict<const T, 3, dim_t> input, dim3_t input_shape,
                                  AccessorRestrict<T, 3, dim_t> output, dim3_t output_shape,
                                  float2_t shift, float22_t matrix,
                                  const geometry::Symmetry& symmetry, float2_t center,
                                  bool normalize, dim_t threads) {
        const dim_t count = symmetry.count();
        const float33_t* sym_matrices = symmetry.get();

        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const cpu::geometry::Interpolator2D interp(input[0], {input_shape[1], input_shape[2]}, 0);

        const float2_t center_shift = float2_t(center + shift);
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(count + 1) : 1;

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(output, output_shape, center, matrix, offset, interp,            \
               center_shift, count, sym_matrices, scaling)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t y = 0; y < output_shape[1]; ++y) {
                for (dim_t x = 0; x < output_shape[2]; ++x) {
                    float2_t pos{y, x};

                    pos -= center;
                    const float2_t coordinates = matrix * pos;
                    T value = interp.template get<INTERP, BORDER_ZERO>(coordinates + center_shift, i * offset);

                    for (dim_t s = 0; s < count; ++s) {
                        const float33_t& m = sym_matrices[s];
                        const float22_t sym_matrix{m[1][1], m[1][2],
                                                   m[2][1], m[2][2]};
                        const float2_t s_coordinates = sym_matrix * coordinates;
                        value += interp.template get<INTERP, BORDER_ZERO>(s_coordinates + center_shift, i * offset);
                    }

                    output(i, y, x) = value * scaling;
                }
            }
        }
    }

    template<InterpMode INTERP, typename T>
    void transformWithSymmetry3D_(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                                  AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape,
                                  float3_t shift, float33_t matrix, const geometry::Symmetry& symmetry, float3_t center,
                                  bool normalize, dim_t threads) {
        const dim_t count = symmetry.count();
        const float33_t* sym_matrices = symmetry.get();

        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const cpu::geometry::Interpolator3D interp(input[0], dim3_t(input_shape.get(1)), 0);

        const float3_t center_shift = float3_t(center + shift);
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(count + 1) : 1;

        #pragma omp parallel for collapse(4) default(none) num_threads(threads)      \
        shared(output, output_shape, center, matrix, offset, interp, \
               center_shift, count, sym_matrices, scaling)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t z = 0; z < output_shape[1]; ++z) {
                for (dim_t y = 0; y < output_shape[2]; ++y) {
                    for (dim_t x = 0; x < output_shape[3]; ++x) {
                        float3_t pos{z, y, x};

                        pos -= center;
                        const float3_t coordinates = matrix * pos;
                        T value = interp.template get<INTERP, BORDER_ZERO>(coordinates + center_shift, i * offset);

                        for (dim_t s = 0; s < count; ++s) {
                            const float3_t s_coordinates = sym_matrices[s] * coordinates;
                            value += interp.template get<INTERP, BORDER_ZERO>(s_coordinates + center_shift, i * offset);
                        }

                        output(i, z, y, x) = value * scaling;
                    }
                }
            }
        }
    }
}

namespace noa::cpu::geometry {
    template<typename T, typename>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), input_strides, input_shape,
                                        output.get(), output_strides, output_shape));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(dim3_t(input_shape.get(1)).ndim() <= 2);
        NOA_ASSERT(dim3_t(output_shape.get(1)).ndim() <= 2);

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            memory::PtrHost<T> buffer;
            const T* tmp;
            dim3_t istrides; // assume Z == 1
            if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
                // FIXME There's no point to support input broadcast since there's only one transform.
                dim4_t shape = input_shape;
                if (input_strides[0] == 0)
                    shape[0] = 1;
                const dim4_t strides = shape.strides();
                buffer = memory::PtrHost<T>(shape.elements());
                bspline::prefilter(input, input_strides, buffer.share(), strides, shape, stream);
                tmp = buffer.get();
                istrides = {strides[0], strides[2], strides[3]};
            } else {
                tmp = input.get();
                istrides = {input_strides[0], input_strides[2], input_strides[3]};
            }

            const dim3_t ishape{input_shape[0], input_shape[2], input_shape[3]};
            const dim3_t oshape{output_shape[0], output_shape[2], output_shape[3]};
            const dim3_t ostrides{output_strides[0], output_strides[2], output_strides[3]};
            switch (interp_mode) {
                case INTERP_NEAREST:
                    return transformWithSymmetry2D_<INTERP_NEAREST, T>(
                            {tmp, istrides}, ishape, {output.get(), ostrides}, oshape,
                            shift, matrix, symmetry, center, normalize, threads);
                case INTERP_LINEAR:
                case INTERP_LINEAR_FAST:
                    return transformWithSymmetry2D_<INTERP_LINEAR, T>(
                            {tmp, istrides}, ishape, {output.get(), ostrides}, oshape,
                            shift, matrix, symmetry, center, normalize, threads);
                case INTERP_COSINE:
                case INTERP_COSINE_FAST:
                    return transformWithSymmetry2D_<INTERP_COSINE, T>(
                            {tmp, istrides}, ishape, {output.get(), ostrides}, oshape,
                            shift, matrix, symmetry, center, normalize, threads);
                case INTERP_CUBIC:
                    return transformWithSymmetry2D_<INTERP_CUBIC, T>(
                            {tmp, istrides}, ishape, {output.get(), ostrides}, oshape,
                            shift, matrix, symmetry, center, normalize, threads);
                case INTERP_CUBIC_BSPLINE:
                case INTERP_CUBIC_BSPLINE_FAST:
                    return transformWithSymmetry2D_<INTERP_CUBIC_BSPLINE, T>(
                            {tmp, istrides}, ishape, {output.get(), ostrides}, oshape,
                            shift, matrix, symmetry, center, normalize, threads);
                default:
                    NOA_THROW("{} is not supported", interp_mode);
            }
        });
    }

    template<typename T, typename>
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), input_strides, input_shape,
                                        output.get(), output_strides, output_shape));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            memory::PtrHost<T> buffer;
            const T* tmp;
            dim4_t tmp_strides;
            if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
                // FIXME There's no point to support input broadcast since there's only one transform.
                dim4_t shape = input_shape;
                if (input_strides[0] == 0)
                    shape[0] = 1;
                const dim4_t strides = shape.strides();
                buffer = memory::PtrHost<T>(shape.elements());
                bspline::prefilter(input, input_strides, buffer.share(), strides, shape, stream);
                tmp = buffer.get();
                tmp_strides = strides;
            } else {
                tmp = input.get();
                tmp_strides = input_strides;
            }

            switch (interp_mode) {
                case INTERP_NEAREST:
                    return transformWithSymmetry3D_<INTERP_NEAREST, T>(
                            {tmp, tmp_strides}, input_shape, {output.get(), output_strides}, output_shape,
                            shift, matrix, symmetry, center, normalize, threads);
                case INTERP_LINEAR:
                case INTERP_LINEAR_FAST:
                    return transformWithSymmetry3D_<INTERP_LINEAR, T>(
                            {tmp, tmp_strides}, input_shape, {output.get(), output_strides}, output_shape,
                            shift, matrix, symmetry, center, normalize, threads);
                case INTERP_COSINE:
                case INTERP_COSINE_FAST:
                    return transformWithSymmetry3D_<INTERP_COSINE, T>(
                            {tmp, tmp_strides}, input_shape, {output.get(), output_strides}, output_shape,
                            shift, matrix, symmetry, center, normalize, threads);
                case INTERP_CUBIC:
                    return transformWithSymmetry3D_<INTERP_CUBIC, T>(
                            {tmp, tmp_strides}, input_shape, {output.get(), output_strides}, output_shape,
                            shift, matrix, symmetry, center, normalize, threads);
                case INTERP_CUBIC_BSPLINE:
                case INTERP_CUBIC_BSPLINE_FAST:
                    return transformWithSymmetry3D_<INTERP_CUBIC_BSPLINE, T>(
                            {tmp, tmp_strides}, input_shape, {output.get(), output_strides}, output_shape,
                            shift, matrix, symmetry, center, normalize, threads);
                default:
                    NOA_THROW("{} is not supported", interp_mode);
            }
        });
    }

    #define NOA_INSTANTIATE_APPLY_(T)                                                                                                                                                                \
    template void transform2D<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float22_t, const Symmetry&, float2_t, InterpMode, bool, bool, Stream&); \
    template void transform3D<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float33_t, const Symmetry&, float3_t, InterpMode, bool, bool, Stream&)

    NOA_INSTANTIATE_APPLY_(float);
    NOA_INSTANTIATE_APPLY_(double);
    NOA_INSTANTIATE_APPLY_(cfloat_t);
    NOA_INSTANTIATE_APPLY_(cdouble_t);
}
