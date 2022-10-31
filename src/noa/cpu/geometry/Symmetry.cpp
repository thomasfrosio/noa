#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/common/Exception.h"
#include "noa/common/geometry/Interpolator.h"

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Prefilter.h"
#include "noa/cpu/geometry/Symmetry.h"

namespace {
    using namespace ::noa;

    template<InterpMode INTERP, typename T>
    void symmetrize_(AccessorRestrict<const T, 4, dim_t> input,
                     AccessorRestrict<T, 4, dim_t> output, dim4_t shape,
                     const geometry::Symmetry& symmetry, float2_t center, bool normalize, dim_t threads) {
        const dim_t count = symmetry.count();
        const float33_t* matrices = symmetry.get();
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(count + 1) : 1;

        // Ignore depth dimension.
        const long3_t istrides(input.stride(0), input.stride(2), input.stride(3));
        const dim3_t ostrides(output.stride(0), output.stride(2), output.stride(3));
        const AccessorRestrict<const T, 3, int64_t> src(input.get(), istrides);
        const AccessorRestrict<T, 3, dim_t> dst(output.get(), ostrides);
        auto interpolator = geometry::interpolator2D<BORDER_ZERO, INTERP>(src, long2_t{shape[2], shape[3]}, T{0});

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(src, dst, shape, center, matrices, count, scaling, interpolator)

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t y = 0; y < shape[2]; ++y) {
                for (dim_t x = 0; x < shape[3]; ++x) {
                    T value = src(i, y, x);

                    float2_t pos{y, x};
                    pos -= center;
                    for (dim_t s = 0; s < count; ++s) {
                        const float33_t& m = matrices[s];
                        float2_t coordinates = float22_t{m[1][1], m[1][2],
                                                         m[2][1], m[2][2]} * pos;
                        coordinates += center;
                        value += interpolator(coordinates, i);
                    }
                    dst(i, y, x) = value * scaling;
                }
            }
        }
    }

    template<InterpMode INTERP, typename T>
    void symmetrize_(AccessorRestrict<const T, 4, dim_t> input,
                     AccessorRestrict<T, 4, dim_t> output, dim4_t shape,
                     const geometry::Symmetry& symmetry, float3_t center, bool normalize, dim_t threads) {
        const dim_t count = symmetry.count();
        const float33_t* matrices = symmetry.get();
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(count + 1) : 1;

        const AccessorRestrict<const T, 4, int64_t> src(input.get(), input.strides());
        auto interpolator = geometry::interpolator3D<BORDER_ZERO, INTERP>(src, long3_t(shape.get(1)), T{0});

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(input, output, shape, center, matrices, count, scaling, interpolator)

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t z = 0; z < shape[1]; ++z) {
                for (dim_t y = 0; y < shape[2]; ++y) {
                    for (dim_t x = 0; x < shape[3]; ++x) {
                        T value = input(i, z, y, x);

                        float3_t pos{z, y, x};
                        pos -= center;
                        for (dim_t s = 0; s < count; ++s) {
                            float3_t coordinates = matrices[s] * pos;
                            coordinates += center;
                            value += interpolator(coordinates, i);
                        }
                        output(i, z, y, x) = value * scaling;
                    }
                }
            }
        }
    }

    template<typename T, typename U>
    void launch_(const AccessorRestrict<const T, 4, dim_t>& input,
                 const AccessorRestrict<T, 4, dim_t>& output, dim4_t shape,
                 const geometry::Symmetry& symmetry, U center, InterpMode interp_mode,
                 bool normalize, dim_t threads) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return symmetrize_<INTERP_NEAREST>(input, output, shape, symmetry, center, normalize, threads);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return symmetrize_<INTERP_LINEAR>(input, output, shape, symmetry, center, normalize, threads);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return symmetrize_<INTERP_COSINE>(input, output, shape, symmetry, center, normalize, threads);
            case INTERP_CUBIC:
                return symmetrize_<INTERP_CUBIC>(input, output, shape, symmetry, center, normalize, threads);
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST:
                return symmetrize_<INTERP_CUBIC_BSPLINE>(input, output, shape, symmetry, center, normalize, threads);
            default:
                NOA_THROW_FUNC("symmetrize(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }

    template<typename T, typename V>
    void symmetrizeND_(const shared_t<T[]>& input, dim4_t input_strides,
                       const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                       const geometry::Symmetry& symmetry, V center, InterpMode interp_mode, bool prefilter,
                       bool normalize, cpu::Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(shape > 0));
        NOA_ASSERT((std::is_same_v<V, float3_t> && dim3_t(shape.get(1)).ndim() <= 3) ||
                   (std::is_same_v<V, float2_t> && dim3_t(shape.get(1)).ndim() <= 2));

        if (!symmetry.count())
            return cpu::memory::copy(input, input_strides, output, output_strides, shape, stream);

        const dim_t threads = stream.threads();
        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                dim4_t new_shape = shape;
                dim4_t new_strides = new_shape.strides();

                if (input_strides[0] == 0) {
                    new_shape[0] = 1; // only one batch in input
                    new_strides[0] = 0;
                }
                cpu::memory::PtrHost<T> buffer{new_shape.elements()};
                cpu::geometry::bspline::prefilter(input, input_strides, buffer.share(), new_strides, new_shape, stream);

                launch_<T>({buffer.get(), new_strides}, {output.get(), output_strides}, shape,
                           symmetry, center, interp_mode, normalize, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_<T>({input.get(), input_strides}, {output.get(), output_strides}, shape,
                           symmetry, center, interp_mode, normalize, threads);
            });
        }
    }
}

namespace noa::cpu::geometry {
    template<typename T, typename>
    void symmetrize2D(const shared_t<T[]>& input, dim4_t input_strides,
                      const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                      const Symmetry& symmetry, float2_t center, InterpMode interp_mode,
                      bool prefilter, bool normalize, Stream& stream) {
        symmetrizeND_(input, input_strides, output, output_strides, shape,
                      symmetry, center, interp_mode, prefilter, normalize, stream);
    }

    template<typename T, typename>
    void symmetrize3D(const shared_t<T[]>& input, dim4_t input_strides,
                      const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                      const Symmetry& symmetry, float3_t center, InterpMode interp_mode,
                      bool prefilter, bool normalize, Stream& stream) {
        symmetrizeND_(input, input_strides, output, output_strides, shape,
                      symmetry, center, interp_mode, prefilter, normalize, stream);
    }

    #define NOA_INSTANTIATE_SYM_(T)                                                                                                                                         \
    template void symmetrize2D<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const Symmetry&, float2_t, InterpMode, bool, bool, Stream&);    \
    template void symmetrize3D<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const Symmetry&, float3_t, InterpMode, bool, bool, Stream&)

    NOA_INSTANTIATE_SYM_(float);
    NOA_INSTANTIATE_SYM_(double);
    NOA_INSTANTIATE_SYM_(cfloat_t);
    NOA_INSTANTIATE_SYM_(cdouble_t);
}
