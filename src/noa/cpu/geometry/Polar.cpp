#include "noa/common/geometry/Polar.h"
#include "noa/common/geometry/Interpolator.h"
#include "noa/common/geometry/details/PolarTransform.h"

#include "noa/cpu/geometry/Polar.h"
#include "noa/cpu/geometry/Prefilter.h"
#include "noa/cpu/utils/Iwise.h"

namespace noa::cpu::geometry {
    template<typename Value, typename>
    void cartesian2polar(const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter, Stream& stream) {
        NOA_ASSERT(cartesian && polar && cartesian.get() != polar.get() &&
                   all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        // Broadcast the input to every output batch.
        if (cartesian_shape[0] == 1)
            cartesian_strides[0] = 0;
        else if (cartesian_strides[0] == 0)
            cartesian_shape[0] = 1;

        const auto cartesian_shape_2d = long2_t{cartesian_shape[2], cartesian_shape[3]};
        const auto cartesian_strides_2d = long3_t{cartesian_strides[0], cartesian_strides[2], cartesian_strides[3]};
        const auto polar_shape_2d = long3_t{polar_shape[0], polar_shape[2], polar_shape[3]};
        const auto polar_strides_2d = long3_t{polar_strides[0], polar_strides[2], polar_strides[3]};

        const dim_t threads = stream.threads();
        stream.enqueue([=]() {
            if (prefilter && (interp == INTERP_CUBIC_BSPLINE || interp == INTERP_CUBIC_BSPLINE_FAST)) {
                bspline::prefilter(cartesian.get(), cartesian_strides,
                                   cartesian.get(), cartesian_strides,
                                   cartesian_shape, threads);
            }

            const auto input_accessor = AccessorRestrict<const Value, 3, int64_t>(cartesian.get(), cartesian_strides_2d);
            const auto output_accessor = AccessorRestrict<Value, 3, int64_t>(polar.get(), polar_strides_2d);

            switch (interp) {
                case INTERP_NEAREST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(
                            input_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::details::cartesian2polar<int64_t>(
                            interpolator, output_accessor, polar_shape,
                            cartesian_center, radius_range, angle_range, log);
                    return utils::iwise3D(polar_shape_2d, kernel, threads);
                }
                case INTERP_LINEAR_FAST:
                case INTERP_LINEAR: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                            input_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::details::cartesian2polar<int64_t>(
                            interpolator, output_accessor, polar_shape,
                            cartesian_center, radius_range, angle_range, log);
                    return utils::iwise3D(polar_shape_2d, kernel, threads);
                }
                case INTERP_COSINE_FAST:
                case INTERP_COSINE: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(
                            input_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::details::cartesian2polar<int64_t>(
                            interpolator, output_accessor, polar_shape,
                            cartesian_center, radius_range, angle_range, log);
                    return utils::iwise3D(polar_shape_2d, kernel, threads);
                }
                case INTERP_CUBIC: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC>(
                            input_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::details::cartesian2polar<int64_t>(
                            interpolator, output_accessor, polar_shape,
                            cartesian_center, radius_range, angle_range, log);
                    return utils::iwise3D(polar_shape_2d, kernel, threads);
                }
                case INTERP_CUBIC_BSPLINE_FAST:
                case INTERP_CUBIC_BSPLINE: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(
                            input_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::details::cartesian2polar<int64_t>(
                            interpolator, output_accessor, polar_shape,
                            cartesian_center, radius_range, angle_range, log);
                    return utils::iwise3D(polar_shape_2d, kernel, threads);
                }
            }
        });
    }

    template<typename Value, typename>
    void polar2cartesian(const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter, Stream& stream) {
        NOA_ASSERT(cartesian && polar && cartesian.get() != polar.get() &&
                   all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        // Broadcast the input to every output batch.
        if (polar_shape[0] == 1)
            polar_strides[0] = 0;
        else if (polar_strides[0] == 0)
            polar_shape[0] = 1;

        const auto polar_shape_2d = long2_t{cartesian_shape[2], cartesian_shape[3]};
        const auto polar_strides_2d = long3_t{polar_strides[0], polar_strides[2], polar_strides[3]};
        const auto cartesian_shape_2d = long3_t{cartesian_shape[0], cartesian_shape[2], cartesian_shape[3]};
        const auto cartesian_strides_2d = long3_t{cartesian_strides[0], cartesian_strides[2], cartesian_strides[3]};

        const dim_t threads = stream.threads();
        stream.enqueue([=]() {

            if (prefilter && (interp == INTERP_CUBIC_BSPLINE || interp == INTERP_CUBIC_BSPLINE_FAST)) {
                bspline::prefilter(polar.get(), polar_strides,
                                   polar.get(), polar_strides,
                                   polar_shape, threads);
            }

            const auto input_accessor = AccessorRestrict<const Value, 3, int64_t>(polar.get(), polar_strides_2d);
            const auto output_accessor = AccessorRestrict<Value, 3, int64_t>(cartesian.get(), cartesian_strides_2d);

            switch (interp) {
                case INTERP_NEAREST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(
                            input_accessor, polar_shape_2d);
                    const auto kernel = noa::geometry::details::polar2cartesian<int64_t>(
                            interpolator, polar_shape, output_accessor,
                            cartesian_center, radius_range, angle_range, log);
                    return utils::iwise3D(cartesian_shape_2d, kernel, threads);
                }
                case INTERP_LINEAR_FAST:
                case INTERP_LINEAR: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                            input_accessor, polar_shape_2d);
                    const auto kernel = noa::geometry::details::polar2cartesian<int64_t>(
                            interpolator, polar_shape, output_accessor,
                            cartesian_center, radius_range, angle_range, log);
                    return utils::iwise3D(cartesian_shape_2d, kernel, threads);
                }
                case INTERP_COSINE_FAST:
                case INTERP_COSINE: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(
                            input_accessor, polar_shape_2d);
                    const auto kernel = noa::geometry::details::polar2cartesian<int64_t>(
                            interpolator, polar_shape, output_accessor,
                            cartesian_center, radius_range, angle_range, log);
                    return utils::iwise3D(cartesian_shape_2d, kernel, threads);
                }
                case INTERP_CUBIC: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC>(
                            input_accessor, polar_shape_2d);
                    const auto kernel = noa::geometry::details::polar2cartesian<int64_t>(
                            interpolator, polar_shape, output_accessor,
                            cartesian_center, radius_range, angle_range, log);
                    return utils::iwise3D(cartesian_shape_2d, kernel, threads);
                }
                case INTERP_CUBIC_BSPLINE_FAST:
                case INTERP_CUBIC_BSPLINE: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(
                            input_accessor, polar_shape_2d);
                    const auto kernel = noa::geometry::details::polar2cartesian<int64_t>(
                            interpolator, polar_shape, output_accessor,
                            cartesian_center, radius_range, angle_range, log);
                    return utils::iwise3D(cartesian_shape_2d, kernel, threads);
                }
            }
        });
    }

    #define INSTANTIATE_POLAR(T)                \
    template void cartesian2polar<T, void>(     \
        const shared_t<T[]>&, dim4_t, dim4_t,   \
        const shared_t<T[]>&, dim4_t, dim4_t,   \
        float2_t, float2_t, float2_t, bool,     \
        InterpMode, bool, Stream&);             \
    template void polar2cartesian<T, void>(     \
        const shared_t<T[]>&, dim4_t, dim4_t,   \
        const shared_t<T[]>&, dim4_t, dim4_t,   \
        float2_t, float2_t, float2_t, bool,     \
        InterpMode, bool, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(double);
    INSTANTIATE_POLAR(cfloat_t);
    INSTANTIATE_POLAR(cdouble_t);
}
