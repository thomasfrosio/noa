#include "noa/common/geometry/Polar.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Polar.h"
#include "noa/cpu/geometry/Prefilter.h"
#include "noa/cpu/geometry/Interpolator.h"

namespace {
    using namespace ::noa;

    template<typename T, InterpMode INTERP>
    void cartesian2polar_(const T* cartesian, size3_t cartesian_strides, size3_t cartesian_shape,
                          T* polar, size3_t polar_strides, size3_t polar_shape,
                          float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                          bool log, size_t threads) {
        const size_t offset = cartesian_shape[0] == 1 ? 0 : cartesian_strides[0];
        const size2_t strides(cartesian_strides.get(1));
        const size2_t shape(cartesian_shape.get(1));
        const cpu::geometry::Interpolator2D<T> interp(cartesian, strides, shape, 0);

        const float2_t size{polar_shape[1] - 1, polar_shape[2] - 1}; // endpoint = true, so N-1
        const float start_angle = angle_range[0];
        const float start_radius = radius_range[0];
        const float step_angle = (angle_range[1] - angle_range[0]) / size[0];
        const float step_radius = log ?
                                     math::log(radius_range[1] - radius_range[0]) / size[1] :
                                     (radius_range[1] - radius_range[0]) / size[1];

        #pragma omp parallel for collapse(3) default(none) num_threads(threads)          \
        shared(polar, polar_strides, polar_shape, cartesian_center, log, interp, offset, \
               step_angle, step_radius, start_radius, start_angle)

        for (size_t batch = 0; batch < polar_shape[0]; ++batch) {
            for (size_t y = 0; y < polar_shape[1]; ++y) {
                for (size_t x = 0; x < polar_shape[2]; ++x) {

                    const float2_t polar_coordinate{y, x};
                    const float phi = polar_coordinate[0] * step_angle + start_angle;
                    const float rho = log ?
                                      math::exp(polar_coordinate[1] * step_radius) - 1 + start_radius :
                                      (polar_coordinate[1] * step_radius) + start_radius;

                    float2_t cartesian_coordinates{math::sin(phi), math::cos(phi)};
                    cartesian_coordinates *= rho;
                    cartesian_coordinates += cartesian_center;

                    polar[indexing::at(batch, y, x, polar_strides)] =
                            interp.template get<INTERP, BORDER_ZERO>(cartesian_coordinates, offset * batch);
                }
            }
        }
    }

    template<typename T, InterpMode INTERP>
    void polar2cartesian_(const T* polar, size3_t polar_strides, size3_t polar_shape,
                          T* cartesian, size3_t cartesian_strides, size3_t cartesian_shape,
                          float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                          bool log, size_t threads) {
        const size_t offset = polar_shape[0] == 1 ? 0 : polar_strides[0];
        const size2_t strides(polar_strides.get(1));
        const size2_t shape(polar_shape.get(1));
        const cpu::geometry::Interpolator2D<T> interp(polar, strides, shape, 0);

        const float2_t size{polar_shape[1] - 1, polar_shape[2] - 1}; // endpoint = true, so N-1
        const float start_angle = angle_range[0];
        const float start_radius = radius_range[0];
        const float step_angle = (angle_range[1] - angle_range[0]) / size[0];
        const float step_radius = log ?
                                     math::log(radius_range[1] - radius_range[0]) / size[1] :
                                     (radius_range[1] - radius_range[0]) / size[1];

        #pragma omp parallel for collapse(3) default(none) num_threads(threads)              \
        shared(cartesian, cartesian_strides, cartesian_shape, cartesian_center, log, interp, \
               offset, step_angle, step_radius, start_radius, start_angle)

        for (size_t batch = 0; batch < cartesian_shape[0]; ++batch) {
            for (size_t y = 0; y < cartesian_shape[1]; ++y) {
                for (size_t x = 0; x < cartesian_shape[2]; ++x) {

                    float2_t cartesian_coordinate{y, x};
                    cartesian_coordinate -= cartesian_center;

                    const float phi = geometry::cartesian2phi(cartesian_coordinate);
                    const float rho = geometry::cartesian2rho(cartesian_coordinate);

                    const float py = (phi - start_angle) / step_angle;
                    const float px = log ?
                                      math::log(rho + 1 - start_radius) / step_radius :
                                      (rho - start_radius) / step_radius;
                    const float2_t polar_coordinate{py, px};

                    cartesian[indexing::at(batch, y, x, cartesian_strides)] =
                            interp.template get<INTERP, BORDER_ZERO>(polar_coordinate, offset * batch);
                }
            }
        }
    }
}

namespace noa::cpu::geometry {
    template<typename T, typename>
    void cartesian2polar(const shared_t<T[]>& cartesian, size4_t cartesian_strides, size4_t cartesian_shape,
                         const shared_t<T[]>& polar, size4_t polar_strides, size4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter, Stream& stream) {
        NOA_ASSERT(cartesian.get() != polar.get());
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        const size3_t src_shape{cartesian_shape[0], cartesian_shape[2], cartesian_shape[3]};
        const size3_t src_strides{cartesian_strides[0], cartesian_strides[2], cartesian_strides[3]};
        const size3_t dst_shape{polar_shape[0], polar_shape[2], polar_shape[3]};
        const size3_t dst_strides{polar_strides[0], polar_strides[2], polar_strides[3]};

        const size_t threads = stream.threads();
        switch (interp) {
            case INTERP_NEAREST:
                return stream.enqueue([=]() {
                    cartesian2polar_<T, INTERP_NEAREST>(
                            cartesian.get(), src_strides, src_shape, polar.get(), dst_strides, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR:
                return stream.enqueue([=]() {
                    cartesian2polar_<T, INTERP_LINEAR>(
                            cartesian.get(), src_strides, src_shape, polar.get(), dst_strides, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_COSINE_FAST:
            case INTERP_COSINE:
                return stream.enqueue([=]() {
                    cartesian2polar_<T, INTERP_COSINE>(
                            cartesian.get(), src_strides, src_shape, polar.get(), dst_strides, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_CUBIC:
                return stream.enqueue([=]() {
                    cartesian2polar_<T, INTERP_CUBIC>(
                            cartesian.get(), src_strides, src_shape, polar.get(), dst_strides, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_CUBIC_BSPLINE_FAST:
            case INTERP_CUBIC_BSPLINE:
                return stream.enqueue([=]() mutable {
                    memory::PtrHost<T> buffer;
                    const T* src = cartesian.get();
                    size3_t src_strides_ = src_strides;
                    if (prefilter) {
                        if (cartesian_strides[0] == 0)
                            cartesian_shape[0] = 1;
                        const size4_t strides = cartesian_shape.strides();
                        buffer = memory::PtrHost<T>(cartesian_shape.elements());
                        bspline::prefilter(cartesian, cartesian_strides,
                                           buffer.share(), strides,
                                           cartesian_shape, stream);
                        src = buffer.get();
                        src_strides_ = {strides[0], strides[2], strides[3]};
                    }
                    cartesian2polar_<T, INTERP_CUBIC_BSPLINE>(
                            src, src_strides_, src_shape, polar.get(), dst_strides, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
        }
    }

    template<typename T, typename>
    void polar2cartesian(const shared_t<T[]>& polar, size4_t polar_strides, size4_t polar_shape,
                         const shared_t<T[]>& cartesian, size4_t cartesian_strides, size4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter, Stream& stream) {
        NOA_ASSERT(cartesian.get() != polar.get());
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        const size3_t src_shape{polar_shape[0], polar_shape[2], polar_shape[3]};
        const size3_t src_strides{polar_strides[0], polar_strides[2], polar_strides[3]};
        const size3_t dst_shape{cartesian_shape[0], cartesian_shape[2], cartesian_shape[3]};
        const size3_t dst_strides{cartesian_strides[0], cartesian_strides[2], cartesian_strides[3]};

        const size_t threads = stream.threads();
        switch (interp) {
            case INTERP_NEAREST:
                return stream.enqueue([=]() {
                    polar2cartesian_<T, INTERP_NEAREST>(
                            polar.get(), src_strides, src_shape, cartesian.get(), dst_strides, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR:
                return stream.enqueue([=]() {
                    polar2cartesian_<T, INTERP_LINEAR>(
                            polar.get(), src_strides, src_shape, cartesian.get(), dst_strides, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_COSINE_FAST:
            case INTERP_COSINE:
                return stream.enqueue([=]() {
                    polar2cartesian_<T, INTERP_COSINE>(
                            polar.get(), src_strides, src_shape, cartesian.get(), dst_strides, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_CUBIC:
                return stream.enqueue([=]() {
                    polar2cartesian_<T, INTERP_CUBIC>(
                            polar.get(), src_strides, src_shape, cartesian.get(), dst_strides, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_CUBIC_BSPLINE_FAST:
            case INTERP_CUBIC_BSPLINE:
                return stream.enqueue([=]() mutable {
                    memory::PtrHost<T> buffer;
                    const T* src = polar.get();
                    size3_t src_strides_ = src_strides;
                    if (prefilter) {
                        if (polar_strides[0] == 0)
                            polar_shape[0] = 1;
                        const size4_t strides = polar_shape.strides();
                        buffer = memory::PtrHost<T>(polar_shape.elements());
                        bspline::prefilter(polar, polar_strides, buffer.share(), strides, polar_shape, stream);
                        src = buffer.get();
                        src_strides_ = {strides[0], strides[2], strides[3]};
                    }
                    polar2cartesian_<T, INTERP_CUBIC_BSPLINE>(
                            src, src_strides_, src_shape, cartesian.get(), dst_strides, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
        }
    }

    #define INSTANTIATE_POLAR(T) \
    template void cartesian2polar<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, bool, Stream&);  \
    template void polar2cartesian<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, bool, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(double);
    INSTANTIATE_POLAR(cfloat_t);
    INSTANTIATE_POLAR(cdouble_t);
}
