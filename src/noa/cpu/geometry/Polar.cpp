#include "noa/common/geometry/Polar.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Polar.h"
#include "noa/cpu/geometry/Prefilter.h"
#include "noa/cpu/geometry/Interpolator.h"

namespace {
    using namespace ::noa;

    template<typename T, InterpMode INTERP>
    void cartesian2polar_(const T* cartesian, size3_t cartesian_stride, size3_t cartesian_shape,
                          T* polar, size3_t polar_stride, size3_t polar_shape,
                          float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                          bool log, size_t threads) {
        const size_t offset = cartesian_shape[0] == 1 ? 0 : cartesian_stride[0];
        const size2_t stride{cartesian_stride.get() + 1};
        const size2_t shape{cartesian_shape.get() + 1};
        const cpu::geometry::Interpolator2D<T> interp{cartesian, stride, shape, 0};

        const float2_t size{polar_shape[1] - 1, polar_shape[2] - 1}; // endpoint = true, so N-1
        const float start_angle = angle_range[0];
        const float start_radius = radius_range[0];
        const float step_angle = (angle_range[1] - angle_range[0]) / size[0];
        const float step_radius = log ?
                                     math::log(radius_range[1] - radius_range[0]) / size[1] :
                                     (radius_range[1] - radius_range[0]) / size[1];

        #pragma omp parallel for collapse(3) default(none) num_threads(threads)         \
        shared(polar, polar_stride, polar_shape, cartesian_center, log, interp, offset, \
               step_angle, step_radius, start_radius, start_angle)

        for (size_t batch = 0; batch < polar_shape[0]; ++batch) {
            for (size_t phi = 0; phi < polar_shape[1]; ++phi) {
                for (size_t rho = 0; rho < polar_shape[2]; ++rho) {

                    const float2_t polar_coordinate{phi, rho};
                    const float angle_rad = polar_coordinate[0] * step_angle + start_angle;
                    const float magnitude = log ?
                                            math::exp(polar_coordinate[1] * step_radius) - 1 + start_radius :
                                            (polar_coordinate[1] * step_radius) + start_radius;

                    float2_t cartesian_coordinates{math::sin(angle_rad), math::cos(angle_rad)};
                    cartesian_coordinates *= magnitude;
                    cartesian_coordinates += cartesian_center;

                    polar[indexing::at(batch, phi, rho, polar_stride)] =
                            interp.template get<INTERP, BORDER_ZERO>(cartesian_coordinates, offset * batch);
                }
            }
        }
    }

    template<typename T, InterpMode INTERP>
    void polar2cartesian_(const T* polar, size3_t polar_stride, size3_t polar_shape,
                          T* cartesian, size3_t cartesian_stride, size3_t cartesian_shape,
                          float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                          bool log, size_t threads) {
        const size_t offset = polar_shape[0] == 1 ? 0 : polar_stride[0];
        const size2_t stride{polar_stride.get() + 1};
        const size2_t shape{polar_shape.get() + 1};
        const cpu::geometry::Interpolator2D<T> interp{polar, stride, shape, 0};

        const float2_t size{polar_shape[1] - 1, polar_shape[2] - 1}; // endpoint = true, so N-1
        const float start_angle = angle_range[0];
        const float start_radius = radius_range[0];
        const float step_angle = (angle_range[1] - angle_range[0]) / size[0];
        const float step_radius = log ?
                                     math::log(radius_range[1] - radius_range[0]) / size[1] :
                                     (radius_range[1] - radius_range[0]) / size[1];

        #pragma omp parallel for collapse(3) default(none) num_threads(threads)             \
        shared(cartesian, cartesian_stride, cartesian_shape, cartesian_center, log, interp, \
               offset, step_angle, step_radius, start_radius, start_angle)

        for (size_t batch = 0; batch < cartesian_shape[0]; ++batch) {
            for (size_t y = 0; y < cartesian_shape[1]; ++y) {
                for (size_t x = 0; x < cartesian_shape[2]; ++x) {

                    float2_t cartesian_coordinate{y, x};
                    cartesian_coordinate -= cartesian_center;

                    const float angle_rad = geometry::cartesian2angle(cartesian_coordinate);
                    const float magnitude = geometry::cartesian2magnitude(cartesian_coordinate);

                    const float phi = (angle_rad - start_angle) / step_angle;
                    const float rho = log ?
                                      math::log(magnitude + 1 - start_radius) / step_radius :
                                      (magnitude - start_radius) / step_radius;
                    const float2_t polar_coordinate{phi, rho};

                    cartesian[indexing::at(batch, y, x, cartesian_stride)] =
                            interp.template get<INTERP, BORDER_ZERO>(polar_coordinate, offset * batch);
                }
            }
        }
    }
}

namespace noa::cpu::geometry {
    template<bool PREFILTER, typename T, typename>
    void cartesian2polar(const shared_t<T[]>& cartesian, size4_t cartesian_stride, size4_t cartesian_shape,
                         const shared_t<T[]>& polar, size4_t polar_stride, size4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream) {
        NOA_ASSERT(cartesian.get() != polar.get());
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        const size3_t src_shape{cartesian_shape[0], cartesian_shape[2], cartesian_shape[3]};
        const size3_t src_stride{cartesian_stride[0], cartesian_stride[2], cartesian_stride[3]};
        const size3_t dst_shape{polar_shape[0], polar_shape[2], polar_shape[3]};
        const size3_t dst_stride{polar_stride[0], polar_stride[2], polar_stride[3]};

        const size_t threads = stream.threads();
        switch (interp) {
            case INTERP_NEAREST:
                return stream.enqueue([=]() {
                    cartesian2polar_<T, INTERP_NEAREST>(
                            cartesian.get(), src_stride, src_shape, polar.get(), dst_stride, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR:
                return stream.enqueue([=]() {
                    cartesian2polar_<T, INTERP_LINEAR>(
                            cartesian.get(), src_stride, src_shape, polar.get(), dst_stride, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_COSINE_FAST:
            case INTERP_COSINE:
                return stream.enqueue([=]() {
                    cartesian2polar_<T, INTERP_COSINE>(
                            cartesian.get(), src_stride, src_shape, polar.get(), dst_stride, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_CUBIC:
                return stream.enqueue([=]() {
                    cartesian2polar_<T, INTERP_CUBIC>(
                            cartesian.get(), src_stride, src_shape, polar.get(), dst_stride, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_CUBIC_BSPLINE_FAST:
            case INTERP_CUBIC_BSPLINE:
                return stream.enqueue([=]() mutable {
                    memory::PtrHost<T> buffer;
                    const T* src = cartesian.get();
                    size3_t src_stride_ = src_stride;
                    if constexpr (PREFILTER) {
                        if (cartesian_stride[0] == 0)
                            cartesian_shape[0] = 1;
                        const size4_t stride = cartesian_shape.stride();
                        buffer = memory::PtrHost<T>{cartesian_shape.elements()};
                        bspline::prefilter(cartesian, cartesian_stride, buffer.share(), stride, cartesian_shape, stream);
                        src = buffer.get();
                        src_stride_ = {stride[0], stride[2], stride[3]};
                    }
                    cartesian2polar_<T, INTERP_CUBIC_BSPLINE>(
                            src, src_stride_, src_shape, polar.get(), dst_stride, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
        }
    }

    template<bool PREFILTER, typename T, typename>
    void polar2cartesian(const shared_t<T[]>& polar, size4_t polar_stride, size4_t polar_shape,
                         const shared_t<T[]>& cartesian, size4_t cartesian_stride, size4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream) {
        NOA_ASSERT(cartesian.get() != polar.get());
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        const size3_t src_shape{polar_shape[0], polar_shape[2], polar_shape[3]};
        const size3_t src_stride{polar_stride[0], polar_stride[2], polar_stride[3]};
        const size3_t dst_shape{cartesian_shape[0], cartesian_shape[2], cartesian_shape[3]};
        const size3_t dst_stride{cartesian_stride[0], cartesian_stride[2], cartesian_stride[3]};

        const size_t threads = stream.threads();
        switch (interp) {
            case INTERP_NEAREST:
                return stream.enqueue([=]() {
                    polar2cartesian_<T, INTERP_NEAREST>(
                            polar.get(), src_stride, src_shape, cartesian.get(), dst_stride, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR:
                return stream.enqueue([=]() {
                    polar2cartesian_<T, INTERP_LINEAR>(
                            polar.get(), src_stride, src_shape, cartesian.get(), dst_stride, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_COSINE_FAST:
            case INTERP_COSINE:
                return stream.enqueue([=]() {
                    polar2cartesian_<T, INTERP_COSINE>(
                            polar.get(), src_stride, src_shape, cartesian.get(), dst_stride, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_CUBIC:
                return stream.enqueue([=]() {
                    polar2cartesian_<T, INTERP_CUBIC>(
                            polar.get(), src_stride, src_shape, cartesian.get(), dst_stride, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
            case INTERP_CUBIC_BSPLINE_FAST:
            case INTERP_CUBIC_BSPLINE:
                return stream.enqueue([=]() mutable {
                    memory::PtrHost<T> buffer;
                    const T* src = polar.get();
                    size3_t src_stride_ = src_stride;
                    if constexpr (PREFILTER) {
                        if (polar_stride[0] == 0)
                            polar_shape[0] = 1;
                        const size4_t stride = polar_shape.stride();
                        buffer = memory::PtrHost<T>{polar_shape.elements()};
                        bspline::prefilter(polar, polar_stride, buffer.share(), stride, polar_shape, stream);
                        src = buffer.get();
                        src_stride_ = {stride[0], stride[2], stride[3]};
                    }
                    polar2cartesian_<T, INTERP_CUBIC_BSPLINE>(
                            src, src_stride_, src_shape, cartesian.get(), dst_stride, dst_shape,
                            cartesian_center, radius_range, angle_range, log, threads);
                });
        }
    }

    #define INSTANTIATE_POLAR(T) \
    template void cartesian2polar<true,T,void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&); \
    template void cartesian2polar<false,T,void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&);\
    template void polar2cartesian<true,T,void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&); \
    template void polar2cartesian<false,T,void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(double);
    INSTANTIATE_POLAR(cfloat_t);
    INSTANTIATE_POLAR(cdouble_t);
}
