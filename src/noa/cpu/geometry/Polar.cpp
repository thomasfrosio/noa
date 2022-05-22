#include "noa/common/geometry/Polar.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Polar.h"
#include "noa/cpu/geometry/Prefilter.h"
#include "noa/cpu/geometry/Interpolator.h"

namespace {
    using namespace ::noa;

    template<typename T, InterpMode INTERP>
    void cartesian2polar_(const T* input, size3_t input_stride, size3_t input_shape,
                          T* output, size3_t output_stride, size3_t output_shape,
                          float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                          bool log, size_t threads) {
        const size_t offset = input_shape[0] == 1 ? 0 : input_stride[0];
        const size2_t stride{input_stride.get() + 1};
        const size2_t shape{input_shape.get() + 1};
        const cpu::geometry::Interpolator2D<T> interp{input, stride, shape, 0};

        using real_t = traits::value_type_t<T>;
        const Float2<real_t> center{cartesian_center};
        const Float2<real_t> radius{radius_range};
        const Float2<real_t> angle{angle_range};

        const auto size_phi = static_cast<real_t>(output_shape[1] - 1);
        const auto step_angle = (angle[1] - angle[0]) / size_phi;

        const auto size_rho = static_cast<real_t>(output_shape[2] - 1);
        const auto step_magnitude = log ?
                                    math::log(radius[1] - radius[0]) / size_rho :
                                    (radius[1] - radius[0]) / size_rho;

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(input, output, output_stride, output_shape, log, interp,         \
               offset, step_angle, step_magnitude, radius, angle, center)

        for (size_t batch = 0; batch < output_shape[0]; ++batch) {
            for (size_t phi = 0; phi < output_shape[1]; ++phi) {
                for (size_t rho = 0; rho < output_shape[2]; ++rho) {

                    // (phi, rho) -> (angle, magnitude)
                    const real_t angle_rad = static_cast<real_t>(phi) * step_angle + angle[0];
                    const real_t magnitude = log ?
                                             math::exp(static_cast<real_t>(rho) * step_magnitude) - 1 + radius[0] :
                                             (static_cast<real_t>(rho) * step_magnitude) + radius[0];

                    // (angle, magnitude) -> (y, x)
                    const float2_t coordinates{center[0] + magnitude * math::sin(angle_rad),
                                               center[1] + magnitude * math::cos(angle_rad)};

                    output[indexing::at(batch, phi, rho, output_stride)] =
                            interp.template get<INTERP, BORDER_ZERO>(coordinates, offset * batch);
                }
            }
        }
    }

    template<typename T, InterpMode INTERP>
    void polar2cartesian_(const T* input, size3_t input_stride, size3_t input_shape,
                          T* output, size3_t output_stride, size3_t output_shape,
                          float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                          bool log, size_t threads) {
        const size_t offset = input_shape[0] == 1 ? 0 : input_stride[0];
        const size2_t stride{input_stride.get() + 1};
        const size2_t shape{input_shape.get() + 1};
        const cpu::geometry::Interpolator2D<T> interp{input, stride, shape, 0};

        using real_t = traits::value_type_t<T>;
        using vec2_t = Float2<real_t>;
        const vec2_t center{cartesian_center};
        const vec2_t radius{radius_range};
        const vec2_t angle{angle_range};

        const auto size_phi = static_cast<real_t>(input_shape[1] - 1);
        const auto step_angle = (angle[1] - angle[0]) / size_phi;

        const auto size_rho = static_cast<real_t>(input_shape[2] - 1);
        const auto step_magnitude = log ?
                                    math::log(radius[1] - radius[0]) / size_rho :
                                    (radius[1] - radius[0]) / size_rho;

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(input, output, output_stride, output_shape, log, interp,         \
               offset, step_angle, step_magnitude, radius, angle, center)

        for (size_t batch = 0; batch < output_shape[0]; ++batch) {
            for (size_t y = 0; y < output_shape[1]; ++y) {
                for (size_t x = 0; x < output_shape[2]; ++x) {
                    vec2_t pos{y, x};
                    pos -= center;

                    const real_t angle_rad = geometry::cartesian2angle(pos);
                    const real_t phi = (angle_rad - angle[0]) / step_angle;

                    const real_t magnitude = geometry::cartesian2magnitude(pos);
                    const real_t rho = log ?
                                       math::log(magnitude + 1 - radius[0]) / step_magnitude :
                                       (magnitude - radius[0]) / step_magnitude;

                    output[indexing::at(batch, y, x, output_stride)] =
                            interp.template get<INTERP, BORDER_ZERO>({phi, rho}, offset * batch);
                }
            }
        }
    }
}

namespace noa::cpu::geometry {
    template<bool PREFILTER, typename T>
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

    template<bool PREFILTER, typename T>
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
    template void cartesian2polar<true, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&); \
    template void cartesian2polar<false, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&);\
    template void polar2cartesian<true, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&); \
    template void polar2cartesian<false, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(double);
    INSTANTIATE_POLAR(cfloat_t);
    INSTANTIATE_POLAR(cdouble_t);
}
