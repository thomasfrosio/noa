#include "noa/cpu/geometry/fft/Polar.h"
#include "noa/cpu/geometry/Interpolator.h"

namespace {
    using namespace ::noa;

    template<typename T, InterpMode INTERP>
    void cartesian2polar_(const T* input, size3_t input_stride, size3_t input_shape,
                          T* output, size3_t output_stride, size3_t output_shape,
                          float2_t frequency_range, float2_t angle_range,
                          bool log, size_t threads) {
        const size_t offset = input_shape[0] == 1 ? 0 : input_stride[0];
        const size2_t stride{input_stride.get() + 1};
        const size2_t shape{input_shape.get() + 1};
        const cpu::geometry::Interpolator2D<T> interp{input, stride, shape.fft(), 0};

        // Since the aspect ratio of the input transform might not be 1,
        // each axis has should have its own magnitude.
        using real_t = traits::value_type_t<T>;
        using vec2_t = Float2<real_t>;
        const vec2_t half_shape{shape / 2};
        const vec2_t frequency_range_{frequency_range};
        const vec2_t radius_y_range{frequency_range_ * 2 * half_shape[0]};
        const vec2_t radius_x_range{frequency_range_ * 2 * half_shape[1]};
        const vec2_t angle_range_{angle_range};

        const auto size_phi = static_cast<real_t>(output_shape[1] - 1);
        const auto step_angle = (angle_range_[1] - angle_range_[0]) / size_phi;

        const auto size_rho = static_cast<real_t>(output_shape[2] - 1);
        real_t step_magnitude_y, step_magnitude_x;
        if (log) {
            step_magnitude_y = math::log(radius_y_range[1] - radius_y_range[0]) / size_rho;
            step_magnitude_x = math::log(radius_x_range[1] - radius_x_range[0]) / size_rho;
        } else {
            step_magnitude_y = (radius_y_range[1] - radius_y_range[0]) / size_rho;
            step_magnitude_x = (radius_x_range[1] - radius_x_range[0]) / size_rho;
        }

        const vec2_t start_radius{radius_y_range[0], radius_x_range[1]};
        const real_t center = half_shape[0];

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(input, output, output_stride, output_shape, log, interp,         \
               offset, step_angle, step_magnitude_y, step_magnitude_x,          \
               angle_range_, start_radius, center)

        for (size_t batch = 0; batch < output_shape[0]; ++batch) {
            for (size_t phi = 0; phi < output_shape[1]; ++phi) {
                for (size_t rho = 0; rho < output_shape[2]; ++rho) {

                    // (phi, rho) -> (angle, magnitude)
                    const vec2_t polar_coordinate{phi, rho};
                    const real_t angle_rad = polar_coordinate[0] * step_angle + angle_range_[0];
                    real_t magnitude_y, magnitude_x;
                    if (log) {
                        magnitude_y = math::exp(polar_coordinate[1] * step_magnitude_y) - 1 + start_radius[0];
                        magnitude_x = math::exp(polar_coordinate[1] * step_magnitude_x) - 1 + start_radius[1];
                    } else {
                        magnitude_y = polar_coordinate[1] * step_magnitude_y + start_radius[0];
                        magnitude_x = polar_coordinate[1] * step_magnitude_x + start_radius[1];
                    }

                    // (angle, magnitude) -> (y, x)
                    float2_t cartesian_coordinates{magnitude_y * math::sin(angle_rad) + center,
                                                   magnitude_x * math::cos(angle_rad)}; // center_x = 0
                    [[maybe_unused]] real_t conj = 1;
                    if (cartesian_coordinates[1] < 0) {
                        cartesian_coordinates = -cartesian_coordinates;
                        if constexpr (traits::is_complex_v<T>)
                            conj = -1;
                    }

                    T value = interp.template get<INTERP, BORDER_ZERO>(cartesian_coordinates, offset * batch);
                    if constexpr (traits::is_complex_v<T>)
                        value.imag *= conj;

                    output[indexing::at(batch, phi, rho, output_stride)] = value;
                }
            }
        }
    }
}

namespace noa::cpu::geometry::fft {
    template<Remap, typename T>
    void cartesian2polar(const shared_t<T[]>& cartesian, size4_t cartesian_stride, size4_t cartesian_shape,
                         const shared_t<T[]>& polar, size4_t polar_stride, size4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
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
                            frequency_range, angle_range, log, threads);
                });
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR:
                return stream.enqueue([=]() {
                    cartesian2polar_<T, INTERP_LINEAR>(
                            cartesian.get(), src_stride, src_shape, polar.get(), dst_stride, dst_shape,
                            frequency_range, angle_range, log, threads);
                });
            case INTERP_COSINE_FAST:
            case INTERP_COSINE:
                return stream.enqueue([=]() {
                    cartesian2polar_<T, INTERP_COSINE>(
                            cartesian.get(), src_stride, src_shape, polar.get(), dst_stride, dst_shape,
                            frequency_range, angle_range, log, threads);
                });
            case INTERP_CUBIC:
                return stream.enqueue([=]() {
                    cartesian2polar_<T, INTERP_CUBIC>(
                            cartesian.get(), src_stride, src_shape, polar.get(), dst_stride, dst_shape,
                            frequency_range, angle_range, log, threads);
                });
            case INTERP_CUBIC_BSPLINE_FAST:
            case INTERP_CUBIC_BSPLINE:
                NOA_THROW("{} is not supported", interp);
        }
    }

    #define INSTANTIATE_POLAR(T) \
    template void cartesian2polar<Remap::HC2FC, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, bool, InterpMode, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(double);
    INSTANTIATE_POLAR(cfloat_t);
    INSTANTIATE_POLAR(cdouble_t);
}
