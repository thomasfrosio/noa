#include "noa/common/geometry/Polar.h"
#include "noa/cpu/signal/Shape.h"

namespace {
    using namespace ::noa;

    template<bool INVERT>
    float getSoftMask_(float irho, float erho, float taper_size) {
        constexpr float PI = math::Constants<float>::PI;
        if constexpr (INVERT) {
            if (irho > erho + taper_size)
                return 1.f;
            else if (irho <= erho)
                return 0.f;
            else
                return (1.f - math::cos(PI * (irho - erho) / taper_size)) * 0.5f;
        } else {
            if (irho > erho + taper_size)
                return 0.f;
            else if (irho <= erho)
                return 1.f;
            else
                return (1.f + math::cos(PI * (irho - erho) / taper_size)) * 0.5f;
        }
    }

    template<bool INVERT>
    float getHardMask_(float rho) {
        if constexpr (INVERT)
            return static_cast<float>(rho > 1);
        else
            return static_cast<float>(rho <= 1);
    }

    template<bool INVERT, typename T>
    void ellipse_(Accessor<const T, 4, dim_t> input,
                  Accessor<T, 4, dim_t> output,
                  dim3_t start, dim3_t end, dim_t batches,
                  float3_t center, float3_t radius) {
        using real_t = traits::value_type_t<T>;

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t j = start[0]; j < end[0]; ++j) {
                for (dim_t k = start[1]; k < end[1]; ++k) {
                    for (dim_t l = start[2]; l < end[2]; ++l) {

                        const float z = math::pow(static_cast<float>(j) - center[0] / radius[0], 2.f);
                        const float y = math::pow(static_cast<float>(k) - center[1] / radius[1], 2.f);
                        const float x = math::pow(static_cast<float>(l) - center[2] / radius[2], 2.f);
                        const float rho = z + y + x;
                        const auto mask = static_cast<real_t>(getHardMask_<INVERT>(rho));

                        output(i, j, k, l) = input ? input(i, j, k, l) * mask : mask;
                    }
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void ellipseOMP_(Accessor<const T, 4, dim_t> input,
                     Accessor<T, 4, dim_t> output,
                     dim3_t start, dim3_t end, dim_t batches,
                     float3_t center, float3_t radius, dim_t threads) {
        using real_t = traits::value_type_t<T>;

        #pragma omp parallel for default(none) collapse(4) num_threads(threads) \
        shared(input, output, start, end, batches, center, radius)

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t j = start[0]; j < end[0]; ++j) {
                for (dim_t k = start[1]; k < end[1]; ++k) {
                    for (dim_t l = start[2]; l < end[2]; ++l) {

                        float3_t coords(j, k, l);
                        coords -= center;
                        coords /= radius;
                        const float rho = math::dot(coords, coords);
                        const auto mask = static_cast<real_t>(getHardMask_<INVERT>(rho));

                        output(i, j, k, l) = input ? input(i, j, k, l) * mask : mask;
                    }
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void ellipse2DSmoothOMP_(Accessor<const T, 3, dim_t> input,
                             Accessor<T, 3, dim_t> output,
                             dim2_t start, dim2_t end, dim_t batches,
                             float2_t center, float2_t radius, float taper_size, dim_t threads) {
        using real_t = traits::value_type_t<T>;
        const float2_t radius_sqd = radius * radius;

        #pragma omp parallel for default(none) collapse(3) num_threads(threads) \
        shared(input, output, start, end, batches, center, radius, taper_size, radius_sqd)

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t k = start[0]; k < end[0]; ++k) {
                for (dim_t l = start[1]; l < end[1]; ++l) {

                    float2_t cartesian{k, l};
                    cartesian -= center;

                    // Current spherical coordinate:
                    const float irho = geometry::cartesian2rho(cartesian);
                    const float iphi = geometry::cartesian2phi<false>(cartesian);

                    // Radius of the ellipse at (iphi, itheta):
                    const float cos2phi = math::pow(math::cos(iphi), 2.f);
                    const float sin2phi = math::pow(math::sin(iphi), 2.f);
                    const float erho = 1.f / math::sqrt(cos2phi / radius_sqd[1] +
                                                        sin2phi / radius_sqd[0]);

                    // Get mask value for this radius.
                    const auto mask = static_cast<real_t>(getSoftMask_<INVERT>(irho, erho, taper_size));

                    output(i, k, l) = input ? input(i, k, l) * mask : mask;
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void ellipse3DSmoothOMP_(Accessor<const T, 4, dim_t> input,
                             Accessor<T, 4, dim_t> output,
                             dim3_t start, dim3_t end, dim_t batches,
                             float3_t center, float3_t radius, float taper_size, dim_t threads) {
        using real_t = traits::value_type_t<T>;
        const float3_t radius_sqd = radius * radius;

        #pragma omp parallel for default(none) collapse(4) num_threads(threads) \
        shared(input, output, start, end, batches, center, radius, taper_size, radius_sqd)

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t j = start[0]; j < end[0]; ++j) {
                for (dim_t k = start[1]; k < end[1]; ++k) {
                    for (dim_t l = start[2]; l < end[2]; ++l) {

                        float3_t cartesian{j, k, l};
                        cartesian -= center;

                        // Current spherical coordinate:
                        const float irho = geometry::cartesian2rho(cartesian);
                        const float iphi = geometry::cartesian2phi<false>(cartesian);
                        const float itheta = geometry::cartesian2theta(cartesian);

                        // Radius of the ellipse at (iphi, itheta):
                        const float cos2phi = math::pow(math::cos(iphi), 2.f);
                        const float sin2phi = math::pow(math::sin(iphi), 2.f);
                        const float cos2theta = math::pow(math::cos(itheta), 2.f);
                        const float sin2theta = math::pow(math::sin(itheta), 2.f);
                        const float erho = 1.f / math::sqrt(cos2phi * sin2theta / radius_sqd[2] +
                                                            sin2phi * sin2theta / radius_sqd[1] +
                                                            cos2theta / radius_sqd[0]);

                        // Get mask value for this radius.
                        const auto mask = static_cast<real_t>(getSoftMask_<INVERT>(irho, erho, taper_size));

                        output(i, j, k, l) = input ? input(i, j, k, l) * mask : mask;
                    }
                }
            }
        }
    }
}

namespace noa::cpu::signal {
    template<bool INVERT, typename T, typename>
    void ellipse(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t center, float3_t radius, float taper_size, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
        }

        dim3_t start{0}, end(shape.get(1));
        if (INVERT && input.get() == output.get()) {
            start = dim3_t(noa::math::clamp(int3_t(center - (radius + taper_size)), int3_t{}, int3_t(end)));
            end = dim3_t(noa::math::clamp(int3_t(center + (radius + taper_size) + 1), int3_t{}, int3_t(end)));
            if (any(end <= start))
                return;
        }

        const dim_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (taper) {
            stream.enqueue([=]() {
                if (shape[1] == 1) {
                    const dim3_t istrides_2d{input_strides[0], input_strides[2], input_strides[3]};
                    const dim3_t ostrides_2d{output_strides[0], output_strides[2], output_strides[3]};
                    ellipse2DSmoothOMP_<INVERT, T>({input.get(), istrides_2d}, {output.get(), ostrides_2d},
                                                   dim2_t(start.get(1)), dim2_t(end.get(1)), shape[0],
                                                   float2_t(center.get(1)), float2_t(radius.get(1)), taper_size,
                                                   threads);
                } else {
                    ellipse3DSmoothOMP_<INVERT, T>({input.get(), input_strides}, {output.get(), output_strides},
                                                   start, end, shape[0], center, radius, taper_size, threads);
                };
            });
        } else {
            stream.enqueue([=]() {
                if (threads) {
                    ellipseOMP_<INVERT, T>({input.get(), input_strides}, {output.get(), output_strides},
                                           start, end, shape[0], center, radius, threads);
                } else {
                    ellipse_<INVERT, T>({input.get(), input_strides}, {output.get(), output_strides},
                                        start, end, shape[0], center, radius);
                }
            });
        }
    }

    #define NOA_INSTANTIATE_ELLIPSE_(T)                                                                                                             \
    template void ellipse<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&);   \
    template void ellipse<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_ELLIPSE_(half_t);
    NOA_INSTANTIATE_ELLIPSE_(float);
    NOA_INSTANTIATE_ELLIPSE_(double);
    NOA_INSTANTIATE_ELLIPSE_(chalf_t);
    NOA_INSTANTIATE_ELLIPSE_(cfloat_t);
    NOA_INSTANTIATE_ELLIPSE_(cdouble_t);
}
