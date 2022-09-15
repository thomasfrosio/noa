#include "noa/common/Math.h"
#include "noa/common/geometry/Polar.h"
#include "noa/gpu/cuda/signal/Shape.h"

// TODO Add vectorized loads/stores?
namespace {
    using namespace noa;
    constexpr dim3 BLOCK_SIZE(32, 8);

    template<bool INVERT>
    __device__ __forceinline__ float getSoftMask_(float irho, float erho, float taper_size) {
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
    __device__ __forceinline__ float getHardMask_(float rho) {
        if constexpr (INVERT)
            return static_cast<float>(rho > 1);
        else
            return static_cast<float>(rho <= 1);
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void ellipse_(const T* input, uint4_t input_strides, T* output, uint4_t output_strides,
                  uint3_t start, uint2_t end, uint batches,
                  float3_t center, float3_t radius) {
        const uint3_t gid{blockIdx.z + start[0],
                          blockIdx.y * BLOCK_SIZE.y + threadIdx.y + start[1],
                          blockIdx.x * BLOCK_SIZE.x + threadIdx.x + start[2]};
        if (gid[1] >= end[0] || gid[2] >= end[1])
            return;

        float3_t coords(gid);
        coords -= center;
        coords /= radius;
        const float rho = math::dot(coords, coords);
        const float mask = getHardMask_<INVERT>(rho);

        using real_t = traits::value_type_t<T>;
        const uint offset = gid[0] * input_strides[1] + gid[1] * input_strides[2] + gid[2] * input_strides[3];
        output += gid[0] * output_strides[1] + gid[1] * output_strides[2] + gid[2] * output_strides[3];
        for (uint batch = 0; batch < batches; ++batch) {
            output[batch * output_strides[0]] =
                    input ?
                    input[batch * input_strides[0] + offset] * static_cast<real_t>(mask) :
                    static_cast<real_t>(mask);
        }
    }

    // FIXME Benchmark if it is really worth having a specific kernel for the 2D case. Just use the 3D version.
    template<bool INVERT, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void ellipse2DSmooth_(const T* input, uint3_t input_strides, T* output, uint3_t output_strides,
                          uint2_t start, uint2_t end, uint batches,
                          float2_t center, float2_t radius_sqd, float taper_size) {
        const uint2_t gid{blockIdx.y * BLOCK_SIZE.y + threadIdx.y + start[0],
                          blockIdx.x * BLOCK_SIZE.x + threadIdx.x + start[1]};
        if (gid[0] >= end[0] || gid[1] >= end[1])
            return;

        float2_t cartesian(gid);
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
        const float mask = getSoftMask_<INVERT>(irho, erho, taper_size);

        using real_t = traits::value_type_t<T>;
        const uint offset = gid[0] * input_strides[1] + gid[1] * input_strides[2];
        output += gid[0] * output_strides[1] + gid[1] * output_strides[2];
        for (uint batch = 0; batch < batches; ++batch) {
            output[batch * output_strides[0]] =
                    input ?
                    input[batch * input_strides[0] + offset] * static_cast<real_t>(mask) :
                    static_cast<real_t>(mask);
        }
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void ellipse3DSmooth_(const T* input, uint4_t input_strides, T* output, uint4_t output_strides,
                          uint3_t start, uint2_t end, uint batches,
                          float3_t center, float3_t radius_sqd, float taper_size) {
        const uint3_t gid{blockIdx.z + start[0],
                          blockIdx.y * BLOCK_SIZE.y + threadIdx.y + start[1],
                          blockIdx.x * BLOCK_SIZE.x + threadIdx.x + start[2]};
        if (gid[1] >= end[0] || gid[2] >= end[1])
            return;

        float3_t cartesian(gid);
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
        const float mask = getSoftMask_<INVERT>(irho, erho, taper_size);

        using real_t = traits::value_type_t<T>;
        const uint offset = gid[0] * input_strides[1] + gid[1] * input_strides[2] + gid[2] * input_strides[3];
        output += gid[0] * output_strides[1] + gid[1] * output_strides[2] + gid[2] * output_strides[3];
        for (uint batch = 0; batch < batches; ++batch) {
            output[batch * output_strides[0]] =
                    input ?
                    input[batch * input_strides[0] + offset] * static_cast<real_t>(mask) :
                    static_cast<real_t>(mask);
        }
    }
}

namespace noa::cuda::signal {
    template<bool INVERT, typename T, typename>
    void ellipse(const shared_t<T[]>& input, size4_t input_strides,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                 float3_t center, float3_t radius, float taper_size, Stream& stream) {
        const size3_t order_3d = indexing::order(size3_t(output_strides.get(1)), size3_t(shape.get(1)));
        if (any(order_3d != size3_t{0, 1, 2})) {
            const size4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
        }

        uint3_t start{0}, end(shape.get(1));
        if (INVERT && input.get() == output.get()) {
            start = uint3_t(noa::math::clamp(int3_t(center - (radius + taper_size)), int3_t{}, int3_t(end)));
            end = uint3_t(noa::math::clamp(int3_t(center + (radius + taper_size) + 1), int3_t{}, int3_t(end)));
            if (any(end <= start))
                return;
        }
        const uint3_t shape_(end - start);
        const dim3 blocks(math::divideUp(shape_[2], BLOCK_SIZE.x),
                          math::divideUp(shape_[1], BLOCK_SIZE.y),
                          shape_[0]);
        const LaunchConfig config{blocks, BLOCK_SIZE};
        const bool taper = taper_size > 1e-5f;
        if (taper) {
            if (shape[1] == 1) {
                const uint3_t istrides{input_strides[0], input_strides[2], input_strides[3]};
                const uint3_t ostrides{output_strides[0], output_strides[2], output_strides[3]};
                const float2_t radius_(radius.get(1));
                stream.enqueue("signal::ellipse", ellipse2DSmooth_<INVERT, T>, config,
                               input.get(), istrides, output.get(), ostrides,
                               uint2_t(start.get(1)), uint2_t(end.get(1)), shape[0],
                               float2_t(center.get(1)), radius_ * radius_, taper_size);
            } else {
                stream.enqueue("signal::ellipse", ellipse3DSmooth_<INVERT, T>, config,
                               input.get(), uint4_t(input_strides), output.get(), uint4_t(output_strides),
                               start, uint2_t(end.get(1)), shape[0], center, radius * radius, taper_size);
            };
        } else {
            stream.enqueue("signal::ellipse", ellipse_<INVERT, T>, config,
                           input.get(), uint4_t(input_strides), output.get(), uint4_t(output_strides),
                           start, uint2_t(end.get(1)), shape[0], center, radius);
        }

        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_ELLIPSE_(T)                                                                                                                 \
    template void ellipse<true, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float3_t, float, Stream&);    \
    template void ellipse<false, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_ELLIPSE_(half_t);
    NOA_INSTANTIATE_ELLIPSE_(float);
    NOA_INSTANTIATE_ELLIPSE_(double);
    NOA_INSTANTIATE_ELLIPSE_(chalf_t);
    NOA_INSTANTIATE_ELLIPSE_(cfloat_t);
    NOA_INSTANTIATE_ELLIPSE_(cdouble_t);
}
