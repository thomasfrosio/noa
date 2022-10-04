#include "noa/common/Math.h"
#include "noa/common/signal/Shape.h"
#include "noa/gpu/cuda/signal/Shape.h"

// TODO Add vectorized loads/stores?
namespace {
    using namespace noa;
    constexpr dim3 BLOCK_SIZE(32, 8);

    template<typename signal_shape_t, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void applyShape3D_(Accessor<const T, 4, uint32_t> input,
                       Accessor<T, 4, uint32_t> output,
                       uint3_t start, uint2_t end, int32_t batches,
                       signal_shape_t signal_shape) {
        const uint3_t gid{blockIdx.z + start[0],
                          blockIdx.y * BLOCK_SIZE.y + threadIdx.y + start[1],
                          blockIdx.x * BLOCK_SIZE.x + threadIdx.x + start[2]};
        if (gid[1] >= end[0] || gid[2] >= end[1])
            return;

        const auto mask = signal_shape(float3_t(gid));
        for (int32_t batch = 0; batch < batches; ++batch)
            output(batch, gid[0], gid[1], gid[2]) = input ? input(batch, gid[0], gid[1], gid[2]) * mask : mask;
    }

    template<typename signal_shape_t, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void applyShape2D_(Accessor<const T, 3, uint32_t> input,
                       Accessor<T, 3, uint32_t> output,
                       uint2_t start, uint2_t end, int32_t batches,
                       signal_shape_t signal_shape) {
        const uint2_t gid{blockIdx.y * BLOCK_SIZE.y + threadIdx.y + start[0],
                          blockIdx.x * BLOCK_SIZE.x + threadIdx.x + start[1]};
        if (gid[0] >= end[0] || gid[1] >= end[1])
            return;

        const auto mask = signal_shape(float2_t(gid));
        for (int32_t batch = 0; batch < batches; ++batch)
            output(batch, gid[0], gid[1]) = input ? input(batch, gid[0], gid[1]) * mask : mask;
    }
}

namespace noa::cuda::signal {
    template<bool INVERT, typename T, typename>
    void sphere(const shared_t<T[]>& input, dim4_t input_strides,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                float3_t center, float radius, float taper_size, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
        }

        int3_t start{0};
        auto end = safe_cast<int3_t>(dim3_t(shape.get(1)));
        if (INVERT && input.get() == output.get()) {
            start = noa::math::clamp(int3_t(center - (radius + taper_size)), int3_t{}, int3_t(end));
            end = noa::math::clamp(int3_t(center + (radius + taper_size) + 1), int3_t{}, int3_t(end));
            if (any(end <= start))
                return;
        }
        const auto shape_ = safe_cast<uint3_t>(end - start);
        const dim3 blocks(math::divideUp(shape_[2], BLOCK_SIZE.x),
                          math::divideUp(shape_[1], BLOCK_SIZE.y),
                          shape_[0]);
        const LaunchConfig config{blocks, BLOCK_SIZE};

        const Accessor<const T, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
        const Accessor<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));

        using real_t = traits::value_type_t<T>;
        using sphere_t = noa::signal::Sphere<3, real_t, INVERT>;
        using sphere_smooth_t = noa::signal::SphereSmooth<3, real_t, INVERT>;
        const bool taper = taper_size > 1e-5f;
        if (taper) {
            stream.enqueue("signal::sphere",
                           applyShape3D_<sphere_smooth_t, T>, config,
                           input_accessor, output_accessor,
                           uint3_t(start), uint2_t(end.get(1)), shape[0],
                           sphere_smooth_t(center, radius, taper_size));
        } else {
            stream.enqueue("signal::sphere",
                           applyShape3D_<sphere_t, T>, config,
                           input_accessor, output_accessor,
                           uint3_t(start), uint2_t(end.get(1)), shape[0],
                           sphere_t(center, radius));
        }
        stream.attach(input, output);
    }

    template<bool INVERT, typename T, typename>
    void cylinder(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  float3_t center, float radius, float length, float taper_size, Stream& stream) {
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[1], center[2]);
        }

        int3_t start{0};
        auto end = safe_cast<int3_t>(dim3_t(shape.get(1)));
        if (INVERT && input.get() == output.get()) {
            float3_t radius_{length, radius, radius};
            radius_ += taper_size;
            start = noa::math::clamp(int3_t(center - radius_), int3_t{}, int3_t(end));
            end = noa::math::clamp(int3_t(center + radius_ + 1), int3_t{}, int3_t(end));
            if (any(end <= start))
                return;
        }
        const auto shape_ = safe_cast<uint3_t>(end - start);
        const dim3 blocks(math::divideUp(shape_[2], BLOCK_SIZE.x),
                          math::divideUp(shape_[1], BLOCK_SIZE.y),
                          shape_[0]);
        const LaunchConfig config{blocks, BLOCK_SIZE};

        const Accessor<const T, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
        const Accessor<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));

        using real_t = traits::value_type_t<T>;
        using cylinder_t = noa::signal::Cylinder<real_t, INVERT>;
        using cylinder_smooth_t = noa::signal::CylinderSmooth<real_t, INVERT>;
        const bool taper = taper_size > 1e-5f;
        if (taper) {
            stream.enqueue("signal::sphere",
                           applyShape3D_<cylinder_smooth_t, T>, config,
                           input_accessor, output_accessor,
                           uint3_t(start), uint2_t(end.get(1)), shape[0],
                           cylinder_smooth_t(center, radius, length, taper_size));
        } else {
            stream.enqueue("signal::sphere",
                           applyShape3D_<cylinder_t, T>, config,
                           input_accessor, output_accessor,
                           uint3_t(start), uint2_t(end.get(1)), shape[0],
                           cylinder_t(center, radius, length));
        }
        stream.attach(input, output);
    }

    template<bool INVERT, typename T, typename>
    void rectangle(const shared_t<T[]>& input, dim4_t input_strides,
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

        int3_t start{0};
        auto end = safe_cast<int3_t>(dim3_t(shape.get(1)));
        if (INVERT && input.get() == output.get()) {
            start = noa::math::clamp(int3_t(center - (radius + taper_size)), int3_t{}, int3_t(end));
            end = noa::math::clamp(int3_t(center + (radius + taper_size) + 1), int3_t{}, int3_t(end));
            if (any(end <= start))
                return;
        }
        const auto shape_ = safe_cast<uint3_t>(end - start);
        const dim3 blocks(math::divideUp(shape_[2], BLOCK_SIZE.x),
                          math::divideUp(shape_[1], BLOCK_SIZE.y),
                          shape_[0]);
        const LaunchConfig config{blocks, BLOCK_SIZE};

        const Accessor<const T, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
        const Accessor<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));

        using real_t = traits::value_type_t<T>;
        using rectangle_t = noa::signal::Rectangle<3, real_t, INVERT>;
        using rectangle_smooth_t = noa::signal::RectangleSmooth<3, real_t, INVERT>;
        const bool taper = taper_size > 1e-5f;
        if (taper) {
            stream.enqueue("signal::sphere",
                           applyShape3D_<rectangle_smooth_t, T>, config,
                           input_accessor, output_accessor,
                           uint3_t(start), uint2_t(end.get(1)), shape[0],
                           rectangle_smooth_t(center, radius, taper_size));
        } else {
            stream.enqueue("signal::sphere",
                           applyShape3D_<rectangle_t, T>, config,
                           input_accessor, output_accessor,
                           uint3_t(start), uint2_t(end.get(1)), shape[0],
                           rectangle_t(center, radius));
        }
        stream.attach(input, output);
    }

    template<bool INVERT, typename T, typename>
    void ellipse(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t center, float3_t radius, float taper_size, Stream& stream) {
        const size3_t order_3d = indexing::order(size3_t(output_strides.get(1)), size3_t(shape.get(1)));
        if (any(order_3d != size3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
        }

        int3_t start{0};
        auto end = safe_cast<int3_t>(dim3_t(shape.get(1)));
        if (INVERT && input.get() == output.get()) {
            start = noa::math::clamp(int3_t(center - (radius + taper_size)), int3_t{}, int3_t(end));
            end = noa::math::clamp(int3_t(center + (radius + taper_size) + 1), int3_t{}, int3_t(end));
            if (any(end <= start))
                return;
        }
        const auto shape_ = safe_cast<uint3_t>(end - start);
        const dim3 blocks(math::divideUp(shape_[2], BLOCK_SIZE.x),
                          math::divideUp(shape_[1], BLOCK_SIZE.y),
                          shape_[0]);
        const LaunchConfig config{blocks, BLOCK_SIZE};

        const auto input_strides_ = safe_cast<uint4_t>(input_strides);
        const auto output_strides_ = safe_cast<uint4_t>(output_strides);

        using real_t = traits::value_type_t<T>;
        using ellipse_t = noa::signal::Ellipse<3, real_t, INVERT>;
        using ellipse_smooth_t = noa::signal::EllipseSmooth<3, real_t, INVERT>;
        const bool taper = taper_size > 1e-5f;
        if (taper) {
            if (shape[1] == 1 && center[0] == 0.f) {
                const uint3_t istrides{input_strides_[0], input_strides_[2], input_strides_[3]};
                const uint3_t ostrides{output_strides_[0], output_strides_[2], output_strides_[3]};
                const Accessor<const T, 3, uint32_t> input_accessor(input.get(), istrides);
                const Accessor<T, 3, uint32_t> output_accessor(output.get(), ostrides);

                using ellipse_smooth_2d_t = noa::signal::EllipseSmooth<2, real_t, INVERT>;
                const ellipse_smooth_2d_t ellipse(float2_t(center.get(1)), float2_t(radius.get(1)), taper_size);

                stream.enqueue("signal::ellipse", applyShape2D_<ellipse_smooth_2d_t, T>, config,
                               input_accessor, output_accessor,
                               uint2_t(start.get(1)), uint2_t(end.get(1)), shape[0],
                               ellipse);
            } else {
                const Accessor<const T, 4, uint32_t> input_accessor(input.get(), input_strides_);
                const Accessor<T, 4, uint32_t> output_accessor(output.get(), output_strides_);
                const ellipse_smooth_t ellipse(center, radius, taper_size);

                stream.enqueue("signal::ellipse", applyShape3D_<ellipse_smooth_t, T>, config,
                               input_accessor, output_accessor,
                               uint3_t(start), uint2_t(end.get(1)), shape[0], ellipse);
            };
        } else {
            const Accessor<const T, 4, uint32_t> input_accessor(input.get(), input_strides_);
            const Accessor<T, 4, uint32_t> output_accessor(output.get(), output_strides_);
            const ellipse_t ellipse(center, radius);

            stream.enqueue("signal::ellipse", applyShape3D_<ellipse_t, T>, config,
                           input_accessor, output_accessor,
                           uint3_t(start), uint2_t(end.get(1)), shape[0], ellipse);
        }

        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_SHAPE_(T)                                                                                                                   \
    template void sphere<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, Stream&);           \
    template void sphere<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, Stream&);          \
    template void cylinder<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, float, Stream&);  \
    template void cylinder<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, float, Stream&); \
    template void rectangle<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&);     \
    template void rectangle<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&);    \
    template void ellipse<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&);       \
    template void ellipse<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_SHAPE_(half_t);
    NOA_INSTANTIATE_SHAPE_(float);
    NOA_INSTANTIATE_SHAPE_(double);
    NOA_INSTANTIATE_SHAPE_(chalf_t);
    NOA_INSTANTIATE_SHAPE_(cfloat_t);
    NOA_INSTANTIATE_SHAPE_(cdouble_t);
}
