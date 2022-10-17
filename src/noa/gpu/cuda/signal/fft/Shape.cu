#include "noa/common/Math.h"
#include "noa/common/signal/Shape.h"
#include "noa/gpu/cuda/signal/fft/Shape.h"
#include "noa/gpu/cuda/util/Pointers.h"

// TODO Add vectorized loads/stores?
namespace {
    using namespace noa;
    constexpr dim3 BLOCK_SIZE(32, 8);
    struct Empty{};

    template<fft::Remap REMAP, bool TRANSFORM,
             typename geom_shape_t, typename matrix_t, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void kernel3D_(Accessor<const T, 4, uint32_t> input,
                   Accessor<T, 4, uint32_t> output,
                   uint3_t start, uint2_t end, uint4_t shape,
                   geom_shape_t signal_shape, matrix_t inv_transform) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & fft::Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & fft::Layout::DST_CENTERED;
        if constexpr (REMAP_ & fft::Layout::SRC_HALF || REMAP_ & fft::Layout::DST_HALF)
            static_assert(traits::always_false_v<T>);

        const uint3_t gid{blockIdx.z + start[0],
                          blockIdx.y * BLOCK_SIZE.y + threadIdx.y + start[1],
                          blockIdx.x * BLOCK_SIZE.x + threadIdx.x + start[2]};
        if (gid[1] >= end[0] || gid[2] >= end[1])
            return;

        const uint3_t i_idx{IS_SRC_CENTERED ? gid[0] : math::FFTShift(gid[0], shape[1]),
                            IS_SRC_CENTERED ? gid[1] : math::FFTShift(gid[1], shape[2]),
                            IS_SRC_CENTERED ? gid[2] : math::FFTShift(gid[2], shape[3])};
        const uint3_t o_idx{IS_DST_CENTERED ? gid[0] : math::FFTShift(gid[0], shape[1]),
                            IS_DST_CENTERED ? gid[1] : math::FFTShift(gid[1], shape[2]),
                            IS_DST_CENTERED ? gid[2] : math::FFTShift(gid[2], shape[3])};

        float3_t coords{i_idx};
        typename geom_shape_t::value_type mask;
        if constexpr (TRANSFORM)
            mask = signal_shape(coords, inv_transform);
        else
            mask = signal_shape(coords);

        for (int32_t batch = 0; batch < static_cast<int32_t>(shape[0]); ++batch)
            output[batch](o_idx) = input ? input[batch](i_idx) * mask : mask;
    }

    template<fft::Remap REMAP, bool TRANSFORM,
             typename geom_shape_t, typename matrix_t, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void kernel2D_(Accessor<const T, 3, uint32_t> input,
                   Accessor<T, 3, uint32_t> output,
                   uint2_t start, uint2_t end, uint3_t shape,
                   geom_shape_t signal_shape, matrix_t inv_transform) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & fft::Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & fft::Layout::DST_CENTERED;
        if constexpr (REMAP_ & fft::Layout::SRC_HALF || REMAP_ & fft::Layout::DST_HALF)
            static_assert(traits::always_false_v<T>);

        const uint2_t gid{blockIdx.y * BLOCK_SIZE.y + threadIdx.y + start[0],
                          blockIdx.x * BLOCK_SIZE.x + threadIdx.x + start[1]};
        if (gid[0] >= end[0] || gid[1] >= end[1])
            return;

        const uint2_t i_idx{IS_SRC_CENTERED ? gid[0] : math::iFFTShift(gid[0], shape[1]),
                            IS_SRC_CENTERED ? gid[1] : math::iFFTShift(gid[1], shape[2])};
        const uint2_t o_idx{IS_DST_CENTERED ? gid[0] : math::iFFTShift(gid[0], shape[1]),
                            IS_DST_CENTERED ? gid[1] : math::iFFTShift(gid[1], shape[2])};

        float2_t coords{i_idx};
        typename geom_shape_t::value_type mask;
        if constexpr (TRANSFORM)
            mask = signal_shape(coords, inv_transform);
        else
            mask = signal_shape(coords);

        for (int32_t batch = 0; batch < static_cast<int32_t>(shape[0]); ++batch)
            output[batch](o_idx) = input ? input[batch](i_idx) * mask : mask;
    }

    template<fft::Remap REMAP, typename geom_shape_t, typename geom_shape_smooth_t, typename T, typename radius_t>
    void launch3D_(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   const float3_t& center, const radius_t& radius, float edge_size,
                   float33_t inv_transform, bool invert, cuda::Stream& stream) {
        NOA_ASSERT(all(shape > 0) && ((REMAP == fft::F2F || REMAP == fft::FC2FC) || input != output));
        NOA_ASSERT(input.get() == nullptr || ::noa::cuda::util::devicePointer(input.get(), stream.device()) != nullptr);
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        int3_t start{0};
        auto end = safe_cast<int3_t>(dim3_t(shape.get(1)));
        if (invert && input.get() == output.get()) {
            start = noa::math::clamp(int3_t(center - (radius + edge_size)), int3_t{}, int3_t(end));
            end = noa::math::clamp(int3_t(center + (radius + edge_size) + 1), int3_t{}, int3_t(end));
            if (any(end <= start))
                return;
        }
        const auto shape_ = safe_cast<uint3_t>(end - start);
        const dim3 blocks(math::divideUp(shape_[2], BLOCK_SIZE.x),
                          math::divideUp(shape_[1], BLOCK_SIZE.y),
                          shape_[0]);
        const cuda::LaunchConfig config{blocks, BLOCK_SIZE};

        const Accessor<const T, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
        const Accessor<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));

        const bool has_smooth_edge = edge_size > 1e-5f;
        if (has_smooth_edge) {
            const geom_shape_smooth_t geom_shape_smooth(center, radius, edge_size);
            if (float33_t{} == inv_transform) {
                stream.enqueue("signal::shape",
                               kernel3D_<REMAP, false, geom_shape_smooth_t, Empty, T>, config,
                               input_accessor, output_accessor,
                               uint3_t(start), uint2_t(end.get(1)), uint4_t(shape),
                               geom_shape_smooth, Empty{});
            } else {
                stream.enqueue("signal::shape",
                               kernel3D_<REMAP, true, geom_shape_smooth_t, float33_t, T>, config,
                               input_accessor, output_accessor,
                               uint3_t(start), uint2_t(end.get(1)), uint4_t(shape),
                               geom_shape_smooth, inv_transform);
            }
        } else {
            const geom_shape_t geom_shape(center, radius);
            if (float33_t{} == inv_transform) {
                stream.enqueue("signal::shape",
                               kernel3D_<REMAP, false, geom_shape_t, Empty, T>, config,
                               input_accessor, output_accessor,
                               uint3_t(start), uint2_t(end.get(1)), uint4_t(shape),
                               geom_shape, Empty{});
            } else {
                stream.enqueue("signal::shape",
                               kernel3D_<REMAP, true, geom_shape_t, float33_t, T>, config,
                               input_accessor, output_accessor,
                               uint3_t(start), uint2_t(end.get(1)), uint4_t(shape),
                               geom_shape, inv_transform);
            }
        }
    }

    template<fft::Remap REMAP, typename geom_shape_t, typename geom_shape_smooth_t, typename T, typename radius_t>
    void launch2D_(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   const float2_t& center, const radius_t& radius, float edge_size,
                   float22_t inv_transform, bool invert, cuda::Stream& stream) {
        NOA_ASSERT(all(shape > 0) && ((REMAP == fft::F2F || REMAP == fft::FC2FC) || input != output));
        NOA_ASSERT(input.get() == nullptr || ::noa::cuda::util::devicePointer(input.get(), stream.device()) != nullptr);
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        int2_t start{0};
        auto end = safe_cast<int2_t>(dim2_t(shape.get(2)));
        if (invert && input.get() == output.get()) {
            start = noa::math::clamp(int2_t(center - (radius + edge_size)), int2_t{}, int2_t(end));
            end = noa::math::clamp(int2_t(center + (radius + edge_size) + 1), int2_t{}, int2_t(end));
            if (any(end <= start))
                return;
        }
        const auto shape_ = safe_cast<uint2_t>(end - start);
        const dim3 blocks(math::divideUp(shape_[1], BLOCK_SIZE.x),
                          math::divideUp(shape_[0], BLOCK_SIZE.y));
        const cuda::LaunchConfig config{blocks, BLOCK_SIZE};

        const dim3_t i_strides_2d{input_strides[0], input_strides[2], input_strides[3]};
        const dim3_t o_strides_2d{output_strides[0], output_strides[2], output_strides[3]};
        const uint3_t shape_3d{shape[0], shape[2], shape[3]};
        const Accessor<const T, 3, uint32_t> input_accessor(input.get(), safe_cast<uint3_t>(i_strides_2d));
        const Accessor<T, 3, uint32_t> output_accessor(output.get(), safe_cast<uint3_t>(o_strides_2d));

        const bool has_smooth_edge = edge_size > 1e-5f;
        if (has_smooth_edge) {
            const geom_shape_smooth_t geom_shape_smooth(center, radius, edge_size);
            if (float22_t{} == inv_transform) {
                stream.enqueue("signal::shape",
                               kernel2D_<REMAP, false, geom_shape_smooth_t, Empty, T>, config,
                               input_accessor, output_accessor,
                               uint2_t(start), uint2_t(end), shape_3d,
                               geom_shape_smooth, Empty{});
            } else {
                stream.enqueue("signal::shape",
                               kernel2D_<REMAP, true, geom_shape_smooth_t, float22_t, T>, config,
                               input_accessor, output_accessor,
                               uint2_t(start), uint2_t(end), shape_3d,
                               geom_shape_smooth, inv_transform);
            }
        } else {
            const geom_shape_t geom_shape(center, radius);
            if (float22_t{} == inv_transform) {
                stream.enqueue("signal::shape",
                               kernel2D_<REMAP, false, geom_shape_t, Empty, T>, config,
                               input_accessor, output_accessor,
                               uint2_t(start), uint2_t(end), shape_3d,
                               geom_shape, Empty{});
            } else {
                stream.enqueue("signal::shape",
                               kernel2D_<REMAP, true, geom_shape_t, float22_t, T>, config,
                               input_accessor, output_accessor,
                               uint2_t(start), uint2_t(end), shape_3d,
                               geom_shape, inv_transform);
            }
        }
    }
}

namespace noa::cuda::signal::fft {
    // Returns or applies an elliptical mask.
    template<fft::Remap REMAP, typename T, typename>
    void ellipse(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t center, float3_t radius, float edge_size,
                 float33_t inv_transform, bool invert, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
            inv_transform = indexing::reorder(inv_transform, order_3d);
        }

        using real_t = traits::value_type_t<T>;
        if (invert) {
            using ellipse_t = noa::signal::Ellipse<3, real_t, true>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, real_t, true>;
            launch3D_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        } else {
            using ellipse_t = noa::signal::Ellipse<3, real_t, false>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, real_t, false>;
            launch3D_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void ellipse(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float2_t center, float2_t radius, float edge_size, float22_t inv_transform, bool invert, Stream& stream) {
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            std::swap(radius[0], radius[1]);
            inv_transform = indexing::reorder(inv_transform, order_2d);
        }

        using real_t = traits::value_type_t<T>;
        if (invert) {
            using ellipse_t = noa::signal::Ellipse<2, real_t, true>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<2, real_t, true>;
            launch2D_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        } else {
            using ellipse_t = noa::signal::Ellipse<2, real_t, false>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<2, real_t, false>;
            launch2D_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void sphere(const shared_t<T[]>& input, dim4_t input_strides,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                float3_t center, float radius, float edge_size,
                float33_t inv_transform, bool invert, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            inv_transform = indexing::reorder(inv_transform, order_3d);
        }

        using real_t = traits::value_type_t<T>;
        if (invert) {
            using sphere_t = noa::signal::Sphere<3, real_t, true>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, real_t, true>;
            launch3D_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        } else {
            using sphere_t = noa::signal::Sphere<3, real_t, false>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, real_t, false>;
            launch3D_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void sphere(const shared_t<T[]>& input, dim4_t input_strides,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                float2_t center, float radius, float edge_size,
                float22_t inv_transform, bool invert, Stream& stream) {
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            inv_transform = indexing::reorder(inv_transform, order_2d);
        }

        using real_t = traits::value_type_t<T>;
        if (invert) {
            using sphere_t = noa::signal::Sphere<2, real_t, true>;
            using sphere_smooth_t = noa::signal::SphereSmooth<2, real_t, true>;
            launch2D_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        } else {
            using sphere_t = noa::signal::Sphere<2, real_t, false>;
            using sphere_smooth_t = noa::signal::SphereSmooth<2, real_t, false>;
            launch2D_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void rectangle(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   float3_t center, float3_t radius, float edge_size,
                   float33_t inv_transform, bool invert, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
            inv_transform = indexing::reorder(inv_transform, order_3d);
        }

        using real_t = traits::value_type_t<T>;
        if (invert) {
            using rectangle_t = noa::signal::Rectangle<3, real_t, true>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, real_t, true>;
            launch3D_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        } else {
            using rectangle_t = noa::signal::Rectangle<3, real_t, false>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, real_t, false>;
            launch3D_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void rectangle(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   float2_t center, float2_t radius, float edge_size,
                   float22_t inv_transform, bool invert, Stream& stream) {
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            std::swap(radius[0], radius[1]);
            inv_transform = indexing::reorder(inv_transform, order_2d);
        }

        using real_t = traits::value_type_t<T>;
        if (invert) {
            using rectangle_t = noa::signal::Rectangle<2, real_t, true>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<2, real_t, true>;
            launch2D_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        } else {
            using rectangle_t = noa::signal::Rectangle<2, real_t, false>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<2, real_t, false>;
            launch2D_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_transform, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void cylinder(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  float3_t center, float radius, float length, float edge_size,
                  float33_t inv_transform, bool invert, Stream& stream) {
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[1], center[2]);
            inv_transform = indexing::reorder(inv_transform, dim3_t{0, order_2d[0] + 1, order_2d[1] + 1});
        }

        const float3_t radius_{length, radius, radius};
        using real_t = traits::value_type_t<T>;
        if (invert) {
            using cylinder_t = noa::signal::Cylinder<real_t, true>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<real_t, true>;
            launch3D_<REMAP, cylinder_t, cylinder_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius_, edge_size, inv_transform, invert, stream);
        } else {
            using cylinder_t = noa::signal::Cylinder<real_t, false>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<real_t, false>;
            launch3D_<REMAP, cylinder_t, cylinder_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius_, edge_size, inv_transform, invert, stream);
        }
    }

    #define NOA_INSTANTIATE_SHAPE_(R, T)                                                                                                                            \
    template void ellipse<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, float33_t, bool, Stream&);     \
    template void ellipse<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float, float22_t, bool, Stream&);     \
    template void sphere<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, float33_t, bool, Stream&);         \
    template void sphere<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float, float, float22_t, bool, Stream&);         \
    template void rectangle<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, float33_t, bool, Stream&);   \
    template void rectangle<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float, float22_t, bool, Stream&);   \
    template void cylinder<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, float, float33_t, bool, Stream&)

    #define NOA_INSTANTIATE_SHAPE_ALL(T)    \
    NOA_INSTANTIATE_SHAPE_(fft::F2F, T);    \
    NOA_INSTANTIATE_SHAPE_(fft::FC2FC, T);  \
    NOA_INSTANTIATE_SHAPE_(fft::F2FC, T);   \
    NOA_INSTANTIATE_SHAPE_(fft::FC2F, T)

    NOA_INSTANTIATE_SHAPE_ALL(half_t);
    NOA_INSTANTIATE_SHAPE_ALL(float);
    NOA_INSTANTIATE_SHAPE_ALL(double);
    NOA_INSTANTIATE_SHAPE_ALL(chalf_t);
    NOA_INSTANTIATE_SHAPE_ALL(cfloat_t);
    NOA_INSTANTIATE_SHAPE_ALL(cdouble_t);
}
