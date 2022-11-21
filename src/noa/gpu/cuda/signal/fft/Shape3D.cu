#include "noa/common/Math.h"
#include "noa/common/signal/Shape.h"
#include "noa/common/signal/details/Shape.h"
#include "noa/gpu/cuda/signal/fft/Shape.h"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace {
    using namespace noa;

    template<fft::Remap REMAP, typename GeomShape, typename GeomShapeSmooth,
             typename Functor, typename Matrix, typename Value, typename Radius>
    void launch3D_(const shared_t<Value[]>& input, const dim4_t& input_strides,
                   const shared_t<Value[]>& output, const dim4_t& output_strides, const dim4_t& shape,
                   const float3_t& center, const Radius& radius, float edge_size,
                   const Matrix& inv_matrix, const Functor& functor, bool invert, cuda::Stream& stream) {
        NOA_ASSERT_DEVICE_OR_NULL_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        int3_t start{0};
        auto end = safe_cast<int3_t>(dim3_t(shape.get(1)));
        if (invert && input.get() == output.get() && inv_matrix == Matrix{}) {
            start = noa::math::clamp(int3_t(center - (radius + edge_size)), int3_t{}, int3_t(end));
            end = noa::math::clamp(int3_t(center + (radius + edge_size) + 1), int3_t{}, int3_t(end));
            if (any(end <= start))
                return;
        }

        const auto input_accessor = Accessor<const Value, 4, uint32_t>(input.get(), safe_cast<uint4_t>(input_strides));
        const auto output_accessor = Accessor<Value, 4, uint32_t>(output.get(), safe_cast<uint4_t>(output_strides));
        const auto iwise_start = uint3_t(start);
        const auto iwise_end = uint3_t(end);
        const auto shape_3d = uint4_t(shape);

        const bool has_smooth_edge = edge_size > 1e-5f;
        if (has_smooth_edge) {
            const GeomShapeSmooth geom_shape_smooth(center, radius, edge_size);
            if (Matrix{} == inv_matrix) {
                const auto kernel = signal::fft::details::shape3D<REMAP, uint32_t>(
                        input_accessor, output_accessor, shape_3d, geom_shape_smooth, empty_t{}, functor);
                cuda::utils::iwise3D("signal::shape", iwise_start, iwise_end, kernel, stream);
            } else {
                const auto kernel = signal::fft::details::shape3D<REMAP, uint32_t>(
                        input_accessor, output_accessor, shape_3d, geom_shape_smooth, inv_matrix, functor);
                cuda::utils::iwise3D("signal::shape", iwise_start, iwise_end, kernel, stream);
            }
        } else {
            const GeomShape geom_shape(center, radius);
            if (Matrix{} == inv_matrix) {
                const auto kernel = signal::fft::details::shape3D<REMAP, uint32_t>(
                        input_accessor, output_accessor, shape_3d, geom_shape, empty_t{}, functor);
                cuda::utils::iwise3D("signal::shape", iwise_start, iwise_end, kernel, stream);
            } else {
                const auto kernel = signal::fft::details::shape3D<REMAP, uint32_t>(
                        input_accessor, output_accessor, shape_3d, geom_shape, inv_matrix, functor);
                cuda::utils::iwise3D("signal::shape", iwise_start, iwise_end, kernel, stream);
            }
        }
    }
}

namespace noa::cuda::signal::fft {
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename>
    void ellipse(const shared_t<Value[]>& input, dim4_t input_strides,
                 const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t center, float3_t radius, float edge_size,
                 Matrix inv_matrix, Functor functor, bool invert, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
            inv_matrix = indexing::reorder(inv_matrix, order_3d);
        }

        using real_t = traits::value_type_t<Value>;
        if (invert) {
            using ellipse_t = noa::signal::Ellipse<3, real_t, true>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, real_t, true>;
            launch3D_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        } else {
            using ellipse_t = noa::signal::Ellipse<3, real_t, false>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, real_t, false>;
            launch3D_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename>
    void sphere(const shared_t<Value[]>& input, dim4_t input_strides,
                const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                float3_t center, float radius, float edge_size,
                Matrix inv_matrix, Functor functor, bool invert, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            inv_matrix = indexing::reorder(inv_matrix, order_3d);
        }

        using real_t = traits::value_type_t<Value>;
        if (invert) {
            using sphere_t = noa::signal::Sphere<3, real_t, true>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, real_t, true>;
            launch3D_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        } else {
            using sphere_t = noa::signal::Sphere<3, real_t, false>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, real_t, false>;
            launch3D_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename>
    void rectangle(const shared_t<Value[]>& input, dim4_t input_strides,
                   const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                   float3_t center, float3_t radius, float edge_size,
                   Matrix inv_matrix, Functor functor, bool invert, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
            inv_matrix = indexing::reorder(inv_matrix, order_3d);
        }

        using real_t = traits::value_type_t<Value>;
        if (invert) {
            using rectangle_t = noa::signal::Rectangle<3, real_t, true>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, real_t, true>;
            launch3D_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        } else {
            using rectangle_t = noa::signal::Rectangle<3, real_t, false>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, real_t, false>;
            launch3D_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename>
    void cylinder(const shared_t<Value[]>& input, dim4_t input_strides,
                  const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                  float3_t center, float radius, float length, float edge_size,
                  Matrix inv_matrix, Functor functor, bool invert, Stream& stream) {
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[1], center[2]);
            inv_matrix = indexing::reorder(inv_matrix, dim3_t{0, order_2d[0] + 1, order_2d[1] + 1});
        }

        const float3_t radius_{length, radius, radius};
        using real_t = traits::value_type_t<Value>;
        if (invert) {
            using cylinder_t = noa::signal::Cylinder<real_t, true>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<real_t, true>;
            launch3D_<REMAP, cylinder_t, cylinder_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius_, edge_size, inv_matrix, functor, invert, stream);
        } else {
            using cylinder_t = noa::signal::Cylinder<real_t, false>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<real_t, false>;
            launch3D_<REMAP, cylinder_t, cylinder_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius_, edge_size, inv_matrix, functor, invert, stream);
        }
    }

    #define NOA_INSTANTIATE_SHAPE_3D_(R, T, M, F)   \
    template void ellipse<R, T, M, F, void>(        \
        const shared_t<T[]>&, dim4_t,               \
        const shared_t<T[]>&, dim4_t,               \
        dim4_t, float3_t, float3_t, float,          \
        M, F, bool, Stream&);                       \
    template void sphere<R, T, M, F, void>(         \
        const shared_t<T[]>&, dim4_t,               \
        const shared_t<T[]>&, dim4_t,               \
        dim4_t, float3_t, float, float,             \
        M, F, bool, Stream&);                       \
    template void rectangle<R, T, M, F, void>(      \
        const shared_t<T[]>&, dim4_t,               \
        const shared_t<T[]>&, dim4_t,               \
        dim4_t, float3_t, float3_t, float,          \
        M, F, bool, Stream&);                       \
    template void cylinder<R, T, M, F, void>(       \
        const shared_t<T[]>&, dim4_t,               \
        const shared_t<T[]>&, dim4_t,               \
        dim4_t, float3_t, float, float, float,      \
        M, F, bool, Stream&)

    #define NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, M)         \
    NOA_INSTANTIATE_SHAPE_3D_(R, T, M, math::multiply_t);   \
    NOA_INSTANTIATE_SHAPE_3D_(R, T, M, math::plus_t)

    #define NOA_INSTANTIATE_SHAPE_MATRIX_(R, T)         \
    NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, float33_t);    \
    NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, float34_t)

    #define NOA_INSTANTIATE_SHAPE_ALL(T)            \
    NOA_INSTANTIATE_SHAPE_MATRIX_(fft::F2F, T);     \
    NOA_INSTANTIATE_SHAPE_MATRIX_(fft::FC2FC, T);   \
    NOA_INSTANTIATE_SHAPE_MATRIX_(fft::F2FC, T);    \
    NOA_INSTANTIATE_SHAPE_MATRIX_(fft::FC2F, T)

    NOA_INSTANTIATE_SHAPE_ALL(float);
    NOA_INSTANTIATE_SHAPE_ALL(double);
    NOA_INSTANTIATE_SHAPE_ALL(cfloat_t);
    NOA_INSTANTIATE_SHAPE_ALL(cdouble_t);
}
