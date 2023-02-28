#include "noa/core/geometry/Shape.hpp"
#include "noa/algorithms/geometry/Shape.hpp"
#include "noa/gpu/cuda/geometry/fft/Shape.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace {
    using namespace noa;

    template<fft::Remap REMAP, typename GeomShape, typename GeomShapeSmooth,
             typename Functor, typename Matrix, typename Value, typename CValue, typename Radius>
    void launch_3d_(const Value* input, const Strides4<i64>& input_strides,
                    Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                    const Vec3<f32>& center, const Radius& radius, f32 edge_size,
                    const Matrix& inv_matrix, const Functor& functor, CValue cvalue,
                    bool invert, cuda::Stream& stream) {
        NOA_ASSERT_DEVICE_OR_NULL_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        Vec3<i32> start{0};
        auto end = shape.pop_front().as_safe<i32>().vec();
        if (input == output && inv_matrix == Matrix{} &&
            ((std::is_same_v<Functor, noa::multiply_t> && invert && cvalue == CValue{1}) ||
             (std::is_same_v<Functor, noa::plus_t> && !invert))) {
            start = noa::math::clamp(Vec3<i32>(center - (radius + edge_size)), Vec3<i32>{}, end);
            end = noa::math::clamp(Vec3<i32>(center + (radius + edge_size) + 1), Vec3<i32>{}, end);
            if (noa::any(end <= start))
                return;
        }

        const auto input_accessor = Accessor<const Value, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = Accessor<Value, 4, u32>(output, output_strides.as_safe<u32>());
        const auto shape_3d = shape.as_safe<i32>();

        if (edge_size > 1e-5f) {
            const GeomShapeSmooth geom_shape_smooth(center, radius, edge_size, cvalue);
            if (Matrix{} == inv_matrix) {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape_smooth, Empty{}, functor);
                noa::cuda::utils::iwise_3d("geometric_shape_3d", start, end, kernel, stream);
            } else {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape_smooth, inv_matrix, functor);
                noa::cuda::utils::iwise_3d("geometric_shape_3d", start, end, kernel, stream);
            }
        } else {
            const GeomShape geom_shape(center, radius, cvalue);
            if (Matrix{} == inv_matrix) {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape, Empty{}, functor);
                noa::cuda::utils::iwise_3d("geometric_shape_3d", start, end, kernel, stream);
            } else {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape, inv_matrix, functor);
                noa::cuda::utils::iwise_3d("geometric_shape_3d", start, end, kernel, stream);
            }
        }
    }
}

namespace noa::cuda::geometry::fft {
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void ellipse(const Value* input, Strides4<i64> input_strides,
                 Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                 Vec3<f32> center, Vec3<f32> radius, f32 edge_size,
                 Matrix inv_matrix, Functor functor, CValue cvalue,
                 bool invert, Stream& stream) {
        const auto order_3d = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
        if (noa::any(order_3d != Vec3<i64>{0, 1, 2})) {
            const auto order = (order_3d + 1).push_front(0);
            input_strides = noa::indexing::reorder(input_strides, order);
            output_strides = noa::indexing::reorder(output_strides, order);
            shape = noa::indexing::reorder(shape, order);
            center = noa::indexing::reorder(center, order_3d);
            radius = noa::indexing::reorder(radius, order_3d);
            inv_matrix = noa::indexing::reorder(inv_matrix, order_3d);
        }

        if (invert) {
            using ellipse_t = noa::signal::Ellipse<3, CValue, true>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, CValue, true>;
            launch_3d_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor,
                    cvalue, invert, stream);
        } else {
            using ellipse_t = noa::signal::Ellipse<3, CValue, false>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, CValue, false>;
            launch_3d_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor,
                    cvalue, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void sphere(const Value* input, Strides4<i64> input_strides,
                Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                Vec3<f32> center, f32 radius, f32 edge_size,
                Matrix inv_matrix, Functor functor, CValue cvalue,
                bool invert, Stream& stream) {
        const auto order_3d = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
        if (noa::any(order_3d != Vec3<i64>{0, 1, 2})) {
            const auto order = (order_3d + 1).push_front(0);
            input_strides = noa::indexing::reorder(input_strides, order);
            output_strides = noa::indexing::reorder(output_strides, order);
            shape = noa::indexing::reorder(shape, order);
            center = noa::indexing::reorder(center, order_3d);
            inv_matrix = noa::indexing::reorder(inv_matrix, order_3d);
        }

        if (invert) {
            using sphere_t = noa::signal::Sphere<3, CValue, true>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, CValue, true>;
            launch_3d_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor,
                    cvalue, invert, stream);
        } else {
            using sphere_t = noa::signal::Sphere<3, CValue, false>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, CValue, false>;
            launch_3d_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor,
                    cvalue, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void rectangle(const Value* input, Strides4<i64> input_strides,
                   Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                   Vec3<f32> center, Vec3<f32> radius, f32 edge_size,
                   Matrix inv_matrix, Functor functor, CValue cvalue,
                   bool invert, Stream& stream) {
        const auto order_3d = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
        if (noa::any(order_3d != Vec3<i64>{0, 1, 2})) {
            const auto order = (order_3d + 1).push_front(0);
            input_strides = noa::indexing::reorder(input_strides, order);
            output_strides = noa::indexing::reorder(output_strides, order);
            shape = noa::indexing::reorder(shape, order);
            center = noa::indexing::reorder(center, order_3d);
            radius = noa::indexing::reorder(radius, order_3d);
            inv_matrix = noa::indexing::reorder(inv_matrix, order_3d);
        }

        if (invert) {
            using rectangle_t = noa::signal::Rectangle<3, CValue, true>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, CValue, true>;
            launch_3d_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor,
                    cvalue, invert, stream);
        } else {
            using rectangle_t = noa::signal::Rectangle<3, CValue, false>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, CValue, false>;
            launch_3d_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor,
                    cvalue, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void cylinder(const Value* input, Strides4<i64> input_strides,
                  Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                  Vec3<f32> center, f32 radius, f32 length, f32 edge_size,
                  Matrix inv_matrix, Functor functor, CValue cvalue,
                  bool invert, Stream& stream) {
        const auto order_2d = noa::indexing::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (noa::any(order_2d != Vec2<i64>{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[1], center[2]);
            inv_matrix = noa::indexing::reorder(inv_matrix, (order_2d + 1).push_front(0));
        }

        const auto radius_ = Vec3<f32>{length, radius, radius};
        if (invert) {
            using cylinder_t = noa::signal::Cylinder<CValue, true>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<CValue, true>;
            launch_3d_<REMAP, cylinder_t, cylinder_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius_, edge_size, inv_matrix, functor,
                    cvalue, invert, stream);
        } else {
            using cylinder_t = noa::signal::Cylinder<CValue, false>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<CValue, false>;
            launch_3d_<REMAP, cylinder_t, cylinder_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius_, edge_size, inv_matrix, functor,
                    cvalue, invert, stream);
        }
    }

    #define NOA_INSTANTIATE_SHAPE_3D_(R, T, C, M, F)\
    template void ellipse<R, T, M, F, C, void>(     \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>,                          \
        Shape4<i64>, Vec3<f32>, Vec3<f32>, f32,     \
        M, F, C, bool, Stream&);                    \
    template void sphere<R, T, M, F, C, void>(      \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>,                          \
        Shape4<i64>, Vec3<f32>, f32, f32,           \
        M, F, C, bool, Stream&);                    \
    template void rectangle<R, T, M, F, C, void>(   \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>,                          \
        Shape4<i64>, Vec3<f32>, Vec3<f32>, f32,     \
        M, F, C, bool, Stream&);                    \
    template void cylinder<R, T, M, F, C, void>(    \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>,                          \
        Shape4<i64>, Vec3<f32>, f32, f32, f32,      \
        M, F, C, bool, Stream&)

    #define NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, C, M)      \
    NOA_INSTANTIATE_SHAPE_3D_(R, T, C, M, noa::multiply_t); \
    NOA_INSTANTIATE_SHAPE_3D_(R, T, C, M, noa::plus_t)

    #define NOA_INSTANTIATE_SHAPE_MATRIX_(R, T, C)      \
    NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, C, Float33);   \
    NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, C, Float34)

    #define NOA_INSTANTIATE_SHAPE_ALL(T, C)                 \
    NOA_INSTANTIATE_SHAPE_MATRIX_(noa::fft::F2F, T, C);     \
    NOA_INSTANTIATE_SHAPE_MATRIX_(noa::fft::FC2FC, T, C);   \
    NOA_INSTANTIATE_SHAPE_MATRIX_(noa::fft::F2FC, T, C);    \
    NOA_INSTANTIATE_SHAPE_MATRIX_(noa::fft::FC2F, T, C)

    NOA_INSTANTIATE_SHAPE_ALL(f32, f32);
    NOA_INSTANTIATE_SHAPE_ALL(f64, f64);
    NOA_INSTANTIATE_SHAPE_ALL(c32, f32);
    NOA_INSTANTIATE_SHAPE_ALL(c64, f64);
}
