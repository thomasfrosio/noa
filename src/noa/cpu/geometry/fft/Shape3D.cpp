#include "noa/core/geometry/Shape.hpp"
#include "noa/algorithms/geometry/Shape.hpp"
#include "noa/cpu/geometry/fft/Shape.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace {
    using namespace noa;

    template<noa::fft::Remap REMAP, typename GeomShape, typename GeomShapeSmooth,
             typename Functor, typename Value, typename CValue, typename Radius, typename Matrix>
    void launch_3d_(const Value* input, const Strides4<i64>& input_strides,
                    Value* output, const Strides4<i64>& output_strides,
                    const Vec4<i64>& start, const Vec4<i64>& end, const Shape4<i64>& shape,
                    const Vec3<f32>& center, const Radius& radius, f32 edge_size,
                    const Matrix& inv_matrix, const Functor& functor, CValue cvalue, i64 threads) {

        const auto input_accessor = Accessor<const Value, 4, i64>(input, input_strides);
        const auto output_accessor = Accessor<Value, 4, i64>(output, output_strides);

        const bool has_smooth_edge = edge_size > 1e-5f;
        if (has_smooth_edge) {
            const GeomShapeSmooth geom_shape_smooth(center, radius, edge_size, cvalue);
            if (Matrix{} == inv_matrix) {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape, geom_shape_smooth, Empty{}, functor);
                noa::cpu::utils::iwise_4d(start, end, kernel, threads);
            } else {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape, geom_shape_smooth, inv_matrix, functor);
                noa::cpu::utils::iwise_4d(start, end, kernel, threads);
            }
        } else {
            const GeomShape geom_shape(center, radius, cvalue);
            if (Matrix{} == inv_matrix) {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape, geom_shape, Empty{}, functor);
                noa::cpu::utils::iwise_4d(start, end, kernel, threads);
            } else {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape, geom_shape, inv_matrix, functor);
                noa::cpu::utils::iwise_4d(start, end, kernel, threads);
            }
        }
    }

    // Find the smallest [start, end) range for each dimension (2D or 3D).
    template<typename Functor, typename Value, typename Center,
             typename Radius, typename Matrix, typename CValue>
    auto get_iwise_subregion_(const Value* input, Value* output, const Shape4<i64>& shape,
                              const Center& center, const Radius& radius, f32 edge_size,
                              const Matrix& inv_matrix, CValue cvalue, bool invert) {

        // TODO Rotate the boundary box to support matrix?
        if (input == output && inv_matrix == Matrix{} &&
            ((std::is_same_v<Functor, noa::multiply_t> && invert && cvalue == CValue{1}) ||
             (std::is_same_v<Functor, noa::plus_t> && !invert))) {
            const auto shape_3d = shape.filter(1, 2, 3).vec();
            const auto start = noa::math::clamp(Vec3<i64>(center - (radius + edge_size)), Vec3<i64>{}, shape_3d);
            const auto end = noa::math::clamp(Vec3<i64>(center + (radius + edge_size) + 1), Vec3<i64>{}, shape_3d);
            return std::pair{start.push_front(0),
                             end.push_front(shape[0])};
        } else {
            return std::pair{Vec4<i64>{0}, shape.vec()};
        }
    }
}

namespace noa::cpu::geometry::fft {
    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void ellipse(const Value* input, Strides4<i64> input_strides,
                 Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                 Vec3<f32> center, Vec3<f32> radius, f32 edge_size,
                 Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        const auto order_3d = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
        if (noa::any(order_3d != Vec3<i64>{0, 1, 2})) {
            const auto order = (order_3d + 1).push_front(0);
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
            inv_matrix = indexing::reorder(inv_matrix, order_3d);
        }

        const auto[start, end] = get_iwise_subregion_<Functor>(
                input, output, shape, center, radius, edge_size, inv_matrix, cvalue, invert);
        if (noa::any(end <= start))
            return;

        if (invert) {
            using ellipse_t = noa::signal::Ellipse<3, CValue, true>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, CValue, true>;
                launch_3d_<REMAP, ellipse_t, ellipse_smooth_t>(
                        input, input_strides, output, output_strides,
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, cvalue, threads);
        } else {
            using ellipse_t = noa::signal::Ellipse<3, CValue, false>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, CValue, false>;
                launch_3d_<REMAP, ellipse_t, ellipse_smooth_t>(
                        input, input_strides, output, output_strides,
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, cvalue, threads);
        }
    }

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void sphere(const Value* input, Strides4<i64> input_strides,
                Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                Vec3<f32> center, f32 radius, f32 edge_size,
                Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        const auto order_3d = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
        if (noa::any(order_3d != Vec3<i64>{0, 1, 2})) {
            const auto order = (order_3d + 1).push_front(0);
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            inv_matrix = indexing::reorder(inv_matrix, order_3d);
        }

        const auto[start, end] = get_iwise_subregion_<Functor>(
                input, output, shape, center, radius, edge_size, inv_matrix, cvalue, invert);
        if (noa::any(end <= start))
            return;

        if (invert) {
            using sphere_t = noa::signal::Sphere<3, CValue, true>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, CValue, true>;
            launch_3d_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, radius, edge_size,
                    inv_matrix, functor, cvalue, threads);
        } else {
            using sphere_t = noa::signal::Sphere<3, CValue, false>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, CValue, false>;
            launch_3d_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, radius, edge_size,
                    inv_matrix, functor, cvalue, threads);
        }
    }

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void rectangle(const Value* input, Strides4<i64> input_strides,
                   Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                   Vec3<f32> center, Vec3<f32> radius, f32 edge_size,
                   Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        const auto order_3d = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
        if (noa::any(order_3d != Vec3<i64>{0, 1, 2})) {
            const auto order = (order_3d + 1).push_front(0);
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
            inv_matrix = indexing::reorder(inv_matrix, order_3d);
        }

        const auto[start, end] = get_iwise_subregion_<Functor>(
                input, output, shape, center, radius, edge_size, inv_matrix, cvalue, invert);
        if (noa::any(end <= start))
            return;

        if (invert) {
            using rectangle_t = noa::signal::Rectangle<3, CValue, true>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, CValue, true>;
            launch_3d_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, radius, edge_size,
                    inv_matrix, functor, cvalue, threads);
        } else {
            using rectangle_t = noa::signal::Rectangle<3, CValue, false>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, CValue, false>;
            launch_3d_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, radius, edge_size,
                    inv_matrix, functor, cvalue, threads);
        }
    }

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void cylinder(const Value* input, Strides4<i64> input_strides,
                  Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                  Vec3<f32> center, f32 radius, f32 length, f32 edge_size,
                  Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        const auto order_2d = noa::indexing::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (noa::any(order_2d != Vec2<i64>{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[1], center[2]);
            inv_matrix = noa::indexing::reorder(inv_matrix, (order_2d + 1).push_front(0));
        }

        const auto radius_ = Vec3<f32>{length, radius, radius};
        const auto[start, end] = get_iwise_subregion_<Functor>(
                input, output, shape, center, radius_, edge_size, inv_matrix, cvalue, invert);
        if (noa::any(end <= start))
            return;

        if (invert) {
            using cylinder_t = noa::signal::Cylinder<CValue, true>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<CValue, true>;
            launch_3d_<REMAP, cylinder_t, cylinder_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, Vec2<f32>{length, radius}, edge_size,
                    inv_matrix, functor, cvalue, threads);
        } else {
            using cylinder_t = noa::signal::Cylinder<CValue, false>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<CValue, false>;
            launch_3d_<REMAP, cylinder_t, cylinder_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, Vec2<f32>{length, radius}, edge_size,
                    inv_matrix, functor, cvalue, threads);
        }
    }

    #define NOA_INSTANTIATE_SHAPE_3D_(R, T, C, M, F)\
    template void ellipse<R, T, M, F, C, void>(     \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>,                          \
        Shape4<i64>, Vec3<f32>, Vec3<f32>, f32,     \
        M, F, C, bool, i64);                        \
    template void sphere<R, T, M, F, C, void>(      \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>,                          \
        Shape4<i64>, Vec3<f32>, f32, f32,           \
        M, F, C, bool, i64);                        \
    template void rectangle<R, T, M, F, C, void>(   \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>,                          \
        Shape4<i64>, Vec3<f32>, Vec3<f32>, f32,     \
        M, F, C, bool, i64);                        \
    template void cylinder<R, T, M, F, C, void>(    \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>,                          \
        Shape4<i64>, Vec3<f32>, f32, f32, f32,      \
        M, F, C, bool, i64)

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
