#include "noa/core/geometry/Shape.hpp"
#include "noa/algorithms/geometry/Shape.hpp"
#include "noa/cpu/geometry/fft/Shape.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace {
    using namespace noa;

    template<noa::fft::Remap REMAP, typename GeomShape, typename GeomShapeSmooth,
             typename Functor, typename Value, typename CValue, typename Radius, typename Matrix>
    void launch_2d_(const Value* input, const Strides4<i64>& input_strides,
                    Value* output, const Strides4<i64>& output_strides,
                    const Vec3<i64>& start, const Vec3<i64>& end, const Shape4<i64>& shape,
                    const Vec2<f32>& center, const Radius& radius, f32 edge_size,
                    const Matrix& inv_matrix, const Functor& functor, CValue cvalue, i64 threads) {

        const auto input_accessor = Accessor<const Value, 3, i64>(input, input_strides.filter(0, 2, 3));
        const auto output_accessor = Accessor<Value, 3, i64>(output, output_strides.filter(0, 2, 3));
        const auto shape_3d = shape.filter(0, 2, 3);
        const bool has_smooth_edge = edge_size > 1e-5f;
        if (has_smooth_edge) {
            const GeomShapeSmooth geom_shape_smooth(center, radius, edge_size, cvalue);
            if (Matrix{} == inv_matrix) {
                const auto kernel = noa::algorithm::geometry::shape_2d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape_smooth, Empty{}, functor);
                noa::cpu::utils::iwise_3d(start, end, kernel, threads);
            } else {
                const auto kernel = noa::algorithm::geometry::shape_2d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape_smooth, inv_matrix, functor);
                noa::cpu::utils::iwise_3d(start, end, kernel, threads);
            }
        } else {
            const GeomShape geom_shape(center, radius, cvalue);
            if (Matrix{} == inv_matrix) {
                const auto kernel = noa::algorithm::geometry::shape_2d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape, Empty{}, functor);
                noa::cpu::utils::iwise_3d(start, end, kernel, threads);
            } else {
                const auto kernel = noa::algorithm::geometry::shape_2d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape, inv_matrix, functor);
                noa::cpu::utils::iwise_3d(start, end, kernel, threads);
            }
        }
    }

    // Find the smallest [start, end) range for each dimension (2D or 3D).
    template<typename Functor, typename Value, typename Center,
             typename Radius, typename Matrix, typename CValue>
    auto get_iwise_subregion_(const Value* input, Value* output, const Shape4<i64>& shape,
                          const Center& center, const Radius& radius, f32 edge_size,
                          const Matrix& inv_matrix, CValue cvalue, bool invert) {

        // In this case, we have to loop through the entire array.
        // TODO Rotate the boundary box to support matrix?
        if (input == output && inv_matrix == Matrix{} &&
            ((std::is_same_v<Functor, noa::multiply_t> && invert && cvalue == CValue{1}) ||
             (std::is_same_v<Functor, noa::plus_t> && !invert))) {
            const auto shape_2d = shape.filter(2, 3).vec();
            const auto start = noa::math::clamp(Vec2<i64>(center - (radius + edge_size)), Vec2<i64>{}, shape_2d);
            const auto end = noa::math::clamp(Vec2<i64>(center + (radius + edge_size) + 1), Vec2<i64>{}, shape_2d);
            return std::pair{start.push_front(0),
                             end.push_front(shape[0])};
        } else {
            return std::pair{Vec3<i64>{0}, shape.filter(0, 2, 3).vec()};
        }
    }
}

namespace noa::cpu::signal::fft {
    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void ellipse(const Value* input, Strides4<i64> input_strides,
                 Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                 Vec2<f32> center, Vec2<f32> radius, f32 edge_size,
                 Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        const auto order_2d = noa::indexing::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (noa::any(order_2d != Vec2<i64>{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            std::swap(radius[0], radius[1]);
            inv_matrix = noa::indexing::reorder(inv_matrix, order_2d);
        }

        const auto[start, end] = get_iwise_subregion_<Functor>(
                input, output, shape, center, radius, edge_size, inv_matrix, cvalue, invert);
        if (noa::any(end <= start))
            return;

        if (invert) {
            using ellipse_t = noa::signal::Ellipse<2, CValue, true>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<2, CValue, true>;
            launch_2d_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, radius, edge_size,
                    inv_matrix, functor, cvalue, threads);
        } else {
            using ellipse_t = noa::signal::Ellipse<2, CValue, false>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<2, CValue, false>;
            launch_2d_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, radius, edge_size,
                    inv_matrix, functor, cvalue, threads);
        }
    }

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void sphere(const Value* input, Strides4<i64> input_strides,
                Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                Vec2<f32> center, f32 radius, f32 edge_size,
                Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        const auto order_2d = noa::indexing::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (noa::any(order_2d != Vec2<i64>{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            inv_matrix = indexing::reorder(inv_matrix, order_2d);
        }

        const auto[start, end] = get_iwise_subregion_<Functor>(
                input, output, shape, center, radius, edge_size, inv_matrix, cvalue, invert);
        if (noa::any(end <= start))
            return;

        if (invert) {
            using sphere_t = noa::signal::Sphere<2, CValue, true>;
            using sphere_smooth_t = noa::signal::SphereSmooth<2, CValue, true>;
            launch_2d_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, radius, edge_size,
                    inv_matrix, functor, cvalue, threads);
        } else {
            using sphere_t = noa::signal::Sphere<2, CValue, false>;
            using sphere_smooth_t = noa::signal::SphereSmooth<2, CValue, false>;
            launch_2d_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, radius, edge_size,
                    inv_matrix, functor, cvalue, threads);
        }
    }

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void rectangle(const Value* input, Strides4<i64> input_strides,
                   Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                   Vec2<f32> center, Vec2<f32> radius, f32 edge_size,
                   Matrix inv_matrix, Functor functor, CValue cvalue, bool invert, i64 threads) {
        const auto order_2d = noa::indexing::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (noa::any(order_2d != Vec2<i64>{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            std::swap(radius[0], radius[1]);
            inv_matrix = indexing::reorder(inv_matrix, order_2d);
        }

        const auto[start, end] = get_iwise_subregion_<Functor>(
                input, output, shape, center, radius, edge_size, inv_matrix, cvalue, invert);
        if (noa::any(end <= start))
            return;

        if (invert) {
            using rectangle_t = noa::signal::Rectangle<2, CValue, true>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<2, CValue, true>;
            launch_2d_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, radius, edge_size,
                    inv_matrix, functor, cvalue, threads);
        } else {
            using rectangle_t = noa::signal::Rectangle<2, CValue, false>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<2, CValue, false>;
            launch_2d_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides,
                    start, end, shape, center, radius, edge_size,
                    inv_matrix, functor, cvalue, threads);
        }
    }

    #define NOA_INSTANTIATE_SHAPE_2D_(R, T, C, M, F)\
    template void ellipse<R, T, M, F, C, void>(     \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>, Shape4<i64>,             \
        Vec2<f32>, Vec2<f32>, f32,                  \
        M, F, C, bool, i64);                        \
    template void sphere<R, T, M, F, C, void>(      \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>, Shape4<i64>,             \
        Vec2<f32>, f32, f32,                        \
        M, F, C, bool, i64);                        \
    template void rectangle<R, T, M, F, C, void>(   \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>, Shape4<i64>,             \
        Vec2<f32>, Vec2<f32>, f32,                  \
        M, F, C, bool, i64)

    #define NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, C, M)      \
    NOA_INSTANTIATE_SHAPE_2D_(R, T, C, M, noa::multiply_t); \
    NOA_INSTANTIATE_SHAPE_2D_(R, T, C, M, noa::plus_t)

    #define NOA_INSTANTIATE_SHAPE_MATRIX_(R, T, C)      \
    NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, C, Float22);   \
    NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, C, Float23)

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
