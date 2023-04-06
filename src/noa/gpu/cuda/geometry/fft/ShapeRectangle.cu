#include "noa/gpu/cuda/geometry/fft/Shape.cuh"

namespace noa::cuda::geometry::fft {
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void rectangle(const Value* input, Strides4<i64> input_strides,
                   Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                   Vec2<f32> center, Vec2<f32> radius, f32 edge_size,
                   const Matrix& inv_matrix, Functor functor, CValue cvalue, bool invert, Stream& stream) {
        Matrix matrices = inv_matrix;
        if constexpr (!std::is_pointer_v<Matrix>) {
            const auto order_2d = noa::indexing::order(output_strides.filter(2, 3), shape.filter(2, 3));
            if (noa::any(order_2d != Vec2<i64>{0, 1})) {
                std::swap(input_strides[2], input_strides[3]);
                std::swap(output_strides[2], output_strides[3]);
                std::swap(shape[2], shape[3]);
                std::swap(center[0], center[1]);
                std::swap(radius[0], radius[1]);
                matrices = noa::indexing::reorder(matrices, order_2d);
            }
        }

        if (invert) {
            using rectangle_t = noa::geometry::Rectangle<2, CValue, true>;
            using rectangle_smooth_t = noa::geometry::RectangleSmooth<2, CValue, true>;
            launch_2d_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, matrices, functor,
                    cvalue, invert, stream);
        } else {
            using rectangle_t = noa::geometry::Rectangle<2, CValue, false>;
            using rectangle_smooth_t = noa::geometry::RectangleSmooth<2, CValue, false>;
            launch_2d_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, matrices, functor,
                    cvalue, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void rectangle(const Value* input, Strides4<i64> input_strides,
                   Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                   Vec3<f32> center, Vec3<f32> radius, f32 edge_size,
                   const Matrix& inv_matrix, Functor functor, CValue cvalue,
                   bool invert, Stream& stream) {

        Matrix matrices = inv_matrix;
        if constexpr (!std::is_pointer_v<Matrix>) {
            const auto order_3d = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
            if (noa::any(order_3d != Vec3<i64>{0, 1, 2})) {
                const auto order = (order_3d + 1).push_front(0);
                input_strides = noa::indexing::reorder(input_strides, order);
                output_strides = noa::indexing::reorder(output_strides, order);
                shape = noa::indexing::reorder(shape, order);
                center = noa::indexing::reorder(center, order_3d);
                radius = noa::indexing::reorder(radius, order_3d);
                matrices = noa::indexing::reorder(matrices, order_3d);
            }
        }

        if (invert) {
            using rectangle_t = noa::geometry::Rectangle<3, CValue, true>;
            using rectangle_smooth_t = noa::geometry::RectangleSmooth<3, CValue, true>;
            launch_3d_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, matrices, functor,
                    cvalue, invert, stream);
        } else {
            using rectangle_t = noa::geometry::Rectangle<3, CValue, false>;
            using rectangle_smooth_t = noa::geometry::RectangleSmooth<3, CValue, false>;
            launch_3d_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, matrices, functor,
                    cvalue, invert, stream);
        }
    }

    #define NOA_INSTANTIATE_SHAPE_2D_(R, T, C, F, M)\
    template void rectangle<R, T, M, F, C, void>(   \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>, Shape4<i64>,             \
        Vec2<f32>, Vec2<f32>, f32,                  \
        M const&, F, C, bool, Stream&)

    #define NOA_INSTANTIATE_SHAPE_3D_(R, T, C, F, M)\
    template void rectangle<R, T, M, F, C, void>(   \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>,                          \
        Shape4<i64>, Vec3<f32>, Vec3<f32>, f32,     \
        M const&, F, C, bool, Stream&)

    #define NOA_INSTANTIATE_SHAPE_MATRIX_(R, T, C, F)       \
    NOA_INSTANTIATE_SHAPE_2D_(R, T, C, F, Float22);         \
    NOA_INSTANTIATE_SHAPE_2D_(R, T, C, F, Float23);         \
    NOA_INSTANTIATE_SHAPE_2D_(R, T, C, F, const Float22*);  \
    NOA_INSTANTIATE_SHAPE_2D_(R, T, C, F, const Float23*);  \
    NOA_INSTANTIATE_SHAPE_3D_(R, T, C, F, Float33);         \
    NOA_INSTANTIATE_SHAPE_3D_(R, T, C, F, Float34);         \
    NOA_INSTANTIATE_SHAPE_3D_(R, T, C, F, const Float33*);  \
    NOA_INSTANTIATE_SHAPE_3D_(R, T, C, F, const Float34*)

    #define NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, C)             \
    NOA_INSTANTIATE_SHAPE_MATRIX_(R, T, C, noa::multiply_t);    \
    NOA_INSTANTIATE_SHAPE_MATRIX_(R, T, C, noa::plus_t)

    #define NOA_INSTANTIATE_SHAPE_ALL(T, C)                 \
    NOA_INSTANTIATE_SHAPE_FUNCTOR_(noa::fft::F2F, T, C);    \
    NOA_INSTANTIATE_SHAPE_FUNCTOR_(noa::fft::FC2FC, T, C)

    NOA_INSTANTIATE_SHAPE_ALL(f32, f32);
    NOA_INSTANTIATE_SHAPE_ALL(f64, f64);
    NOA_INSTANTIATE_SHAPE_ALL(c32, f32);
    NOA_INSTANTIATE_SHAPE_ALL(c64, f64);
}
