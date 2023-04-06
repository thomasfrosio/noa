#include "noa/gpu/cuda/geometry/fft/Shape.cuh"

namespace noa::cuda::geometry::fft {
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename CValue, typename>
    void cylinder(const Value* input, Strides4<i64> input_strides,
                  Value* output, Strides4<i64> output_strides, Shape4<i64> shape,
                  Vec3<f32> center, f32 radius, f32 length, f32 edge_size,
                  const Matrix& inv_matrix, Functor functor, CValue cvalue,
                  bool invert, Stream& stream) {
        Matrix matrices = inv_matrix;
        if constexpr (!std::is_pointer_v<Matrix>) {
            const auto order_2d = noa::indexing::order(output_strides.filter(2, 3), shape.filter(2, 3));
            if (noa::any(order_2d != Vec2<i64>{0, 1})) {
                std::swap(input_strides[2], input_strides[3]);
                std::swap(output_strides[2], output_strides[3]);
                std::swap(shape[2], shape[3]);
                std::swap(center[1], center[2]);
                matrices = noa::indexing::reorder(matrices, (order_2d + 1).push_front(0));
            }
        }

        const auto radius_ = Vec3<f32>{length, radius, radius};
        if (invert) {
            using cylinder_t = noa::geometry::Cylinder<CValue, true>;
            using cylinder_smooth_t = noa::geometry::CylinderSmooth<CValue, true>;
            launch_3d_<REMAP, cylinder_t, cylinder_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius_, edge_size, matrices, functor,
                    cvalue, invert, stream);
        } else {
            using cylinder_t = noa::geometry::Cylinder<CValue, false>;
            using cylinder_smooth_t = noa::geometry::CylinderSmooth<CValue, false>;
            launch_3d_<REMAP, cylinder_t, cylinder_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius_, edge_size, matrices, functor,
                    cvalue, invert, stream);
        }
    }

    #define NOA_INSTANTIATE_SHAPE_3D_(R, T, C, F, M)\
    template void cylinder<R, T, M, F, C, void>(    \
        const T*, Strides4<i64>,                    \
        T*, Strides4<i64>,                          \
        Shape4<i64>, Vec3<f32>, f32, f32, f32,      \
        M const&, F, C, bool, Stream&)

    #define NOA_INSTANTIATE_SHAPE_MATRIX_(R, T, C, F)       \
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
