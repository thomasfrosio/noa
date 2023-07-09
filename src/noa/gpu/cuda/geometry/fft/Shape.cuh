#include "noa/core/geometry/Shape.hpp"
#include "noa/algorithms/geometry/Shape.hpp"
#include "noa/gpu/cuda/geometry/fft/Shape.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace {
    using namespace noa;

    template<fft::Remap REMAP, typename GeomShape, typename GeomShapeSmooth,
             typename Functor, typename Matrix, typename Value, typename CValue, typename Radius>
    void launch_2d_(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Vec2<f32>& center, const Radius& radius, f32 edge_size,
            const Matrix& inv_matrix, const Functor& functor, CValue cvalue,
            bool invert, noa::cuda::Stream& stream
    ) {
        NOA_ASSERT_DEVICE_OR_NULL_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        if constexpr (std::is_pointer_v<Matrix>) {
            NOA_ASSERT_DEVICE_OR_NULL_PTR(inv_matrix, stream.device());
        }

        Vec2<i32> start{0};
        auto end = shape.filter(2, 3).as_safe<i32>().vec();
        if (input == output && inv_matrix == Matrix{} &&
            ((std::is_same_v<Functor, noa::multiply_t> && invert && cvalue == CValue{1}) ||
             (std::is_same_v<Functor, noa::plus_t> && !invert))) {
            start = noa::math::clamp(Vec2<i32>(center - (radius + edge_size)), Vec2<i32>{}, end);
            end = noa::math::clamp(Vec2<i32>(center + (radius + edge_size) + 1), Vec2<i32>{}, end);
            if (noa::any(end <= start))
                return;
        }

        const auto input_accessor = Accessor<const Value, 3, u32>(input, input_strides.filter(0, 2, 3).as_safe<u32>());
        const auto output_accessor = Accessor<Value, 3, u32>(output, output_strides.filter(0, 2, 3).as_safe<u32>());
        const auto shape_2d = shape.filter(0, 2, 3).as_safe<i32>();
        const bool is_multiple_shapes_case = noa::algorithm::geometry::is_multiple_shapes_case(
                input_strides, output_strides, shape, inv_matrix);

        if (edge_size > 1e-5f) {
            const GeomShapeSmooth geom_shape_smooth(center, radius, edge_size, cvalue);
            if constexpr (std::is_pointer_v<Matrix>) {
                if (is_multiple_shapes_case) {
                    const auto kernel = noa::algorithm::geometry::shape_2d<REMAP, f32>(
                            input_accessor, output_accessor, shape_2d, geom_shape_smooth, inv_matrix, functor);
                    return noa::cuda::utils::iwise_2d(start, end, kernel, stream);
                }
            }
            const auto start_3d = start.push_front(0);
            const auto end_3d = end.push_front(shape_2d[0]);
            if (Matrix{} == inv_matrix) {
                const auto kernel = noa::algorithm::geometry::shape_2d<REMAP, f32>(
                        input_accessor, output_accessor, shape_2d, geom_shape_smooth, Empty{}, functor);
                noa::cuda::utils::iwise_3d(start_3d, end_3d, kernel, stream);
            } else {
                const auto kernel = noa::algorithm::geometry::shape_2d<REMAP, f32>(
                        input_accessor, output_accessor, shape_2d, geom_shape_smooth, inv_matrix, functor);
                noa::cuda::utils::iwise_3d(start_3d, end_3d, kernel, stream);
            }
        } else {
            const GeomShape geom_shape(center, radius, cvalue);
            if constexpr (std::is_pointer_v<Matrix>) {
                if (is_multiple_shapes_case) {
                    const auto kernel = noa::algorithm::geometry::shape_2d<REMAP, f32>(
                            input_accessor, output_accessor, shape_2d, geom_shape, inv_matrix, functor);
                    return noa::cuda::utils::iwise_2d(start, end, kernel, stream);
                }
            }
            const auto start_3d = start.push_front(0);
            const auto end_3d = end.push_front(shape_2d[0]);
            if (Matrix{} == inv_matrix) {
                const auto kernel = noa::algorithm::geometry::shape_2d<REMAP, f32>(
                        input_accessor, output_accessor, shape_2d, geom_shape, Empty{}, functor);
                noa::cuda::utils::iwise_3d(start_3d, end_3d, kernel, stream);
            } else {
                const auto kernel = noa::algorithm::geometry::shape_2d<REMAP, f32>(
                        input_accessor, output_accessor, shape_2d, geom_shape, inv_matrix, functor);
                noa::cuda::utils::iwise_3d(start_3d, end_3d, kernel, stream);
            }
        }
    }

    template<fft::Remap REMAP, typename GeomShape, typename GeomShapeSmooth,
             typename Functor, typename Matrix, typename Value, typename CValue, typename Radius>
    void launch_3d_(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Vec3<f32>& center, const Radius& radius, f32 edge_size,
            const Matrix& inv_matrix, const Functor& functor, CValue cvalue,
            bool invert, noa::cuda::Stream& stream
    ) {
        NOA_ASSERT_DEVICE_OR_NULL_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        if constexpr (std::is_pointer_v<Matrix>) {
            NOA_ASSERT_DEVICE_OR_NULL_PTR(inv_matrix, stream.device());
        }

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
        const bool is_multiple_shapes_case = noa::algorithm::geometry::is_multiple_shapes_case(
                input_strides, output_strides, shape, inv_matrix);

        if (edge_size > 1e-5f) {
            const GeomShapeSmooth geom_shape_smooth(center, radius, edge_size, cvalue);
            if constexpr (std::is_pointer_v<Matrix>) {
                if (is_multiple_shapes_case) {
                    const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                            input_accessor, output_accessor, shape_3d, geom_shape_smooth, inv_matrix, functor);
                    return noa::cuda::utils::iwise_3d(start, end, kernel, stream);
                }
            }
            const auto start_4d = start.push_front(0);
            const auto end_4d = end.push_front(shape_3d[0]);
            if (Matrix{} == inv_matrix) {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape_smooth, Empty{}, functor);
                noa::cuda::utils::iwise_4d(start_4d, end_4d, kernel, stream);
            } else {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape_smooth, inv_matrix, functor);
                noa::cuda::utils::iwise_4d(start_4d, end_4d, kernel, stream);
            }
        } else {
            const GeomShape geom_shape(center, radius, cvalue);
            if constexpr (std::is_pointer_v<Matrix>) {
                if (is_multiple_shapes_case) {
                    const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                            input_accessor, output_accessor, shape_3d, geom_shape, inv_matrix, functor);
                    return noa::cuda::utils::iwise_3d(start, end, kernel, stream);
                }
            }
            const auto start_4d = start.push_front(0);
            const auto end_4d = end.push_front(shape_3d[0]);
            if (Matrix{} == inv_matrix) {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape, Empty{}, functor);
                noa::cuda::utils::iwise_4d(start_4d, end_4d, kernel, stream);
            } else {
                const auto kernel = noa::algorithm::geometry::shape_3d<REMAP, f32>(
                        input_accessor, output_accessor, shape_3d, geom_shape, inv_matrix, functor);
                noa::cuda::utils::iwise_4d(start_4d, end_4d, kernel, stream);
            }
        }
    }
}
