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
    void launch2D_(const shared_t<Value[]>& input, const dim4_t& input_strides,
                   const shared_t<Value[]>& output, const dim4_t& output_strides, const dim4_t& shape,
                   const float2_t& center, const Radius& radius, float edge_size,
                   const Matrix& inv_matrix, const Functor& functor, bool invert, cuda::Stream& stream) {
        NOA_ASSERT_DEVICE_OR_NULL_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        int2_t start{0};
        auto end = safe_cast<int2_t>(dim2_t(shape.get(2)));
        if (invert && input.get() == output.get() && inv_matrix == Matrix{}) {
            start = noa::math::clamp(int2_t(center - (radius + edge_size)), int2_t{}, int2_t(end));
            end = noa::math::clamp(int2_t(center + (radius + edge_size) + 1), int2_t{}, int2_t(end));
            if (any(end <= start))
                return;
        }

        const auto i_strides_2d = dim3_t{input_strides[0], input_strides[2], input_strides[3]};
        const auto o_strides_2d = dim3_t{output_strides[0], output_strides[2], output_strides[3]};
        const auto input_accessor = Accessor<const Value, 3, uint32_t>(input.get(), safe_cast<uint3_t>(i_strides_2d));
        const auto output_accessor = Accessor<Value, 3, uint32_t>(output.get(), safe_cast<uint3_t>(o_strides_2d));
        const auto iwise_start = uint2_t(start);
        const auto iwise_end = uint2_t(end);
        const auto shape_2d = uint3_t{shape[0], shape[2], shape[3]};

        const bool has_smooth_edge = edge_size > 1e-5f;
        if (has_smooth_edge) {
            const GeomShapeSmooth geom_shape_smooth(center, radius, edge_size);
            if (Matrix{} == inv_matrix) {
                const auto kernel = signal::fft::details::shape2D<REMAP, uint32_t>(
                        input_accessor, output_accessor, shape_2d, geom_shape_smooth, empty_t{}, functor);
                cuda::utils::iwise2D("signal::shape", iwise_start, iwise_end, kernel, stream);
            } else {
                const auto kernel = signal::fft::details::shape2D<REMAP, uint32_t>(
                        input_accessor, output_accessor, shape_2d, geom_shape_smooth, inv_matrix, functor);
                cuda::utils::iwise2D("signal::shape", iwise_start, iwise_end, kernel, stream);
            }
        } else {
            const GeomShape geom_shape(center, radius);
            if (Matrix{} == inv_matrix) {
                const auto kernel = signal::fft::details::shape2D<REMAP, uint32_t>(
                        input_accessor, output_accessor, shape_2d, geom_shape, empty_t{}, functor);
                cuda::utils::iwise2D("signal::shape", iwise_start, iwise_end, kernel, stream);
            } else {
                const auto kernel = signal::fft::details::shape2D<REMAP, uint32_t>(
                        input_accessor, output_accessor, shape_2d, geom_shape, inv_matrix, functor);
                cuda::utils::iwise2D("signal::shape", iwise_start, iwise_end, kernel, stream);
            }
        }
    }
}

namespace noa::cuda::signal::fft {
    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename>
    void ellipse(const shared_t<Value[]>& input, dim4_t input_strides,
                 const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                 float2_t center, float2_t radius, float edge_size,
                 Matrix inv_matrix, Functor functor, bool invert, Stream& stream) {
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            std::swap(radius[0], radius[1]);
            inv_matrix = indexing::reorder(inv_matrix, order_2d);
        }

        using real_t = traits::value_type_t<Value>;
        if (invert) {
            using ellipse_t = noa::signal::Ellipse<2, real_t, true>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<2, real_t, true>;
            launch2D_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        } else {
            using ellipse_t = noa::signal::Ellipse<2, real_t, false>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<2, real_t, false>;
            launch2D_<REMAP, ellipse_t, ellipse_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename>
    void sphere(const shared_t<Value[]>& input, dim4_t input_strides,
                const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                float2_t center, float radius, float edge_size,
                Matrix inv_matrix, Functor functor, bool invert, Stream& stream) {
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            inv_matrix = indexing::reorder(inv_matrix, order_2d);
        }

        using real_t = traits::value_type_t<Value>;
        if (invert) {
            using sphere_t = noa::signal::Sphere<2, real_t, true>;
            using sphere_smooth_t = noa::signal::SphereSmooth<2, real_t, true>;
            launch2D_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        } else {
            using sphere_t = noa::signal::Sphere<2, real_t, false>;
            using sphere_smooth_t = noa::signal::SphereSmooth<2, real_t, false>;
            launch2D_<REMAP, sphere_t, sphere_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Functor, typename>
    void rectangle(const shared_t<Value[]>& input, dim4_t input_strides,
                   const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                   float2_t center, float2_t radius, float edge_size,
                   Matrix inv_matrix, Functor functor, bool invert, Stream& stream) {
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            std::swap(radius[0], radius[1]);
            inv_matrix = indexing::reorder(inv_matrix, order_2d);
        }

        using real_t = traits::value_type_t<Value>;
        if (invert) {
            using rectangle_t = noa::signal::Rectangle<2, real_t, true>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<2, real_t, true>;
            launch2D_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        } else {
            using rectangle_t = noa::signal::Rectangle<2, real_t, false>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<2, real_t, false>;
            launch2D_<REMAP, rectangle_t, rectangle_smooth_t>(
                    input, input_strides, output, output_strides, shape,
                    center, radius, edge_size, inv_matrix, functor, invert, stream);
        }
    }

    #define NOA_INSTANTIATE_SHAPE_2D_(R, T, M, F)   \
    template void ellipse<R, T, M, F, void>(        \
        const shared_t<T[]>&, dim4_t,               \
        const shared_t<T[]>&, dim4_t,               \
        dim4_t, float2_t, float2_t, float,          \
        M, F, bool, Stream&);                       \
    template void sphere<R, T, M, F, void>(         \
        const shared_t<T[]>&, dim4_t,               \
        const shared_t<T[]>&, dim4_t,               \
        dim4_t, float2_t, float, float,             \
        M, F, bool, Stream&);                       \
    template void rectangle<R, T, M, F, void>(      \
        const shared_t<T[]>&, dim4_t,               \
        const shared_t<T[]>&, dim4_t,               \
        dim4_t, float2_t, float2_t, float,          \
        M, F, bool, Stream&)

    #define NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, M)         \
    NOA_INSTANTIATE_SHAPE_2D_(R, T, M, math::multiply_t);   \
    NOA_INSTANTIATE_SHAPE_2D_(R, T, M, math::plus_t)

    #define NOA_INSTANTIATE_SHAPE_MATRIX_(R, T)         \
    NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, float22_t);    \
    NOA_INSTANTIATE_SHAPE_FUNCTOR_(R, T, float23_t)

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
