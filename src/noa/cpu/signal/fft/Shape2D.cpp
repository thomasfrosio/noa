#include "noa/common/Math.h"
#include "noa/common/Functors.h"
#include "noa/common/signal/Shape.h"
#include "noa/common/signal/details/Shape.h"
#include "noa/cpu/signal/fft/Shape.h"
#include "noa/cpu/utils/Iwise.h"

namespace {
    using namespace noa;

    template<fft::Remap REMAP, typename GeomShape, typename GeomShapeSmooth,
             typename Functor, typename Value, typename Radius, typename Matrix>
    void launch2D_(const Accessor<const Value, 3, dim_t>& input,
                   const Accessor<Value, 3, dim_t>& output,
                   const dim4_t& start, const dim4_t& end, const dim4_t& shape,
                   const float2_t& center, const Radius& radius, float edge_size,
                   const Matrix& inv_matrix, const Functor& functor, dim_t threads) {
        const bool has_smooth_edge = edge_size > 1e-5f;
        const auto shape_2d = dim3_t{shape[0], shape[2], shape[3]};
        const auto iwise_start = dim3_t{0, start[2], start[3]};
        const auto iwise_end = dim3_t{shape[0], end[2], end[3]};

        if (has_smooth_edge) {
            const GeomShapeSmooth geom_shape_smooth(center, radius, edge_size);
            if (Matrix{} == inv_matrix) {
                const auto kernel = signal::fft::details::shape2D<REMAP, dim_t>(
                        input, output, shape_2d, geom_shape_smooth, empty_t{}, functor);
                cpu::utils::iwise3D(iwise_start, iwise_end, kernel, threads);
            } else {
                const auto kernel = signal::fft::details::shape2D<REMAP, dim_t>(
                        input, output, shape_2d, geom_shape_smooth, inv_matrix, functor);
                cpu::utils::iwise3D(iwise_start, iwise_end, kernel, threads);
            }
        } else {
            const GeomShape geom_shape(center, radius);
            if (Matrix{} == inv_matrix) {
                const auto kernel = signal::fft::details::shape2D<REMAP, dim_t>(
                        input, output, shape_2d, geom_shape, empty_t{}, functor);
                cpu::utils::iwise3D(iwise_start, iwise_end, kernel, threads);
            } else {
                const auto kernel = signal::fft::details::shape2D<REMAP, dim_t>(
                        input, output, shape_2d, geom_shape, inv_matrix, functor);
                cpu::utils::iwise3D(iwise_start, iwise_end, kernel, threads);
            }
        }
    }

    // Find the smallest [start, end) range for each dimension (2D or 3D).
    template<typename Value, typename Center, typename Radius, typename Matrix>
    std::pair<dim4_t, dim4_t>
    computeIwiseSubregion(const shared_t<Value[]>& input, const shared_t<Value[]>& output, const dim4_t& shape,
                          const Center& center, const Radius& radius, float edge_size,
                          const Matrix& inv_matrix, bool invert) {

        // In this case, we have to loop through the entire array.
        // TODO Rotate the boundary box to support matrix?
        if (!invert || input != output || inv_matrix != Matrix{})
            return {dim4_t{0}, shape};

        const dim2_t start_(math::clamp(int2_t(center - (radius + edge_size)), int2_t{}, int2_t(shape.get(2))));
        const dim2_t end_(math::clamp(int2_t(center + (radius + edge_size) + 1), int2_t{}, int2_t(shape.get(2))));

        return {dim4_t{0, 0, start_[0], start_[1]},
                dim4_t{shape[0], 1, end_[0], end_[1]}};
    }
}

namespace noa::cpu::signal::fft {
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

        const auto[start, end] = computeIwiseSubregion(
                input, output, shape, center, radius, edge_size, inv_matrix, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<Value>;
        const dim3_t i_strides_2d{input_strides[0], input_strides[2], input_strides[3]};
        const dim3_t o_strides_2d{output_strides[0], output_strides[2], output_strides[3]};
        const dim_t threads = stream.threads();

        if (invert) {
            using ellipse_t = noa::signal::Ellipse<2, real_t, true>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<2, real_t, true>;
            stream.enqueue([=]() {
                launch2D_<REMAP, ellipse_t, ellipse_smooth_t>(
                        Accessor<const Value, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<Value, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
        } else {
            using ellipse_t = noa::signal::Ellipse<2, real_t, false>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<2, real_t, false>;
            stream.enqueue([=]() {
                launch2D_<REMAP, ellipse_t, ellipse_smooth_t>(
                        Accessor<const Value, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<Value, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
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

        const auto[start, end] = computeIwiseSubregion(
                input, output, shape, center, radius, edge_size, inv_matrix, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<Value>;
        const dim3_t i_strides_2d{input_strides[0], input_strides[2], input_strides[3]};
        const dim3_t o_strides_2d{output_strides[0], output_strides[2], output_strides[3]};
        const dim_t threads = stream.threads();

        if (invert) {
            using sphere_t = noa::signal::Sphere<2, real_t, true>;
            using sphere_smooth_t = noa::signal::SphereSmooth<2, real_t, true>;
            stream.enqueue([=]() {
                launch2D_<REMAP, sphere_t, sphere_smooth_t>(
                        Accessor<const Value, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<Value, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
        } else {
            using sphere_t = noa::signal::Sphere<2, real_t, false>;
            using sphere_smooth_t = noa::signal::SphereSmooth<2, real_t, false>;
            stream.enqueue([=]() {
                launch2D_<REMAP, sphere_t, sphere_smooth_t>(
                        Accessor<const Value, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<Value, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
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

        const auto[start, end] = computeIwiseSubregion(
                input, output, shape, center, radius, edge_size, inv_matrix, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<Value>;
        const dim3_t i_strides_2d{input_strides[0], input_strides[2], input_strides[3]};
        const dim3_t o_strides_2d{output_strides[0], output_strides[2], output_strides[3]};
        const dim_t threads = stream.threads();

        if (invert) {
            using rectangle_t = noa::signal::Rectangle<2, real_t, true>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<2, real_t, true>;
            stream.enqueue([=]() {
                launch2D_<REMAP, rectangle_t, rectangle_smooth_t>(
                        Accessor<const Value, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<Value, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
        } else {
            using rectangle_t = noa::signal::Rectangle<2, real_t, false>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<2, real_t, false>;
            stream.enqueue([=]() {
                launch2D_<REMAP, rectangle_t, rectangle_smooth_t>(
                        Accessor<const Value, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<Value, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
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
