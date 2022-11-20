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
    void launch3D_(const Accessor<const Value, 4, dim_t>& input,
                   const Accessor<Value, 4, dim_t>& output,
                   const dim4_t& start, const dim4_t& end, const dim4_t& shape,
                   const float3_t& center, const Radius& radius, float edge_size,
                   const Matrix& inv_matrix, const Functor& functor, dim_t threads) {
        const bool has_smooth_edge = edge_size > 1e-5f;
        if (has_smooth_edge) {
            const GeomShapeSmooth geom_shape_smooth(center, radius, edge_size);
            if (Matrix{} == inv_matrix) {
                const auto kernel = signal::fft::details::shape3D<REMAP, dim_t>(
                        input, output, shape, geom_shape_smooth, empty_t{}, functor);
                cpu::utils::iwise4D(start, end, kernel, threads);
            } else {
                const auto kernel = signal::fft::details::shape3D<REMAP, dim_t>(
                        input, output, shape, geom_shape_smooth, inv_matrix, functor);
                cpu::utils::iwise4D(start, end, kernel, threads);
            }
        } else {
            const GeomShape geom_shape(center, radius);
            if (Matrix{} == inv_matrix) {
                const auto kernel = signal::fft::details::shape3D<REMAP, dim_t>(
                        input, output, shape, geom_shape, empty_t{}, functor);
                cpu::utils::iwise4D(start, end, kernel, threads);
            } else {
                const auto kernel = signal::fft::details::shape3D<REMAP, dim_t>(
                        input, output, shape, geom_shape, inv_matrix, functor);
                cpu::utils::iwise4D(start, end, kernel, threads);
            }
        }
    }

    // Find the smallest [start, end) range for each dimension (2D or 3D).
    template<typename Value, typename Center, typename Radius>
    std::pair<dim4_t, dim4_t>
    computeIwiseSubregion(const shared_t<Value[]>& input, const shared_t<Value[]>& output, const dim4_t& shape,
                          const Center& center, const Radius& radius, float edge_size, bool invert) {

        // In this case, we have to loop through the entire array.
        if (!invert || input != output)
            return {dim4_t{0}, shape};

        const dim3_t start_(math::clamp(int3_t(center - (radius + edge_size)), int3_t{}, int3_t(shape.get(1))));
        const dim3_t end_(math::clamp(int3_t(center + (radius + edge_size) + 1), int3_t{}, int3_t(shape.get(1))));

        return {dim4_t{0, start_[0], start_[1], start_[2]},
                dim4_t{shape[0], end_[0], end_[1], end_[2]}};
    }
}

namespace noa::cpu::signal::fft {
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

        const auto[start, end] = computeIwiseSubregion(input, output, shape, center, radius, edge_size, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<Value>;
        const dim_t threads = stream.threads();

        if (invert) {
            using ellipse_t = noa::signal::Ellipse<3, real_t, true>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, real_t, true>;
            stream.enqueue([=]() {
                launch3D_<REMAP, ellipse_t, ellipse_smooth_t>(
                        Accessor<const Value, 4, dim_t>(input.get(), input_strides),
                        Accessor<Value, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
        } else {
            using ellipse_t = noa::signal::Ellipse<3, real_t, false>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, real_t, false>;
            stream.enqueue([=]() {
                launch3D_<REMAP, ellipse_t, ellipse_smooth_t>(
                        Accessor<const Value, 4, dim_t>(input.get(), input_strides),
                        Accessor<Value, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
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

        const auto[start, end] = computeIwiseSubregion(input, output, shape, center, radius, edge_size, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<Value>;
        const dim_t threads = stream.threads();

        if (invert) {
            using sphere_t = noa::signal::Sphere<3, real_t, true>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, real_t, true>;
            stream.enqueue([=]() {
                launch3D_<REMAP, sphere_t, sphere_smooth_t>(
                        Accessor<const Value, 4, dim_t>(input.get(), input_strides),
                        Accessor<Value, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
        } else {
            using sphere_t = noa::signal::Sphere<3, real_t, false>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, real_t, false>;
            stream.enqueue([=]() {
                launch3D_<REMAP, sphere_t, sphere_smooth_t>(
                        Accessor<const Value, 4, dim_t>(input.get(), input_strides),
                        Accessor<Value, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
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

        const auto[start, end] = computeIwiseSubregion(input, output, shape, center, radius, edge_size, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<Value>;
        const dim_t threads = stream.threads();

        if (invert) {
            using rectangle_t = noa::signal::Rectangle<3, real_t, true>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, real_t, true>;
            stream.enqueue([=]() {
                launch3D_<REMAP, rectangle_t, rectangle_smooth_t>(
                        Accessor<const Value, 4, dim_t>(input.get(), input_strides),
                        Accessor<Value, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
        } else {
            using rectangle_t = noa::signal::Rectangle<3, real_t, false>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, real_t, false>;
            stream.enqueue([=]() {
                launch3D_<REMAP, rectangle_t, rectangle_smooth_t>(
                        Accessor<const Value, 4, dim_t>(input.get(), input_strides),
                        Accessor<Value, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size,
                        inv_matrix, functor, threads);
            });
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
        const auto[start, end] = computeIwiseSubregion(input, output, shape, center, radius_, edge_size, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<Value>;
        const dim_t threads = stream.threads();

        if (invert) {
            using cylinder_t = noa::signal::Cylinder<real_t, true>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<real_t, true>;
            stream.enqueue([=]() {
                launch3D_<REMAP, cylinder_t, cylinder_smooth_t>(
                        Accessor<const Value, 4, dim_t>(input.get(), input_strides),
                        Accessor<Value, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, float2_t{length, radius}, edge_size,
                        inv_matrix, functor, threads);
            });
        } else {
            using cylinder_t = noa::signal::Cylinder<real_t, false>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<real_t, false>;
            stream.enqueue([=]() {
                launch3D_<REMAP, cylinder_t, cylinder_smooth_t>(
                        Accessor<const Value, 4, dim_t>(input.get(), input_strides),
                        Accessor<Value, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, float2_t{length, radius}, edge_size,
                        inv_matrix, functor, threads);
            });
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
