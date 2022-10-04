#include "noa/common/Math.h"
#include "noa/common/signal/Shape.h"
#include "noa/cpu/signal/Shape.h"

namespace {
    using namespace noa;

    template<typename signal_shape_t, typename T>
    void applyMask3DOMP_(Accessor<const T, 4, dim_t> input,
                         Accessor<T, 4, dim_t> output,
                         dim3_t start, dim3_t end, dim_t batches,
                         signal_shape_t signal_shape, dim_t threads) {

        #pragma omp parallel for default(none) collapse(4) num_threads(threads) \
        shared(input, output, start, end, batches, signal_shape)

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t j = start[0]; j < end[0]; ++j) {
                for (dim_t k = start[1]; k < end[1]; ++k) {
                    for (dim_t l = start[2]; l < end[2]; ++l) {
                        const float3_t coords{j, k, l};
                        const auto mask = signal_shape(coords);
                        output(i, j, k, l) = input ? input(i, j, k, l) * mask : mask;
                    }
                }
            }
        }
    }

    template<typename signal_shape_t, typename T>
    void applyMask2DOMP_(Accessor<const T, 3, dim_t> input,
                         Accessor<T, 3, dim_t> output,
                         dim2_t start, dim2_t end, dim_t batches,
                         signal_shape_t signal_shape, dim_t threads) {

        #pragma omp parallel for default(none) collapse(3) num_threads(threads) \
        shared(input, output, start, end, batches, signal_shape)

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t k = start[0]; k < end[0]; ++k) {
                for (dim_t l = start[1]; l < end[1]; ++l) {
                    const float2_t coords{k, l};
                    const auto mask = signal_shape(coords);
                    output(i, k, l) = input ? input(i, k, l) * mask : mask;
                }
            }
        }
    }

    template<typename signal_shape_t, typename T>
    void applyMask3D_(Accessor<const T, 4, dim_t> input,
                      Accessor<T, 4, dim_t> output,
                      dim3_t start, dim3_t end, dim_t batches,
                      signal_shape_t signal_shape, dim_t) {

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t j = start[0]; j < end[0]; ++j) {
                for (dim_t k = start[1]; k < end[1]; ++k) {
                    for (dim_t l = start[2]; l < end[2]; ++l) {
                        const float3_t coords{j, k, l};
                        const auto mask = signal_shape(coords);
                        output(i, j, k, l) = input ? input(i, j, k, l) * mask : mask;
                    }
                }
            }
        }
    }

    template<typename signal_shape_t, typename T>
    void applyMask2D_(Accessor<const T, 3, dim_t> input,
                      Accessor<T, 3, dim_t> output,
                      dim2_t start, dim2_t end, dim_t batches,
                      signal_shape_t signal_shape, dim_t) {

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t k = start[0]; k < end[0]; ++k) {
                for (dim_t l = start[1]; l < end[1]; ++l) {
                    const float2_t coords{k, l};
                    const auto mask = signal_shape(coords);
                    output(i, k, l) = input ? input(i, k, l) * mask : mask;
                }
            }
        }
    }

    template<typename ...Args>
    void launch3D_(bool OMP, Args&& ...args) {
        if (OMP) {
            applyMask3DOMP_(std::forward<Args>(args)...);
        } else {
            applyMask3D_(std::forward<Args>(args)...);
        }
    }

    template<typename ...Args>
    void launch2D_(bool OMP, Args&& ...args) {
        if (OMP) {
            applyMask2DOMP_(std::forward<Args>(args)...);
        } else {
            applyMask2D_(std::forward<Args>(args)...);
        }
    }
}

namespace noa::cpu::signal {
    template<bool INVERT, typename T, typename>
    void sphere(const shared_t<T[]>& input, dim4_t input_strides,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                float3_t center, float radius, float taper_size, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
        }

        dim3_t start{0}, end(shape.get(1));
        if (INVERT && input.get() == output.get()) {
            start = dim3_t(noa::math::clamp(int3_t(center - (radius + taper_size)), int3_t{}, int3_t(end)));
            end = dim3_t(noa::math::clamp(int3_t(center + (radius + taper_size) + 1), int3_t{}, int3_t(end)));
            if (any(end <= start))
                return;
        }

        using real_t = traits::value_type_t<T>;
        using sphere_t = noa::signal::Sphere<3, real_t, INVERT>;
        using sphere_smooth_t = noa::signal::SphereSmooth<3, real_t, INVERT>;
        const dim_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (taper) {
            stream.enqueue([=]() {
                launch3D_(
                        threads > 1,
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape[0],
                        sphere_smooth_t(center, radius, taper_size), threads);
            });
        } else {
            stream.enqueue([=]() {
                launch3D_(
                        threads > 1,
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape[0],
                        sphere_t(center, radius), threads);
            });
        }
    }

    template<bool INVERT, typename T, typename>
    void cylinder(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  float3_t center, float radius, float length, float taper_size, Stream& stream) {
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[1], center[2]);
        }

        dim3_t start{0}, end(shape.get(1));
        if (INVERT && input.get() == output.get()) {
            float3_t radius_{length, radius, radius};
            radius_ += taper_size;
            start = dim3_t(noa::math::clamp(int3_t(center - radius_), int3_t{}, int3_t(end)));
            end = dim3_t(noa::math::clamp(int3_t(center + radius_ + 1), int3_t{}, int3_t(end)));
            if (any(end <= start))
                return;
        }

        using real_t = traits::value_type_t<T>;
        using cylinder_t = noa::signal::Cylinder<real_t, INVERT>;
        using cylinder_smooth_t = noa::signal::CylinderSmooth<real_t, INVERT>;
        const dim_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (taper) {
            stream.enqueue([=]() {
                launch3D_(
                        threads > 1,
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape[0],
                        cylinder_smooth_t(center, radius, length, taper_size), threads);
            });
        } else {
            stream.enqueue([=]() {
                launch3D_(
                        threads > 1,
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape[0],
                        cylinder_t(center, radius, length), threads);
            });
        }
    }

    template<bool INVERT, typename T, typename>
    void rectangle(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   float3_t center, float3_t radius, float taper_size, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
        }

        dim3_t start{0}, end(shape.get(1));
        if (INVERT && input.get() == output.get()) {
            start = dim3_t(noa::math::clamp(int3_t(center - (radius + taper_size)), int3_t{}, int3_t(end)));
            end = dim3_t(noa::math::clamp(int3_t(center + (radius + taper_size) + 1), int3_t{}, int3_t(end)));
            if (any(end <= start))
                return;
        }

        using real_t = traits::value_type_t<T>;
        using rectangle_t = noa::signal::Rectangle<3, real_t, INVERT>;
        using rectangle_smooth_t = noa::signal::RectangleSmooth<3, real_t, INVERT>;
        const dim_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (taper) {
            stream.enqueue([=]() {
                launch3D_(
                        threads > 1,
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape[0],
                        rectangle_smooth_t(center, radius, taper_size), threads);
            });
        } else {
            stream.enqueue([=]() {
                launch3D_(
                        threads > 1,
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape[0],
                        rectangle_t(center, radius), threads);
            });
        }
    }

    template<bool INVERT, typename T, typename>
    void ellipse(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t center, float3_t radius, float taper_size, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
        }

        dim3_t start{0}, end(shape.get(1));
        if (INVERT && input.get() == output.get()) {
            start = dim3_t(noa::math::clamp(int3_t(center - (radius + taper_size)), int3_t{}, int3_t(end)));
            end = dim3_t(noa::math::clamp(int3_t(center + (radius + taper_size) + 1), int3_t{}, int3_t(end)));
            if (any(end <= start))
                return;
        }

        using real_t = traits::value_type_t<T>;
        using ellipse_t = noa::signal::Ellipse<3, real_t, INVERT>;
        using ellipse_smooth_t = noa::signal::EllipseSmooth<3, real_t, INVERT>;
        const dim_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (taper && shape[1] == 1 && center[0] == 0.f) {
            const dim3_t i_strides_2d{input_strides[0], input_strides[2], input_strides[3]};
            const dim3_t o_strides_2d{output_strides[0], output_strides[2], output_strides[3]};
            const float2_t center_2d(center.get(1));
            const float2_t radius_2d(radius.get(1));

            // TODO Benchmark if it is actually worth detecting 2D case.
            using ellipse_smooth_2d_t = noa::signal::EllipseSmooth<2, real_t, INVERT>;
            stream.enqueue([=]() {
                launch2D_(
                        threads > 1,
                        Accessor<const T, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<T, 3, dim_t>(output.get(), o_strides_2d),
                        dim2_t(start.get(1)), dim2_t(end.get(1)), shape[0],
                        ellipse_smooth_2d_t(center_2d, radius_2d, taper_size),
                        threads);
            });
        } else {
            if (taper) {
                stream.enqueue([=]() {
                    launch3D_(
                            threads > 1,
                            Accessor<const T, 4, dim_t>(input.get(), input_strides),
                            Accessor<T, 4, dim_t>(output.get(), output_strides),
                            start, end, shape[0],
                            ellipse_smooth_t(center, radius, taper_size), threads);
                });
            } else {
                stream.enqueue([=]() {
                    launch3D_(
                            threads > 1,
                            Accessor<const T, 4, dim_t>(input.get(), input_strides),
                            Accessor<T, 4, dim_t>(output.get(), output_strides),
                            start, end, shape[0],
                            ellipse_t(center, radius), threads);
                });
            }
        }
    }

    #define NOA_INSTANTIATE_SHAPE_(T)                                                                                                                   \
    template void sphere<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, Stream&);           \
    template void sphere<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, Stream&);          \
    template void cylinder<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, float, Stream&);  \
    template void cylinder<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, float, Stream&); \
    template void rectangle<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&);     \
    template void rectangle<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&);    \
    template void ellipse<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&);       \
    template void ellipse<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_SHAPE_(half_t);
    NOA_INSTANTIATE_SHAPE_(float);
    NOA_INSTANTIATE_SHAPE_(double);
    NOA_INSTANTIATE_SHAPE_(chalf_t);
    NOA_INSTANTIATE_SHAPE_(cfloat_t);
    NOA_INSTANTIATE_SHAPE_(cdouble_t);
}
