#include "noa/common/Math.h"
#include "noa/common/signal/Shape.h"
#include "noa/cpu/signal/fft/Shape.h"
#include "noa/cpu/utils/Loops.h"

namespace {
    using namespace noa;
    struct Empty{};

    template<fft::Remap REMAP, bool TRANSFORM,
             typename geom_shape_t, typename matrix_t, typename T, typename void_ = void>
    void kernel3D_(Accessor<const T, 4, dim_t> input,
                   Accessor<T, 4, dim_t> output,
                   const dim4_t& start, const dim4_t& end, const dim4_t& shape,
                   geom_shape_t signal_shape, matrix_t inv_transform,
                   dim_t threads) {

        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & fft::Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & fft::Layout::DST_CENTERED;
        if constexpr (REMAP_ & fft::Layout::SRC_HALF || REMAP_ & fft::Layout::DST_HALF)
            static_assert(traits::always_false_v<void_>);

        auto op = [=](dim_t i, dim_t j, dim_t k, dim_t l) {
            dim3_t index{IS_SRC_CENTERED ? j : math::FFTShift(j, shape[1]),
                         IS_SRC_CENTERED ? k : math::FFTShift(k, shape[2]),
                         IS_SRC_CENTERED ? l : math::FFTShift(l, shape[3])};

            float3_t coords{index};
            typename geom_shape_t::value_type mask;
            if constexpr (TRANSFORM)
                mask = signal_shape(coords, inv_transform);
            else
                mask = signal_shape(coords);

            const auto value = input ? input(i, index[0], index[1], index[2]) * mask : mask;

            if constexpr (IS_SRC_CENTERED != IS_DST_CENTERED) {
                index[0] = IS_DST_CENTERED ? j : math::FFTShift(j, shape[1]);
                index[1] = IS_DST_CENTERED ? k : math::FFTShift(k, shape[2]);
                index[2] = IS_DST_CENTERED ? l : math::FFTShift(l, shape[3]);
            }
            output(i, index[0], index[1], index[2]) = value;
        };
        cpu::utils::iwise4D(start, end, op, threads);
    }

    template<fft::Remap REMAP, typename geom_shape_t, typename geom_shape_smooth_t, typename T, typename radius_t>
    void launch3D_(Accessor<const T, 4, dim_t> input,
                   Accessor<T, 4, dim_t> output,
                   const dim4_t& start, const dim4_t& end, const dim4_t& shape,
                   const float3_t& center, const radius_t& radius, float edge_size,
                   float33_t inv_transform,
                   dim_t threads) {
        const bool has_smooth_edge = edge_size > 1e-5f;
        if (has_smooth_edge) {
            const geom_shape_smooth_t geom_shape_smooth(center, radius, edge_size);
            if (float33_t{} == inv_transform)
                kernel3D_<REMAP, false>(input, output, start, end, shape, geom_shape_smooth, Empty{}, threads);
            else
                kernel3D_<REMAP, true>(input, output, start, end, shape, geom_shape_smooth, inv_transform, threads);
        } else {
            const geom_shape_t geom_shape(center, radius);
            if (float33_t{} == inv_transform)
                kernel3D_<REMAP, false>(input, output, start, end, shape, geom_shape, Empty{}, threads);
            else
                kernel3D_<REMAP, true>(input, output, start, end, shape, geom_shape, inv_transform, threads);
        }
    }

    template<fft::Remap REMAP, bool TRANSFORM,
            typename geom_shape_t, typename matrix_t, typename T, typename void_ = void>
    void kernel2D_(Accessor<const T, 3, dim_t> input,
                   Accessor<T, 3, dim_t> output,
                   const dim4_t& start, const dim4_t& end, const dim4_t& shape,
                   const geom_shape_t& signal_shape, matrix_t inv_transform,
                   dim_t threads) {

        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & fft::Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & fft::Layout::DST_CENTERED;
        if constexpr (REMAP_ & fft::Layout::SRC_HALF || REMAP_ & fft::Layout::DST_HALF)
            static_assert(traits::always_false_v<void_>);

        auto op = [=](dim_t i, dim_t j, dim_t k) {
            dim2_t index{IS_SRC_CENTERED ? j : math::FFTShift(j, shape[2]),
                         IS_SRC_CENTERED ? k : math::FFTShift(k, shape[3])};

            float2_t coords{index};
            typename geom_shape_t::value_type mask;
            if constexpr (TRANSFORM)
                mask = signal_shape(coords, inv_transform);
            else
                mask = signal_shape(coords);

            const auto value = input ? input(i, index[0], index[1]) * mask : mask;

            if constexpr (IS_SRC_CENTERED != IS_DST_CENTERED) {
                index[0] = IS_DST_CENTERED ? j : math::FFTShift(j, shape[2]);
                index[1] = IS_DST_CENTERED ? k : math::FFTShift(k, shape[3]);
            }
            output(i, index[0], index[1]) = value;
        };
        cpu::utils::iwise3D(dim3_t{0, start[2], start[3]},
                            dim3_t{shape[0], end[2], end[3]}, op, threads);
    }

    template<fft::Remap REMAP, typename geom_shape_t, typename geom_shape_smooth_t, typename T, typename radius_t>
    void launch2D_(Accessor<const T, 3, dim_t> input,
                   Accessor<T, 3, dim_t> output,
                   const dim4_t& start, const dim4_t& end, const dim4_t& shape,
                   const float2_t& center, const radius_t& radius, float edge_size,
                   const float22_t& inv_transform,
                   dim_t threads) {
        const bool has_smooth_edge = edge_size > 1e-5f;
        if (has_smooth_edge) {
            const geom_shape_smooth_t geom_shape_smooth(center, radius, edge_size);
            if (float22_t{} == inv_transform)
                kernel2D_<REMAP, false>(input, output, start, end, shape, geom_shape_smooth, Empty{}, threads);
            else
                kernel2D_<REMAP, true>(input, output, start, end, shape, geom_shape_smooth, inv_transform, threads);
        } else {
            const geom_shape_t geom_shape(center, radius);
            if (float22_t{} == inv_transform)
                kernel2D_<REMAP, false>(input, output, start, end, shape, geom_shape, Empty{}, threads);
            else
                kernel2D_<REMAP, true>(input, output, start, end, shape, geom_shape, inv_transform, threads);
        }
    }

    // Find the smallest [start, end) range for each dimension (2D or 3D).
    template<typename T, typename center_t, typename radius_t>
    auto computeIwiseSubregion(const shared_t<T[]>& input, const shared_t<T[]>& output, const dim4_t& shape,
                               const center_t& center, const radius_t& radius, float edge_size, bool invert) {
        constexpr bool IS_2D = center_t::COUNT == 2;
        using intX_t = std::conditional_t<IS_2D, int2_t, int3_t>;
        using dimX_t = std::conditional_t<IS_2D, dim2_t, dim3_t>;
        using pair_t = std::pair<dim4_t, dim4_t>;
        constexpr dim_t OFFSET = 4 - center_t::COUNT;

        // In this case, we have to loop through the entire array.
        if (!invert || input != output)
            return pair_t{dim4_t{0}, shape};

        const dimX_t start_(math::clamp(intX_t(center - (radius + edge_size)), intX_t{}, intX_t(shape.get(OFFSET))));
        const dimX_t end_(math::clamp(intX_t(center + (radius + edge_size) + 1), intX_t{}, intX_t(shape.get(OFFSET))));

        if constexpr (IS_2D) {
            return pair_t{dim4_t{0, 0, start_[0], start_[1]},
                          dim4_t{shape[0], 1, end_[0], end_[1]}};
        } else {
            return pair_t{dim4_t{0, start_[0], start_[1], start_[2]},
                          dim4_t{shape[0], end_[0], end_[1], end_[2]}};
        }
    }
}

namespace noa::cpu::signal::fft {
    // Returns or applies an elliptical mask.
    template<fft::Remap REMAP, typename T, typename>
    void ellipse(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t center, float3_t radius, float edge_size,
                 float33_t inv_transform, bool invert, Stream& stream) {
        NOA_ASSERT((REMAP == fft::F2F || REMAP == fft::FC2FC) || input != output);
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
            inv_transform = indexing::reorder(inv_transform, order_3d);
        }

        const auto[start, end] = computeIwiseSubregion(input, output, shape, center, radius, edge_size, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<T>;
        const dim_t threads = stream.threads();

        if (invert) {
            using ellipse_t = noa::signal::Ellipse<3, real_t, true>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, real_t, true>;
            stream.enqueue([=]() {
                launch3D_<REMAP, ellipse_t, ellipse_smooth_t>(
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        } else {
            using ellipse_t = noa::signal::Ellipse<3, real_t, false>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<3, real_t, false>;
            stream.enqueue([=]() {
                launch3D_<REMAP, ellipse_t, ellipse_smooth_t>(
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void ellipse(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float2_t center, float2_t radius, float edge_size, float22_t inv_transform, bool invert, Stream& stream) {
        NOA_ASSERT((REMAP == fft::F2F || REMAP == fft::FC2FC) || input != output);
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            std::swap(radius[0], radius[1]);
            inv_transform = indexing::reorder(inv_transform, order_2d);
        }

        const auto[start, end] = computeIwiseSubregion(input, output, shape, center, radius, edge_size, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<T>;
        const dim3_t i_strides_2d{input_strides[0], input_strides[2], input_strides[3]};
        const dim3_t o_strides_2d{output_strides[0], output_strides[2], output_strides[3]};
        const dim_t threads = stream.threads();

        if (invert) {
            using ellipse_t = noa::signal::Ellipse<2, real_t, true>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<2, real_t, true>;
            stream.enqueue([=]() {
                launch2D_<REMAP, ellipse_t, ellipse_smooth_t>(
                        Accessor<const T, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<T, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        } else {
            using ellipse_t = noa::signal::Ellipse<2, real_t, false>;
            using ellipse_smooth_t = noa::signal::EllipseSmooth<2, real_t, false>;
            stream.enqueue([=]() {
                launch2D_<REMAP, ellipse_t, ellipse_smooth_t>(
                        Accessor<const T, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<T, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void sphere(const shared_t<T[]>& input, dim4_t input_strides,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                float3_t center, float radius, float edge_size,
                float33_t inv_transform, bool invert, Stream& stream) {
        NOA_ASSERT((REMAP == fft::F2F || REMAP == fft::FC2FC) || input != output);
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            inv_transform = indexing::reorder(inv_transform, order_3d);
        }

        const auto[start, end] = computeIwiseSubregion(input, output, shape, center, radius, edge_size, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<T>;
        const dim_t threads = stream.threads();

        if (invert) {
            using sphere_t = noa::signal::Sphere<3, real_t, true>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, real_t, true>;
            stream.enqueue([=]() {
                launch3D_<REMAP, sphere_t, sphere_smooth_t>(
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        } else {
            using sphere_t = noa::signal::Sphere<3, real_t, false>;
            using sphere_smooth_t = noa::signal::SphereSmooth<3, real_t, false>;
            stream.enqueue([=]() {
                launch3D_<REMAP, sphere_t, sphere_smooth_t>(
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void sphere(const shared_t<T[]>& input, dim4_t input_strides,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                float2_t center, float radius, float edge_size,
                float22_t inv_transform, bool invert, Stream& stream) {
        NOA_ASSERT((REMAP == fft::F2F || REMAP == fft::FC2FC) || input != output);
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            inv_transform = indexing::reorder(inv_transform, order_2d);
        }

        const auto[start, end] = computeIwiseSubregion(input, output, shape, center, radius, edge_size, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<T>;
        const dim3_t i_strides_2d{input_strides[0], input_strides[2], input_strides[3]};
        const dim3_t o_strides_2d{output_strides[0], output_strides[2], output_strides[3]};
        const dim_t threads = stream.threads();

        if (invert) {
            using sphere_t = noa::signal::Sphere<2, real_t, true>;
            using sphere_smooth_t = noa::signal::SphereSmooth<2, real_t, true>;
            stream.enqueue([=]() {
                launch2D_<REMAP, sphere_t, sphere_smooth_t>(
                        Accessor<const T, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<T, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        } else {
            using sphere_t = noa::signal::Sphere<2, real_t, false>;
            using sphere_smooth_t = noa::signal::SphereSmooth<2, real_t, false>;
            stream.enqueue([=]() {
                launch2D_<REMAP, sphere_t, sphere_smooth_t>(
                        Accessor<const T, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<T, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void rectangle(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   float3_t center, float3_t radius, float edge_size,
                   float33_t inv_transform, bool invert, Stream& stream) {
        NOA_ASSERT((REMAP == fft::F2F || REMAP == fft::FC2FC) || input != output);
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
            inv_transform = indexing::reorder(inv_transform, order_3d);
        }

        const auto[start, end] = computeIwiseSubregion(input, output, shape, center, radius, edge_size, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<T>;
        const dim_t threads = stream.threads();

        if (invert) {
            using rectangle_t = noa::signal::Rectangle<3, real_t, true>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, real_t, true>;
            stream.enqueue([=]() {
                launch3D_<REMAP, rectangle_t, rectangle_smooth_t>(
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        } else {
            using rectangle_t = noa::signal::Rectangle<3, real_t, false>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<3, real_t, false>;
            stream.enqueue([=]() {
                launch3D_<REMAP, rectangle_t, rectangle_smooth_t>(
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void rectangle(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   float2_t center, float2_t radius, float edge_size,
                   float22_t inv_transform, bool invert, Stream& stream) {
        NOA_ASSERT((REMAP == fft::F2F || REMAP == fft::FC2FC) || input != output);
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[0], center[1]);
            std::swap(radius[0], radius[1]);
            inv_transform = indexing::reorder(inv_transform, order_2d);
        }

        const auto[start, end] = computeIwiseSubregion(input, output, shape, center, radius, edge_size, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<T>;
        const dim3_t i_strides_2d{input_strides[0], input_strides[2], input_strides[3]};
        const dim3_t o_strides_2d{output_strides[0], output_strides[2], output_strides[3]};
        const dim_t threads = stream.threads();

        if (invert) {
            using rectangle_t = noa::signal::Rectangle<2, real_t, true>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<2, real_t, true>;
            stream.enqueue([=]() {
                launch2D_<REMAP, rectangle_t, rectangle_smooth_t>(
                        Accessor<const T, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<T, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        } else {
            using rectangle_t = noa::signal::Rectangle<2, real_t, false>;
            using rectangle_smooth_t = noa::signal::RectangleSmooth<2, real_t, false>;
            stream.enqueue([=]() {
                launch2D_<REMAP, rectangle_t, rectangle_smooth_t>(
                        Accessor<const T, 3, dim_t>(input.get(), i_strides_2d),
                        Accessor<T, 3, dim_t>(output.get(), o_strides_2d),
                        start, end, shape, center, radius, edge_size, inv_transform, threads);
            });
        }
    }

    template<fft::Remap REMAP, typename T, typename>
    void cylinder(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  float3_t center, float radius, float length, float edge_size,
                  float33_t inv_transform, bool invert, Stream& stream) {
        NOA_ASSERT((REMAP == fft::F2F || REMAP == fft::FC2FC) || input != output);
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[1], center[2]);
            inv_transform = indexing::reorder(inv_transform, dim3_t{0, order_2d[0] + 1, order_2d[1] + 1});
        }

        const float3_t radius_{length, radius, radius};
        const auto[start, end] = computeIwiseSubregion(input, output, shape, center, radius_, edge_size, invert);
        if (any(end <= start))
            return;

        using real_t = traits::value_type_t<T>;
        const dim_t threads = stream.threads();

        if (invert) {
            using cylinder_t = noa::signal::Cylinder<real_t, true>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<real_t, true>;
            stream.enqueue([=]() {
                launch3D_<REMAP, cylinder_t, cylinder_smooth_t>(
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, float2_t{length, radius}, edge_size, inv_transform, threads);
            });
        } else {
            using cylinder_t = noa::signal::Cylinder<real_t, false>;
            using cylinder_smooth_t = noa::signal::CylinderSmooth<real_t, false>;
            stream.enqueue([=]() {
                launch3D_<REMAP, cylinder_t, cylinder_smooth_t>(
                        Accessor<const T, 4, dim_t>(input.get(), input_strides),
                        Accessor<T, 4, dim_t>(output.get(), output_strides),
                        start, end, shape, center, float2_t{length, radius}, edge_size, inv_transform, threads);
            });
        }
    }

    #define NOA_INSTANTIATE_SHAPE_(R, T)                                                                                                                            \
    template void ellipse<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, float33_t, bool, Stream&);     \
    template void ellipse<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float, float22_t, bool, Stream&);     \
    template void sphere<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, float33_t, bool, Stream&);         \
    template void sphere<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float, float, float22_t, bool, Stream&);         \
    template void rectangle<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, float33_t, bool, Stream&);   \
    template void rectangle<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float, float22_t, bool, Stream&);   \
    template void cylinder<R, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, float, float33_t, bool, Stream&)

    #define NOA_INSTANTIATE_SHAPE_ALL(T)    \
    NOA_INSTANTIATE_SHAPE_(fft::F2F, T);    \
    NOA_INSTANTIATE_SHAPE_(fft::FC2FC, T);  \
    NOA_INSTANTIATE_SHAPE_(fft::F2FC, T);   \
    NOA_INSTANTIATE_SHAPE_(fft::FC2F, T)

    NOA_INSTANTIATE_SHAPE_ALL(half_t);
    NOA_INSTANTIATE_SHAPE_ALL(float);
    NOA_INSTANTIATE_SHAPE_ALL(double);
    NOA_INSTANTIATE_SHAPE_ALL(chalf_t);
    NOA_INSTANTIATE_SHAPE_ALL(cfloat_t);
    NOA_INSTANTIATE_SHAPE_ALL(cdouble_t);
}
