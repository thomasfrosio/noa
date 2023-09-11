#include "noa/cpu/geometry/fft/Project.hpp"
#include "noa/cpu/utils/Iwise.hpp"

#include "noa/algorithms/geometry/FourierInsertRasterize.hpp"

namespace {
    using namespace ::noa;

    template<typename PointerOrValue>
    auto wrap_into_accessor_(
            PointerOrValue pointer_or_value,
            const Strides3<i64>& strides
    ) {
        using value_t = std::remove_pointer_t<PointerOrValue>;
        if constexpr (!std::is_pointer_v<PointerOrValue>) {
            return pointer_or_value;
        } else {
            return AccessorRestrict<value_t, 3, i64>(pointer_or_value, strides);
        }
    }
}

namespace noa::cpu::geometry::fft {
    template<noa::fft::Remap REMAP,
             typename InputValue, typename InputWeight,
             typename OutputValue, typename OutputWeight,
             typename Scale, typename Rotate>
    void insert_rasterize_3d(
            InputValue input_slice, const Strides4<i64>& input_slice_strides,
            InputWeight input_weight, const Strides4<i64>& input_weight_strides,
            const Shape4<i64>& input_slice_shape,
            OutputValue output_volume, const Strides4<i64>& output_volume_strides,
            OutputWeight output_weight, const Strides4<i64>& output_weight_strides,
            const Shape4<i64>& output_volume_shape,
            const Scale& inv_scaling, const Rotate& fwd_rotation,
            f32 fftfreq_cutoff, const Shape4<i64>& target_shape,
            const Vec2<f32>& ews_radius, i64 n_threads
    ) {
        const auto input_slice_accessor = wrap_into_accessor_(input_slice, input_slice_strides.filter(0, 2, 3));
        const auto input_weight_accessor = wrap_into_accessor_(input_weight, input_weight_strides.filter(0, 2, 3));

        const auto output_volume_accessor = wrap_into_accessor_(output_volume, output_volume_strides.filter(1, 2, 3));
        const auto output_weight_accessor = wrap_into_accessor_(output_weight, output_weight_strides.filter(1, 2, 3));

        const auto iwise_shape = input_slice_shape.filter(0, 2, 3).rfft();
        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = inv_scaling != Scale{};

        if (apply_ews || apply_scale) {
            const auto op = noa::geometry::fourier_insert_rasterize_op<REMAP>(
                    input_slice_accessor, input_weight_accessor, input_slice_shape,
                    output_volume_accessor, output_weight_accessor, output_volume_shape,
                    inv_scaling, fwd_rotation, fftfreq_cutoff, target_shape, ews_radius);
            noa::cpu::utils::iwise_3d(iwise_shape, op, n_threads);
        } else {
            const auto op = noa::geometry::fourier_insert_rasterize_op<REMAP>(
                    input_slice_accessor, input_weight_accessor, input_slice_shape,
                    output_volume_accessor, output_weight_accessor, output_volume_shape,
                    Empty{}, fwd_rotation, fftfreq_cutoff, target_shape, Empty{});
            noa::cpu::utils::iwise_3d(iwise_shape, op, n_threads);
        }
    }

    #define NOA_INSTANTIATE_INSERT_(REMAP, IValue, IWeight, OValue, OWeight, Scale, Rotate)     \
    template void insert_rasterize_3d<REMAP, IValue, IWeight, OValue, OWeight, Scale, Rotate>(  \
        IValue, const Strides4<i64>&, IWeight, const Strides4<i64>&, const Shape4<i64>&,        \
        OValue, const Strides4<i64>&, OWeight, const Strides4<i64>&, const Shape4<i64>&,        \
        Scale const&, Rotate const&, f32, const Shape4<i64>&, const Vec2<f32>&, i64)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(IValue, IWeight, OValue, OWeight, Scale, Rotate)    \
    NOA_INSTANTIATE_INSERT_(noa::fft::Remap::HC2H, IValue, IWeight, OValue, OWeight, Scale, Rotate);    \
    NOA_INSTANTIATE_INSERT_(noa::fft::Remap::HC2HC, IValue, IWeight, OValue, OWeight, Scale, Rotate);   \
    NOA_INSTANTIATE_INSERT_(noa::fft::Remap::H2H, IValue, IWeight, OValue, OWeight, Scale, Rotate);     \
    NOA_INSTANTIATE_INSERT_(noa::fft::Remap::H2HC, IValue, IWeight, OValue, OWeight, Scale, Rotate)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(IValue, IWeight, OValue, OWeight, Rotate)   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(IValue, IWeight, OValue, OWeight, Float22, Rotate); \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(IValue, IWeight, OValue, OWeight, const Float22*, Rotate)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(IValue, IWeight, OValue, OWeight)          \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(IValue, IWeight, OValue, OWeight, Float33);         \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(IValue, IWeight, OValue, OWeight, const Float33*);  \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(IValue, IWeight, OValue, OWeight, const Quaternion<f32>*)

    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f32*, Empty, f32*, Empty);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c32*, Empty, c32*, Empty);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f64*, Empty, f64*, Empty);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c64*, Empty, c64*, Empty);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f32*, const f32*, f32*, f32*);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c32*, const f32*, c32*, f32*);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f64*, const f64*, f64*, f64*);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c64*, const f64*, c64*, f64*);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f32*, f32, f32*, f32*);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c32*, f32, c32*, f32*);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f64*, f64, f64*, f64*);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c64*, f64, c64*, f64*);
}
