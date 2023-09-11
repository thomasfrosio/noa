#include "noa/cpu/geometry/fft/Project.hpp"
#include "noa/cpu/utils/Iwise.hpp"

#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/geometry/InterpolatorValue.hpp"
#include "noa/algorithms/geometry/FourierInsertInterpolate.hpp"

namespace {
    using namespace ::noa;

    template<typename PointerOrValue>
    auto wrap_into_interpolator_(
            PointerOrValue pointer_or_value,
            const Strides4<i64>& strides,
            const Shape4<i64>& shape
    ) {
        using value_t = nt::remove_pointer_cv_t<PointerOrValue>;
        if constexpr (std::is_empty_v<value_t>) {
            return Empty{};
        } else {
            using interpolator_array_t = noa::geometry::Interpolator2D<
                    BorderMode::ZERO, InterpMode::LINEAR, value_t, i64, f32,
                    3, PointerTraits::RESTRICT, StridesTraits::STRIDED>;

            using interpolator_value_t = noa::geometry::InterpolatorValue2D<
                    BorderMode::ZERO, InterpMode::LINEAR, value_t, i64, f32>;

            using interpolator_t = std::conditional_t<
                    std::is_pointer_v<PointerOrValue>, interpolator_array_t, interpolator_value_t>;

            interpolator_t interpolator;
            if constexpr (std::is_pointer_v<PointerOrValue>) {
                interpolator = interpolator_array_t(
                        AccessorRestrict<const value_t, 3, i64>(pointer_or_value, strides.filter(0, 2, 3)),
                        shape.filter(2, 3).rfft(), value_t{0});
            } else {
                interpolator = interpolator_value_t(
                        pointer_or_value, shape.filter(2, 3).rfft(), value_t{0});
            }
            return interpolator;
        }
    }

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
    void insert_interpolate_3d(
            InputValue input_slice, const Strides4<i64>& input_slice_strides,
            InputWeight input_weight, const Strides4<i64>& input_weight_strides,
            const Shape4<i64>& input_slice_shape,
            OutputValue output_volume, const Strides4<i64>& output_volume_strides,
            OutputWeight output_weight, const Strides4<i64>& output_weight_strides,
            const Shape4<i64>& output_volume_shape,
            const Scale& fwd_scaling, const Rotate& inv_rotation,
            f32 fftfreq_sinc, f32 fftfreq_blackman, f32 fftfreq_cutoff,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, i64 n_threads
    ) {
        const auto input_slice_interpolator = wrap_into_interpolator_(input_slice, input_slice_strides, input_slice_shape);
        const auto input_weight_interpolator = wrap_into_interpolator_(input_weight, input_weight_strides, input_slice_shape);
        const auto output_volume_accessor = wrap_into_accessor_(output_volume, output_volume_strides.filter(1, 2, 3));
        const auto output_weight_accessor = wrap_into_accessor_(output_weight, output_weight_strides.filter(1, 2, 3));

        const auto iwise_shape = output_volume_shape.pop_front().rfft();
        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = fwd_scaling != Scale{};

        if (apply_ews || apply_scale) {
            const auto op = noa::geometry::fourier_insert_interpolate_op<REMAP>(
                    input_slice_interpolator, input_weight_interpolator, input_slice_shape,
                    output_volume_accessor, output_weight_accessor, output_volume_shape,
                    fwd_scaling, inv_rotation, fftfreq_sinc, fftfreq_blackman, fftfreq_cutoff,
                    target_shape, ews_radius);
            noa::cpu::utils::iwise_3d(iwise_shape, op, n_threads);
        } else {
            const auto op = noa::geometry::fourier_insert_interpolate_op<REMAP>(
                    input_slice_interpolator, input_weight_interpolator, input_slice_shape,
                    output_volume_accessor, output_weight_accessor, output_volume_shape,
                    Empty{}, inv_rotation, fftfreq_sinc, fftfreq_blackman, fftfreq_cutoff,
                    target_shape, Empty{});
            noa::cpu::utils::iwise_3d(iwise_shape, op, n_threads);
        }
    }

    #define NOA_INSTANTIATE_INSERT_(REMAP, IValue, IWeight, OValue, OWeight, Scale, Rotate)         \
    template void insert_interpolate_3d<REMAP, IValue, IWeight, OValue, OWeight, Scale, Rotate>(    \
        IValue, const Strides4<i64>&, IWeight, const Strides4<i64>&, const Shape4<i64>&,            \
        OValue, const Strides4<i64>&, OWeight, const Strides4<i64>&, const Shape4<i64>&,            \
        Scale const&, Rotate const&, f32, f32, f32, const Shape4<i64>&, const Vec2<f32>&, i64)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(IValue, IWeight, OValue, OWeight, Scale, Rotate)    \
    NOA_INSTANTIATE_INSERT_(noa::fft::Remap::HC2H, IValue, IWeight, OValue, OWeight, Scale, Rotate);    \
    NOA_INSTANTIATE_INSERT_(noa::fft::Remap::HC2HC, IValue, IWeight, OValue, OWeight, Scale, Rotate)

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
