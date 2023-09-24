#include "noa/cpu/geometry/fft/Project.hpp"
#include "noa/cpu/memory/Set.hpp"
#include "noa/cpu/utils/Iwise.hpp"

#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/geometry/InterpolatorValue.hpp"
#include "noa/algorithms/geometry/FourierInsertExtract.hpp"

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
            const Strides4<i64>& strides
    ) {
        using value_t = std::remove_pointer_t<PointerOrValue>;
        if constexpr (std::is_empty_v<value_t>) {
            return Empty{};
        } else {
            return AccessorRestrict<value_t, 3, i64>(pointer_or_value, strides.filter(0, 2, 3));
        }
    }

    template<typename OutputValue, typename OutputWeight>
    auto fill_with_zeros_(
            OutputValue output_slice, const Strides4<i64>& output_slice_strides,
            OutputWeight output_weight, const Strides4<i64>& output_weight_strides,
            const Shape4<i64>& output_slice_shape
    ) {
        noa::cpu::memory::set(
                output_slice, output_slice_strides, output_slice_shape.rfft(),
                nt::remove_pointer_cv_t<OutputValue>{0}, /*threads=*/ 1);
        if constexpr (!std::is_empty_v<OutputWeight>) {
            noa::cpu::memory::set(
                    output_weight, output_weight_strides, output_slice_shape.rfft(),
                    nt::remove_pointer_cv_t<OutputWeight>{0}, /*threads=*/ 1);
        }
    }
}

namespace noa::cpu::geometry::fft {
    template<noa::fft::Remap REMAP,
            typename InputValue, typename InputWeight,
            typename OutputValue, typename OutputWeight,
            typename Scale0, typename Scale1,
            typename Rotate0, typename Rotate1>
    void insert_interpolate_and_extract_3d(
            InputValue input_slice, const Strides4<i64>& input_slice_strides,
            InputWeight input_weight, const Strides4<i64>& input_weight_strides,
            const Shape4<i64>& input_slice_shape,
            OutputValue output_slice, const Strides4<i64>& output_slice_strides,
            OutputWeight output_weight, const Strides4<i64>& output_weight_strides,
            const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling, const Rotate0& insert_inv_rotation,
            const Scale1& extract_inv_scaling, const Rotate1& extract_fwd_rotation,
            f32 fftfreq_input_sinc, f32 fftfreq_input_blackman,
            f32 fftfreq_z_sinc, f32 fftfreq_z_blackman,
            f32 fftfreq_cutoff, bool add_to_output, bool correct_multiplicity,
            const Vec2<f32>& ews_radius, i64 n_threads
    ) {
        const auto input_slice_interpolator = wrap_into_interpolator_(input_slice, input_slice_strides, input_slice_shape);
        const auto input_weight_interpolator = wrap_into_interpolator_(input_weight, input_weight_strides, input_slice_shape);
        const auto output_slice_accessor = wrap_into_accessor_(output_slice, output_slice_strides);
        const auto output_weight_accessor = wrap_into_accessor_(output_weight, output_weight_strides);

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = insert_fwd_scaling != Scale0{};

        if (apply_ews || apply_scale) {
            const auto op = noa::geometry::fourier_insert_and_extract_op<REMAP>(
                    input_slice_interpolator, input_weight_interpolator, input_slice_shape,
                    output_slice_accessor, output_weight_accessor, output_slice_shape,
                    insert_fwd_scaling, insert_inv_rotation,
                    extract_inv_scaling, extract_fwd_rotation,
                    fftfreq_input_sinc, fftfreq_input_blackman,
                    fftfreq_z_sinc, fftfreq_z_blackman, fftfreq_cutoff,
                    add_to_output, correct_multiplicity, ews_radius);
            if (op.is_iwise_4d()) {
                if (!add_to_output) {
                    fill_with_zeros_(
                            output_slice, output_slice_strides,
                            output_weight, output_weight_strides,
                            output_slice_shape);
                }
                noa::cpu::utils::iwise_4d(output_slice_shape.rfft().set<1>(op.output_window_size()), op, n_threads);
            } else {
                noa::cpu::utils::iwise_3d(output_slice_shape.filter(0, 2, 3).rfft(), op, n_threads);
            }
        } else {
            const auto op = noa::geometry::fourier_insert_and_extract_op<REMAP>(
                    input_slice_interpolator, input_weight_interpolator, input_slice_shape,
                    output_slice_accessor, output_weight_accessor, output_slice_shape,
                    Empty{}, insert_inv_rotation,
                    extract_inv_scaling, extract_fwd_rotation,
                    fftfreq_input_sinc, fftfreq_input_blackman,
                    fftfreq_z_sinc, fftfreq_z_blackman, fftfreq_cutoff,
                    add_to_output, correct_multiplicity, Empty{});
            if (op.is_iwise_4d()) {
                if (!add_to_output) {
                    fill_with_zeros_(
                            output_slice, output_slice_strides,
                            output_weight, output_weight_strides,
                            output_slice_shape);
                }
                noa::cpu::utils::iwise_4d(output_slice_shape.rfft().set<1>(op.output_window_size()), op, n_threads);
            } else {
                noa::cpu::utils::iwise_3d(output_slice_shape.filter(0, 2, 3).rfft(), op, n_threads);
            }
        }
    }

    #define NOA_INSTANTIATE_INSERT_EXTRACT_(REMAP, IValue, IWeight, OValue, OWeight, IScale, OScale, IRotate, ORotate)          \
    template void insert_interpolate_and_extract_3d<REMAP, IValue, IWeight, OValue, OWeight, IScale, OScale, IRotate, ORotate>( \
        IValue, const Strides4<i64>&, IWeight, const Strides4<i64>&, const Shape4<i64>&,                                        \
        OValue, const Strides4<i64>&, OWeight, const Strides4<i64>&, const Shape4<i64>&,                                        \
        IScale const&, IRotate const&, OScale const&, ORotate const&,                                                           \
        f32, f32, f32, f32, f32, bool, bool, const Vec2<f32>&, i64)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(IValue, IWeight, OValue, OWeight, IScale, OScale, IRotate, ORotate)         \
    NOA_INSTANTIATE_INSERT_EXTRACT_(noa::fft::Remap::HC2H, IValue, IWeight, OValue, OWeight, IScale, OScale, IRotate, ORotate)
//    NOA_INSTANTIATE_INSERT_EXTRACT_(noa::fft::Remap::HC2HC, IValue, IWeight, OValue, OWeight, IScale, OScale, IRotate, ORotate)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(IValue, IWeight, OValue, OWeight, IRotate, ORotate)                     \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(IValue, IWeight, OValue, OWeight, Float22, Float22, IRotate, ORotate)
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(IValue, IWeight, OValue, OWeight, const Float22*, Float22, IRotate, ORotate);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(IValue, IWeight, OValue, OWeight, Float22, const Float22*, IRotate, ORotate);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(IValue, IWeight, OValue, OWeight, const Float22*, const Float22*, IRotate, ORotate)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(IValue, IWeight, OValue, OWeight)                                \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(IValue, IWeight, OValue, OWeight, Float33, Float33);                      \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(IValue, IWeight, OValue, OWeight, const Quaternion<f32>*, Float33)
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(IValue, IWeight, OValue, OWeight, const Float33*, Float33);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(IValue, IWeight, OValue, OWeight, Float33, const Float33*);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(IValue, IWeight, OValue, OWeight, const Float33*, const Float33*);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(IValue, IWeight, OValue, OWeight, const Quaternion<f32>*, const Float33*);

//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f32*, Empty, f32*, Empty);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c32*, Empty, c32*, Empty);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f64*, Empty, f64*, Empty);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c64*, Empty, c64*, Empty);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f32*, const f32*, f32*, f32*);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c32*, const f32*, c32*, f32*);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f64*, const f64*, f64*, f64*);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c64*, const f64*, c64*, f64*);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f32*, f32, f32*, f32*);
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c32*, f32, c32*, f32*);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const f64*, f64, f64*, f64*);
//    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(const c64*, f64, c64*, f64*);
}
