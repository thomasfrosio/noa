#include "noa/cpu/geometry/fft/Project.hpp"
#include "noa/cpu/memory/Set.hpp"
#include "noa/cpu/utils/Iwise.hpp"

#include "noa/core/geometry/Interpolator.hpp"
#include "noa/algorithms/geometry/FourierExtract.hpp"

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
            using interpolator_t = noa::geometry::Interpolator3D<
                    BorderMode::ZERO, InterpMode::LINEAR, value_t, i64, f32,
                    3, PointerTraits::RESTRICT, StridesTraits::STRIDED>;
            return interpolator_t(
                    AccessorRestrict<const value_t, 3, i64>(pointer_or_value, strides.filter(1, 2, 3)),
                    shape.filter(1, 2, 3).rfft(), value_t{0});
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
            typename Scale, typename Rotate>
    void extract_3d(
            InputValue input_volume, const Strides4<i64>& input_volume_strides,
            InputWeight input_weight, const Strides4<i64>& input_weight_strides,
            const Shape4<i64>& input_volume_shape,
            OutputValue output_slice, const Strides4<i64>& output_slice_strides,
            OutputWeight output_weight, const Strides4<i64>& output_weight_strides,
            const Shape4<i64>& output_slice_shape,
            const Scale& inv_scaling, const Rotate& fwd_rotation,
            f32 fftfreq_z_sinc, f32 fftfreq_z_blackman, f32 fftfreq_cutoff,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, i64 n_threads
    ) {
        const auto input_volume_interpolator = wrap_into_interpolator_(
                input_volume, input_volume_strides, input_volume_shape);
        const auto input_weight_interpolator = wrap_into_interpolator_(
                input_weight, input_weight_strides, input_volume_shape);
        const auto output_slice_accessor = wrap_into_accessor_(output_slice, output_slice_strides);
        const auto output_weight_accessor = wrap_into_accessor_(output_weight, output_weight_strides);

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = inv_scaling != Scale{};

        if (apply_ews || apply_scale) {
            const auto op = noa::geometry::fourier_extract_op<REMAP>(
                    input_volume_interpolator, input_weight_interpolator, input_volume_shape,
                    output_slice_accessor, output_weight_accessor, output_slice_shape,
                    inv_scaling, fwd_rotation, fftfreq_z_sinc, fftfreq_z_blackman, fftfreq_cutoff,
                    target_shape, ews_radius);
            if (op.windowed_sinc_size() > 1) {
                fill_with_zeros_(
                        output_slice, output_slice_strides,
                        output_weight, output_weight_strides,
                        output_slice_shape);
                noa::cpu::utils::iwise_4d(output_slice_shape.rfft().set<1>(op.windowed_sinc_size()), op, n_threads);
            } else {
                noa::cpu::utils::iwise_3d(output_slice_shape.filter(0, 2, 3).rfft(), op, n_threads);
            }
        } else {
            const auto op = noa::geometry::fourier_extract_op<REMAP>(
                    input_volume_interpolator, input_weight_interpolator, input_volume_shape,
                    output_slice_accessor, output_weight_accessor, output_slice_shape,
                    Empty{}, fwd_rotation, fftfreq_z_sinc, fftfreq_z_blackman, fftfreq_cutoff,
                    target_shape, Empty{});
            if (op.windowed_sinc_size() > 1) {
                fill_with_zeros_(
                        output_slice, output_slice_strides,
                        output_weight, output_weight_strides,
                        output_slice_shape);
                noa::cpu::utils::iwise_4d(output_slice_shape.rfft().set<1>(op.windowed_sinc_size()), op, n_threads);
            } else {
                noa::cpu::utils::iwise_3d(output_slice_shape.filter(0, 2, 3).rfft(), op, n_threads);
            }
        }
    }

    #define NOA_INSTANTIATE_INSERT_(REMAP, IValue, IWeight, OValue, OWeight, Scale, Rotate)         \
    template void extract_3d<REMAP, IValue, IWeight, OValue, OWeight, Scale, Rotate>(    \
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
}
