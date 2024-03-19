#pragma once

#include "noa/core/fft/RemapInterface.hpp"
#include "noa/core/geometry/Euler.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/core/geometry/Quaternion.hpp"
#include "noa/core/geometry/FourierInsertRasterize.hpp"
#include "noa/core/geometry/FourierInsertInterpolate.hpp"
#include "noa/core/geometry/FourierInsertExtract.hpp"
#include "noa/core/geometry/FourierExtract.hpp"
#include "noa/core/geometry/FourierGriddingCorrection.hpp"

#include "noa/unified/Array.hpp"
#include "noa/unified/Texture.hpp"
#include "noa/unified/Utilities.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Ewise.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Interpolator.hpp"
#endif

namespace noa::geometry::guts {
    template<typename Input, typename Output>
    constexpr bool is_valid_projection_input_output_value_v =
            (not std::is_const_v<Output>) and
            (nt::are_real_v<Input, Output> or
             nt::are_complex_v<Input, Output> or
             (nt::is_complex_v<Input> and nt::is_real_v<Output>));

    template<typename Input, typename Output, bool AllowTexture, bool AllowValue,
             typename InputValue = nt::value_type_t<Input>,
             typename OutputValue = nt::value_type_t<Output>>
    constexpr bool is_valid_projection_input_output_v =
            (nt::are_varray_v<Input, Output> and
             is_valid_projection_input_output_value_v<InputValue, OutputValue>)
            or
            (AllowTexture and nt::is_texture_v<Input> and nt::is_varray_v<Output> and
             is_valid_projection_input_output_value_v<InputValue, OutputValue>)
            or
            (AllowValue and nt::is_varray_v<Output> and
             is_valid_projection_input_output_value_v<Input, OutputValue>);

    template<typename Input, typename Output, bool AllowTexture, bool AllowValue>
    constexpr bool is_valid_projection_input_output_weight_v =
            (nt::is_varray_of_real_v<Output> or std::is_empty_v<Output>) and
            (nt::is_varray_of_real_v<Input> or std::is_empty_v<Output> or
             (AllowTexture and nt::is_texture_of_real_v<Input>) or
             (AllowValue and nt::is_real_v<Input>));

    template<typename Scale, typename Rotation,
             typename Coord = nt::mutable_value_type_twice_t<Rotation>>
    constexpr bool is_valid_projection_transform_v =
            (nt::is_any_v<Rotation, Mat33<f32>, Mat33<f64>, Quaternion<f32>, Quaternion<f64>> or
             nt::is_varray_of_almost_any_v<Rotation, Mat33<f32>, Mat33<f64>, Quaternion<f32>, Quaternion<f64>>) and
            (nt::is_any_v<Scale, Mat22<Coord>> or nt::is_varray_of_almost_any_v<Scale, Mat22<Coord>>);


    enum class ProjectionType { INSERT_RASTERIZE, INSERT_INTERPOLATE, EXTRACT, INSERT_EXTRACT };

    template<ProjectionType DIRECTION,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename InputScale, typename InputRotate,
             typename OutputScale = Mat22<f64>, typename OutputRotate = Mat33<f64>>
    void fourier_projection_check_parameters(
            const Input& input, const InputWeight& input_weight, const Shape4<i64>& input_shape,
            const Output& output, const OutputWeight& output_weight, const Shape4<i64>& output_shape,
            const Shape4<i64>& target_shape,
            const InputScale& input_scaling,
            const InputRotate& input_rotation,
            const OutputScale& output_scaling = {},
            const OutputRotate& output_rotation = {}
    ) {
        check(not output.is_empty(), "Empty array detected");
        const Device output_device = output.device();
        check(all(output.shape() == output_shape.rfft()),
              "The shape of the rfft output does not match the expected shape. Got output:shape={} and expected:shape={}",
              output.shape(), output_shape.rfft());

        auto check_input = [&](const auto& array, std::string_view name) {
            if constexpr (nt::is_varray_v<decltype(array)> or nt::is_texture_v<decltype(array)>) {
                check(not array.is_empty(), "Empty array detected");
                if constexpr (nt::is_varray_v<Input>)
                    check(not ni::are_overlapped(array, output), "Input and output arrays should not overlap");

                if constexpr (nt::is_texture_v<Input>) {
                    check(array.border_mode() == Border::ZERO,
                          "The texture border mode should be {}, but got {}",
                          Border::ZERO, array.border_mode());
                }
                const Device device = array.device();
                check(device == output_device,
                      "The arrays should be on the same device but got {}:device={} and output:device={}",
                      name, device, output_device);
                check(all(array.shape() == input_shape.rfft()),
                      "The shape of the rfft does not match the expected shape. Got {}:shape={} and shape={}",
                      name, array.shape(), input_shape.rfft());
            }
        };
        check_input(input, "input");
        check_input(input_weight, "input_weight");

        if constexpr (not std::is_empty_v<OutputWeight>) {
            check(not output_weight.is_empty(), "Empty array detected");
            check(not ni::are_overlapped(output_weight, output), "Output arrays should not overlap");
            check(output_weight.device() == output_device,
                  "The arrays should be on the same device, but got output_weight:device={} and output:device={}",
                  output_weight.device(), output_device);
            check(all(output_weight.shape() == output_shape.rfft()),
                  "The shape of the rfft does not match the expected shape. Got output_weight:shape={} and shape={}",
                  output_weight.shape(), output_shape.rfft());
        }

        if constexpr (DIRECTION == ProjectionType::INSERT_RASTERIZE or
                      DIRECTION == ProjectionType::INSERT_INTERPOLATE) {
            check(input_shape[1] == 1, "2d input slices are expected, but got shape={}", input_shape);
            check(output_shape[0] == 1 and target_shape[0] <= 1,
                  "A single 3d volume is expected, but got output_shape={} and target_shape={} (optional)",
                  output_shape, target_shape);
        } else if constexpr (DIRECTION == ProjectionType::EXTRACT) {
            check(input_shape[0] == 1 and target_shape[0] <= 1,
                  "A single 3d volume is expected, but got input_shape={} and target_shape={} (optional)",
                  input_shape, target_shape);
            check(output_shape[1] == 1, "2d input slices are expected but got shape {}", output_shape);
        } else { // INSERT_EXTRACT
            check(input_shape[1] == 1 and output_shape[1] == 1,
                  "2d slices are expected but got shape input:shape={} and output:shape={}",
                  input_shape, output_shape);
        }

        auto check_transform = [&]<bool IsOptional>(const auto& transform, i64 required_size, std::string_view name) {
            if constexpr (IsOptional) {
                if (transform.is_empty())
                    return;
            } else {
                check(not transform.is_empty(), "{} should not be empty", name);
            }
            check(ni::is_contiguous_vector(transform) and transform.elements() == required_size,
                  "{} should be a contiguous vector of n_slices={}, but got {}:shape={}, {}:strides={}",
                  name, required_size, name, transform.shape(), name, transform.strides());
            check(transform.device() == output_device, "{} should be on the compute device", name);
        };

        const auto required_count = DIRECTION == ProjectionType::EXTRACT ? output_shape[0] : input_shape[0];
        if constexpr (nt::is_varray_v<InputScale>)
            check_transform<true>(input_scaling, required_count, "input_scaling");
        if constexpr (nt::is_varray_v<InputRotate>)
            check_transform<false>(input_rotation, required_count, "input_rotation");

        // Only for INSERT_EXTRACT.
        if constexpr (nt::is_varray_v<OutputScale>)
            check_transform<true>(output_scaling, output_shape[0], "output_scaling");
        if constexpr (nt::is_varray_v<OutputRotate>)
            check_transform<false>(output_rotation, output_shape[0], "output_rotation");
    }

    auto fourier_insert_extract_interp_mode(const auto& input, const auto& input_weight) -> Interp {
        Interp interp{Interp::LINEAR};
        if constexpr (nt::is_texture_v<decltype(input)>) {
            interp = input.interp_mode();
            if constexpr (nt::is_texture_v<decltype(input_weight)>) {
                check(input_weight.interp_mode() == interp,
                      "Input GPU textures are currently limited to using the same interpolation mode, "
                      "but got input:interp_mode={} and input_weight:interp_mode={}",
                      interp, input_weight.interp_mode());
            }
        } else if constexpr (nt::is_texture_v<decltype(input_weight)>) {
            interp = input_weight.interp_mode();
        }
        check(interp == Interp::LINEAR or interp == Interp::LINEAR_FAST,
              "Only the linear interpolation methods are currently supported, but got interp_mode={}",
              interp);
        return interp;
    }

    template<bool IsCPU, typename Index, Interp InterpMode, typename Coord, size_t N, typename T, typename... I>
    auto fourier_insert_extract_to_interpolator(const T& object, const Shape<i64, N>& shape, I... filter) {
        using mutable_value_t = nt::mutable_value_type_t<T>;
        if constexpr (nt::is_varray_v<T> or (nt::is_texture_v<T> and IsCPU)) {
            using accessor_t = AccessorRestrict<const mutable_value_t, N, Index>;
            using interpolator_t = InterpolatorNd<N, Border::ZERO, InterpMode, accessor_t>;
            return interpolator_t(accessor_t(object.get(), object.strides().filter(filter...).template as<Index>()),
                                  shape.template as<Index>());

        } else if constexpr (nt::is_texture_v<T> and not IsCPU) {
            #ifdef NOA_ENABLE_CUDA
            constexpr bool is_layered = N == 2;
            using interpolator_t = noa::cuda::geometry::InterpolatorNd<N, InterpMode, mutable_value_t, false, is_layered, Coord>;
            return interpolator_t(object.gpu()->texture);
            #else
            panic("No GPU backend detected");
            #endif

        } else if constexpr (std::is_empty_v<T>) {
            return object;

        } else { // real or complex
            using accessor_t = AccessorValue<const mutable_value_t, Index>;
            using interpolator_t = InterpolatorNd<N, Border::ZERO, InterpMode, accessor_t>;
            return interpolator_t(accessor_t(object), shape.template as<Index>());
        }
    }

    template<bool EnforceConst, bool Contiguous, size_t N, typename Index, typename T, typename... I>
    constexpr auto fourier_insert_extract_to_accessor(const T& object, I... filter) {
        if constexpr (nt::is_varray_v<T>) {
            using mutable_value_t = nt::mutable_value_type_t<T>;
            using value_t = std::conditional_t<EnforceConst, const mutable_value_t, mutable_value_t>;
            constexpr auto stride_trait = Contiguous ? StridesTraits::CONTIGUOUS : StridesTraits::STRIDED;
            using accessor_t = AccessorRestrict<value_t, N, Index, stride_trait>;
            return accessor_t(object.get(), object.strides().filter(filter...).template as<Index>());
        } else { // value or empty
            using value_t = std::conditional_t<EnforceConst, const std::decay_t<T>, T>;
            return AccessorValue<value_t>(object);
        }
    }

    template<typename T>
    constexpr bool fourier_project_has_scale(const T& scale) {
        if constexpr (nt::is_mat22_v<T>)
            return scale != T::eye(1);
        else if constexpr (nt::is_varray_v<T> and nt::is_mat22_v<nt::value_type_t<T>>)
            return true;
        else if constexpr (std::is_empty_v<T>)
            return false;
        else
            static_assert(nt::always_false_v<T>);
    }

    template<noa::fft::RemapInterface Remap, typename Index,
            typename Input, typename InputWeight,
            typename Output, typename OutputWeight,
            typename Scale, typename Rotate>
    void launch_fourier_insert_rasterize_3d(
            const Input& slice, const InputWeight& slice_weight, const Shape4<i64> slice_shape,
            const Output& volume, const OutputWeight& volume_weight, const Shape4<i64> volume_shape,
            const Scale& scaling, const Rotate& rotation,
            f64 fftfreq_cutoff, Shape4<i64> target_shape, Vec2<f64> ews_radius
    ) {
        using coord_t = nt::value_type_twice_t<Rotate>;
        auto slice_accessor = fourier_insert_extract_to_accessor<true, false, 3, Index>(slice, 0, 2, 3);
        auto slice_weight_accessor = fourier_insert_extract_to_accessor<true, false, 3, Index>(slice_weight, 0, 2, 3);
        auto volume_accessor = fourier_insert_extract_to_accessor<false, false, 3, Index>(volume, 1, 2, 3);
        auto volume_weight_accessor = fourier_insert_extract_to_accessor<false, false, 3, Index>(volume_weight, 1, 2, 3);
        auto scaling_accessor = fourier_insert_extract_to_accessor<true, true, 1, Index>(scaling);
        auto rotation_accessor = fourier_insert_extract_to_accessor<true, true, 1, Index>(rotation);
        using slice_at = decltype(slice_accessor);
        using slice_weight_at = decltype(slice_weight_accessor);
        using volume_at = decltype(volume_accessor);
        using volume_weight_at = decltype(volume_weight_accessor);
        using scale_at = decltype(scaling_accessor);
        using rotate_at = decltype(rotation_accessor);

        const auto s_input_slice_shape = slice_shape.as<Index>();
        const auto s_volume_shape = volume_shape.as<Index>();
        const auto s_target_shape = target_shape.as<Index>();
        const auto iwise_shape = s_input_slice_shape.filter(0, 2, 3).rfft();
        const auto has_ews = any(ews_radius != 0);
        const bool has_scale = fourier_project_has_scale(scaling);
        const auto fftfreq_cutoff_ = static_cast<coord_t>(fftfreq_cutoff);
        const auto ews_radius_ = ews_radius.template as<coord_t>();
        const auto device = volume.device();

        if (has_ews or has_scale) {
            auto op = FourierInsertRasterize
                    <Remap.remap, Index, scale_at, rotate_at, Vec2<coord_t>,
                     slice_at, slice_weight_at, volume_at, volume_weight_at>
                    (slice_accessor, slice_weight_accessor, s_input_slice_shape,
                     volume_accessor, volume_weight_accessor, s_volume_shape,
                     scaling_accessor, rotation_accessor,
                     fftfreq_cutoff_, s_target_shape, ews_radius_);
            iwise(iwise_shape, device, op, slice, slice_weight, volume, volume_weight, scaling, rotation);
        } else {
            auto op = FourierInsertRasterize
                    <Remap.remap, Index, Empty, rotate_at, Empty,
                     slice_at, slice_weight_at, volume_at, volume_weight_at>
                    (slice_accessor, slice_weight_accessor, s_input_slice_shape,
                     volume_accessor, volume_weight_accessor, s_volume_shape,
                     {}, rotation_accessor, fftfreq_cutoff_, s_target_shape, {});
            iwise(iwise_shape, device, op, slice, slice_weight, volume, volume_weight, rotation);
        }
    }

    template<noa::fft::RemapInterface Remap, typename Index, bool IsCPU,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename Scale, typename Rotate>
    void launch_fourier_insert_interpolate_3d(
            const Input& slice, const InputWeight& slice_weight, const Shape4<i64> slice_shape,
            const Output& volume, const OutputWeight& volume_weight, const Shape4<i64> volume_shape,
            const Scale& scaling, const Rotate& rotation, Interp interp_mode,
            f64 fftfreq_sinc, f64 fftfreq_blackman, f64 fftfreq_cutoff,
            Shape4<i64> target_shape, Vec2<f64> ews_radius
    ) {
        using coord_t = nt::value_type_twice_t<Rotate>;
        auto volume_accessor = fourier_insert_extract_to_accessor<false, false, 3, Index>(volume, 1, 2, 3);
        auto volume_weight_accessor = fourier_insert_extract_to_accessor<false, false, 3, Index>(volume_weight, 1, 2, 3);
        auto scaling_accessor = fourier_insert_extract_to_accessor<true, true, 1, Index>(scaling);
        auto rotation_accessor = fourier_insert_extract_to_accessor<true, true, 1, Index>(rotation);
        using volume_at = decltype(volume_accessor);
        using weight_at = decltype(volume_weight_accessor);
        using scale_at = decltype(scaling_accessor);
        using rotate_at = decltype(rotation_accessor);

        const auto s_slice_shape = slice_shape.as<Index>();
        const auto s_volume_shape = volume_shape.as<Index>();
        const auto s_target_shape = target_shape.as<Index>();
        const auto iwise_shape = s_slice_shape.filter(0, 2, 3).rfft();
        const auto fftfreq_sinc_ = static_cast<coord_t>(fftfreq_sinc);
        const auto fftfreq_blackman_ = static_cast<coord_t>(fftfreq_blackman);
        const auto fftfreq_cutoff_ = static_cast<coord_t>(fftfreq_cutoff);
        const auto ews_radius_ = ews_radius.template as<coord_t>();
        const auto has_ews = any(ews_radius != 0);
        const bool has_scale = fourier_project_has_scale(scaling);
        const auto device = volume.device();

        auto launch = [&]<typename T, typename U>(const T& slice_interpolator, const U& slice_weight_interpolator){
            if (has_ews or has_scale) {
                auto op = FourierInsertInterpolate
                        <Remap.remap, Index, scale_at, rotate_at, Vec2<coord_t>, T, U, volume_at, weight_at>
                        (slice_interpolator, slice_weight_interpolator, s_slice_shape,
                        volume_accessor, volume_weight_accessor, s_volume_shape,
                        scaling_accessor, rotation_accessor,
                        fftfreq_sinc_, fftfreq_blackman_, fftfreq_cutoff_, s_target_shape, ews_radius_);
                iwise(iwise_shape, device, op, slice, slice_weight, volume, volume_weight, scaling, rotation);
            } else {
                auto op = FourierInsertInterpolate
                        <Remap.remap, Index, Empty, rotate_at, Empty, T, U, volume_at, weight_at>
                        (slice_interpolator, slice_weight_interpolator, s_slice_shape,
                         volume_accessor, volume_weight_accessor, s_volume_shape,
                         {}, rotation_accessor,
                         fftfreq_sinc_, fftfreq_blackman_, fftfreq_cutoff_, s_target_shape, {});
                iwise(iwise_shape, device, op, slice, slice_weight, volume, volume_weight, rotation);
            }
        };

        auto slice_shape_2d = slice_shape.filter(2, 3);
        if (interp_mode == Interp::LINEAR) {
            auto input_slice_interpolator = fourier_insert_extract_to_interpolator
                    <IsCPU, Index, Interp::LINEAR, coord_t>(slice, slice_shape_2d, 0, 2, 3);
            auto input_slice_weight_interpolator = fourier_insert_extract_to_interpolator
                    <IsCPU, Index, Interp::LINEAR, coord_t>(slice_weight, slice_shape_2d, 0, 2, 3);
            launch(input_slice_interpolator, input_slice_weight_interpolator);
        } else {
            auto input_slice_interpolator = fourier_insert_extract_to_interpolator
                    <IsCPU, Index, Interp::LINEAR_FAST, coord_t>(slice, slice_shape_2d, 0, 2, 3);
            auto input_slice_weight_interpolator = fourier_insert_extract_to_interpolator
                    <IsCPU, Index, Interp::LINEAR_FAST, coord_t>(slice_weight, slice_shape_2d, 0, 2, 3);
            launch(input_slice_interpolator, input_slice_weight_interpolator);
        }
    }

    template<noa::fft::RemapInterface Remap, typename Index, bool IsCPU,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename Scale, typename Rotate>
    void launch_fourier_extract_3d(
            const Input& volume, const InputWeight& volume_weight, const Shape4<i64> volume_shape,
            const Output& slice, const OutputWeight& slice_weight, const Shape4<i64> slice_shape,
            const Scale& scaling, const Rotate& rotation, Interp interp_mode,
            f64 fftfreq_sinc, f64 fftfreq_blackman, f64 fftfreq_cutoff,
            Shape4<i64> target_shape, Vec2<f64> ews_radius
    ) {
        using coord_t = nt::value_type_twice_t<Rotate>;
        auto slice_accessor = fourier_insert_extract_to_accessor<false, false, 3, Index>(slice, 0, 2, 3);
        auto slice_weight_accessor = fourier_insert_extract_to_accessor<false, false, 3, Index>(slice_weight, 0, 2, 3);
        auto scaling_accessor = fourier_insert_extract_to_accessor<true, true, 1, Index>(scaling);
        auto rotation_accessor = fourier_insert_extract_to_accessor<true, true, 1, Index>(rotation);
        using slice_at = decltype(slice_accessor);
        using weight_at = decltype(slice_weight_accessor);
        using scale_at = decltype(scaling_accessor);
        using rotate_at = decltype(rotation_accessor);

        const auto s_slice_shape = slice_shape.as<Index>();
        const auto s_volume_shape = volume_shape.as<Index>();
        const auto s_target_shape = target_shape.as<Index>();
        const auto fftfreq_sinc_ = static_cast<coord_t>(fftfreq_sinc);
        const auto fftfreq_blackman_ = static_cast<coord_t>(fftfreq_blackman);
        const auto fftfreq_cutoff_ = static_cast<coord_t>(fftfreq_cutoff);
        const auto ews_radius_ = ews_radius.template as<coord_t>();
        const auto has_ews = any(ews_radius != 0);
        const bool has_scale = fourier_project_has_scale(scaling);
        const auto device = volume.device();

        auto fill_with_zeros = [&] {
            if constexpr (std::is_empty_v<OutputWeight>)
                ewise({}, wrap(slice), ZeroInitialize{});
            else
                ewise({}, wrap(slice, slice_weight), ZeroInitialize{});
        };

        auto launch = [&]<typename T, typename U>(const T& volume_interpolator, const U& volume_weight_interpolator){
            if (has_ews or has_scale) {
                auto op = FourierExtract<Remap.remap, Index, scale_at, rotate_at, Vec2<coord_t>, T, U, slice_at, weight_at>
                        (volume_interpolator, volume_weight_interpolator, s_volume_shape,
                         slice_accessor, slice_weight_accessor, s_slice_shape,
                         scaling_accessor, rotation_accessor,
                         fftfreq_sinc_, fftfreq_blackman_, fftfreq_cutoff_, s_target_shape, ews_radius_);
                if (op.windowed_sinc_size() > 1) {
                    fill_with_zeros();
                    const auto iwise_shape = s_slice_shape.set<1>(op.windowed_sinc_size()).rfft();
                    iwise(iwise_shape, device, op, slice, slice_weight, volume, volume_weight, scaling, rotation);
                } else {
                    const auto iwise_shape = s_slice_shape.filter(0, 2, 3).rfft();
                    iwise(iwise_shape, device, op, slice, slice_weight, volume, volume_weight, scaling, rotation);
                }
            } else {
                auto op = FourierExtract<Remap.remap, Index, Empty, rotate_at, Empty, T, U, slice_at, weight_at>
                        (volume_interpolator, volume_weight_interpolator, s_volume_shape,
                         slice_accessor, slice_weight_accessor, s_slice_shape,
                         {}, rotation_accessor,
                         fftfreq_sinc_, fftfreq_blackman_, fftfreq_cutoff_, s_target_shape, {});
                if (op.windowed_sinc_size() > 1) {
                    fill_with_zeros();
                    const auto iwise_shape = s_slice_shape.set<1>(op.windowed_sinc_size()).rfft();
                    iwise(iwise_shape, device, op, slice, slice_weight, volume, volume_weight, rotation);
                } else {
                    const auto iwise_shape = s_slice_shape.filter(0, 2, 3).rfft();
                    iwise(iwise_shape, device, op, slice, slice_weight, volume, volume_weight, rotation);
                }
            }
        };

        auto volume_shape_3d = volume_shape.filter(1, 2, 3);
        if (interp_mode == Interp::LINEAR) {
            auto input_volume_interpolator = fourier_insert_extract_to_interpolator
                    <IsCPU, Index, Interp::LINEAR, coord_t>(volume, volume_shape_3d, 1, 2, 3);
            auto input_volume_weight_interpolator = fourier_insert_extract_to_interpolator
                    <IsCPU, Index, Interp::LINEAR, coord_t>(volume_weight, volume_shape_3d, 1, 2, 3);
            launch(input_volume_interpolator, input_volume_weight_interpolator);
        } else {
            auto input_volume_interpolator = fourier_insert_extract_to_interpolator
                    <IsCPU, Index, Interp::LINEAR_FAST, coord_t>(volume, volume_shape_3d, 1, 2, 3);
            auto input_volume_weight_interpolator = fourier_insert_extract_to_interpolator
                    <IsCPU, Index, Interp::LINEAR_FAST, coord_t>(volume_weight, volume_shape_3d, 1, 2, 3);
            launch(input_volume_interpolator, input_volume_weight_interpolator);
        }
    }
}

namespace noa::geometry {
    struct FourierInsertRasterizeOptions {
        /// Frequency cutoff of the output volume, in cycle/pix.
        /// Frequencies above this are left unchanged.
        f64 fftfreq_cutoff{0.5};

        /// Actual BDHW logical shape of the 3d volume.
        /// The function normalizes the slice and volume dimensions, and works with normalized frequencies.
        /// By default (empty target_shape or target_shape == volume_shape), the slice frequencies are mapped onto
        /// the volume frequencies. If the volume is larger than the slices, the slices are implicitly stretched
        /// (over-sampling case). If the volume is smaller than the slices, the slices are shrank (under-sampling case).
        /// However, if target_shape is specified, the slice frequencies are instead mapped onto the frequencies of a
        /// 3d FFT volume of target_shape. In this case, volume is the region to "render" within the volume, defined
        /// by target_shape, centered on the DC. This can be useful for instance to only render a subregion of
        /// target_shape.
        Shape4<i64> target_shape{};

        /// HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
        /// If negative, the negative curve is computed. If {0,0}, the slices are projections.
        /// In order to have both left and right beams assigned to different values, the function only computes one
        /// "side" of the EWS, as specified by ews_radius. To insert the other side, one has to call the function
        /// a second time with ews_radius * -1.
        Vec2<f64> ews_radius{};
    };

    /// Inserts 2d Fourier central-slice(s) into a 3d Fourier volume, using tri-linear rasterization.
    /// \details Fourier-insertion using rasterization/gridding to insert central-slices in a volume.
    ///          This method is mostly used for cases with a lot of central-slices (where errors are averaged-out)
    ///          and is likely the most efficient way of implementing backward projection. Note however that this
    ///          method is not the most accurate (central-slices are modeled using a simple trilinear-pulse for the
    ///          rasterization). Gridding correction can be beneficial as post-processing one the real-space output.
    ///          A density correction (i.e. normalization) is required. This can easily be achieved by inserting the
    ///          per-slice weights into another volume to keep track of what was inserted and where.
    ///
    /// \tparam REMAP               Remapping from the slice to the volume layout.
    ///                             Should be H2H, H2HC, HC2H or HC2HC.
    /// \tparam Input               f32|f64|c32|c64, or a varray of any of these types.
    /// \tparam InputWeight         f32|f64, or a varray of any of these types, or empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64, or empty.
    /// \tparam Scale               Empty, Mat22 or a varray of this type.
    /// \tparam Rotate              Mat33, Quaternion, or a varray of these types.
    ///                             Sets the floating-point precision of the transformation.
    ///
    /// \param[in] slice            2d-rfft central-slice(s) to insert (can be a constant value).
    /// \param[in] slice_weight     (optional) Varray|value associated with \p slice. Default to ones.
    /// \param slice_shape          BDHW logical shape of \p slice.
    /// \param[out] volume          3d-rfft volume inside which the slices are inserted.
    /// \param[out] volume_weight   (optional) Varray associated with \p volume.
    /// \param volume_shape         BDHW logical shape of \p volume.
    /// \param[in] inv_scaling      2x2 HW \e inverse real-space scaling matrix to apply to the slices
    ///                             before the rotation. If an array is passed, it can be empty or have
    ///                             one matrix per slice. Otherwise the same scaling matrix is applied
    ///                             to every slice.
    /// \param[in] fwd_rotation     3x3 DHW \e forward rotation-matrices or quaternions to apply to the slices.
    ///                             If an array is passed, it should have one element per slice.
    ///                             Otherwise the same rotation is applied to every slice.
    /// \param options              Insertion options.
    ///
    /// \note The scaling and the rotation are kept separated from one another in order to properly compute the
    ///       curve of the Ewald sphere. Indeed, the scaling is applied first to correct for magnification, so that the
    ///       EWS is computed using the original frequencies (from the scattering) and is therefore spherical even
    ///       under anisotropic magnification. If \p options.ews_radius is 0, the scaling factors can be merged with
    ///       the rotations (if a rotation-matrix is passed).
    template<noa::fft::RemapInterface Remap,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale = Empty, typename Rotate>
    requires (guts::is_valid_projection_input_output_v<Input, Output, false, true> and
              guts::is_valid_projection_input_output_weight_v<InputWeight, OutputWeight, false, true> and
              guts::is_valid_projection_transform_v<Scale, Rotate> and
              Remap.is_hx2xx() and Remap.is_xx2hx())
    void fourier_insert_rasterize_3d(
            const Input& slice,
            const InputWeight& slice_weight,
            const Shape4<i64> slice_shape,
            const Output& volume,
            const OutputWeight& volume_weight,
            const Shape4<i64> volume_shape,
            const Scale& inv_scaling,
            const Rotate& fwd_rotation,
            const FourierInsertRasterizeOptions& options = {}
    ) {
        guts::fourier_projection_check_parameters<guts::ProjectionType::INSERT_RASTERIZE>(
                slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
                options.target_shape, inv_scaling, fwd_rotation);

        if (volume.device().is_gpu()) {
            check(ng::is_accessor_access_safe<i32>(slice, slice.shape()) and
                  ng::is_accessor_access_safe<i32>(slice_weight, slice_weight.shape()) and
                  ng::is_accessor_access_safe<i32>(volume, volume.shape()) and
                  ng::is_accessor_access_safe<i32>(volume_weight, volume_weight.shape()),
                  "GPU backend only instantiate i32-based accessor indexing, "
                  "which is unsafe for the given input and output arrays. "
                  "Please report this.");

            guts::launch_fourier_insert_rasterize_3d<Remap, i32>(
                    slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
                    inv_scaling, fwd_rotation, options.fftfreq_cutoff, options.target_shape, options.ews_radius);
        } else {
            guts::launch_fourier_insert_rasterize_3d<Remap, i64>(
                    slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
                    inv_scaling, fwd_rotation, options.fftfreq_cutoff, options.target_shape, options.ews_radius);
        }
    }

    /// Settings for the windowed-sinc convolution of the central-slice.\n
    /// \b Fourier-insertion: Central slices are inserted in a (virtual) volume. This parameter defines
    /// the windowed-sinc that is convolved along the normal of the perfectly thin slice(s) to insert.\n
    /// \b Fourier-extraction: Central slices are extracted from a (virtual) volume. This parameter defines the
    /// windowed-sinc that is convolved, along the z of the reconstruction, with the perfectly thin slice(s) to
    /// extract. This is used to effectively apply an horizontal (smooth) rectangular mask centered on the object
    /// _before_ the forward projection. The current API doesn't allow to change the orientation of this sinc
    /// (it is always along z) since its only purpose was originally to improve projections from tomograms by
    /// masking out the noise from above and below the sample.
    struct WindowedSinc {
        /// Frequency, in cycle/pix, of the first zero of the sinc.
        /// This is clamped to ensure a minimum of 1 pixel diameter,
        /// which is usually want we want for Fourier insertion.
        f64 fftfreq_sinc{-1};

        /// Frequency, in cycle/pix, where the blackman window stops (weight is 0 at this frequency).
        /// This parameter is usually here to control the accuracy/performance ratio, but it can also be used
        /// to control the smoothness of the corresponding real-space mask. Usually this value should be a
        /// multiple of the sinc-cutoff. The larger this multiple, the sharper the step window, but the slower
        /// it is to compute the slice.
        /// This is clamped to ensure the window stops at least to the first sinc-cutoff.
        /// So if both frequencies are left to their default value (-1), a 1 pixel thick slice
        /// is generated, which is usually want we want for Fourier insertion.
        f64 fftfreq_blackman{-1};
    };

    struct FourierInsertInterpolateOptions {
        /// Windowed-sinc convolved along the normal of the input slice(s).
        WindowedSinc windowed_sinc{};

        /// Frequency cutoff in the output volume, in cycle/pix.
        /// Frequencies above this are left unchanged.
        f64 fftfreq_cutoff{0.5};

        /// Actual BDHW logical shape of the 3d volume.
        /// See FourierInsertRasterizeOptions for more details.
        Shape4<i64> target_shape{};

        /// HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
        /// See FourierInsertRasterizeOptions for more details.
        Vec2<f64> ews_radius{};
    };

    /// Fourier-insertion using 2d-interpolation to insert central-slices in the volume.
    /// \details This method is the most accurate one but is certainly slower than rasterization. Here, instead of
    ///          calling every pixel in the central-slices for rasterization, every voxel in the volume is sampled
    ///          by collecting the contribution of every central-slice for each output voxel. The advantage is that
    ///          it allows to use a more accurate model for the central-slices, i.e., a windowed-sinc. Indeed, slices
    ///          are now effectively convolved with a windowed-sinc (both the sinc frequency and window size can be
    ///          controlled) along their normal before the insertion. Note that this (windowed) sinc translates to
    ///          a (smooth) rectangular mask in real-space, along the normal of the slice (an interesting property
    ///          that can be useful for some applications).
    /// \warning This function computes the inverse transformation compared to the overload above using rasterization.
    ///
    /// \tparam REMAP               Remapping from the slice to the volume layout. Should be HC2H or HC2HC.
    /// \tparam Input               f32|f64|c32|c64, or a varray|texture of any of these types.
    /// \tparam InputWeight         f32|f64, a varray|texture of any of these types, or empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64, or empty.
    /// \tparam Scale               Mat22, a varray of Mat22, or empty.
    /// \tparam Rotate              Mat33|Quaternion, or a varray of this type.
    ///                             Sets the floating-point precision of the transformation.
    ///
    /// \param[in] slice            2d-rfft central-slice(s) to insert (can be a constant value).
    /// \param[in] slice_weight     (options) Varray|texture|value associated with \p slice. Default to ones.
    /// \param slice_shape          BDHW logical shape of \p slice.
    /// \param[out] volume          3d-rfft volume inside which the slices are to be inserted.
    /// \param[out] volume_weight   (options) Varray associated with \p volume.
    /// \param volume_shape         BDHW logical shape of \p volume.
    /// \param[in] fwd_scaling      2x2 HW \e forward real-space scaling matrix to apply to the slices
    ///                             before the rotation. If an array is passed, it can be empty or have
    ///                             one matrix per slice. Otherwise the same scaling matrix is applied
    ///                             to every slice.
    /// \param[in] inv_rotation     3x3 DHW \e inverse rotation-matrices or quaternions to apply to the slices.
    ///                             If an array is passed, it should have one rotation per slice.
    ///                             Otherwise the same rotation is applied to every slice.
    /// \param options              Insertion options.
    template<noa::fft::RemapInterface Remap,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale = Empty, typename Rotate>
    requires (guts::is_valid_projection_input_output_v<Input, Output, true, true> and
              guts::is_valid_projection_input_output_weight_v<InputWeight, OutputWeight, true, true> and
              guts::is_valid_projection_transform_v<Scale, Rotate> and
              Remap.is_hc2xx() and Remap.is_xx2hx())
    void insert_interpolate_3d(
            const Input& slice,
            const InputWeight& slice_weight,
            const Shape4<i64> slice_shape,
            const Output& volume,
            const OutputWeight& volume_weight,
            const Shape4<i64> volume_shape,
            const Scale& fwd_scaling,
            const Rotate& inv_rotation,
            const FourierInsertInterpolateOptions& options = {}
    ) {
        guts::fourier_projection_check_parameters<guts::ProjectionType::INSERT_INTERPOLATE>(
                slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
                options.target_shape, fwd_scaling, inv_rotation);

        const Interp interp_mode = guts::fourier_insert_extract_interp_mode(slice, slice_weight);

        if (volume.device().is_gpu()) {
            check(ng::is_accessor_access_safe<i32>(slice, slice.shape()) and
                  ng::is_accessor_access_safe<i32>(slice_weight, slice_weight.shape()) and
                  ng::is_accessor_access_safe<i32>(volume, volume.shape()) and
                  ng::is_accessor_access_safe<i32>(volume_weight, volume_weight.shape()),
                  "GPU backend only instantiate i32-based accessor indexing, "
                  "which is unsafe for the given input and output arrays. "
                  "Please report this.");

            guts::launch_fourier_insert_interpolate_3d<Remap, i32, false>(
                    slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
                    fwd_scaling, inv_rotation, interp_mode,
                    options.windowed_sinc.fftfreq_sinc, options.windowed_sinc.fftfreq_blackman,
                    options.fftfreq_cutoff, options.target_shape, options.ews_radius);
        } else {
            guts::launch_fourier_insert_interpolate_3d<Remap, i64, true>(
                    slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
                    fwd_scaling, inv_rotation, interp_mode,
                    options.windowed_sinc.fftfreq_sinc, options.windowed_sinc.fftfreq_blackman,
                    options.fftfreq_cutoff, options.target_shape, options.ews_radius);
        }
    }

    struct FourierExtractOptions {
        /// Windowed-sinc convolved along the w (the depth) of the Fourier volume.
        WindowedSinc w_windowed_sinc{};

        /// Frequency cutoff in the output volume, in cycle/pix.
        /// Frequencies above this are set to zero.
        f64 fftfreq_cutoff{0.5};

        /// Actual BDHW logical shape of the 3d volume.
        /// See FourierInsertRasterizeOptions for more details.
        Shape4<i64> target_shape{};

        /// HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
        /// See FourierInsertRasterizeOptions for more details.
        Vec2<f64> ews_radius{};
    };

    /// Extracts 2d central-slice(s) from a volume.
    /// \details This is the reverse operation of the Fourier insertion. There are two main behaviors (both
    ///          controlled by the \p z_windowed_sinc parameter): 1. (default) A simple and fast slice extraction,
    ///          where every pixel of the slice(s) are sampled from the volume using 3d-interpolation.
    ///          2. A z-windowed-sinc slice extraction. This is similar, but instead of simply extracting the slice
    ///          from the volume, it convolves the volume with a 1d windowed-sinc along the z-axis of the volume.
    ///          Note that the convolution is simplified to a simple per-slice weighted-mean along the z-axis of the
    ///          volume. This windowed-sinc convolution translates to a (smooth) rectangular mask along the z-axis
    ///          and centered on the ifft of the volume. As such, if such masking is required, this method can replace
    ///          the real-space masking, which could be advantageous in scenarios where going back to real-space
    ///          is expensive.
    ///
    /// \tparam REMAP               Remapping from the volume to the slice layout. Should be HC2H or HC2HC.
    /// \tparam Input               A varray|texture of (const) f32|f64|c32|c64.
    /// \tparam InputWeight         A varray|texture of (const) f32|f64, or empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64, or empty.
    /// \tparam Scale               Mat22, a varray of Mat22, or empty.
    /// \tparam Rotate              Mat33|Quaternion, or a varray of this type.
    ///
    /// \param[in] volume           3d-centered-rfft volume from which to extract the slices.
    /// \param[in] volume_weight    (optional) Varray|texture associated with \p volume.
    /// \param volume_shape         BDHW logical shape of \p volume.
    /// \param[out] slice           2d-rfft central-slice(s) to extract.
    /// \param[out] slice_weight    (optional) Varray associated with \p slice.
    /// \param slice_shape          BDHW logical shape of \p slice.
    /// \param[in] inv_scaling      2x2 HW \e inverse real-space scaling to apply to the slices before the rotation.
    ///                             If an array is passed, it can be empty or have one matrix per slice.
    ///                             Otherwise the same scaling matrix is applied to every slice.
    /// \param[in] fwd_rotation     3x3 DHW \e forward rotation-matrices or quaternions to apply to the slices.
    ///                             If an array is passed, it should have one rotation per slice.
    ///                             Otherwise the same rotation is applied to every slice.
    /// \param options              Extraction options.
    template<noa::fft::RemapInterface Remap,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale = Empty, typename Rotate>
    requires (guts::is_valid_projection_input_output_v<Input, Output, true, false> and
              guts::is_valid_projection_input_output_weight_v<InputWeight, OutputWeight, true, false> and
              guts::is_valid_projection_transform_v<Scale, Rotate> and
              Remap.is_hc2xx() and Remap.is_xx2hx())
    void fourier_extract_3d(
            const Input& volume,
            const InputWeight& volume_weight,
            const Shape4<i64> volume_shape,
            const Output& slice,
            const OutputWeight& slice_weight,
            const Shape4<i64> slice_shape,
            const Scale& inv_scaling,
            const Rotate& fwd_rotation,
            const FourierExtractOptions& options = {}
    ) {
        guts::fourier_projection_check_parameters<guts::ProjectionType::EXTRACT>(
                volume, volume_weight, volume_shape, slice, slice_weight, slice_shape,
                options.target_shape, inv_scaling, fwd_rotation);

        const Interp interp_mode = guts::fourier_insert_extract_interp_mode(volume, volume_weight);

        if (volume.device().is_gpu()) {
            check(ng::is_accessor_access_safe<i32>(slice, slice.shape()) and
                  ng::is_accessor_access_safe<i32>(slice_weight, slice_weight.shape()) and
                  ng::is_accessor_access_safe<i32>(volume, volume.shape()) and
                  ng::is_accessor_access_safe<i32>(volume_weight, volume_weight.shape()),
                  "GPU backend only instantiate i32-based accessor indexing, "
                  "which is unsafe for the given input and output arrays. "
                  "Please report this.");

            guts::launch_fourier_extract_3d<Remap, i32, false>(
                    volume, volume_weight, volume_shape, slice, slice_weight, slice_shape,
                    inv_scaling, fwd_rotation, interp_mode,
                    options.w_windowed_sinc.fftfreq_sinc, options.w_windowed_sinc.fftfreq_blackman,
                    options.fftfreq_cutoff, options.target_shape, options.ews_radius);
        } else {
            guts::launch_fourier_extract_3d<Remap, i64, true>(
                    volume, volume_weight, volume_shape, slice, slice_weight, slice_shape,
                    inv_scaling, fwd_rotation, interp_mode,
                    options.w_windowed_sinc.fftfreq_sinc, options.w_windowed_sinc.fftfreq_blackman,
                    options.fftfreq_cutoff, options.target_shape, options.ews_radius);
        }
    }

    struct FourierInsertExtractOptions {
        /// Windowed-sinc convolved along the normal of the input slice(s).
        WindowedSinc input_windowed_sinc{};

        /// Windowed-sinc convolved along the w (the depth) of the virtual Fourier volume.
        WindowedSinc w_windowed_sinc{};

        /// Whether the contribution of the input slices should be added to the output.
        /// By default, the operator _sets_ the output_(slice|weight). With this option enabled, it instead adds the
        /// contribution of input_(slice|weight) to the signal already present in the output_(slice|weight), allowing
        /// to progressively build the output_(slice|weight).
        bool add_to_output{false};

        /// Correct for the multiplicity, i.e. divide the output sampled values by their corresponding weight.
        /// This is equivalent of doing `output_slice/max(1, output_weight)` after the function and assumes all of
        /// the input slices are included in the same function call. It is not compatible with add_to_output=true or
        /// with a non-default w_windowed_sinc.
        /// This can be useful for cases where there's no real-space mask to apply before the forward projection and
        /// if the weights are to be applied right away and not used for anything else (in which case output_weight
        /// doesn't need to be computed and can be left empty).
        bool correct_multiplicity{false};

        /// Frequency cutoff of the virtual 3d volume, in cycle/pix.
        f64 fftfreq_cutoff{0.5};

        /// HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
        /// If negative, the negative curve is computed.
        /// If {0,0}, the slices are projections.
        Vec2<f64> ews_radius{};
    };

    /// Extracts 2d central-slice(s) from a virtual volume filled by other central-slices.
    /// \details This function effectively combines the insertion and extraction, but instead of actually inserting
    ///          slices into a volume, it directly inserts them in the extracted slices. In other words, it builds a
    ///          virtual volume, made of central-slices, and this virtual volume is then sampled at (exactly) the
    ///          frequency of the central-slices to extract. This has massive performance benefits, because it only
    ///          samples the frequency of the output slices and never allocates/reconstructs the volume. It is also
    ///          more accurate since the volume is never actually discretized (thus skipping a layer of interpolation).
    ///          Note that these performance benefits are expected to disappear if thousands (possibly hundreds?) of
    ///          slices are extracted. Indeed, for every output slice, the operator needs to sample the volume by
    ///          collecting the signal of every input slice using 2d-interpolation. This is as opposed to the other
    ///          extract method, where the volume is already sampled, making the extraction much cheaper (and constant
    ///          cost: it's a simple 3d-interpolation).
    ///
    /// \tparam REMAP                   Remapping from the input slice to the output slice layout. Should be HC2H or HC2HC.
    /// \tparam Input                   A varray|texture of (const) f32|f64|c32|c64.
    /// \tparam InputWeight             A varray|texture of (const) f32|f64|c32|c64, or empty.
    /// \tparam Output                  VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight            VArray of type f32|f64|c32|c64, or empty.
    /// \tparam InputScale              Mat22 or a varray of this type.
    /// \tparam InputRotate             Mat33|Quaternion, or a varray of this type.
    /// \tparam OutputScale             Mat22 or a varray of this type.
    /// \tparam OutputRotate            Mat33|Quaternion, or a varray of this type.
    ///
    /// \param[in] input_slice          2d-rfft central-slice(s) to insert (can be a constant value).
    /// \param[in] input_weight         Another optional varray|texture|value associated with \p input_slice.
    /// \param input_slice_shape        BDHW logical shape of \p input_slice.
    /// \param[in,out] output_slice     2d-rfft central-slice(s) to extract. See \p add_to_output.
    /// \param[in,out] output_weight    Another optional varray|value associated with \p output_slice.
    /// \param output_slice_shape       BDHW logical shape of \p output_slice.
    /// \param[in] input_fwd_scaling    2x2 HW \e forward real-space scaling matrix to apply to the input slices
    ///                                 before the rotation. If an array is passed, it can be empty or have one matrix
    ///                                 per slice. Otherwise the same scaling matrix is applied to every slice.
    /// \param[in] input_inv_rotation   3x3 DHW \e inverse rotation-matrices or quaternions to apply to the input slices.
    ///                                 If an array is passed, it should have one rotation per slice.
    ///                                 Otherwise the same rotation is applied to every slice.
    /// \param[in] output_inv_scaling   2x2 HW \e inverse real-space scaling matrix to apply to the output slices
    ///                                 before the rotation. If an array is passed, it can be empty or have one matrix
    ///                                 per slice. Otherwise the same scaling matrix is applied to every slice.
    /// \param[in] output_fwd_rotation  3x3 DHW \e forward rotation-matrices or quaternions to apply to the output slices.
    ///                                 If an array is passed, it should have one rotation per slice.
    ///                                 Otherwise the same rotation is applied to every slice.
    /// \param options                  Operator options.
    template<noa::fft::RemapInterface REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename InputScale, typename InputRotate,
             typename OutputScale, typename OutputRotate>
    void fourier_insert_interpolate_and_extract_3d(
            const Input& input_slice,
            const InputWeight& input_weight,
            const Shape4<i64>& input_slice_shape,
            const Output& output_slice,
            const OutputWeight& output_weight,
            const Shape4<i64>& output_slice_shape,
            const InputScale& input_fwd_scaling,
            const InputRotate& input_inv_rotation,
            const OutputScale& output_inv_scaling,
            const OutputRotate& output_fwd_rotation,
            const FourierInsertExtractOptions& options = {}
    ) {

        // check correct_multiplicity is false if w_window is more than 1 or if add_to_output=true
    }

    /// Corrects for the gridding, assuming the Fourier insertion was done using tri-linear rasterization.
    /// \details During direct Fourier insertion of central-slices S into a volume B, two problems arises:
    ///          1) The insertion is not uniform (e.g. inherently more dense at low frequencies). This can be
    ///             easily corrected by inserting the data as well as its associated weights and normalizing the
    ///             inserted data with the inserted weights. This is often referred to as density or multiplicity
    ///             correction. This function is not about that.
    ///          2) The data-points can be inserted in Fourier space by rasterization, a process also called gridding,
    ///             which is essentially a convolution between the data points and the interpolation filter
    ///             (e.g. triangle pulse for linear interpolation). The interpolation filter is often referred to as
    ///             the gridding kernel. Since convolution in frequency space corresponds to a multiplication in
    ///             real-space, the resulting inverse Fourier transform of the volume B is the product of the final
    ///             wanted reconstruction and the apodization function. The apodization function is the Fourier
    ///             transform of the gridding kernel (e.g. sinc^2 for linear interpolation). This function is there
    ///             to correct for this gridding artefact, assuming tri-linear interpolation.
    /// \param[in] input        Inverse Fourier transform of the 3d volume used for direct Fourier insertion.
    /// \param[out] output      Gridding-corrected output. Can be equal to \p input.
    /// \param post_correction  Whether the correction is the post- or pre-correction.
    ///                         Post correction is meant to be applied on the volume that was just back-projected,
    ///                         whereas pre-correction is meant to be applied on the volume that is about to be
    ///                         forward projected.
    template<typename Input, typename Output, typename = std::enable_if_t<
             nt::is_varray_of_almost_any_v<Input, f32, f64> &&
             nt::is_varray_of_any_v<Output, f32, f64> &&
             nt::are_almost_same_value_type_v<Input, Output>>>
    void gridding_correction(const Input& input, const Output& output, bool post_correction) {

    }
}
