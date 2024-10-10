#pragma once

#include <valarray>

#include "noa/core/Enums.hpp"
#include "noa/core/geometry/Euler.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/core/geometry/Quaternion.hpp"
#include "noa/core/geometry/FourierInsertRasterize.hpp"
#include "noa/core/geometry/FourierInsertInterpolate.hpp"
#include "noa/core/geometry/FourierInsertExtract.hpp"
#include "noa/core/geometry/FourierExtract.hpp"
#include "noa/core/geometry/FourierGriddingCorrection.hpp"

#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Interpolation.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Texture.hpp"
#include "noa/unified/Utilities.hpp"

namespace noa::geometry::guts {
    template<bool AllowTexture, bool AllowValue,
             typename Input, typename Output,
             typename InputValue = nt::value_type_t<Input>,
             typename OutputValue = nt::value_type_t<Output>>
    concept fourier_projection_input_output_value =
        nt::writable_varray<Output> and
        ((nt::readable_varray<Input> and nt::spectrum_types<InputValue, OutputValue>) or
         (AllowTexture and nt::texture<Input> and nt::spectrum_types<InputValue, OutputValue>) or
         (AllowValue and nt::spectrum_types<Input, OutputValue>));

    template<bool AllowTexture, bool AllowValue, typename Input, typename Output>
    concept fourier_projection_input_output_weight =
        (nt::writable_varray_of_real<Output> or nt::empty<Output>) and
        (nt::readable_varray_of_real<Input> or nt::empty<Input> or
         (AllowTexture and nt::texture_of_real<Input>) or
         (AllowValue and nt::real<Input>));

    template<bool AllowTexture, bool AllowValue,
             typename Input, typename Output, typename InputWeight, typename OutputWeight>
    concept fourier_projection_input_output =
        fourier_projection_input_output_value<AllowTexture, AllowValue, std::decay_t<Input>, std::decay_t<Output>> and
        fourier_projection_input_output_weight<AllowTexture, AllowValue, std::decay_t<InputWeight>, std::decay_t<OutputWeight>>;

    template<typename Scale, typename Rotation,
             typename Coord = nt::mutable_value_type_twice_t<Rotation>>
    concept fourier_projection_transform =
        (nt::any_of<std::decay_t<Rotation>, Mat33<f32>, Mat33<f64>, Quaternion<f32>, Quaternion<f64>> or
         nt::varray_decay_of_almost_any<Rotation, Mat33<f32>, Mat33<f64>, Quaternion<f32>, Quaternion<f64>>) and
        (nt::any_of<std::decay_t<Scale>, Empty, Mat22<Coord>> or nt::varray_decay_of_almost_any<Scale, Mat22<Coord>>);

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
        check(vall(Equal{}, output.shape(), output_shape.rfft()),
              "The shape of the rfft output does not match the expected shape. Got output:shape={} and expected:shape={}",
              output.shape(), output_shape.rfft());

        auto check_input = [&]<typename T>(const T& array, std::string_view name) {
            if constexpr (nt::varray<T> or nt::texture<T>) {
                check(not array.is_empty(), "Empty array detected");
                if constexpr (nt::varray<Input>)
                    check(not ni::are_overlapped(array, output), "Input and output arrays should not overlap");

                if constexpr (nt::texture<Input>) {
                    check(array.border() == Border::ZERO,
                          "The texture border mode should be {}, but got {}",
                          Border::ZERO, array.border());
                }
                const Device device = array.device();
                check(device == output_device,
                      "The arrays should be on the same device, but got {}:device={} and output:device={}",
                      name, device, output_device);
                check(vall(Equal{}, array.shape(), input_shape.rfft()),
                      "The shape of the rfft does not match the expected shape. Got {}:shape={} and shape={}",
                      name, array.shape(), input_shape.rfft());
            }
        };
        check_input(input, "input");
        check_input(input_weight, "input_weight");

        if constexpr (not nt::empty<OutputWeight>) {
            check(not output_weight.is_empty(), "Empty array detected");
            check(not ni::are_overlapped(output_weight, output), "Output arrays should not overlap");
            check(output_weight.device() == output_device,
                  "The arrays should be on the same device, but got output_weight:device={} and output:device={}",
                  output_weight.device(), output_device);
            check(vall(Equal{}, output_weight.shape(), output_shape.rfft()),
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

        auto check_transform = [&](const auto& transform, i64 required_size, std::string_view name) {
            check(not transform.is_empty(), "{} should not be empty", name);
            check(ni::is_contiguous_vector(transform) and transform.n_elements() == required_size,
                  "{} should be a contiguous vector with n_slices={}, but got {}:shape={}, {}:strides={}",
                  name, required_size, name, transform.shape(), name, transform.strides());
            check(transform.device() == output_device, "{} should be on the compute device", name);
        };

        const auto required_count = DIRECTION == ProjectionType::EXTRACT ? output_shape[0] : input_shape[0];
        if constexpr (nt::varray<InputScale>)
            check_transform(input_scaling, required_count, "input_scaling");
        if constexpr (nt::varray<InputRotate>)
            check_transform(input_rotation, required_count, "input_rotation");

        // Only for INSERT_EXTRACT.
        if constexpr (nt::varray<OutputScale>)
            check_transform(output_scaling, output_shape[0], "output_scaling");
        if constexpr (nt::varray<OutputRotate>)
            check_transform(output_rotation, output_shape[0], "output_rotation");
    }

    template<typename T, typename U, typename V, typename W>
    auto fourier_projection_is_i32_safe_access(
        const T& input, const U& input_weight, const V& output, const W& output_weight
    ) {
        bool is_safe_access{true};
        if constexpr (nt::varray_decay<T>)
            is_safe_access = ng::is_accessor_access_safe<i32>(input, input.shape());
        if constexpr (nt::varray_decay<U>)
            is_safe_access = is_safe_access and ng::is_accessor_access_safe<i32>(input_weight, input_weight.shape());
        if constexpr (nt::varray_decay<V>)
            is_safe_access = is_safe_access and ng::is_accessor_access_safe<i32>(output, output.shape());
        if constexpr (nt::varray_decay<W>)
            is_safe_access = is_safe_access and ng::is_accessor_access_safe<i32>(output_weight, output_weight.shape());
        return is_safe_access;
    }

    template<typename T, typename U>
    auto fourier_insert_extract_interp_mode(const T& input, const U& input_weight, Interp interp) -> Interp {
        if constexpr (nt::texture<T>) {
            interp = input.interp();
            if constexpr (nt::texture<U>) {
                check(input_weight.interp() == interp,
                      "Input textures should have the same interpolation method, "
                      "but got input:interp={} and input_weight:interp={}",
                      interp, input_weight.interp());
            }
        } else if constexpr (nt::texture<U>) {
            interp = input_weight.interp();
        }
        return interp;
    }

    template<size_t N, Remap REMAP, bool IS_GPU, Interp INTERP, typename Coord, typename Index, typename T>
    auto fourier_projection_to_interpolator(const T& input, const Shape<Index, 4>& shape) {
        if constexpr (nt::varray_or_texture<T>) {
            return ng::to_interpolator_spectrum<N, REMAP, INTERP, Coord, IS_GPU>(input, shape);

        } else if constexpr (nt::empty<T>) {
            return input;

        } else { // real or complex
            using accessor_t = AccessorValue<const T, Index>;
            using interpolator_t = InterpolatorSpectrum<N, REMAP, INTERP.erase_fast(), accessor_t>;
            return interpolator_t(accessor_t(input), shape.template filter_nd<N>().pop_front());
        }
    }

    template<typename T>
    constexpr bool fourier_project_has_scale(const T& scale) {
        if constexpr (nt::mat22<T>)
            return scale != T::eye(1);
        else if constexpr (nt::varray<T> and nt::mat22<nt::value_type_t<T>>)
            return true;
        else if constexpr (nt::empty<T>)
            return false;
        else
            static_assert(nt::always_false<T>);
    }

    template<bool ENFORCE_EMPTY, typename Coord>
    constexpr auto fourier_projection_to_ews(const Vec<f64, 2>& ews) {
        if constexpr (ENFORCE_EMPTY)
            return Empty{};
        else
            return ews.as<Coord>();
    }

    template<Remap REMAP, typename Index,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename Scale, typename Rotate>
    void launch_fourier_insert_rasterize_3d(
        Input&& slice, InputWeight&& slice_weight, const Shape4<i64>& slice_shape,
        Output&& volume, OutputWeight&& volume_weight, const Shape4<i64>& volume_shape,
        Scale&& scaling, Rotate&& rotation, const auto& options
    ) {
        constexpr auto input_accessor_config = ng::AccessorConfig<3>{
            .enforce_const=true,
            .enforce_restrict=true,
            .allow_empty=true,
            .filter={0, 2, 3},
        };
        constexpr auto output_accessor_config = ng::AccessorConfig<3>{
            .enforce_restrict=true,
            .allow_empty=true,
            .filter={1, 2, 3},
        };
        auto slice_accessor = ng::to_accessor<input_accessor_config, Index>(slice);
        auto slice_weight_accessor = ng::to_accessor<input_accessor_config, Index>(slice_weight);
        auto volume_accessor = ng::to_accessor<output_accessor_config, Index>(volume);
        auto volume_weight_accessor = ng::to_accessor<output_accessor_config, Index>(volume_weight);

        const auto s_input_slice_shape = slice_shape.as<Index>();
        const auto s_volume_shape = volume_shape.as<Index>();
        auto batched_rotation = ng::to_batched_transform(rotation);
        using coord_t = nt::value_type_twice_t<Rotate>;

        auto launch = [&]<bool NO_EWS_SCALE> {
            auto ews = fourier_projection_to_ews<NO_EWS_SCALE, coord_t>(options.ews_radius);
            auto batched_scaling = ng::to_batched_transform<true, NO_EWS_SCALE>(scaling);

            using op_t = FourierInsertRasterize<
                REMAP, Index,
                decltype(batched_scaling), decltype(batched_rotation), decltype(ews),
                decltype(slice_accessor), decltype(slice_weight_accessor),
                decltype(volume_accessor), decltype(volume_weight_accessor)>;
            auto op = op_t(
                slice_accessor, slice_weight_accessor, s_input_slice_shape,
                volume_accessor, volume_weight_accessor, s_volume_shape,
                batched_scaling, batched_rotation,
                static_cast<coord_t>(options.fftfreq_cutoff),
                options.target_shape.template as<Index>(), ews);

            iwise(s_input_slice_shape.filter(0, 2, 3).rfft(), volume.device(), op,
                  std::forward<Input>(slice), std::forward<InputWeight>(slice_weight),
                  std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight),
                  std::forward<Scale>(scaling), std::forward<Rotate>(rotation));
        };

        const auto has_ews = any(options.ews_radius != 0);
        const bool has_scale = fourier_project_has_scale(scaling);
        if (has_ews or has_scale)
            return launch.template operator()<false>();
        return launch.template operator()<true>();
    }

    template<Remap REMAP, typename Index, bool IS_GPU,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename Scale, typename Rotate>
    void launch_fourier_insert_interpolate_3d(
        Input&& slice, InputWeight&& slice_weight, const Shape4<i64>& slice_shape,
        Output&& volume, OutputWeight&& volume_weight, const Shape4<i64>& volume_shape,
        Scale&& scaling, Rotate&& rotation, const auto& options
    ) {
        constexpr auto accessor_config = ng::AccessorConfig<3>{
            .enforce_restrict = true,
            .allow_empty = true,
            .filter = {1, 2, 3},
        };
        auto volume_accessor = ng::to_accessor<accessor_config, Index>(volume);
        auto volume_weight_accessor = ng::to_accessor<accessor_config, Index>(volume_weight);

        const auto s_slice_shape = slice_shape.as<Index>();
        const auto s_volume_shape = volume_shape.as<Index>();
        auto batched_rotation = ng::to_batched_transform<false>(rotation);
        using coord_t = nt::value_type_twice_t<Rotate>;

        auto launch = [&]<Interp INTERP, bool NO_EWS_SCALE>{
            auto slice_interpolator = fourier_projection_to_interpolator<2, REMAP, IS_GPU, INTERP, coord_t>(slice, s_slice_shape);
            auto slice_weight_interpolator = fourier_projection_to_interpolator<2, REMAP, IS_GPU, INTERP, coord_t>(slice_weight, s_slice_shape);

            auto batched_scaling = ng::to_batched_transform<true, NO_EWS_SCALE>(scaling);
            auto ews = fourier_projection_to_ews<NO_EWS_SCALE, coord_t>(options.ews_radius);

            using op_t = FourierInsertInterpolate<
                REMAP, Index, decltype(batched_scaling), decltype(batched_rotation), decltype(ews),
                decltype(slice_interpolator), decltype(slice_weight_interpolator),
                decltype(volume_accessor), decltype(volume_weight_accessor)>;
            auto op = op_t(
                slice_interpolator, slice_weight_interpolator, s_slice_shape,
                volume_accessor, volume_weight_accessor, s_volume_shape,
                batched_scaling, batched_rotation,
                static_cast<coord_t>(options.windowed_sinc.fftfreq_sinc),
                static_cast<coord_t>(options.windowed_sinc.fftfreq_blackman),
                static_cast<coord_t>(options.fftfreq_cutoff),
                options.target_shape.template as<Index>(), ews);

            iwise(s_slice_shape.filter(0, 2, 3).rfft(), volume.device(), op,
                  std::forward<Input>(slice), std::forward<InputWeight>(slice_weight),
                  std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight),
                  std::forward<Scale>(scaling), std::forward<Rotate>(rotation));
        };

        const auto has_ews = any(options.ews_radius != 0);
        const bool has_scale = fourier_project_has_scale(scaling);
        auto launch_scale = [&]<Interp INTERP>{
            if (has_ews or has_scale)
                return launch.template operator()<INTERP, false>();
            return launch.template operator()<INTERP, true>();
        };

        const Interp interp = guts::fourier_insert_extract_interp_mode(slice, slice_weight, options.interp);
        switch (interp) {
            case Interp::NEAREST:            return launch_scale.template operator()<Interp::NEAREST>();
            case Interp::NEAREST_FAST:       return launch_scale.template operator()<Interp::NEAREST_FAST>();
            case Interp::LINEAR:             return launch_scale.template operator()<Interp::LINEAR>();
            case Interp::LINEAR_FAST:        return launch_scale.template operator()<Interp::LINEAR_FAST>();
            case Interp::CUBIC:              return launch_scale.template operator()<Interp::CUBIC>();
            case Interp::CUBIC_FAST:         return launch_scale.template operator()<Interp::CUBIC_FAST>();
            case Interp::CUBIC_BSPLINE:      return launch_scale.template operator()<Interp::CUBIC_BSPLINE>();
            case Interp::CUBIC_BSPLINE_FAST: return launch_scale.template operator()<Interp::CUBIC_BSPLINE_FAST>();
            case Interp::LANCZOS4:           return launch_scale.template operator()<Interp::LANCZOS4>();
            case Interp::LANCZOS6:           return launch_scale.template operator()<Interp::LANCZOS6>();
            case Interp::LANCZOS8:           return launch_scale.template operator()<Interp::LANCZOS8>();
            case Interp::LANCZOS4_FAST:      return launch_scale.template operator()<Interp::LANCZOS4_FAST>();
            case Interp::LANCZOS6_FAST:      return launch_scale.template operator()<Interp::LANCZOS6_FAST>();
            case Interp::LANCZOS8_FAST:      return launch_scale.template operator()<Interp::LANCZOS8_FAST>();
        }
    }

    template<Remap REMAP, typename Index, bool IS_GPU,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename Scale, typename Rotate>
    void launch_fourier_extract_3d(
        Input&& volume, InputWeight&& volume_weight, const Shape4<i64>& volume_shape,
        Output&& slice, OutputWeight&& slice_weight, const Shape4<i64>& slice_shape,
        Scale&& scaling, Rotate&& rotation, const auto& options
    ) {
        constexpr auto accessor_config = ng::AccessorConfig<3>{
            .enforce_restrict = true,
            .allow_empty = true,
            .filter = {0, 2, 3},
        };
        auto slice_accessor = ng::to_accessor<accessor_config, Index>(slice);
        auto slice_weight_accessor = ng::to_accessor<accessor_config, Index>(slice_weight);
        auto batched_rotation = ng::to_batched_transform(rotation);

        const auto s_slice_shape = slice_shape.as<Index>();
        const auto s_volume_shape = volume_shape.as<Index>();
        const auto s_target_shape = options.target_shape.template as<Index>();

        auto launch = [&]<Interp INTERP, bool NO_EWS_SCALE>{
            using coord_t = nt::value_type_twice_t<Rotate>;
            auto volume_interpolator = fourier_projection_to_interpolator<3, REMAP, IS_GPU, INTERP, coord_t>(volume, s_volume_shape);
            auto volume_weight_interpolator = fourier_projection_to_interpolator<3, REMAP, IS_GPU, INTERP, coord_t>(volume_weight, s_volume_shape);

            auto batched_scale = ng::to_batched_transform<true, NO_EWS_SCALE>(scaling);
            auto ews = fourier_projection_to_ews<NO_EWS_SCALE, coord_t>(options.ews_radius);

            using op_t = FourierExtract<
                REMAP, Index,
                decltype(batched_scale), decltype(batched_rotation), decltype(ews),
                decltype(volume_interpolator), decltype(volume_weight_interpolator),
                decltype(slice_accessor), decltype(slice_weight_accessor)>;
            auto op = op_t(
                volume_interpolator, volume_weight_interpolator, s_volume_shape,
                slice_accessor, slice_weight_accessor, s_slice_shape,
                batched_scale, batched_rotation,
                static_cast<coord_t>(options.w_windowed_sinc.fftfreq_sinc),
                static_cast<coord_t>(options.w_windowed_sinc.fftfreq_blackman),
                static_cast<coord_t>(options.fftfreq_cutoff),
                s_target_shape, ews);

            if (op.windowed_sinc_size() > 1) {
                if constexpr (nt::empty<OutputWeight>)
                    ewise({}, wrap(slice), Zero{});
                else
                    ewise({}, wrap(slice, slice_weight), Zero{});

                const auto iwise_shape = s_slice_shape.template set<1>(op.windowed_sinc_size()).rfft();
                iwise(iwise_shape, volume.device(), op,
                      std::forward<Input>(volume), std::forward<InputWeight>(volume_weight),
                      std::forward<Output>(slice), std::forward<OutputWeight>(slice_weight),
                      std::forward<Scale>(scaling), std::forward<Rotate>(rotation));
            } else {
                const auto iwise_shape = s_slice_shape.filter(0, 2, 3).rfft();
                iwise(iwise_shape, volume.device(), op,
                      std::forward<Input>(volume), std::forward<InputWeight>(volume_weight),
                      std::forward<Output>(slice), std::forward<OutputWeight>(slice_weight),
                      std::forward<Scale>(scaling), std::forward<Rotate>(rotation));
            }
        };

        const auto has_ews = any(options.ews_radius != 0);
        const bool has_scale = fourier_project_has_scale(scaling);
        auto launch_scale = [&]<Interp INTERP>{
            if (has_ews or has_scale)
                return launch.template operator()<INTERP, false>();
            return launch.template operator()<INTERP, true>();
        };

        const Interp interp = guts::fourier_insert_extract_interp_mode(volume, volume_weight, options.interp);
        switch (interp) {
            case Interp::NEAREST:            return launch_scale.template operator()<Interp::NEAREST>();
            case Interp::NEAREST_FAST:       return launch_scale.template operator()<Interp::NEAREST_FAST>();
            case Interp::LINEAR:             return launch_scale.template operator()<Interp::LINEAR>();
            case Interp::LINEAR_FAST:        return launch_scale.template operator()<Interp::LINEAR_FAST>();
            case Interp::CUBIC:              return launch_scale.template operator()<Interp::CUBIC>();
            case Interp::CUBIC_FAST:         return launch_scale.template operator()<Interp::CUBIC_FAST>();
            case Interp::CUBIC_BSPLINE:      return launch_scale.template operator()<Interp::CUBIC_BSPLINE>();
            case Interp::CUBIC_BSPLINE_FAST: return launch_scale.template operator()<Interp::CUBIC_BSPLINE_FAST>();
            case Interp::LANCZOS4:           return launch_scale.template operator()<Interp::LANCZOS4>();
            case Interp::LANCZOS6:           return launch_scale.template operator()<Interp::LANCZOS6>();
            case Interp::LANCZOS8:           return launch_scale.template operator()<Interp::LANCZOS8>();
            case Interp::LANCZOS4_FAST:      return launch_scale.template operator()<Interp::LANCZOS4_FAST>();
            case Interp::LANCZOS6_FAST:      return launch_scale.template operator()<Interp::LANCZOS6_FAST>();
            case Interp::LANCZOS8_FAST:      return launch_scale.template operator()<Interp::LANCZOS8_FAST>();
        }
    }

    template<Remap REMAP, typename Index, bool IS_GPU = false,
             typename Input, typename InputWeight,
             typename Output, typename OutputWeight,
             typename InputScale, typename InputRotate,
             typename OutputScale, typename OutputRotate>
    void launch_fourier_insert_interpolate_extract_3d(
        Input&& input_slice, InputWeight&& input_weight, const Shape4<i64>& input_shape,
        Output&& output_slice, OutputWeight&& output_weight, const Shape4<i64>& output_shape,
        InputScale&& input_scaling, InputRotate&& input_rotation,
        OutputScale&& output_scaling, OutputRotate&& output_rotation,
        const auto& options
    ) {
        constexpr auto output_config = ng::AccessorConfig<3>{
            .enforce_restrict = true,
            .allow_empty = true,
            .filter = {0, 2, 3},
        };
        auto output_slice_accessor = ng::to_accessor<output_config, Index>(output_slice);
        auto output_weight_accessor = ng::to_accessor<output_config, Index>(output_weight);
        auto input_rotation_accessor = ng::to_batched_transform(input_rotation);
        auto output_rotation_accessor = ng::to_batched_transform(output_rotation);

        const auto s_input_shape = input_shape.as<Index>();
        const auto s_output_shape = output_shape.as<Index>();

        auto launch = [&]<Interp INTERP, bool NO_EWS_SCALE>() {
            using coord_t = nt::value_type_twice_t<InputRotate>;
            auto input_interpolator = fourier_projection_to_interpolator<2, REMAP, IS_GPU, INTERP, coord_t>(input_slice, s_input_shape);
            auto input_weight_interpolator = fourier_projection_to_interpolator<2, REMAP, IS_GPU, INTERP, coord_t>(input_weight, s_input_shape);

            auto ews = fourier_projection_to_ews<NO_EWS_SCALE, coord_t>(options.ews_radius);
            auto input_scaling_accessor = ng::to_batched_transform<true, NO_EWS_SCALE>(input_scaling);
            auto output_scaling_accessor = ng::to_batched_transform<true, NO_EWS_SCALE>(output_scaling);

            using op_t = FourierInsertExtract<
                REMAP, Index,
                decltype(input_scaling_accessor), decltype(input_rotation_accessor),
                decltype(output_scaling_accessor), decltype(output_rotation_accessor), decltype(ews),
                decltype(input_interpolator), decltype(input_weight_interpolator),
                decltype(output_slice_accessor), decltype(output_weight_accessor)>;
            auto op = op_t(
                input_interpolator, input_weight_interpolator, s_input_shape,
                output_slice_accessor, output_weight_accessor, s_output_shape,
                input_scaling_accessor, input_rotation_accessor,
                output_scaling_accessor, output_rotation_accessor,
                static_cast<coord_t>(options.input_windowed_sinc.fftfreq_sinc),
                static_cast<coord_t>(options.input_windowed_sinc.fftfreq_blackman),
                static_cast<coord_t>(options.w_windowed_sinc.fftfreq_sinc),
                static_cast<coord_t>(options.w_windowed_sinc.fftfreq_blackman),
                static_cast<coord_t>(options.fftfreq_cutoff),
                options.add_to_output, options.correct_multiplicity, ews);

            if (op.is_iwise_4d()) {
                check(not options.correct_multiplicity, "");
                if (not options.add_to_output) {
                    if constexpr (nt::empty<OutputWeight>)
                        ewise({}, wrap(output_slice), Zero{});
                    else
                        ewise({}, wrap(output_slice, output_weight), Zero{});
                }
                iwise(s_output_shape.template set<1>(op.output_window_size()).rfft(), output_slice.device(), op,
                      std::forward<Input>(input_slice),
                      std::forward<InputWeight>(input_weight),
                      std::forward<Output>(output_slice),
                      std::forward<OutputWeight>(output_weight),
                      std::forward<InputScale>(input_scaling),
                      std::forward<InputRotate>(input_rotation),
                      std::forward<OutputScale>(output_scaling),
                      std::forward<OutputRotate>(output_rotation));
            } else {
                iwise(s_output_shape.filter(0, 2, 3).rfft(), output_slice.device(), op,
                      std::forward<Input>(input_slice),
                      std::forward<InputWeight>(input_weight),
                      std::forward<Output>(output_slice),
                      std::forward<OutputWeight>(output_weight),
                      std::forward<InputScale>(input_scaling),
                      std::forward<InputRotate>(input_rotation),
                      std::forward<OutputScale>(output_scaling),
                      std::forward<OutputRotate>(output_rotation));
            }
        };

        const auto has_ews = any(options.ews_radius != 0);
        const bool has_scale = fourier_project_has_scale(input_scaling) or fourier_project_has_scale(output_scaling);
        auto launch_scale = [&]<Interp INTERP>{
            if (has_ews or has_scale)
                return launch.template operator()<INTERP, false>();
            return launch.template operator()<INTERP, true>();
        };

        const Interp interp = guts::fourier_insert_extract_interp_mode(input_slice, input_weight, options.interp);
        switch (interp) {
            case Interp::NEAREST:            return launch_scale.template operator()<Interp::NEAREST>();
            case Interp::NEAREST_FAST:       return launch_scale.template operator()<Interp::NEAREST_FAST>();
            case Interp::LINEAR:             return launch_scale.template operator()<Interp::LINEAR>();
            case Interp::LINEAR_FAST:        return launch_scale.template operator()<Interp::LINEAR_FAST>();
            case Interp::CUBIC:              return launch_scale.template operator()<Interp::CUBIC>();
            case Interp::CUBIC_FAST:         return launch_scale.template operator()<Interp::CUBIC_FAST>();
            case Interp::CUBIC_BSPLINE:      return launch_scale.template operator()<Interp::CUBIC_BSPLINE>();
            case Interp::CUBIC_BSPLINE_FAST: return launch_scale.template operator()<Interp::CUBIC_BSPLINE_FAST>();
            case Interp::LANCZOS4:           return launch_scale.template operator()<Interp::LANCZOS4>();
            case Interp::LANCZOS6:           return launch_scale.template operator()<Interp::LANCZOS6>();
            case Interp::LANCZOS8:           return launch_scale.template operator()<Interp::LANCZOS8>();
            case Interp::LANCZOS4_FAST:      return launch_scale.template operator()<Interp::LANCZOS4_FAST>();
            case Interp::LANCZOS6_FAST:      return launch_scale.template operator()<Interp::LANCZOS6_FAST>();
            case Interp::LANCZOS8_FAST:      return launch_scale.template operator()<Interp::LANCZOS8_FAST>();
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
        /// As such, if the volume is larger than the slices, the slices are implicitly stretched (over-sampling case).
        /// Similarly, if the volume is smaller than the slices, the slices are shrunk (under-sampling case).
        /// This parameter specifies the size of the volume onto which the slice frequencies should be mapped against.
        /// By default, i.e. empty target_shape or target_shape == volume_shape, the slice frequencies are mapped onto
        /// the volume frequencies, as mentioned above.
        Shape4<i64> target_shape{};

        /// HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
        /// If negative, the negative curve is computed. If {0,0}, the slices are projections.
        /// To have both left and right beams assigned to different values, the function only computes one
        /// "side" of the EWS, as specified by ews_radius. To insert the other side, one has to call the function
        /// a second time with ews_radius * -1.
        /// Note that the scaling and the rotation are kept separated from one another in order to properly compute the
        /// curve of the Ewald sphere. Indeed, the scaling is applied first to correct for magnification, so that the
        /// EWS is computed using the original frequencies (from the scattering) and is therefore spherical even
        /// under anisotropic magnification. If ews_radius is 0, the scaling factors can be merged with the rotation
        /// matrices.
        Vec2<f64> ews_radius{};
    };

    /// Inserts 2d Fourier central-slice(s) into a 3d Fourier volume, using tri-linear rasterization.
    /// \details Fourier-insertion using rasterization to insert central-slices in a volume.
    ///          This method is mostly used for cases with a lot of central-slices, where errors can be averaged-out.
    ///          Indeed, this method is not very accurate as central-slices are modeled using a simple trilinear-pulse
    ///          for rasterization.
    ///
    /// \tparam REMAP               Remapping. Should be HX2HX.
    /// \tparam Input               A varray|value of (const) f32|f64|c32|c64.
    /// \tparam InputWeight         A varray|value of (const) f32|f64, or Empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64, or Empty.
    /// \tparam Scale               Mat22, a varray of Mat22, or Empty.
    /// \tparam Rotate              Mat33|Quaternion, or a varray of this type.
    ///                             Sets the floating-point precision of the transformation.
    ///
    /// \param[in] slice            2d-rfft central-slice(s) to insert (can be a constant value).
    /// \param[in] slice_weight     Optional weights associated with slice. Default to ones.
    /// \param slice_shape          BDHW logical shape of slice.
    /// \param[out] volume          3d-rfft volume inside which the slices are inserted.
    /// \param[out] volume_weight   Optional weights associated with volume.
    /// \param volume_shape         BDHW logical shape of volume.
    /// \param[in] inv_scaling      2x2 HW inverse scaling matrix to apply to the slices before the rotation.
    /// \param[in] fwd_rotation     3x3 DHW forward rotation-matrices or quaternions to apply to the slices.
    /// \param options              Insertion options.
    template<Remap REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale = Empty, typename Rotate>
    requires (guts::fourier_projection_input_output<false, true, Input, Output, InputWeight, OutputWeight> and
              guts::fourier_projection_transform<Scale, Rotate> and
              REMAP.is_hx2hx())
    void fourier_insert_rasterize_3d(
        Input&& slice,
        InputWeight&& slice_weight,
        const Shape4<i64>& slice_shape,
        Output&& volume,
        OutputWeight&& volume_weight,
        const Shape4<i64>& volume_shape,
        Scale&& inv_scaling,
        Rotate&& fwd_rotation,
        const FourierInsertRasterizeOptions& options = {}
    ) {
        guts::fourier_projection_check_parameters<guts::ProjectionType::INSERT_RASTERIZE>(
            slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
            options.target_shape, inv_scaling, fwd_rotation);

        if (volume.device().is_gpu()) {
            #ifdef NOA_ENABLE_CUDA
            check(guts::fourier_projection_is_i32_safe_access(slice, slice_weight, volume, volume_weight),
                  "i64 indexing not instantiated for GPU devices");
            return guts::launch_fourier_insert_rasterize_3d<REMAP, i32>(
                std::forward<Input>(slice), std::forward<InputWeight>(slice_weight), slice_shape,
                std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight), volume_shape,
                std::forward<Scale>(inv_scaling), std::forward<Rotate>(fwd_rotation), options);
            #else
            std::terminate(); // unreachable
            #endif
        }

        guts::launch_fourier_insert_rasterize_3d<REMAP, i64>(
            std::forward<Input>(slice), std::forward<InputWeight>(slice_weight), slice_shape,
            std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight), volume_shape,
            std::forward<Scale>(inv_scaling), std::forward<Rotate>(fwd_rotation), options);
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
        /// which is usually want we want for the Fourier insertion.
        f64 fftfreq_sinc{-1};

        /// Frequency, in cycle/pix, where the blackman window stops (weight is 0 at this frequency).
        /// This parameter is usually here to control the accuracy-performance ratio, but it can also be used
        /// to control the smoothness of the corresponding real-space mask, in which case it is usually a multiple
        /// of the sinc-cutoff. The larger this multiple, the sharper the real-space step window, but the slower it
        /// is to compute the slice.
        /// This is clamped to ensure the window stops at least to the first sinc-cutoff.
        /// So if both frequencies are left to their default value (-1), a 1 pixel thick slice
        /// is generated, which is usually want we want for Fourier insertion.
        f64 fftfreq_blackman{-1};
    };

    struct FourierInsertInterpolateOptions {
        /// Interpolation method.
        /// This is ignored if the input(_weights) is a texture.
        Interp interp{Interp::LINEAR};

        /// Windowed-sinc convolved along the normal of the input slice(s).
        WindowedSinc windowed_sinc{};

        /// Frequency cutoff in the output volume, in cycle/pix.
        /// The frequencies above this are left unchanged.
        f64 fftfreq_cutoff{0.5};

        /// Actual BDHW logical shape of the 3d volume.
        /// See FourierInsertRasterizeOptions for more details.
        Shape4<i64> target_shape{};

        /// HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
        /// See FourierInsertRasterizeOptions for more details.
        Vec2<f64> ews_radius{};
    };

    /// Fourier-insertion using 2d-interpolation to insert central-slices in the volume.
    /// \details This method is the more accurate than rasterizationm but can be slower. Here, instead of
    ///          calling every pixel in the central-slices for rasterization, every voxel in the volume is sampled
    ///          by collecting the contribution of every central-slice for each output voxel. The advantage is that
    ///          it allows to use a more accurate model for the central-slices, i.e., a windowed-sinc. Indeed, slices
    ///          are now effectively convolved with a windowed-sinc (both the sinc frequency and window size can be
    ///          controlled) along their normal before the insertion. Note that this (windowed) sinc translates to
    ///          a (smooth) rectangular mask in real-space, along the normal of the slice (an interesting property
    ///          that can be useful for some applications).
    /// \warning This function computes the inverse transformation compared to the overload above using rasterization.
    ///
    /// \tparam REMAP               Remapping. Should be HX2HX.
    /// \tparam Input               A varray|texture of (const) f32|f64|c32|c64.
    /// \tparam InputWeight         A varray|texture of (const) f32|f64, or Empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64, or Empty.
    /// \tparam Scale               Mat22, a varray of Mat22, or Empty.
    /// \tparam Rotate              Mat33|Quaternion, or a varray of this type.
    ///                             Sets the floating-point precision of the transformation.
    ///
    /// \param[in] slice            2d-rFFT central-slice(s) to insert.
    /// \param[in] slice_weight     Optional weights associated with slice. Default to ones.
    /// \param slice_shape          BDHW logical shape of slice.
    /// \param[out] volume          3d-rFFT volume inside which the slices are to be inserted.
    /// \param[out] volume_weight   Optional weights associated with volume.
    /// \param volume_shape         BDHW logical shape of volume.
    /// \param[in] fwd_scaling      2x2 HW forward scaling matrix to apply to the slices before the rotation.
    /// \param[in] inv_rotation     3x3 DHW inverse rotation-matrices or quaternions to apply to the slices.
    /// \param options              Insertion options.
    template<Remap REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale = Empty, typename Rotate>
    requires (guts::fourier_projection_input_output<true, true, Input, Output, InputWeight, OutputWeight> and
              guts::fourier_projection_transform<Scale, Rotate> and
              REMAP.is_hx2hx())
    void fourier_insert_interpolate_3d(
        Input&& slice,
        InputWeight&& slice_weight,
        const Shape4<i64>& slice_shape,
        Output&& volume,
        OutputWeight&& volume_weight,
        const Shape4<i64>& volume_shape,
        Scale&& fwd_scaling,
        Rotate&& inv_rotation,
        const FourierInsertInterpolateOptions& options = {}
    ) {
        guts::fourier_projection_check_parameters<guts::ProjectionType::INSERT_INTERPOLATE>(
            slice, slice_weight, slice_shape, volume, volume_weight, volume_shape,
            options.target_shape, fwd_scaling, inv_rotation);

        if (volume.device().is_gpu()) {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (
                (nt::texture_decay<Input> and nt::any_of<nt::mutable_value_type_t<Input>, f64, c64>) or
                (nt::texture_decay<InputWeight> and nt::any_of<nt::mutable_value_type_t<InputWeight>, f64, c64>)) {
                std::terminate(); // unreachable
            } else {
                check(guts::fourier_projection_is_i32_safe_access(slice, slice_weight, volume, volume_weight),
                      "i64 indexing not instantiated for GPU devices");
                return guts::launch_fourier_insert_interpolate_3d<REMAP, i32, true>(
                    std::forward<Input>(slice), std::forward<InputWeight>(slice_weight), slice_shape,
                    std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight), volume_shape,
                    std::forward<Scale>(fwd_scaling), std::forward<Rotate>(inv_rotation), options);
            }
            #else
            std::terminate(); // unreachable
            #endif
        }

        guts::launch_fourier_insert_interpolate_3d<REMAP, i64, false>(
            std::forward<Input>(slice), std::forward<InputWeight>(slice_weight), slice_shape,
            std::forward<Output>(volume), std::forward<OutputWeight>(volume_weight), volume_shape,
            std::forward<Scale>(fwd_scaling), std::forward<Rotate>(inv_rotation), options);
    }

    struct FourierExtractOptions {
        /// Interpolation method.
        /// This is ignored if the input(_weights) is a texture.
        Interp interp{Interp::LINEAR};

        /// Windowed-sinc convolved along the w (the depth) of the Fourier volume.
        /// This is used to apply a real-space step mask along the z (the depth) of the reconstruction
        /// before the forward projection.
        WindowedSinc w_windowed_sinc{};

        /// Frequency cutoff in the volume, in cycle/pix.
        /// The frequencies above this cutoff are assumed to be zero.
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
    ///          controlled by the \p options.w_windowed_sinc parameter):
    ///          1. (default) A simple and fast slice extraction, where every pixel of the output slice(s) are sampled
    ///          from the volume using 3d-interpolation.
    ///          2. A w-windowed-sinc slice extraction. This is similar, but instead of simply extracting the slice
    ///          from the volume, it convolves the volume with a 1d windowed-sinc along the w-axis of the volume.
    ///          Note that the convolution is simplified to a simple per-slice weighted-mean along the w-axis of the
    ///          volume. This windowed-sinc convolution translates to a (smooth) centered rectangular mask along the
    ///          z-axis of the reconstruction.
    ///
    /// \tparam REMAP               Remapping. Should be HX2HX.
    /// \tparam Input               A varray|texture of (const) f32|f64|c32|c64.
    /// \tparam InputWeight         A varray|texture of (const) f32|f64, or Empty.
    /// \tparam Output              VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight        VArray of type f32|f64, or Empty.
    /// \tparam Scale               Mat22, a varray of Mat22, or Empty.
    /// \tparam Rotate              Mat33|Quaternion, or a varray of this type.
    ///                             Sets the floating-point precision of the transformation.
    ///
    /// \param[in] volume           3d-rFFT volume from which to extract the slices.
    /// \param[in] volume_weight    Optional weights associated with volume.
    /// \param volume_shape         BDHW logical shape of volume.
    /// \param[out] slice           2d-rFFT central-slice(s) to extract.
    /// \param[out] slice_weight    Optional weights associated with slice.
    /// \param slice_shape          BDHW logical shape of slice.
    /// \param[in] inv_scaling      2x2 HW inverse scaling to apply to the slices before the rotation.
    /// \param[in] fwd_rotation     3x3 DHW forward rotation-matrices or quaternions to apply to the slices.
    /// \param options              Extraction options.
    template<Remap REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename Scale = Empty, typename Rotate>
    requires (guts::fourier_projection_input_output<true, false, Input, Output, InputWeight, OutputWeight> and
              guts::fourier_projection_transform<Scale, Rotate> and
              REMAP.is_hx2hx())
    void fourier_extract_3d(
        Input&& volume,
        InputWeight&& volume_weight,
        const Shape4<i64>& volume_shape,
        Output&& slice,
        OutputWeight&& slice_weight,
        const Shape4<i64>& slice_shape,
        Scale&& inv_scaling,
        Rotate&& fwd_rotation,
        const FourierExtractOptions& options = {}
    ) {
        guts::fourier_projection_check_parameters<guts::ProjectionType::EXTRACT>(
            volume, volume_weight, volume_shape, slice, slice_weight, slice_shape,
            options.target_shape, inv_scaling, fwd_rotation);

        if (volume.device().is_gpu()) {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (
                (nt::texture_decay<Input> and nt::any_of<nt::mutable_value_type_t<Input>, f64, c64>) or
                (nt::texture_decay<InputWeight> and nt::any_of<nt::mutable_value_type_t<InputWeight>, f64, c64>)) {
                std::terminate(); // unreachable
            } else {
                check(guts::fourier_projection_is_i32_safe_access(slice, slice_weight, volume, volume_weight),
                      "i64 indexing not instantiated for GPU devices");
                return guts::launch_fourier_extract_3d<REMAP, i32, true>(
                    std::forward<Input>(volume), std::forward<InputWeight>(volume_weight), volume_shape,
                    std::forward<Output>(slice), std::forward<OutputWeight>(slice_weight), slice_shape,
                    std::forward<Scale>(inv_scaling), std::forward<Rotate>(fwd_rotation), options);
            }
            #else
            std::terminate(); // unreachable
            #endif
        }

        guts::launch_fourier_extract_3d<REMAP, i64, false>(
            std::forward<Input>(volume), std::forward<InputWeight>(volume_weight), volume_shape,
            std::forward<Output>(slice), std::forward<OutputWeight>(slice_weight), slice_shape,
            std::forward<Scale>(inv_scaling), std::forward<Rotate>(fwd_rotation), options);
    }

    struct FourierInsertExtractOptions {
        /// Interpolation method.
        /// This is ignored if input(_weights) is a texture.
        Interp interp{Interp::LINEAR};

        /// Windowed-sinc convolved along the normal of the input slice(s).
        WindowedSinc input_windowed_sinc{};

        /// Windowed-sinc convolved along the w (the depth) of the virtual Fourier volume.
        /// This is used to apply a real-space step mask along the z (the depth) of the reconstruction
        /// before the forward projection.
        WindowedSinc w_windowed_sinc{};

        /// Whether the contribution of the input slices should be added to the output.
        /// By default, the operator sets the output_{slice|weight}. With this option enabled, it instead adds the
        /// contribution of input_{slice|weight} to the signal already present in the output_{slice|weight}, allowing
        /// to progressively build the output_{slice|weight}.
        bool add_to_output{false};

        /// Correct for the multiplicity, i.e. divide the output sampled values by their corresponding weight.
        /// This is the equivalent of doing `output_slice/max(1, output_weight)` after the function and assumes all
        /// the input slices are included in the same function call. It is not compatible with add_to_output=true or
        /// with a non-default w_windowed_sinc.
        /// This can be useful for cases where there's no real-space mask to apply before the forward projection and
        /// if the weights are to be applied right away and not used for anything else (in which case output_weight
        /// doesn't need to be saved and can be left empty).
        bool correct_multiplicity{false};

        /// Frequency cutoff of the virtual 3d volume, in cycle/pix.
        /// The frequencies above this cutoff are assumed to be zero.
        f64 fftfreq_cutoff{0.5};

        /// HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
        /// See FourierInsertRasterizeOptions for more details.
        Vec2<f64> ews_radius{};
    };

    /// Extracts 2d central-slice(s) from a virtual volume filled by other central-slices.
    /// \details This function effectively combines the insertion and extraction, but instead of actually inserting
    ///          slices into a volume, it directly inserts them in the extracted slices. In other words, it builds a
    ///          virtual volume, made of central-slices, and this virtual volume is then sampled at (exactly) the
    ///          frequency of the central-slices to extract. This has major performance benefits because it only
    ///          samples the frequency of the output slices and never allocates/reconstructs the volume. It is also
    ///          more accurate since the volume is never actually discretized (thus skipping an interpolation step).
    ///          Note that these performance benefits are expected to disappear if thousands (possibly hundreds?) of
    ///          slices are extracted. Indeed, for every output slice, the operator needs to sample the volume by
    ///          collecting the signal of every input slice using 2d-interpolation. This is as opposed to the other
    ///          extract method, where the volume is already sampled, making the extraction much cheaper (and constant
    ///          cost: it's a simple 3d-interpolation).
    ///
    /// \tparam REMAP                   Remapping. Should be HX2HX.
    /// \tparam Input                   A varray|texture|value of (const) f32|f64|c32|c64.
    /// \tparam InputWeight             A varray|texture|value of (const) f32|f64, or Empty.
    /// \tparam Output                  VArray of type f32|f64|c32|c64.
    /// \tparam OutputWeight            VArray of type f32|f64, or Empty.
    /// \tparam InputScale              Mat22 or a varray of this type, or Empty
    /// \tparam InputRotate             Mat33|Quaternion, or a varray of this type.
    /// \tparam OutputScale             Mat22 or a varray of this type, or Empty
    /// \tparam OutputRotate            Mat33|Quaternion, or a varray of this type.
    ///
    /// \param[in] input_slice          2d central-slice(s) to insert.
    /// \param[in] input_weight         Optional weights associated with input_slice. Defaults to ones.
    /// \param input_slice_shape        BDHW logical shape of input_slice.
    /// \param[in,out] output_slice     2d central-slice(s) to extract. See options.add_to_output.
    /// \param[in,out] output_weight    Optional weights associated with output_slice.
    /// \param output_slice_shape       BDHW logical shape of output_slice.
    /// \param[in] input_fwd_scaling    2x2 HW forward scaling matrices to apply to the input slices before the rotation.
    /// \param[in] input_inv_rotation   3x3 DHW inverse rotation-matrices or quaternions to apply to the input slices.
    /// \param[in] output_inv_scaling   2x2 HW inverse scaling matrix to apply to the output slices before the rotation.
    /// \param[in] output_fwd_rotation  3x3 DHW forward rotation-matrices or quaternions to apply to the output slices.
    /// \param options                  Operator options.
    template<Remap REMAP,
             typename Input, typename InputWeight = Empty,
             typename Output, typename OutputWeight = Empty,
             typename InputScale = Empty, typename InputRotate,
             typename OutputScale = Empty, typename OutputRotate>
    requires (guts::fourier_projection_input_output<true, true, Input, Output, InputWeight, OutputWeight> and
              guts::fourier_projection_transform<InputScale, InputRotate> and
              guts::fourier_projection_transform<OutputScale, OutputRotate> and
              REMAP.is_hx2hx())
    void fourier_insert_interpolate_and_extract_3d(
        Input&& input_slice,
        InputWeight&& input_weight,
        const Shape4<i64>& input_slice_shape,
        Output&& output_slice,
        OutputWeight&& output_weight,
        const Shape4<i64>& output_slice_shape,
        InputScale&& input_fwd_scaling,
        InputRotate&& input_inv_rotation,
        OutputScale&& output_inv_scaling,
        OutputRotate&& output_fwd_rotation,
        const FourierInsertExtractOptions& options = {}
    ) {
        guts::fourier_projection_check_parameters<guts::ProjectionType::INSERT_EXTRACT>(
            input_slice, input_weight, input_slice_shape, output_slice, output_weight, output_slice_shape,
            {}, input_fwd_scaling, input_inv_rotation, output_inv_scaling, output_fwd_rotation);

        using coord_t = nt::value_type_twice_t<OutputRotate>;
        const auto volume_z = static_cast<coord_t>(min(output_slice_shape.filter(2, 3)));
        const auto fftfreq_blackman = static_cast<coord_t>(options.w_windowed_sinc.fftfreq_blackman);
        const auto w_blackman_size = guts::blackman_window_size<i64>(fftfreq_blackman, volume_z);
        check(not options.correct_multiplicity or (not options.add_to_output and w_blackman_size == 1),
              "options.correct_multiplicity=true is not compatible with "
              "options.add_to_output=true and options.w_windowed_sinc.fftfreq_blackman={} (={} pixels)",
              options.w_windowed_sinc.fftfreq_blackman, w_blackman_size);

        if (output_slice.device().is_gpu()) {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (
                (nt::texture_decay<Input> and nt::any_of<nt::mutable_value_type_t<Input>, f64, c64>) or
                (nt::texture_decay<InputWeight> and nt::any_of<nt::mutable_value_type_t<InputWeight>, f64, c64>)) {
                std::terminate(); // unreachable
            } else {
                check(guts::fourier_projection_is_i32_safe_access(input_slice, input_weight, output_slice, output_weight),
                      "i64 indexing not instantiated for GPU devices");
                return guts::launch_fourier_insert_interpolate_extract_3d<REMAP, i32, true>(
                    std::forward<Input>(input_slice), std::forward<InputWeight>(input_weight), input_slice_shape,
                    std::forward<Output>(output_slice), std::forward<OutputWeight>(output_weight), output_slice_shape,
                    std::forward<InputScale>(input_fwd_scaling), std::forward<InputRotate>(input_inv_rotation),
                    std::forward<OutputScale>(output_inv_scaling), std::forward<OutputRotate>(output_fwd_rotation),
                    options);
            }
            #else
            std::terminate(); // unreachable
            #endif
        }

        guts::launch_fourier_insert_interpolate_extract_3d<REMAP, i64>(
            std::forward<Input>(input_slice), std::forward<InputWeight>(input_weight), input_slice_shape,
            std::forward<Output>(output_slice), std::forward<OutputWeight>(output_weight), output_slice_shape,
            std::forward<InputScale>(input_fwd_scaling), std::forward<InputRotate>(input_inv_rotation),
            std::forward<OutputScale>(output_inv_scaling), std::forward<OutputRotate>(output_fwd_rotation),
            options);
    }

    /// Corrects for the interpolation kernel applied to Fourier transforms.
    /// \details When interpolating Fourier transforms, we effectively convolves the input Fourier components with
    ///          an interpolation kernel. As such, the resulting iFT of the interpolated output is the product of the
    ///          final wanted output and the iFT of the interpolation kernel. This function corrects for the effect of
    ///          the interpolation kernel in real-space.
    /// \param[in] input        Inverse Fourier transform of the 3d volume used for direct Fourier insertion.
    /// \param[out] output      Gridding-corrected output. Can be equal to \p input.
    /// \param interp           Interpolation method.
    /// \param post_correction  Whether the correction is the post- or pre-correction. Post correction is meant to be
    ///                         applied to the interpolated output, whereas pre-correction is meant to be applied to
    ///                         the input about to be interpolated.
    template<nt::varray_decay_of_almost_any<f32, f64> Input,
             nt::varray_decay_of_any<f32, f64> Output>
    void fourier_interpolation_correction(Input&& input, Output&& output, Interp interp, bool post_correction) {
        /// TODO Add correction for other interpolation methods.
        check(interp.is_almost_any(Interp::LINEAR), "{} is currently not supported", interp);

        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using coord_t = std::conditional_t<nt::any_of<f64, input_value_t, output_value_t>, f64, f32>;
        const auto output_shape = output.shape();
        const auto input_strides = ng::broadcast_strides(input, output);
        const auto input_accessor = Accessor<const input_value_t, 4, i64>(input.get(), input_strides.template as<i64>());
        const auto output_accessor = Accessor<output_value_t, 4, i64>(output.get(), output.strides().template as<i64>());

        if (post_correction) {
            const auto op = guts::GriddingCorrection<true, coord_t, decltype(input_accessor), decltype(output_accessor)>(
                input_accessor, output_accessor, output_shape);
            iwise(output_shape, output.device(), op,
                  std::forward<Input>(input), std::forward<Output>(output));
        } else {
            const auto op = guts::GriddingCorrection<false, coord_t, decltype(input_accessor), decltype(output_accessor)>(
                input_accessor, output_accessor, output_shape);
            iwise(output_shape, output.device(), op,
                  std::forward<Input>(input), std::forward<Output>(output));
        }
    }
}
