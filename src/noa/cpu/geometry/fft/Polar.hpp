#pragma once

#include "noa/algorithms/signal/CTF.hpp"
#include "noa/core/Types.hpp"

namespace noa::cpu::geometry::fft::details {
    using namespace ::noa::fft;

    template<Remap REMAP, typename Input, typename Output>
    constexpr bool is_valid_polar_xform_v =
            nt::is_any_v<Input, f32, f64, c32, c64> &&
            (nt::are_all_same_v<Input, Output> ||
             (nt::is_complex_v<Input> &&
              nt::is_real_v<Output> &&
              nt::are_same_value_type_v<Input, Output>)) &&
            REMAP == HC2FC;

    template<Remap REMAP, typename Input, typename Ctf, typename Output, typename Weight>
    constexpr bool is_valid_rotational_average_v =
            (REMAP == Remap::H2H || REMAP == Remap::HC2H || REMAP == Remap::F2H || REMAP == Remap::FC2H) &&
            (nt::is_ctf_anisotropic_v<Ctf> ||
             (std::is_pointer_v<Ctf> && nt::is_ctf_anisotropic_v<nt::remove_pointer_cv_t<Ctf>>) ||
             std::is_empty_v<Ctf>) &&
            (nt::are_same_value_type_v<Input, Output, Weight> &&
             ((nt::are_all_same_v<Input, Output> &&
               nt::are_real_or_complex_v<Input, Output>) ||
              (nt::is_complex_v<Input> &&
               nt::is_real_v<Output>)));
}

namespace noa::cpu::geometry::fft {
    template<noa::fft::Remap REMAP, typename Input, typename Output,
             typename = std::enable_if_t<details::is_valid_polar_xform_v<REMAP, Input, Output>>>
    void cartesian2polar(
            const Input* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
            Output* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, bool frequency_range_endpoint,
            const Vec2<f32>& angle_range, bool angle_range_endpoint,
            InterpMode interp_mode, i64 threads);

    template<noa::fft::Remap REMAP, typename Input, typename Ctf, typename Output, typename Weight,
             typename = std::enable_if_t<details::is_valid_rotational_average_v<REMAP, Input, Ctf, Output, Weight>>>
    void rotational_average(
            const Input* input, Strides4<i64> input_strides, Shape4<i64> input_shape, const Ctf& input_ctf,
            Output* output, i64 output_batch_stride, Weight* weight, i64 weight_batch_stride, i64 n_shells,
            const Vec2<f32>& frequency_range, bool frequency_range_endpoint, bool average, i64 threads);
}
