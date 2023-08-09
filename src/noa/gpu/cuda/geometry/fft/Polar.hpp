#pragma once

#include "noa/core/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::geometry::fft::details {
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
            (noa::algorithm::signal::fft::is_valid_aniso_ctf_v<Ctf> || std::is_empty_v<Ctf>) &&
            (nt::are_same_value_type_v<Input, Output, Weight> &&
             ((nt::are_all_same_v<Input, Output> &&
               nt::are_real_or_complex_v<Input, Output>) ||
              (nt::is_complex_v<Input> &&
               nt::is_real_v<Output>)));
}

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename Input, typename Output,
             typename = std::enable_if_t<details::is_valid_polar_xform_v<REMAP, Input, Output>>>
    void cartesian2polar(
            const Input* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
            Output* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, bool frequency_range_endpoint,
            const Vec2<f32>& angle_range, bool angle_range_endpoint,
            InterpMode interp, Stream& stream);

    template<typename Input, typename Output, typename = std::enable_if_t<
             nt::is_any_v<Output, f32, c32> ||
             details::is_valid_polar_xform_v<noa::fft::HC2FC, Input, Output>>>
    void cartesian2polar(
            cudaArray* array, cudaTextureObject_t cartesian,
            InterpMode cartesian_interp, const Shape4<i64>& cartesian_shape,
            Output* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, bool frequency_range_endpoint,
            const Vec2<f32>& angle_range, bool angle_range_endpoint,
            Stream& stream);

    template<noa::fft::Remap REMAP, typename Input, typename Ctf, typename Output, typename Weight,
             typename = std::enable_if_t<details::is_valid_rotational_average_v<REMAP, Input, Ctf, Output, Weight>>>
    void rotational_average(
            const Input* input,
            const Strides4<i64>& input_strides, const Shape4<i64>& input_shape, const Ctf& input_ctf,
            Output* output, i64 output_batch_stride, Weight* weight, i64 weight_batch_stride, i64 n_shells,
            const Vec2<f32>& frequency_range, bool frequency_range_endpoint, bool average, Stream& stream);
}
