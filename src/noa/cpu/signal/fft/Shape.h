#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::signal::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_shape_v =
            traits::is_any_v<T, float, cfloat_t, double, cdouble_t> &&
            (REMAP == F2F || REMAP == FC2FC || REMAP == F2FC || REMAP == FC2F);
}

namespace noa::cpu::signal::fft {
    using namespace ::noa::fft;

    // Returns or applies an elliptical mask.
    template<fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shape_v<REMAP, T>>>
    void ellipse(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t center, float3_t radius, float edge_size,
                 float33_t inv_transform, bool invert, Stream& stream);

    // Returns or applies a 2D elliptical mask.
    template<fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shape_v<REMAP, T>>>
    void ellipse(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float2_t center, float2_t radius, float edge_size,
                 float22_t inv_transform, bool invert, Stream& stream);

    // Returns or applies a spherical mask.
    template<fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shape_v<REMAP, T>>>
    void sphere(const shared_t<T[]>& input, dim4_t input_strides,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                float3_t center, float radius, float edge_size,
                float33_t inv_transform, bool invert, Stream& stream);

    // Returns or applies a 2D spherical mask.
    template<fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shape_v<REMAP, T>>>
    void sphere(const shared_t<T[]>& input, dim4_t input_strides,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                float2_t center, float radius, float edge_size,
                float22_t inv_transform, bool invert, Stream& stream);

    // Returns or applies a rectangular mask.
    template<fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shape_v<REMAP, T>>>
    void rectangle(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   float3_t center, float3_t radius, float edge_size,
                   float33_t inv_transform, bool invert, Stream& stream);

    // Returns or applies a 2D rectangular mask.
    template<fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shape_v<REMAP, T>>>
    void rectangle(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   float2_t center, float2_t radius, float edge_size,
                   float22_t inv_transform, bool invert, Stream& stream);

    // Returns or applies a cylindrical mask.
    template<fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shape_v<REMAP, T>>>
    void cylinder(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  float3_t center, float radius, float length, float edge_size,
                  float33_t inv_transform, bool invert, Stream& stream);
}
