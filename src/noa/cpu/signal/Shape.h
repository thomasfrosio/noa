#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::signal {
    // Returns or applies an elliptical mask.
    template<bool INVERT = false, typename T,
             typename = std::enable_if_t<traits::is_float_v<T> || traits::is_complex_v<T>>>
    void ellipse(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t center, float3_t radius, float taper_size, Stream& stream);

    // Returns or applies a 2D elliptical mask.
    template<bool INVERT = false, typename T,
             typename = std::enable_if_t<traits::is_float_v<T> || traits::is_complex_v<T>>>
    void ellipse(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float2_t center, float2_t radius, float taper_size, Stream& stream) {
        return ellipse<INVERT>(input, input_strides, output, output_strides, shape,
                               float3_t{0, center[0], center[1]},
                               float3_t{1, radius[0], radius[1]},
                               taper_size, stream);
    }

    // Returns or applies a spherical mask.
    template<bool INVERT = false, typename T,
             typename = std::enable_if_t<traits::is_float_v<T> || traits::is_complex_v<T>>>
    void sphere(const shared_t<T[]>& input, dim4_t input_strides,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                float3_t center, float radius, float taper_size, Stream& stream);

    // Returns or applies a 2D spherical mask.
    template<bool INVERT = false, typename T,
             typename = std::enable_if_t<traits::is_float_v<T> || traits::is_complex_v<T>>>
    void sphere(const shared_t<T[]>& input, dim4_t input_strides,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                float2_t center, float radius, float taper_size, Stream& stream) {
        return sphere<INVERT>(input, input_strides, output, output_strides, shape,
                              float3_t{0, center[0], center[1]}, radius,
                              taper_size, stream);
    }

    // Returns or applies a rectangular mask.
    template<bool INVERT = false, typename T,
             typename = std::enable_if_t<traits::is_float_v<T> || traits::is_complex_v<T>>>
    void rectangle(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   float3_t center, float3_t radius, float taper_size, Stream& stream);

    // Returns or applies a 2D rectangular mask.
    template<bool INVERT = false, typename T,
             typename = std::enable_if_t<traits::is_float_v<T> || traits::is_complex_v<T>>>
    void rectangle(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   float2_t center, float2_t radius, float taper_size, Stream& stream) {
        return rectangle<INVERT>(input, input_strides, output, output_strides, shape,
                                 float3_t{0, center[0], center[1]}, float3_t{1, radius[0], radius[1]},
                                 taper_size, stream);
    }

    // Returns or applies a cylindrical mask.
    template<bool INVERT = false, typename T,
             typename = std::enable_if_t<traits::is_float_v<T> || traits::is_complex_v<T>>>
    void cylinder(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  float3_t center, float radius, float length, float taper_size, Stream& stream);
}
