/// \file noa/gpu/cuda/Types.h
/// \brief Overloads of the CUDA tex1D, tex2D, tex3D functions to support different interpolation methods.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021
/// \details The following interpolation methods, using CUDA texture, are supported:
///          INTERP_(NEAREST|LINEAR|COSINE|CUBIC_BSPLINE).

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

// Forward declarations
namespace noa::cuda::transform::details {
    template<typename T> NOA_FD T tex1D(cudaTextureObject_t tex, float x);
    template<typename T> NOA_FD T tex2D(cudaTextureObject_t tex, float x, float y);
    template<typename T> NOA_FD T tex3D(cudaTextureObject_t tex, float x, float y, float z);

    namespace cosine {
        template<typename T> NOA_DEVICE T tex1D(cudaTextureObject_t tex, float x);
        template<typename T> NOA_DEVICE T tex2D(cudaTextureObject_t tex, float x, float y);
        template<typename T> NOA_DEVICE T tex3D(cudaTextureObject_t tex, float x, float y, float z);
    }

    namespace bspline {
        template<typename T> NOA_DEVICE T tex1D(cudaTextureObject_t tex, float x);
        template<typename T> NOA_DEVICE T tex2D(cudaTextureObject_t tex, float x, float y);
        template<typename T> NOA_DEVICE T tex3D(cudaTextureObject_t tex, float x, float y, float z);
    }
}

namespace noa::cuda::transform {
    /// 1D interpolation of the data in \a texture at the texture coordinate \a x.
    /// \tparam MODE            Interpolation method to use.
    /// \param[out] fetched     Interpolated output value.
    /// \param texture          Valid CUDA texture object.
    /// \param x                First dimension coordinate.
    ///
    /// \details
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), un-normalized coordinates are expected.
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), \a texture should be in linear mode.
    template<InterpMode MODE>
    NOA_FD void tex1D(float* fetched, cudaTextureObject_t texture, float x) {
        static_assert(MODE == INTERP_NEAREST || MODE == INTERP_LINEAR ||
                      MODE == INTERP_COSINE || MODE == INTERP_CUBIC_BSPLINE);

        if constexpr (MODE == INTERP_NEAREST || MODE == INTERP_LINEAR) {
            *fetched = details::tex1D<float>(texture, x);
        } else if constexpr (MODE == INTERP_COSINE) {
            *fetched = details::cosine::tex1D<float>(texture, x);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE) {
            *fetched = details::bspline::tex1D<float>(texture, x);
        }
    }

    /// 1D interpolation of the data in \a texture at the texture coordinate \a x.
    /// \tparam MODE            Interpolation method to use.
    /// \param[out] fetched     Interpolated output value.
    /// \param texture          Valid CUDA texture object. The channel descriptor should be float2.
    /// \param x                First dimension coordinate.
    ///
    /// \details
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), un-normalized coordinates are expected.
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), \a texture should be in linear mode.
    ///
    /// \note An overload of `cudaCreateChannelDesc<>(::noa::cfloat_t)` is added by "noa/gpu/cuda/Types.h", so
    ///       if this file is included before the \a texture (or the underlying CUDA array) creation, or if it was
    ///       created by PtrTexture<> (or PtrArray<>), the channel descriptor will be correctly set for this function.
    template<InterpMode MODE>
    NOA_FD void tex1D(cfloat_t* interpolated_value, cudaTextureObject_t texture, float x) {
        static_assert(MODE == INTERP_NEAREST || MODE == INTERP_LINEAR ||
                      MODE == INTERP_COSINE || MODE == INTERP_CUBIC_BSPLINE);

        float2 tmp;
        if constexpr (MODE == INTERP_NEAREST || MODE == INTERP_LINEAR) {
            tmp = details::tex1D<float2>(texture, x);
        } else if constexpr (MODE == INTERP_COSINE) {
            tmp = details::cosine::tex1D<float2>(texture, x);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE) {
            tmp = details::bspline::tex1D<float2>(texture, x);
        }
        interpolated_value->real(tmp.x);
        interpolated_value->imag(tmp.y);
    }

    /// 2D interpolation of the data in \a texture at the texture coordinate \a x.
    /// \tparam MODE            Interpolation method to use.
    /// \param[out] fetched     Interpolated output value.
    /// \param texture          Valid CUDA texture object.
    /// \param x                First dimension coordinate.
    /// \param y                Second dimension coordinate.
    ///
    /// \details
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), un-normalized coordinates are expected.
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), \a texture should be in linear mode.
    template<InterpMode MODE>
    NOA_FD void tex2D(float* fetched, cudaTextureObject_t texture, float x, float y) {
        static_assert(MODE == INTERP_NEAREST || MODE == INTERP_LINEAR ||
                      MODE == INTERP_COSINE || MODE == INTERP_CUBIC_BSPLINE);

        if constexpr (MODE == INTERP_NEAREST || MODE == INTERP_LINEAR) {
            *fetched = details::tex2D<float>(texture, x, y);
        } else if constexpr (MODE == INTERP_COSINE) {
            *fetched = details::cosine::tex2D<float>(texture, x, y);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE) {
            *fetched = details::bspline::tex2D<float>(texture, x, y);
        }
    }

    /// 2D interpolation of the data in \a texture at the texture coordinate \a x.
    /// \tparam MODE            Interpolation method to use.
    /// \param[out] fetched     Interpolated output value.
    /// \param texture          Valid CUDA texture object. The channel descriptor should be float2.
    /// \param x                First dimension coordinate.
    /// \param y                Second dimension coordinate.
    ///
    /// \details
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), un-normalized coordinates are expected.
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), \a texture should be in linear mode.
    ///
    /// \note An overload of `cudaCreateChannelDesc<>(::noa::cfloat_t)` is added by "noa/gpu/cuda/Types.h", so
    ///       if this file is included before the \a texture (or the underlying CUDA array) creation, or if it was
    ///       created by PtrTexture<> (or PtrArray<>), the channel descriptor will be correctly set for this function.
    template<InterpMode MODE>
    NOA_FD void tex2D(cfloat_t* fetched, cudaTextureObject_t texture, float x, float y) {
        static_assert(MODE == INTERP_NEAREST || MODE == INTERP_LINEAR ||
                      MODE == INTERP_COSINE || MODE == INTERP_CUBIC_BSPLINE);

        float2 tmp;
        if constexpr (MODE == INTERP_NEAREST || MODE == INTERP_LINEAR) {
            tmp = details::tex2D<float2>(texture, x, y);
        } else if constexpr (MODE == INTERP_COSINE) {
            tmp = details::cosine::tex2D<float2>(texture, x, y);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE) {
            tmp = details::bspline::tex2D<float2>(texture, x, y);
        }
        fetched->real(tmp.x);
        fetched->imag(tmp.y);
    }

    /// 3D interpolation of the data in \a texture at the texture coordinate \a x.
    /// \tparam MODE            Interpolation method to use.
    /// \param[out] fetched     Interpolated output value.
    /// \param texture          Valid CUDA texture object.
    /// \param x                First dimension coordinate.
    /// \param y                Second dimension coordinate.
    /// \param z                Third dimension coordinate.
    ///
    /// \details
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), un-normalized coordinates are expected.
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), \a texture should be in linear mode.
    template<InterpMode MODE>
    NOA_FD void tex3D(float* fetched, cudaTextureObject_t texture, float x, float y, float z) {
        static_assert(MODE == INTERP_NEAREST || MODE == INTERP_LINEAR ||
                      MODE == INTERP_COSINE || MODE == INTERP_CUBIC_BSPLINE);

        if constexpr (MODE == INTERP_NEAREST || MODE == INTERP_LINEAR) {
            *fetched = details::tex3D<float>(texture, x, y, z);
        } else if constexpr (MODE == INTERP_COSINE) {
            *fetched = details::cosine::tex3D<float>(texture, x, y, z);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE) {
            *fetched = details::bspline::tex3D<float>(texture, x, y, z);
        }
    }

    /// 3D interpolation of the data in \a texture at the texture coordinate \a x.
    /// \tparam MODE            Interpolation method to use.
    /// \param[out] fetched     Interpolated output value.
    /// \param texture          Valid CUDA texture object. The channel descriptor should be float2.
    /// \param x                First dimension coordinate.
    /// \param y                Second dimension coordinate.
    /// \param z                Third dimension coordinate.
    ///
    /// \details
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), un-normalized coordinates are expected.
    ///     - If MODE == INTERP_(COSINE|CUBIC_BSPLINE), \a texture should be in linear mode.
    ///
    /// \note An overload of `cudaCreateChannelDesc<>(::noa::cfloat_t)` is added by "noa/gpu/cuda/Types.h", so
    ///       if this file is included before the \a texture (or the underlying CUDA array) creation, or if it was
    ///       created by PtrTexture<> (or PtrArray<>), the channel descriptor will be correctly set for this function.
    template<InterpMode MODE>
    NOA_FD void tex3D(cfloat_t* fetched, cudaTextureObject_t texture, float x, float y, float z) {
        static_assert(MODE == INTERP_NEAREST || MODE == INTERP_LINEAR ||
                      MODE == INTERP_COSINE || MODE == INTERP_CUBIC_BSPLINE);

        float2 tmp;
        if constexpr (MODE == INTERP_NEAREST || MODE == INTERP_LINEAR) {
            tmp = details::tex3D<float2>(texture, x, y, z);
        } else if constexpr (MODE == INTERP_COSINE) {
            tmp = details::cosine::tex3D<float2>(texture, x, y, z);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE) {
            tmp = details::bspline::tex3D<float2>(texture, x, y, z);
        }
        fetched->real(tmp.x);
        fetched->imag(tmp.y);
    }
}

namespace noa::cuda::transform::bspline {
    /// Applies a 2D prefilter to \a inputs so that the interpolated cubic B-spline values will pass through the sample data.
    /// \tparam T               float or cfloat_t.
    /// \param inputs           Input arrays. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \a inputs.
    /// \param outputs          Output arrays. One per batch. Can be equal to \a inputs, otherwise, they shouldn't overlap.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs and \a outputs, ignoring the batches.
    ///                         The last dimension is ignored and assumed to be 1.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///
    /// \details From Danny Ruijters:
    ///          "When the approach described above is directly applied, it will result in smoothened images.
    ///          This is caused by the fact that the cubic B-spline filtering yields a function that does not
    ///          pass through its coefficients (i.e. texture values). In order to wind up with a cubic B-spline
    ///          interpolated image that passes through the original samples, we need to pre-filter the texture".
    /// \see http://www.dannyruijters.nl/cubicinterpolation/ for more details.
    template<typename T>
    NOA_HOST void prefilter2D(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                              size3_t shape, uint batches, Stream& stream);

    /// Applies a 3D prefilter to \a inputs so that the interpolated cubic B-spline values will pass through the sample data.
    /// \tparam T               float or cfloat_t.
    /// \param inputs           Input arrays. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \a inputs.
    /// \param outputs          Output arrays. One per batch. Can be equal to \a inputs, otherwise, they shouldn't overlap.
    /// \param outputs_pitch    Pitch, in elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs and \a outputs, ignoring the batches.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///
    /// \details From Danny Ruijters:
    ///          "When the approach described above is directly applied, it will result in smoothened images.
    ///          This is caused by the fact that the cubic B-spline filtering yields a function that does not
    ///          pass through its coefficients (i.e. texture values). In order to wind up with a cubic B-spline
    ///          interpolated image that passes through the original samples, we need to pre-filter the texture".
    /// \see http://www.dannyruijters.nl/cubicinterpolation/ for more details.
    template<typename T>
    NOA_HOST void prefilter3D(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                              size3_t shape, uint batches, Stream& stream);

    /// Applies a prefilter to \a inputs so that the interpolated cubic B-spline values will pass through the sample data.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void prefilter(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                          size3_t shape, uint batches, Stream& stream) {
        uint ndim = getNDim(shape);
        if (ndim == 3)
            prefilter3D(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, stream);
        else if (ndim == 2)
            prefilter2D(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, stream);
        else
            NOA_THROW("Cubic B-spline pre-filtering is only available for 2D and 3D arrays");
    }
}

// -- Texture fetching implementation -- //

namespace noa::cuda::transform::details {
    template<typename T>
    NOA_FD T tex1D(cudaTextureObject_t tex, float x) {
        #ifdef __CUDA_ARCH__
        return ::tex1D<T>(tex, x);
        #endif
    }

    template<typename T>
    NOA_FD T tex2D(cudaTextureObject_t tex, float x, float y) {
        #ifdef __CUDA_ARCH__
        return ::tex2D<T>(tex, x, y);
        #endif
    }

    template<typename T>
    NOA_FD T tex3D(cudaTextureObject_t tex, float x, float y, float z) {
        #ifdef __CUDA_ARCH__
        return ::tex3D<T>(tex, x, y, z);
        #endif
    }
}

// These cudaTextureObject_t should be set to INTERP_LINEAR, unnormalized coordinates.
namespace noa::cuda::transform::details::cosine {
    template<typename T>
    NOA_DEVICE T tex1D(cudaTextureObject_t tex, float x) {
        const float index = math::floor(x);
        float fraction = x - index;
        fraction = (1.f - math::cos(fraction * math::Constants<float>::PI)) / 2.f; // cosine smoothing
        return details::tex1D<T>(tex, index + fraction); // fetch the linear interpolation
    }

    template<typename T>
    NOA_DEVICE T tex2D(cudaTextureObject_t tex, float x, float y) {
        const float2 index = {math::floor(x), math::floor(y)};
        float2 fraction = {x - index.x, y - index.y};
        fraction.x = (1.f - math::cos(fraction.x * math::Constants<float>::PI)) / 2.f;
        fraction.y = (1.f - math::cos(fraction.y * math::Constants<float>::PI)) / 2.f;
        return details::tex2D<T>(tex, index.x + fraction.x, index.y + fraction.y);
    }

    template<typename T>
    NOA_DEVICE T tex3D(cudaTextureObject_t tex, float x, float y, float z) {
        const float3 index = {math::floor(x), math::floor(y), math::floor(z)};
        float3 fraction = {x - index.x, y - index.y, z - index.z};
        fraction.x = (1.f - math::cos(fraction.x * math::Constants<float>::PI)) / 2.f;
        fraction.y = (1.f - math::cos(fraction.y * math::Constants<float>::PI)) / 2.f;
        fraction.z = (1.f - math::cos(fraction.z * math::Constants<float>::PI)) / 2.f;
        return details::tex3D<T>(tex, index.x + fraction.x, index.y + fraction.y, index.z + fraction.z);
    }
}

// This is from https://github.com/DannyRuijters/CubicInterpolationCUDA
// See licences/CubicInterpolationCUDA.txt
// This is very much like "cuda/samples/3_Imaging/bicubicTexture/bicubicTexture_kernel.cuh"
namespace noa::cuda::transform::details::bspline {
    // Computes the bspline convolution weights. fraction is from 0 to 1.
    template<typename T>
    NOA_ID void weights(T fraction, T* w0, T* w1, T* w2, T* w3) {
        const T one_frac = 1.0f - fraction;
        const T squared = fraction * fraction;
        const T one_sqd = one_frac * one_frac;

        *w0 = 1.0f / 6.0f * one_sqd * one_frac;
        *w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
        *w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
        *w3 = 1.0f / 6.0f * squared * fraction;
    }

    // Bicubic interpolated texture lookup, using unnormalized coordinates.
    // Fast implementation, using 2 linear lookups.
    template<typename T>
    NOA_DEVICE T tex1D(cudaTextureObject_t tex, float x) {
        // x is expected to have the +0.5 offset to compensate for CUDA linear filtering convention,
        // so to get the fraction and compute the weights, remove this offset temporarily.
        const float coord_grid = x - 0.5f;
        const float index = math::floor(coord_grid);
        const float fraction = coord_grid - index;
        float w0, w1, w2, w3;
        weights(fraction, &w0, &w1, &w2, &w3);

        const float g0 = w0 + w1;
        const float g1 = w2 + w3;
        const float h0 = (w1 / g0) - 0.5f + index; // h0 = w1/g0 - 1 + index, +0.5 to add the offset back
        const float h1 = (w3 / g1) + 1.5f + index; // h1 = w3/g1 + 1 + index, +0.5 to add the offset back

        // fetch the two linear interpolations
        T tex0 = details::tex1D<T>(tex, h0);
        T tex1 = details::tex1D<T>(tex, h1);

        // weight along the x-direction
        return (g0 * tex0 + g1 * tex1);
    }

    template<typename T>
    NOA_DEVICE T tex2D(cudaTextureObject_t tex, float x, float y) {
        const float2_t coord_grid(x - 0.5f, y - 0.5f);
        const float2_t index(math::floor(coord_grid));
        const float2_t fraction(coord_grid - index);
        float2_t w0, w1, w2, w3;
        weights(fraction, &w0, &w1, &w2, &w3);

        const float2_t g0(w0 + w1);
        const float2_t g1(w2 + w3);
        const float2_t h0(w1 / g0 - 0.5f + index);
        const float2_t h1(w3 / g1 + 1.5f + index);

        // fetch the four linear interpolations
        T tex00 = details::tex2D<T>(tex, h0.x, h0.y);
        T tex10 = details::tex2D<T>(tex, h1.x, h0.y);
        T tex01 = details::tex2D<T>(tex, h0.x, h1.y);
        T tex11 = details::tex2D<T>(tex, h1.x, h1.y);

        // weight along the y-direction
        tex00 = g0.y * tex00 + g1.y * tex01;
        tex10 = g0.y * tex10 + g1.y * tex11;

        // weight along the x-direction
        return (g0.x * tex00 + g1.x * tex10);
    }

    template<typename T>
    NOA_DEVICE T tex3D(cudaTextureObject_t tex, float x, float y, float z) {
        const float3_t coord_grid(x - 0.5f, y - 0.5f, z - 0.5f);
        const float3_t index(math::floor(coord_grid));
        const float3_t fraction(coord_grid - index);
        float3_t w0, w1, w2, w3;
        weights(fraction, &w0, &w1, &w2, &w3);

        const float3_t g0(w0 + w1);
        const float3_t g1(w2 + w3);
        const float3_t h0(w1 / g0 - 0.5f + index);
        const float3_t h1(w3 / g1 + 1.5f + index);

        // fetch the eight linear interpolations
        // weighting and fetching is interleaved for performance and stability reasons
        T tex000 = details::tex3D<T>(tex, h0.x, h0.y, h0.z);
        T tex100 = details::tex3D<T>(tex, h1.x, h0.y, h0.z);
        tex000 = g0.x * tex000 + g1.x * tex100; // weight along the x-direction
        T tex010 = details::tex3D<T>(tex, h0.x, h1.y, h0.z);
        T tex110 = details::tex3D<T>(tex, h1.x, h1.y, h0.z);
        tex010 = g0.x * tex010 + g1.x * tex110; // weight along the x-direction
        tex000 = g0.y * tex000 + g1.y * tex010; // weight along the y-direction
        T tex001 = details::tex3D<T>(tex, h0.x, h0.y, h1.z);
        T tex101 = details::tex3D<T>(tex, h1.x, h0.y, h1.z);
        tex001 = g0.x * tex001 + g1.x * tex101; // weight along the x-direction
        T tex011 = details::tex3D<T>(tex, h0.x, h1.y, h1.z);
        T tex111 = details::tex3D<T>(tex, h1.x, h1.y, h1.z);
        tex011 = g0.x * tex011 + g1.x * tex111; // weight along the x-direction
        tex001 = g0.y * tex001 + g1.y * tex011; // weight along the y-direction

        return (g0.z * tex000 + g1.z * tex001); // weight along the z-direction
    }
}
