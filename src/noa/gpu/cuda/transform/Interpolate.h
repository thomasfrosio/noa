/// \file noa/gpu/cuda/Types.h
/// \brief Overloads of the CUDA tex1D, tex2D, tex3D functions to support different interpolation methods.
/// \puthor Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::transform::bspline {
    /// Applies a 1D prefilter to \p inputs so that the cubic B-spline values will pass through the sample data.
    /// \tparam T               float or cfloat_t.
    /// \param inputs           On the \b device. Input arrays. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param outputs          On the \b device. Output arrays. One per batch. Can be equal to \p inputs.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param size             Size, in elements, of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches in \p inputs and \p outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note The implementation requires a single thread to go through the entire 1D array. This is not very efficient
    ///       compared to the CPU implementation. However, when multiple batches are processes, a warp can process
    ///       simultaneously as many batches as it has threads, which is more efficient.
    ///
    /// \details From Danny Ruijters:
    ///          "When the approach described above is directly applied, it will result in smoothened images.
    ///          This is caused by the fact that the cubic B-spline filtering yields a function that does not
    ///          pass through its coefficients (i.e. texture values). In order to wind up with a cubic B-spline
    ///          interpolated image that passes through the original samples, we need to pre-filter the texture".
    /// \see http://www.dannyruijters.nl/cubicinterpolation/ for more details.
    template<typename T>
    NOA_HOST void prefilter1D(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                              size_t size, uint batches, Stream& stream);

    /// Applies a 2D prefilter to \p inputs so that the cubic B-spline values will pass through the sample data.
    /// \tparam T               float or cfloat_t.
    /// \param inputs           On the \b device. Input arrays. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param outputs          On the \b device. Output arrays. One per batch. Can be equal to \p inputs.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches in \p inputs and \p outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see cuda::transform::bspline::prefilter1D() for more details.
    template<typename T>
    NOA_HOST void prefilter2D(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                              size2_t shape, uint batches, Stream& stream);

    /// Applies a 3D prefilter to \p inputs so that the cubic B-spline values will pass through the sample data.
    /// \tparam T               float or cfloat_t.
    /// \param inputs           On the \b device. Input arrays. One per batch.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param outputs          On the \b device. Output arrays. One per batch. Can be equal to \p inputs.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param batches          Number of batches in \p inputs and \p outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see cuda::transform::bspline::prefilter1D() for more details.
    template<typename T>
    NOA_HOST void prefilter3D(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                              size3_t shape, uint batches, Stream& stream);

    /// Applies a prefilter to \p inputs so that the cubic B-spline values will pass through the sample data.
    /// \tparam SIZE    size_t, size2_t, size3_t.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see Calls prefilter1D, prefilter2D or prefilter3D depending on the dimensionality of the inputs.
    template<typename T, typename SIZE>
    NOA_IH void prefilter(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                          SIZE shape, uint batches, Stream& stream) {
        if constexpr (std::is_same_v<SIZE, size_t>) {
            prefilter1D(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, stream);
        } else if constexpr (std::is_same_v<SIZE, size2_t>) {
            prefilter2D(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, stream);
        } else if constexpr (std::is_same_v<SIZE, size3_t>) {
            size_t dim = ndim(shape);
            NOA_ASSERT(dim && dim <= 3);
            switch (dim) {
                case 1:
                    prefilter1D(inputs, inputs_pitch, outputs, outputs_pitch, shape.x, batches, stream);
                    break;
                case 2:
                    prefilter2D(inputs, inputs_pitch, outputs, outputs_pitch, {shape.x, shape.y}, batches);
                    break;
                case 3:
                    prefilter3D(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, stream);
                    break;
                default:
                    break;
            }
        } else {
            static_assert(traits::always_false_v<T>);
        }
    }
}

// Forward declarations
namespace noa::cuda::transform::details {
    template<typename T> NOA_FD T tex1D(cudaTextureObject_t tex, float x);
    template<typename T> NOA_FD T tex2D(cudaTextureObject_t tex, float x, float y);
    template<typename T> NOA_FD T tex3D(cudaTextureObject_t tex, float x, float y, float z);

    namespace linear {
        template<typename T> NOA_DEVICE T tex1DAccurate(cudaTextureObject_t tex, float x);
        template<typename T> NOA_DEVICE T tex2DAccurate(cudaTextureObject_t tex, float x, float y);
        template<typename T> NOA_DEVICE T tex3DAccurate(cudaTextureObject_t tex, float x, float y, float z);
    }

    namespace cosine {
        template<typename T> NOA_DEVICE T tex1D(cudaTextureObject_t tex, float x);
        template<typename T> NOA_DEVICE T tex2D(cudaTextureObject_t tex, float x, float y);
        template<typename T> NOA_DEVICE T tex3D(cudaTextureObject_t tex, float x, float y, float z);
        template<typename T> NOA_DEVICE T tex1DAccurate(cudaTextureObject_t tex, float x);
        template<typename T> NOA_DEVICE T tex2DAccurate(cudaTextureObject_t tex, float x, float y);
        template<typename T> NOA_DEVICE T tex3DAccurate(cudaTextureObject_t tex, float x, float y, float z);
    }

    namespace cubic {
        template<typename T> NOA_DEVICE T tex1DAccurate(cudaTextureObject_t tex, float x);
        template<typename T> NOA_DEVICE T tex2DAccurate(cudaTextureObject_t tex, float x, float y);
        template<typename T> NOA_DEVICE T tex3DAccurate(cudaTextureObject_t tex, float x, float y, float z);
    }

    namespace bspline {
        template<typename T> NOA_DEVICE T tex1D(cudaTextureObject_t tex, float x);
        template<typename T> NOA_DEVICE T tex2D(cudaTextureObject_t tex, float x, float y);
        template<typename T> NOA_DEVICE T tex3D(cudaTextureObject_t tex, float x, float y, float z);
        template<typename T> NOA_DEVICE T tex1DAccurate(cudaTextureObject_t tex, float x);
        template<typename T> NOA_DEVICE T tex2DAccurate(cudaTextureObject_t tex, float x, float y);
        template<typename T> NOA_DEVICE T tex3DAccurate(cudaTextureObject_t tex, float x, float y, float z);
    }
}

namespace noa::cuda::transform {
    /// 1D interpolation of the data in \p texture at the texture coordinate \p x, using ::tex1D.
    /// \tparam T               float or cfloat_t.
    /// \tparam MODE            Interpolation method to use. Any of InterpMode.
    /// \param[out] fetched     Interpolated output value.
    /// \param texture          Valid CUDA texture object. The channel descriptor should be float2 if \p T is cfloat_t.
    /// \param x                First dimension coordinate.
    ///
    /// \note \p texture is expected to have the correct filter and addressing mode, as well as the correct coordinate
    ///       mode (normalized or unnormalized). See PtrTexture<T>::setDescription() for more details.
    /// \note An overload of `cudaCreateChannelDesc<>(::noa::cfloat_t)` is added by "noa/gpu/cuda/Types.h", so
    ///       if this file is included before the \p texture (or the underlying CUDA array) creation, or if it was
    ///       created by PtrTexture<> (or PtrArray<>), the channel descriptor will be correctly set for cfloat_t.
    template<typename T, InterpMode MODE>
    NOA_FD T tex1D(cudaTextureObject_t texture, float x) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, cfloat_t>);

        if constexpr (MODE == INTERP_NEAREST) {
            return details::tex1D<T>(texture, x); // can use normalized coordinates
        } else if constexpr (MODE == INTERP_LINEAR) {
            return details::linear::tex1DAccurate<T>(texture, x);
        } else if constexpr (MODE == INTERP_COSINE) {
            return details::cosine::tex1DAccurate<T>(texture, x);
        } else if constexpr (MODE == INTERP_CUBIC) {
            return details::cubic::tex1DAccurate<T>(texture, x);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE) {
            return details::bspline::tex1DAccurate<T>(texture, x);

        } else if constexpr (MODE == INTERP_LINEAR_FAST) {
            return details::tex1D<T>(texture, x); // can use normalized coordinates
        } else if constexpr (MODE == INTERP_COSINE_FAST) {
            return details::cosine::tex1D<T>(texture, x);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE_FAST) {
            return details::bspline::tex1D<T>(texture, x);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return 0; // unreachable, to fix spurious warning: https://stackoverflow.com/questions/64523302
    }

    /// 2D interpolation of the data in \p texture at the texture coordinates, using ::tex2D.
    /// \tparam T               float or cfloat_t.
    /// \tparam MODE            Interpolation method to use. Any of InterpMode.
    /// \param[out] fetched     Interpolated output value.
    /// \param texture          Valid CUDA texture object. The channel descriptor should be float2 if \p T is cfloat_t.
    /// \param x,y              Coordinates.
    ///
    /// \note \p texture is expected to have the correct filter and addressing mode, as well as the correct coordinate
    ///       mode (normalized or unnormalized). See PtrTexture<T>::setDescription() for more details.
    /// \note An overload of `cudaCreateChannelDesc<>(::noa::cfloat_t)` is added by "noa/gpu/cuda/Types.h", so
    ///       if this file is included before the \p texture (or the underlying CUDA array) creation, or if it was
    ///       created by PtrTexture<> (or PtrArray<>), the channel descriptor will be correctly set for cfloat_t.
    template<typename T, InterpMode MODE>
    NOA_FD T tex2D(cudaTextureObject_t texture, float x, float y) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, cfloat_t>);

        if constexpr (MODE == INTERP_NEAREST) {
            return details::tex2D<T>(texture, x, y);
        } else if constexpr (MODE == INTERP_LINEAR) {
            return details::linear::tex2DAccurate<T>(texture, x, y);
        } else if constexpr (MODE == INTERP_COSINE) {
            return details::cosine::tex2DAccurate<T>(texture, x, y);
        } else if constexpr (MODE == INTERP_CUBIC) {
            return details::cubic::tex2DAccurate<T>(texture, x, y);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE) {
            return details::bspline::tex2DAccurate<T>(texture, x, y);

        } else if constexpr (MODE == INTERP_LINEAR_FAST) {
            return details::tex2D<T>(texture, x, y);
        } else if constexpr (MODE == INTERP_COSINE_FAST) {
            return details::cosine::tex2D<T>(texture, x, y);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE_FAST) {
            return details::bspline::tex2D<T>(texture, x, y);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return 0;
    }

    /// 2D interpolation of the data in \p texture at the texture coordinates, using ::tex2D.
    template<typename T, InterpMode MODE>
    NOA_FD T tex2D(cudaTextureObject_t texture, float2_t coordinates) {
        return tex2D<T, MODE>(texture, coordinates.x, coordinates.y);
    }

    /// 3D interpolation of the data in \p texture at the texture coordinates, using ::tex3D.
    /// \tparam T               float or cfloat_t.
    /// \tparam MODE            Interpolation method to use. Any of InterpMode.
    /// \param[out] fetched     Interpolated output value.
    /// \param texture          Valid CUDA texture object. The channel descriptor should be float2 if \p T is cfloat_t.
    /// \param x,y,z            Coordinates.
    ///
    /// \note \p texture is expected to have the correct filter and addressing mode, as well as the correct coordinate
    ///       mode (normalized or unnormalized). See PtrTexture<T>::setDescription() for more details.
    /// \note An overload of `cudaCreateChannelDesc<>(::noa::cfloat_t)` is added by "noa/gpu/cuda/Types.h", so
    ///       if this file is included before the \p texture (or the underlying CUDA array) creation, or if it was
    ///       created by PtrTexture<> (or PtrArray<>), the channel descriptor will be correctly set for cfloat_t.
    template<typename T, InterpMode MODE>
    NOA_FD T tex3D(cudaTextureObject_t texture, float x, float y, float z) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, cfloat_t>);

        if constexpr (MODE == INTERP_NEAREST) {
            return details::tex3D<T>(texture, x, y, z);
        } else if constexpr (MODE == INTERP_LINEAR) {
            return details::linear::tex3DAccurate<T>(texture, x, y, z);
        } else if constexpr (MODE == INTERP_COSINE) {
            return details::cosine::tex3DAccurate<T>(texture, x, y, z);
        } else if constexpr (MODE == INTERP_CUBIC) {
            return details::cubic::tex3DAccurate<T>(texture, x, y, z);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE) {
            return details::bspline::tex3DAccurate<T>(texture, x, y, z);

        } else if constexpr (MODE == INTERP_LINEAR_FAST) {
            return details::tex3D<T>(texture, x, y, z);
        } else if constexpr (MODE == INTERP_COSINE_FAST) {
            return details::cosine::tex3D<T>(texture, x, y, z);
        } else if constexpr (MODE == INTERP_CUBIC_BSPLINE_FAST) {
            return details::bspline::tex3D<T>(texture, x, y, z);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return 0;
    }

    /// 3D interpolation of the data in \p texture at the texture coordinates, using ::tex3D.
    template<typename T, InterpMode MODE>
    NOA_FD T tex3D(cudaTextureObject_t texture, float3_t coordinates) {
        return tex3D<T, MODE>(texture, coordinates.x, coordinates.y, coordinates.z);
    }
}

// -- Texture interpolation implementation -- //
// These are device only functions and should only be
// compiled if the compilation is steered by nvcc.

#ifdef __CUDACC__
namespace noa::cuda::transform::details {
    template<>
    NOA_FD float tex1D<float>(cudaTextureObject_t tex, float x) {
        return ::tex1D<float>(tex, x);
    }

    template<>
    NOA_FD float tex2D<float>(cudaTextureObject_t tex, float x, float y) {
        return ::tex2D<float>(tex, x, y);
    }

    template<>
    NOA_FD float tex3D<float>(cudaTextureObject_t tex, float x, float y, float z) {
        return ::tex3D<float>(tex, x, y, z);
    }

    template<>
    NOA_FD cfloat_t tex1D<cfloat_t>(cudaTextureObject_t tex, float x) {
        auto tmp = ::tex1D<float2>(tex, x);
        return {tmp.x, tmp.y};
    }

    template<>
    NOA_FD cfloat_t tex2D<cfloat_t>(cudaTextureObject_t tex, float x, float y) {
        auto tmp = ::tex2D<float2>(tex, x, y);
        return {tmp.x, tmp.y};
    }

    template<>
    NOA_FD cfloat_t tex3D<cfloat_t>(cudaTextureObject_t tex, float x, float y, float z) {
        auto tmp = ::tex3D<float2>(tex, x, y, z);
        return {tmp.x, tmp.y};
    }

    template<typename T>
    NOA_FD T linear1D(T v0, T v1, float r) {
        return r * (v1 - v0) + v0;
    }

    template<typename T>
    NOA_FD T linear2D(T v00, T v01, T v10, T v11, float rx, float ry) {
        T tmp1 = linear1D(v00, v01, rx);
        T tmp2 = linear1D(v10, v11, rx);
        return linear1D(tmp1, tmp2, ry);
    }
}

namespace noa::cuda::transform::details::linear {
    template<typename T>
    NOA_DEVICE T tex1DAccurate(cudaTextureObject_t tex, float x) {
        x -= 0.5f;
        float index = noa::math::floor(x);
        float fraction = x - index;
        index += 0.5f;
        return linear1D(details::tex1D<T>(tex, index), details::tex1D<T>(tex, index + 1.f), fraction);
    }

    template<typename T>
    NOA_DEVICE T tex2DAccurate(cudaTextureObject_t tex, float x, float y) {
        x -= 0.5f;
        y -= 0.5f;
        float2 index{noa::math::floor(x), noa::math::floor(y)};
        const float2 fraction{x - index.x, y - index.y};
        index.x += 0.5f;
        index.y += 0.5f;

        const T v0 = linear1D(details::tex2D<T>(tex, index.x, index.y),
                              details::tex2D<T>(tex, index.x + 1.0f, index.y), fraction.x);
        const T v1 = linear1D(details::tex2D<T>(tex, index.x, index.y + 1.0f),
                              details::tex2D<T>(tex, index.x + 1.0f, index.y + 1.0f), fraction.x);
        return linear1D(v0, v1, fraction.y);
    }

    template<typename T>
    NOA_DEVICE T tex3DAccurate(cudaTextureObject_t tex, float x, float y, float z) {
        x -= 0.5f;
        y -= 0.5f;
        z -= 0.5f;
        float3_t index{noa::math::floor(x),
                       noa::math::floor(y),
                       noa::math::floor(z)};
        const float3 fraction{x - index.x,
                              y - index.y,
                              z - index.z};
        index.x += 0.5f;
        index.y += 0.5f;
        index.z += 0.5f;

        const T y0 = linear2D(details::tex3D<T>(tex, index.x, index.y, index.z),
                              details::tex3D<T>(tex, index.x + 1.0f, index.y, index.z),
                              details::tex3D<T>(tex, index.x, index.y + 1.0f, index.z),
                              details::tex3D<T>(tex, index.x + 1.0f, index.y + 1.0f, index.z),
                              fraction.x, fraction.y);
        const T y1 = linear2D(details::tex3D<T>(tex, index.x, index.y, index.z + 1.0f),
                              details::tex3D<T>(tex, index.x + 1.0f, index.y, index.z + 1.0f),
                              details::tex3D<T>(tex, index.x, index.y + 1.0f, index.z + 1.0f),
                              details::tex3D<T>(tex, index.x + 1.0f, index.y + 1.0f, index.z + 1.0f),
                              fraction.x, fraction.y);
        return linear1D(y0, y1, fraction.z);
    }
}

namespace noa::cuda::transform::details::cosine {
    // Fast 1D cosine interpolation using 1 linear lookup and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex1D(cudaTextureObject_t tex, float x) {
        const float coord_grid = x - 0.5f; // remove texture offset
        const float index = noa::math::floor(coord_grid);
        float fraction = coord_grid - index;
        fraction = (1.f - noa::math::cos(fraction * noa::math::Constants<float>::PI)) / 2.f; // cosine smoothing
        return details::tex1D<T>(tex, index + fraction + 0.5f); // add texture offset and fetch the linear interpolation
    }

    // Fast 2D cosine interpolation using 1 linear lookup and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex2D(cudaTextureObject_t tex, float x, float y) {
        const float2 coord_grid{x - 0.5f, y - 0.5f};
        const float2 index{noa::math::floor(coord_grid.x),
                           noa::math::floor(coord_grid.y)};
        float2 fraction{coord_grid.x - index.x,
                        coord_grid.y - index.y};
        fraction.x = (1.f - noa::math::cos(fraction.x * noa::math::Constants<float>::PI)) / 2.f;
        fraction.y = (1.f - noa::math::cos(fraction.y * noa::math::Constants<float>::PI)) / 2.f;
        return details::tex2D<T>(tex,
                                 index.x + fraction.x + 0.5f,
                                 index.y + fraction.y + 0.5f);
    }

    // Fast 3D cosine interpolation using 1 linear lookup and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex3D(cudaTextureObject_t tex, float x, float y, float z) {
        const float3 coord_grid{x - 0.5f, y - 0.5f, z - 0.5f};
        const float3 index{noa::math::floor(coord_grid.x),
                           noa::math::floor(coord_grid.y),
                           noa::math::floor(coord_grid.z)};
        float3 fraction{coord_grid.x - index.x,
                        coord_grid.y - index.y,
                        coord_grid.z - index.z};
        fraction.x = (1.f - noa::math::cos(fraction.x * noa::math::Constants<float>::PI)) / 2.f;
        fraction.y = (1.f - noa::math::cos(fraction.y * noa::math::Constants<float>::PI)) / 2.f;
        fraction.z = (1.f - noa::math::cos(fraction.z * noa::math::Constants<float>::PI)) / 2.f;
        return details::tex3D<T>(tex,
                                 index.x + fraction.x + 0.5f,
                                 index.y + fraction.y + 0.5f,
                                 index.z + fraction.z + 0.5f);
    }

    // Slow but precise 1D cosine interpolation using
    // 2 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex1DAccurate(cudaTextureObject_t tex, float x) {
        x -= 0.5f;
        float index = noa::math::floor(x);
        float fraction = x - index;
        index += 0.5f;
        fraction = (1.f - noa::math::cos(fraction * noa::math::Constants<float>::PI)) / 2.f;
        return linear1D(details::tex1D<T>(tex, index), details::tex1D<T>(tex, index + 1.f), fraction);
    }

    // Slow but precise 2D cosine interpolation using
    // 4 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex2DAccurate(cudaTextureObject_t tex, float x, float y) {
        x -= 0.5f;
        y -= 0.5f;
        float2 index{noa::math::floor(x), noa::math::floor(y)};
        float2 fraction{x - index.x, y - index.y};
        index.x += 0.5f;
        index.y += 0.5f;

        fraction.x = (1.f - noa::math::cos(fraction.x * noa::math::Constants<float>::PI)) / 2.f;
        fraction.y = (1.f - noa::math::cos(fraction.y * noa::math::Constants<float>::PI)) / 2.f;
        const T v0 = linear1D(details::tex2D<T>(tex, index.x, index.y),
                              details::tex2D<T>(tex, index.x + 1.0f, index.y), fraction.x);
        const T v1 = linear1D(details::tex2D<T>(tex, index.x, index.y + 1.0f),
                              details::tex2D<T>(tex, index.x + 1.0f, index.y + 1.0f), fraction.x);
        return linear1D(v0, v1, fraction.y);
    }

    // Slow but precise 3D cosine interpolation using
    // 8 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex3DAccurate(cudaTextureObject_t tex, float x, float y, float z) {
        x -= 0.5f;
        y -= 0.5f;
        z -= 0.5f;
        float3_t index{noa::math::floor(x), noa::math::floor(y), noa::math::floor(z)};
        float3 fraction{x - index.x, y - index.y, z - index.z};
        index.x += 0.5f;
        index.y += 0.5f;
        index.z += 0.5f;

        fraction.x = (1.f - noa::math::cos(fraction.x * noa::math::Constants<float>::PI)) / 2.f;
        fraction.y = (1.f - noa::math::cos(fraction.y * noa::math::Constants<float>::PI)) / 2.f;
        fraction.z = (1.f - noa::math::cos(fraction.z * noa::math::Constants<float>::PI)) / 2.f;
        const T y0 = linear2D(details::tex3D<T>(tex, index.x, index.y, index.z),
                              details::tex3D<T>(tex, index.x + 1.0f, index.y, index.z),
                              details::tex3D<T>(tex, index.x, index.y + 1.0f, index.z),
                              details::tex3D<T>(tex, index.x + 1.0f, index.y + 1.0f, index.z),
                              fraction.x, fraction.y);
        const T y1 = linear2D(details::tex3D<T>(tex, index.x, index.y, index.z + 1.0f),
                              details::tex3D<T>(tex, index.x + 1.0f, index.y, index.z + 1.0f),
                              details::tex3D<T>(tex, index.x, index.y + 1.0f, index.z + 1.0f),
                              details::tex3D<T>(tex, index.x + 1.0f, index.y + 1.0f, index.z + 1.0f),
                              fraction.x, fraction.y);
        return linear1D(y0, y1, fraction.z);
    }
}

namespace noa::cuda::transform::details::cubic {
    template<typename T>
    NOA_DEVICE T cubic1D(T v0, T v1, T v2, T v3, float r) {
        T a0 = v3 - v2 - v0 + v1;
        T a1 = v0 - v1 - a0;
        T a2 = v2 - v0;
        // a3 = v1
        float r2 = r * r;
        return a0 * r2 * r + a1 * r2 + a2 * r + v1;
    }

    // Slow but precise 1D cubic interpolation using
    // 4 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex1DAccurate(cudaTextureObject_t tex, float x) {
        x -= 0.5f;
        float index = noa::math::floor(x);
        float fraction = x - index;
        index += 0.5f;
        return cubic1D(details::tex1D<T>(tex, index - 1.0f),
                       details::tex1D<T>(tex, index),
                       details::tex1D<T>(tex, index + 1.0f),
                       details::tex1D<T>(tex, index + 2.f),
                       fraction);
    }

    // Slow but precise 2D cubic interpolation using
    // 16 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex2DAccurate(cudaTextureObject_t tex, float x, float y) {
        const float2_t coord_grid(x - 0.5f, y - 0.5f);
        float2_t index(noa::math::floor(coord_grid));
        const float2_t fraction(coord_grid - index);
        index += 0.5f;

        T v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float i_y = index.y + static_cast<float>(i - 1);
            v[i] = cubic1D(details::tex2D<T>(tex, index.x - 1.f, i_y),
                           details::tex2D<T>(tex, index.x, i_y),
                           details::tex2D<T>(tex, index.x + 1.f, i_y),
                           details::tex2D<T>(tex, index.x + 2.f, i_y),
                           fraction.x);
        }
        return cubic1D(v[0], v[1], v[2], v[3], fraction.y);
    }

    // Slow but precise 3D cubic interpolation using
    // 64 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex3DAccurate(cudaTextureObject_t tex, float x, float y, float z) {
        const float3_t coord_grid(x - 0.5f, y - 0.5f, z - 0.5f);
        float3_t index(noa::math::floor(coord_grid));
        const float3_t fraction(coord_grid - index);
        index += 0.5f;

        T v[4];
        T tmp[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float i_z = index.z + static_cast<float>(j - 1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float i_y = index.y + static_cast<float>(i - 1);
                tmp[i] = cubic1D(details::tex3D<T>(tex, index.x - 1.f, i_y, i_z),
                                 details::tex3D<T>(tex, index.x, i_y, i_z),
                                 details::tex3D<T>(tex, index.x + 1.f, i_y, i_z),
                                 details::tex3D<T>(tex, index.x + 2.f, i_y, i_z),
                                 fraction.x);
            }
            v[j] = cubic1D(tmp[0], tmp[1], tmp[2], tmp[3], fraction.y);
        }
        return cubic1D(v[0], v[1], v[2], v[3], fraction.z);
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

    // 1D bicubic interpolated texture lookup, using unnormalized coordinates.
    // Fast implementation, using 2 linear lookups.
    template<typename T>
    NOA_DEVICE T tex1D(cudaTextureObject_t tex, float x) {
        // x is expected to have the +0.5 offset to compensate for CUDA linear filtering convention,
        // so to get the fraction and compute the weights, remove this offset temporarily.
        const float coord_grid = x - 0.5f;
        const float index = noa::math::floor(coord_grid);
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
        return g0 * tex0 + g1 * tex1;
    }

    // 2D bicubic interpolated texture lookup, using unnormalized coordinates.
    // Fast implementation, using 4 linear lookups.
    template<typename T>
    NOA_DEVICE T tex2D(cudaTextureObject_t tex, float x, float y) {
        const float2_t coord_grid(x - 0.5f, y - 0.5f);
        const float2_t index(noa::math::floor(coord_grid));
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
        return g0.x * tex00 + g1.x * tex10;
    }

    // 3D bicubic interpolated texture lookup, using unnormalized coordinates.
    // Fast implementation, using 8 linear lookups.
    template<typename T>
    NOA_DEVICE T tex3D(cudaTextureObject_t tex, float x, float y, float z) {
        const float3_t coord_grid(x - 0.5f, y - 0.5f, z - 0.5f);
        const float3_t index(noa::math::floor(coord_grid));
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

        return g0.z * tex000 + g1.z * tex001; // weight along the z-direction
    }

    // Slow but precise 1D cubic B-spline interpolation using
    // 4 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex1DAccurate(cudaTextureObject_t tex, float x) {
        x -= 0.5f;
        float idx = noa::math::floor(x);
        float f = x - idx;
        idx += 0.5f;

        float w0, w1, w2, w3;
        weights(f, &w0, &w1, &w2, &w3);
        return details::tex1D<T>(tex, idx - 1.f) * w0 +
               details::tex1D<T>(tex, idx) * w1 +
               details::tex1D<T>(tex, idx + 1.f) * w2 +
               details::tex1D<T>(tex, idx + 2.f) * w3;
    }

    // Slow but precise 2D cubic B-spline interpolation using
    // 16 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex2DAccurate(cudaTextureObject_t tex, float x, float y) {
        const float2_t coord_grid(x - 0.5f, y - 0.5f);
        float2_t index(noa::math::floor(coord_grid));
        const float2_t fraction(coord_grid - index);
        index += 0.5f;
        float w0, w1, w2, w3;
        weights(fraction.x, &w0, &w1, &w2, &w3);

        T v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float i_y = index.y + static_cast<float>(i - 1);
            v[i] = details::tex2D<T>(tex, index.x - 1.f, i_y) * w0 +
                   details::tex2D<T>(tex, index.x, i_y) * w1 +
                   details::tex2D<T>(tex, index.x + 1.f, i_y) * w2 +
                   details::tex2D<T>(tex, index.x + 2.f, i_y) * w3;
        }
        weights(fraction.y, &w0, &w1, &w2, &w3);
        return v[0] * w0 + v[1] * w1 + v[2] * w2 + v[3] * w3;
    }

    // Slow but precise 3D cubic B-spline interpolation using
    // 64 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex3DAccurate(cudaTextureObject_t tex, float x, float y, float z) {
        const float3_t coord_grid(x - 0.5f, y - 0.5f, z - 0.5f);
        float3_t index(noa::math::floor(coord_grid));
        const float3_t fraction(coord_grid - index);
        index += 0.5f;
        float2_t w0, w1, w2, w3; // compute only the x and y weights for now, leave z weights for later
        weights(float2_t{fraction.x, fraction.y}, &w0, &w1, &w2, &w3);

        T v[4];
        T tmp[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float i_z = index.z + static_cast<float>(j - 1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float i_y = index.y + static_cast<float>(i - 1);
                tmp[i] = details::tex3D<T>(tex, index.x - 1.f, i_y, i_z) * w0.x +
                         details::tex3D<T>(tex, index.x, i_y, i_z) * w1.x +
                         details::tex3D<T>(tex, index.x + 1.f, i_y, i_z) * w2.x +
                         details::tex3D<T>(tex, index.x + 2.f, i_y, i_z) * w3.x;
            }
            v[j] = tmp[0] * w0.y + tmp[1] * w1.y + tmp[2] * w2.y + tmp[3] * w3.y;
        }
        weights(fraction.z, &w0.x, &w1.x, &w2.x, &w3.x);
        return v[0] * w0.x + v[1] * w1.x + v[2] * w2.x + v[3] * w3.x;
    }
}

#endif // __CUDACC__
