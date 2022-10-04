#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

// Forward declarations
namespace noa::cuda::geometry::details {
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

namespace noa::cuda::geometry {
    // - The texture is expected to have the correct filter and addressing mode, as well as the correct coordinate
    // mode (normalized or unnormalized). See PtrTexture::description() for more details.
    // - An overload of `cudaCreateChannelDesc<>(::noa::cfloat_t)` is added by "noa/gpu/cuda/Types.h", so
    // if this file is included before the "texture" (or the underlying CUDA array) creation, or if it was
    // created by PtrTexture<> (or PtrArray<>), the channel descriptor will be correctly set for cfloat_t.

    // 1D interpolation of the data in "texture" at the texture coordinate "x", using ::tex1D.
    // T:       float or cfloat_t.
    // MODE:    Interpolation method to use. Any of InterpMode.
    // texture: Valid CUDA texture object. The channel descriptor should be float2 if \p T is cfloat_t.
    // x:       Width coordinate.
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

    // 2D interpolation of the data in "texture" at the texture coordinate "x", using ::tex1D.
    // T:       float or cfloat_t.
    // MODE:    Interpolation method to use. Any of InterpMode.
    // texture: Valid CUDA texture object. The channel descriptor should be float2 if \p T is cfloat_t.
    // y,x:     Height and width coordinates.
    template<typename T, InterpMode MODE>
    NOA_FD T tex2D(cudaTextureObject_t texture, float y, float x) {
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

    // 2D interpolation of the data in "texture" at the texture coordinates, using ::tex2D.
    template<typename T, InterpMode MODE>
    NOA_FD T tex2D(cudaTextureObject_t texture, float2_t coordinates) {
        return tex2D<T, MODE>(texture, coordinates[0], coordinates[1]);
    }

    // 3D interpolation of the data in "texture" at the texture coordinate "x", using ::tex1D.
    // T:       float or cfloat_t.
    // MODE:    Interpolation method to use. Any of InterpMode.
    // texture: Valid CUDA texture object. The channel descriptor should be float2 if \p T is cfloat_t.
    // z,y,x:   Depth, height and width coordinates.
    template<typename T, InterpMode MODE>
    NOA_FD T tex3D(cudaTextureObject_t texture, float z, float y, float x) {
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

    // 3D interpolation of the data in "texture" at the texture coordinates, using ::tex3D.
    template<typename T, InterpMode MODE>
    NOA_FD T tex3D(cudaTextureObject_t texture, float3_t coordinates) {
        return tex3D<T, MODE>(texture, coordinates[0], coordinates[1], coordinates[2]);
    }
}

// -- Texture interpolation implementation -- //
// These are device only functions and should only be
// compiled if the compilation is steered by nvcc.

#ifdef __CUDACC__
namespace noa::cuda::geometry::details {
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

    template<typename T>
    NOA_FD T linear3D(T v000, T v001, T v010, T v011,
                      T v100, T v101, T v110, T v111,
                      float rx, float ry, float rz) noexcept {
        T tmp1 = linear2D(v000, v001, v010, v011, rx, ry);
        T tmp2 = linear2D(v100, v101, v110, v111, rx, ry);
        return linear1D(tmp1, tmp2, rz);
    }
}

namespace noa::cuda::geometry::details::linear {
    // Slow but precise 1D linear interpolation using
    // 2 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex1DAccurate(cudaTextureObject_t tex, float x) {
        x -= 0.5f;
        float index = noa::math::floor(x);
        float fraction = x - index;
        index += 0.5f;
        return linear1D(details::tex1D<T>(tex, index), details::tex1D<T>(tex, index + 1.f), fraction);
    }

    // Slow but precise 2D linear interpolation using
    // 4 nearest neighbour lookups and unnormalized coordinates.
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

    // Slow but precise 3D linear interpolation using
    // 8 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex3DAccurate(cudaTextureObject_t tex, float x, float y, float z) {
        x -= 0.5f;
        y -= 0.5f;
        z -= 0.5f;
        float3 index{noa::math::floor(x),
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

namespace noa::cuda::geometry::details::cosine {
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
        float3 index{noa::math::floor(x), noa::math::floor(y), noa::math::floor(z)};
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

namespace noa::cuda::geometry::details::cubic {
    template<typename T>
    NOA_DEVICE T cubic1D(T v0, T v1, T v2, T v3, float r) {
        const T a0 = v3 - v2 - v0 + v1;
        const T a1 = v0 - v1 - a0;
        const T a2 = v2 - v0;
        // a3 = v1
        const float r2 = r * r;
        return a0 * r2 * r + a1 * r2 + a2 * r + v1;
    }

    // Slow but precise 1D cubic interpolation using
    // 4 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex1DAccurate(cudaTextureObject_t tex, float x) {
        x -= 0.5f;
        float index = noa::math::floor(x);
        const float fraction = x - index;
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
        for (int32_t i = 0; i < 4; ++i) {
            const float i_y = index[1] + static_cast<float>(i - 1);
            v[i] = cubic1D(details::tex2D<T>(tex, index[0] - 1.f, i_y),
                           details::tex2D<T>(tex, index[0], i_y),
                           details::tex2D<T>(tex, index[0] + 1.f, i_y),
                           details::tex2D<T>(tex, index[0] + 2.f, i_y),
                           fraction[0]);
        }
        return cubic1D(v[0], v[1], v[2], v[3], fraction[1]);
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
        for (int32_t j = 0; j < 4; ++j) {
            const float i_z = index[2] + static_cast<float>(j - 1);
            #pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
                const float i_y = index[1] + static_cast<float>(i - 1);
                tmp[i] = cubic1D(details::tex3D<T>(tex, index[0] - 1.f, i_y, i_z),
                                 details::tex3D<T>(tex, index[0], i_y, i_z),
                                 details::tex3D<T>(tex, index[0] + 1.f, i_y, i_z),
                                 details::tex3D<T>(tex, index[0] + 2.f, i_y, i_z),
                                 fraction[0]);
            }
            v[j] = cubic1D(tmp[0], tmp[1], tmp[2], tmp[3], fraction[1]);
        }
        return cubic1D(v[0], v[1], v[2], v[3], fraction[2]);
    }
}

// This is from https://github.com/DannyRuijters/CubicInterpolationCUDA
// See licences/CubicInterpolationCUDA.txt
// This is very much like "cuda/samples/3_Imaging/bicubicTexture/bicubicTexture_kernel.cuh"
namespace noa::cuda::geometry::details::bspline {
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
        const T tex0 = details::tex1D<T>(tex, h0);
        const T tex1 = details::tex1D<T>(tex, h1);

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
        T tex00 = details::tex2D<T>(tex, h0[0], h0[1]);
        T tex10 = details::tex2D<T>(tex, h1[0], h0[1]);
        T tex01 = details::tex2D<T>(tex, h0[0], h1[1]);
        T tex11 = details::tex2D<T>(tex, h1[0], h1[1]);

        // weight along the y-direction
        tex00 = g0[1] * tex00 + g1[1] * tex01;
        tex10 = g0[1] * tex10 + g1[1] * tex11;

        // weight along the x-direction
        return g0[0] * tex00 + g1[0] * tex10;
    }

    // 3D bicubic interpolated texture lookup, using unnormalized coordinates.
    // Fast implementation, using 8 linear lookups.
    template<typename T>
    NOA_DEVICE T tex3D(cudaTextureObject_t tex, float x, float y, float z) {
        const float3_t coord_grid{x - 0.5f, y - 0.5f, z - 0.5f};
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
        T tex000 = details::tex3D<T>(tex, h0[0], h0[1], h0[2]);
        T tex100 = details::tex3D<T>(tex, h1[0], h0[1], h0[2]);
        tex000 = g0[0] * tex000 + g1[0] * tex100; // weight along the x-direction
        T tex010 = details::tex3D<T>(tex, h0[0], h1[1], h0[2]);
        T tex110 = details::tex3D<T>(tex, h1[0], h1[1], h0[2]);
        tex010 = g0[0] * tex010 + g1[0] * tex110; // weight along the x-direction
        tex000 = g0[1] * tex000 + g1[1] * tex010; // weight along the y-direction
        T tex001 = details::tex3D<T>(tex, h0[0], h0[1], h1[2]);
        T tex101 = details::tex3D<T>(tex, h1[0], h0[1], h1[2]);
        tex001 = g0[0] * tex001 + g1[0] * tex101; // weight along the x-direction
        T tex011 = details::tex3D<T>(tex, h0[0], h1[1], h1[2]);
        T tex111 = details::tex3D<T>(tex, h1[0], h1[1], h1[2]);
        tex011 = g0[0] * tex011 + g1[0] * tex111; // weight along the x-direction
        tex001 = g0[1] * tex001 + g1[1] * tex011; // weight along the y-direction

        return g0[2] * tex000 + g1[2] * tex001; // weight along the z-direction
    }

    // Slow but precise 1D cubic B-spline interpolation using
    // 4 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex1DAccurate(cudaTextureObject_t tex, float x) {
        x -= 0.5f;
        float idx = noa::math::floor(x);
        const float f = x - idx;
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
        const float2_t coord_grid{x - 0.5f, y - 0.5f};
        float2_t index(noa::math::floor(coord_grid));
        const float2_t fraction(coord_grid - index);
        index += 0.5f;
        float w0, w1, w2, w3;
        weights(fraction[0], &w0, &w1, &w2, &w3);

        T v[4];
        #pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            const float i_y = index[1] + static_cast<float>(i - 1);
            v[i] = details::tex2D<T>(tex, index[0] - 1.f, i_y) * w0 +
                   details::tex2D<T>(tex, index[0], i_y) * w1 +
                   details::tex2D<T>(tex, index[0] + 1.f, i_y) * w2 +
                   details::tex2D<T>(tex, index[0] + 2.f, i_y) * w3;
        }
        weights(fraction[1], &w0, &w1, &w2, &w3);
        return v[0] * w0 + v[1] * w1 + v[2] * w2 + v[3] * w3;
    }

    // Slow but precise 3D cubic B-spline interpolation using
    // 64 nearest neighbour lookups and unnormalized coordinates.
    template<typename T>
    NOA_DEVICE T tex3DAccurate(cudaTextureObject_t tex, float x, float y, float z) {
        const float3_t coord_grid{x - 0.5f, y - 0.5f, z - 0.5f};
        float3_t index(noa::math::floor(coord_grid));
        const float3_t fraction(coord_grid - index);
        index += 0.5f;
        float2_t w0, w1, w2, w3; // compute only the x and y weights for now, leave z weights for later
        weights(float2_t{fraction[0], fraction[1]}, &w0, &w1, &w2, &w3);

        T v[4];
        T tmp[4];
        #pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            const float i_z = index[2] + static_cast<float>(j - 1);
            #pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
                const float i_y = index[1] + static_cast<float>(i - 1);
                tmp[i] = details::tex3D<T>(tex, index[0] - 1.f, i_y, i_z) * w0[0] +
                         details::tex3D<T>(tex, index[0], i_y, i_z) * w1[0] +
                         details::tex3D<T>(tex, index[0] + 1.f, i_y, i_z) * w2[0] +
                         details::tex3D<T>(tex, index[0] + 2.f, i_y, i_z) * w3[0];
            }
            v[j] = tmp[0] * w0[1] + tmp[1] * w1[1] + tmp[2] * w2[1] + tmp[3] * w3[1];
        }
        weights(fraction[2], &w0[0], &w1[0], &w2[0], &w3[0]);
        return v[0] * w0[0] + v[1] * w1[0] + v[2] * w2[0] + v[3] * w3[0];
    }
}

#endif // __CUDACC__
