#pragma once

#include "noa/common/Types.h"
#include "noa/common/geometry/Interpolate.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"

namespace noa::cuda::geometry::details {
    template<typename Data, InterpMode INTERP_MODE, bool NORMALIZED, bool LAYERED>
    constexpr void validateTexture(cudaTextureObject_t texture) {
        cudaArray* array = cuda::memory::PtrTexture::array(texture);
        const bool is_layered = cuda::memory::PtrArray<Data>::isLayered(array);
        NOA_CHECK(is_layered == LAYERED, "The input texture is not layered, but a layered interpolator was created");

        const cudaTextureDesc description = cuda::memory::PtrTexture::description(texture);
        if constexpr (INTERP_MODE == INTERP_NEAREST ||
                      INTERP_MODE == INTERP_LINEAR ||
                      INTERP_MODE == INTERP_COSINE ||
                      INTERP_MODE == INTERP_CUBIC ||
                      INTERP_MODE == INTERP_CUBIC_BSPLINE) {
            NOA_CHECK(description.filterMode == cudaFilterModePoint,
                      "The input texture is not using mode-point lookups, which is required for {}", INTERP_MODE);
        } else if constexpr (INTERP_MODE == INTERP_LINEAR_FAST ||
                             INTERP_MODE == INTERP_COSINE_FAST ||
                             INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST) {
            NOA_CHECK(description.filterMode == cudaFilterModeLinear,
                      "The input texture is not using linear lookups, which is required for {}", INTERP_MODE);
        }

        NOA_CHECK(NORMALIZED == description.normalizedCoords,
                  "The input texture is not using normalized, which doesn't match the interpolator type");
    }
}

namespace noa::cuda::geometry {
    template<InterpMode INTERP_MODE, typename Data,
             bool NORMALIZED = false, bool LAYERED = false, typename Coord = float>
    class Interpolator2D {
    public:
        static_assert(traits::is_any_v<Data, float, cfloat_t> && !std::is_const_v<Data>);
        static_assert(traits::is_float_v<Coord> && !std::is_const_v<Coord>);
        static_assert(!NORMALIZED || (INTERP_MODE == INTERP_NEAREST || INTERP_MODE == INTERP_LINEAR_FAST));

        using data_type = Data;
        using real_type = traits::value_type_t<data_type>;
        using real2_type = Float2<real_type>;
        using coord_type = Coord;
        using coord2_type = Float2<coord_type>;
        using f_shape_type = std::conditional_t<NORMALIZED, coord2_type, empty_t>;

    public:
        constexpr Interpolator2D() = default;

        template<typename Void = void, typename = std::enable_if_t<!NORMALIZED && std::is_same_v<Void, void>>>
        constexpr explicit Interpolator2D(cudaTextureObject_t texture)
                : m_texture(texture) {
            details::validateTexture<data_type, INTERP_MODE, NORMALIZED, LAYERED>(m_texture);
        }

        template<typename Void = void, typename = std::enable_if_t<NORMALIZED && std::is_same_v<Void, void>>>
        constexpr Interpolator2D(cudaTextureObject_t texture, f_shape_type shape)
                : m_texture(texture), m_shape(shape) {
            details::validateTexture<data_type, INTERP_MODE, NORMALIZED, LAYERED>(m_texture);
        }

        template<typename Int, typename = std::enable_if_t<traits::is_int_v<Int>>>
        constexpr NOA_HD data_type operator()(coord2_type coordinate, Int layer = Int{0}) const noexcept {
            if constexpr (!LAYERED) {
                if constexpr (INTERP_MODE == INTERP_NEAREST) {
                    return nearest_(coordinate, 0);
                } else if constexpr (INTERP_MODE == INTERP_LINEAR) {
                    return linearAccurate_(coordinate, 0);
                } else if constexpr (INTERP_MODE == INTERP_COSINE) {
                    return cosineAccurate_(coordinate, 0);
                } else if constexpr (INTERP_MODE == INTERP_CUBIC) {
                    return cubicAccurate_(coordinate, 0);
                } else if constexpr (INTERP_MODE == INTERP_CUBIC_BSPLINE) {
                    return cubicBSplineAccurate_(coordinate, 0);
                } else if constexpr (INTERP_MODE == INTERP_LINEAR_FAST) {
                    return linearFast_(coordinate, 0);
                } else if constexpr (INTERP_MODE == INTERP_COSINE_FAST) {
                    return cosineFast_(coordinate, 0);
                } else if constexpr (INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST) {
                    return cubicBSplineFast_(coordinate, 0);
                } else {
                    static_assert(traits::always_false_v<data_type>);
                }
            } else {
                if constexpr (INTERP_MODE == INTERP_NEAREST) {
                    return nearest_(coordinate, static_cast<int32_t>(layer));
                } else if constexpr (INTERP_MODE == INTERP_LINEAR) {
                    return linearAccurate_(coordinate, static_cast<int32_t>(layer));
                } else if constexpr (INTERP_MODE == INTERP_COSINE) {
                    return cosineAccurate_(coordinate, static_cast<int32_t>(layer));
                } else if constexpr (INTERP_MODE == INTERP_CUBIC) {
                    return cubicAccurate_(coordinate, static_cast<int32_t>(layer));
                } else if constexpr (INTERP_MODE == INTERP_CUBIC_BSPLINE) {
                    return cubicBSplineAccurate_(coordinate, static_cast<int32_t>(layer));
                } else if constexpr (INTERP_MODE == INTERP_LINEAR_FAST) {
                    return linearFast_(coordinate, static_cast<int32_t>(layer));
                } else if constexpr (INTERP_MODE == INTERP_COSINE_FAST) {
                    return cosineFast_(coordinate, static_cast<int32_t>(layer));
                } else if constexpr (INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST) {
                    return cubicBSplineFast_(coordinate, static_cast<int32_t>(layer));
                } else {
                    static_assert(traits::always_false_v<data_type>);
                }
            }
            return data_type{};
        }

    private:
        NOA_FD data_type fetch2D_(coord_type x, coord_type y, int32_t layer) const noexcept {
            #ifdef __CUDACC__
            if constexpr (std::is_same_v<data_type, float>) {
                if constexpr (LAYERED) {
                    return ::tex2DLayered<float>(m_texture, static_cast<float>(x), static_cast<float>(y), layer);
                } else {
                    (void) layer;
                    return ::tex2D<float>(m_texture, static_cast<float>(x), static_cast<float>(y));
                }
            } else if constexpr (std::is_same_v<data_type, cfloat_t>) {
                if constexpr (LAYERED) {
                    auto tmp = ::tex2DLayered<float2>(m_texture, static_cast<float>(x), static_cast<float>(y), layer);
                    return {tmp.x, tmp.y};
                } else {
                    (void) layer;
                    auto tmp = ::tex2D<float2>(m_texture, static_cast<float>(x), static_cast<float>(y));
                    return {tmp.x, tmp.y};
                }
            } else {
                static_assert(traits::always_false_v<data_type>);
            }
            #else
            (void) x;
            (void) y;
            (void) index;
            return {};
            #endif
        }

        constexpr NOA_HD data_type nearest_(coord2_type coordinate, int32_t layer) const noexcept {
            coordinate += coord_type{0.5};
            if constexpr (NORMALIZED)
                coordinate /= m_shape;
            return fetch2D_(coordinate[1], coordinate[0], layer);
        }

        constexpr NOA_HD data_type linearFast_(coord2_type coordinate, int32_t layer) const noexcept {
            return nearest_(coordinate, layer);
        }

        // Slow but precise 2D linear interpolation using
        // 4 nearest neighbour lookups and unnormalized coordinates.
        constexpr NOA_HD data_type linearAccurate_(coord2_type coordinate, int32_t layer) const noexcept {
            static_assert(!NORMALIZED);
            coord2_type index = ::noa::math::floor(coordinate);
            coord2_type fraction = coordinate - index;
            index += coord_type{0.5};

            if constexpr (INTERP_MODE == INTERP_COSINE) {
                constexpr coord_type PI = ::noa::math::Constants<coord_type>::PI;
                fraction = (coord_type{1} - ::noa::math::cos(fraction * PI)) / coord_type{2};
            }

            const data_type t00 = fetch2D_(index[1], index[0], layer);
            const data_type t01 = fetch2D_(index[1] + 1, index[0], layer);
            const data_type v0 = ::noa::geometry::interpolate::lerp1D(t00, t01, fraction[1]);

            const data_type t10 = fetch2D_(index[1], index[0] + 1, layer);
            const data_type t11 = fetch2D_(index[1] + 1, index[0] + 1, layer);
            const data_type v1 = ::noa::geometry::interpolate::lerp1D(t10, t11, fraction[1]);

            return ::noa::geometry::interpolate::lerp1D(v0, v1, fraction[0]);
        }

        // Fast 2D cosine interpolation using 1 linear lookup and unnormalized coordinates.
        constexpr NOA_HD data_type cosineFast_(coord2_type coordinate, int32_t layer) const noexcept {
            static_assert(!NORMALIZED);
            coord2_type index = ::noa::math::floor(coordinate);
            coord2_type fraction = coordinate - index;

            constexpr coord_type PI = ::noa::math::Constants<coord_type>::PI;
            fraction = (coord_type{1} - ::noa::math::cos(fraction * PI)) / coord_type{2};

            index += fraction + coord_type{0.5};
            return fetch2D_(index[1], index[0], layer);
        }

        // Slow but precise 2D cosine interpolation using
        // 4 nearest neighbour lookups and unnormalized coordinates.
        constexpr NOA_HD data_type cosineAccurate_(coord2_type coordinate, int32_t layer) const noexcept {
            return linearAccurate_(coordinate, layer);
        }

        // Slow but precise 2D cubic interpolation using
        // 16 nearest neighbour lookups and unnormalized coordinates.
        constexpr NOA_HD data_type cubicAccurate_(coord2_type coordinate, int32_t layer) const noexcept {
            static_assert(!NORMALIZED);
            coord2_type index = ::noa::math::floor(coordinate);
            coord2_type fraction = coordinate - index;
            index += coord_type{0.5};

            data_type v[4];
            #pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
                const coord_type index_y = index[0] + static_cast<coord_type>(i - 1);
                const data_type t0 = fetch2D_(index[1] - 1, index_y, layer);
                const data_type t1 = fetch2D_(index[1] + 0, index_y, layer);
                const data_type t2 = fetch2D_(index[1] + 1, index_y, layer);
                const data_type t3 = fetch2D_(index[1] + 2, index_y, layer);
                v[i] = ::noa::geometry::interpolate::cubic1D(t0, t1, t2, t3, fraction[1]);
            }
            return ::noa::geometry::interpolate::cubic1D(v[0], v[1], v[2], v[3], fraction[0]);
        }

        // 2D bicubic interpolated texture lookup, using unnormalized coordinates.
        // Fast implementation, using 4 linear lookups.
        constexpr NOA_HD data_type cubicBSplineFast_(coord2_type coordinate, int32_t layer) const noexcept {
            static_assert(!NORMALIZED);
            coord2_type index = ::noa::math::floor(coordinate);
            coord2_type fraction = coordinate - index;

            real2_type w0, w1, w2, w3;
            ::noa::geometry::interpolate::details::bsplineWeights(fraction, &w0, &w1, &w2, &w3);

            const real2_type g0 = w0 + w1;
            const real2_type g1 = w2 + w3;
            const real2_type h0 = w1 / g0 - real_type{0.5} + real2_type(index);
            const real2_type h1 = w3 / g1 + real_type{1.5} + real2_type(index);

            // Fetch the four linear interpolations.
            const data_type v00 = fetch2D_(h0[1], h0[0], layer);
            const data_type v10 = fetch2D_(h1[1], h0[0], layer);
            const data_type v01 = fetch2D_(h0[1], h1[0], layer);
            const data_type v11 = fetch2D_(h1[1], h1[0], layer);

            // Weight along the y-direction.
            const data_type v0 = g0[0] * v00 + g1[0] * v01;
            const data_type v1 = g0[0] * v10 + g1[0] * v11;

            // Weight along the x-direction.
            return g0[1] * v0 + g1[1] * v1;
        }

        // Slow but precise 2D cubic B-spline interpolation using
        // 16 nearest neighbour lookups and unnormalized coordinates.
        constexpr NOA_HD data_type cubicBSplineAccurate_(coord2_type coordinate, int32_t layer) const noexcept {
            static_assert(!NORMALIZED);
            coord2_type index = ::noa::math::floor(coordinate);
            coord2_type fraction = coordinate - index;
            index += coord_type{0.5};

            real_type w0, w1, w2, w3;
            ::noa::geometry::interpolate::details::bsplineWeights(fraction[1], &w0, &w1, &w2, &w3);

            data_type v[4];
            #pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
                const coord_type index_y = index[0] + static_cast<coord_type>(i - 1);
                v[i] = fetch2D_(index[1] - 1, index_y, layer) * w0 +
                       fetch2D_(index[1] + 0, index_y, layer) * w1 +
                       fetch2D_(index[1] + 1, index_y, layer) * w2 +
                       fetch2D_(index[1] + 2, index_y, layer) * w3;
            }

            ::noa::geometry::interpolate::details::bsplineWeights(fraction[0], &w0, &w1, &w2, &w3);
            return v[0] * w0 + v[1] * w1 + v[2] * w2 + v[3] * w3;
        }

    private:
        cudaTextureObject_t m_texture{};
        f_shape_type m_shape{};
    };
}

namespace noa::cuda::geometry {
    template<InterpMode INTERP_MODE, typename Data,
             bool NORMALIZED = false, typename Coord = float>
    class Interpolator3D {
    public:
        static_assert(traits::is_any_v<Data, float, cfloat_t> && !std::is_const_v<Data>);
        static_assert(traits::is_float_v<Coord> && !std::is_const_v<Coord>);
        static_assert(!NORMALIZED || (INTERP_MODE == INTERP_NEAREST || INTERP_MODE == INTERP_LINEAR_FAST));

        using data_type = Data;
        using coord_type = Coord;
        using real_type = traits::value_type_t<data_type>;
        using real2_type = Float2<real_type>;
        using real3_type = Float3<real_type>;
        using coord2_type = Float2<coord_type>;
        using coord3_type = Float3<coord_type>;
        using f_shape_type = std::conditional_t<NORMALIZED, coord3_type, empty_t>;

    public:
        constexpr Interpolator3D() = default;

        template<typename Void = void, typename = std::enable_if_t<!NORMALIZED && std::is_same_v<Void, void>>>
        constexpr explicit Interpolator3D(cudaTextureObject_t texture)
                : m_texture(texture) {
            details::validateTexture<data_type, INTERP_MODE, NORMALIZED, false>(m_texture);
        }

        template<typename Void = void, typename = std::enable_if_t<NORMALIZED && std::is_same_v<Void, void>>>
        constexpr Interpolator3D(cudaTextureObject_t texture, f_shape_type shape)
                : m_texture(texture), m_shape(shape) {
            details::validateTexture<data_type, INTERP_MODE, NORMALIZED, false>(m_texture);
        }

        template<typename Int = int32_t>
        constexpr NOA_HD data_type operator()(coord3_type coordinate, Int = Int{0}) const noexcept {
            if constexpr (INTERP_MODE == INTERP_NEAREST) {
                return nearest_(coordinate);
            } else if constexpr (INTERP_MODE == INTERP_LINEAR) {
                return linearAccurate_(coordinate);
            } else if constexpr (INTERP_MODE == INTERP_COSINE) {
                return cosineAccurate_(coordinate);
            } else if constexpr (INTERP_MODE == INTERP_CUBIC) {
                return cubicAccurate_(coordinate);
            } else if constexpr (INTERP_MODE == INTERP_CUBIC_BSPLINE) {
                return cubicBSplineAccurate_(coordinate);
            } else if constexpr (INTERP_MODE == INTERP_LINEAR_FAST) {
                return linearFast_(coordinate);
            } else if constexpr (INTERP_MODE == INTERP_COSINE_FAST) {
                return cosineFast_(coordinate);
            } else if constexpr (INTERP_MODE == INTERP_CUBIC_BSPLINE_FAST) {
                return cubicBSplineFast_(coordinate);
            } else {
                static_assert(traits::always_false_v<data_type>);
            }
            return data_type{};
        }

    private:
        NOA_FD data_type fetch3D_(coord_type x, coord_type y, coord_type z) const noexcept {
            #ifdef __CUDACC__
            if constexpr (std::is_same_v<data_type, float>) {
                return ::tex3D<float>(m_texture,
                                      static_cast<float>(x),
                                      static_cast<float>(y),
                                      static_cast<float>(z));
            } else if constexpr (std::is_same_v<data_type, cfloat_t>) {
                auto tmp = ::tex3D<float2>(m_texture,
                                           static_cast<float>(x),
                                           static_cast<float>(y),
                                           static_cast<float>(z));
                return {tmp.x, tmp.y};
            } else {
                static_assert(traits::always_false_v<data_type>);
            }
            #else
            (void) x;
            (void) y;
            (void) z;
            return {};
            #endif
        }

        constexpr NOA_HD data_type nearest_(coord3_type coordinate) const noexcept {
            coordinate += coord_type{0.5};
            if constexpr (NORMALIZED)
                coordinate /= m_shape;
            return fetch3D_(coordinate[2], coordinate[1], coordinate[0]);
        }

        constexpr NOA_HD data_type linearFast_(coord3_type coordinate) const noexcept {
            return nearest_(coordinate);
        }

        // Slow but precise 3D linear interpolation using
        // 8 nearest neighbour lookups and unnormalized coordinates.
        constexpr NOA_HD data_type linearAccurate_(coord3_type coordinate) const noexcept {
            static_assert(!NORMALIZED);
            coord3_type index = ::noa::math::floor(coordinate);
            coord3_type fraction = coordinate - index;
            index += coord_type{0.5};

            if constexpr (INTERP_MODE == INTERP_COSINE) {
                constexpr coord_type PI = ::noa::math::Constants<coord_type>::PI;
                fraction = (coord_type{1} - ::noa::math::cos(fraction * PI)) / coord_type{2};
            }

            const data_type v000 = fetch3D_(index[2] + 0, index[1] + 0, index[0]);
            const data_type v001 = fetch3D_(index[2] + 1, index[1] + 0, index[0]);
            const data_type v010 = fetch3D_(index[2] + 0, index[1] + 1, index[0]);
            const data_type v011 = fetch3D_(index[2] + 1, index[1] + 1, index[0]);
            const data_type v0 = ::noa::geometry::interpolate::lerp2D(v000, v001, v010, v011, fraction[2], fraction[1]);

            const data_type v100 = fetch3D_(index[2] + 0, index[1] + 0, index[0] + 1);
            const data_type v101 = fetch3D_(index[2] + 1, index[1] + 0, index[0] + 1);
            const data_type v110 = fetch3D_(index[2] + 0, index[1] + 1, index[0] + 1);
            const data_type v111 = fetch3D_(index[2] + 1, index[1] + 1, index[0] + 1);
            const data_type v1 = ::noa::geometry::interpolate::lerp2D(v100, v101, v110, v111, fraction[2], fraction[1]);

            return ::noa::geometry::interpolate::lerp1D(v0, v1, fraction[0]);
        }

        // Fast 3D cosine interpolation using 1 linear lookup and unnormalized coordinates.
        constexpr NOA_HD data_type cosineFast_(coord3_type coordinate) const noexcept {
            static_assert(!NORMALIZED);
            coord3_type index = ::noa::math::floor(coordinate);
            coord3_type fraction = coordinate - index;

            constexpr coord_type PI = ::noa::math::Constants<coord_type>::PI;
            fraction = (coord_type{1} - ::noa::math::cos(fraction * PI)) / coord_type{2};

            index += fraction + coord_type{0.5};
            return fetch3D_(index[2], index[1], index[0]);
        }

        // Slow but precise 3D cosine interpolation using
        // 8 nearest neighbour lookups and unnormalized coordinates.
        constexpr NOA_HD data_type cosineAccurate_(coord3_type coordinate) const noexcept {
            return linearAccurate_(coordinate);
        }

        // Slow but precise 3D cubic interpolation using
        // 64 nearest neighbour lookups and unnormalized coordinates.
        constexpr NOA_HD data_type cubicAccurate_(coord3_type coordinate) const noexcept {
            static_assert(!NORMALIZED);
            coord3_type index = ::noa::math::floor(coordinate);
            coord3_type fraction = coordinate - index;
            index += coord_type{0.5};

            data_type vz[4];
            data_type vy[4];
            #pragma unroll
            for (int32_t y = 0; y < 4; ++y) {
                const coord_type offset_z = index[0] + static_cast<coord_type>(y - 1);
                #pragma unroll
                for (int32_t x = 0; x < 4; ++x) {
                    const coord_type offset_y = index[1] + static_cast<coord_type>(x - 1);
                    const data_type v0 = fetch3D_(index[2] - 1, offset_y, offset_z);
                    const data_type v1 = fetch3D_(index[2] + 0, offset_y, offset_z);
                    const data_type v2 = fetch3D_(index[2] + 1, offset_y, offset_z);
                    const data_type v3 = fetch3D_(index[2] + 2, offset_y, offset_z);
                    vy[x] = ::noa::geometry::interpolate::cubic1D(v0, v1, v2, v3, fraction[2]);
                }
                vz[y] = ::noa::geometry::interpolate::cubic1D(vy[0], vy[1], vy[2], vy[3], fraction[1]);
            }
            return ::noa::geometry::interpolate::cubic1D(vz[0], vz[1], vz[2], vz[3], fraction[0]);
        }

        // 3D bicubic interpolated texture lookup, using unnormalized coordinates.
        // Fast implementation, using 8 linear lookups.
        constexpr NOA_HD data_type cubicBSplineFast_(coord3_type coordinate) const noexcept {
            static_assert(!NORMALIZED);
            coord3_type index = ::noa::math::floor(coordinate);
            coord3_type fraction = coordinate - index;

            real3_type w0, w1, w2, w3;
            ::noa::geometry::interpolate::details::bsplineWeights(fraction, &w0, &w1, &w2, &w3);

            const real3_type g0 = w0 + w1;
            const real3_type g1 = w2 + w3;
            const real3_type h0 = w1 / g0 - real_type{0.5} + real3_type(index);
            const real3_type h1 = w3 / g1 + real_type{1.5} + real3_type(index);

            // Fetch the eight linear interpolations.
            const data_type v000 = fetch3D_(h0[2], h0[1], h0[0]);
            const data_type v001 = fetch3D_(h1[2], h0[1], h0[0]);
            const data_type x00 = g0[2] * v000 + g1[2] * v001;
            const data_type v010 = fetch3D_(h0[2], h1[1], h0[0]);
            const data_type v011 = fetch3D_(h1[2], h1[1], h0[0]);
            const data_type x01 = g0[2] * v010 + g1[2] * v011;
            const data_type y0 = g0[1] * x00 + g1[1] * x01;

            const data_type v100 = fetch3D_(h0[2], h0[1], h1[0]);
            const data_type v101 = fetch3D_(h1[2], h0[1], h1[0]);
            const data_type x10 = g0[2] * v100 + g1[2] * v101;
            const data_type v110 = fetch3D_(h0[2], h1[1], h1[0]);
            const data_type v111 = fetch3D_(h1[2], h1[1], h1[0]);
            const data_type x11 = g0[2] * v110 + g1[2] * v111;
            const data_type y1 = g0[1] * x10 + g1[1] * x11;

            return g0[0] * y0 + g1[0] * y1;
        }

        // Slow but precise 3D cubic B-spline interpolation using
        // 64 nearest neighbour lookups and unnormalized coordinates.
        constexpr NOA_HD data_type cubicBSplineAccurate_(coord3_type coordinate) const noexcept {
            static_assert(!NORMALIZED);
            coord3_type index = ::noa::math::floor(coordinate);
            coord3_type fraction = coordinate - index;
            index += coord_type{0.5};

            real2_type w00, w01, w02, w03;
            ::noa::geometry::interpolate::details::bsplineWeights(
                    coord2_type(fraction.get(1)), &w00, &w01, &w02, &w03);

            data_type vz[4];
            data_type vy[4];
            #pragma unroll
            for (int32_t z = 0; z < 4; ++z) {
                const coord_type offset_z = index[0] + static_cast<coord_type>(z - 1);
                #pragma unroll
                for (int32_t y = 0; y < 4; ++y) {
                    const coord_type offset_y = index[1] + static_cast<coord_type>(y - 1);
                    vy[y] = fetch3D_(index[2] - 1, offset_y, offset_z) * w00[1] +
                            fetch3D_(index[2] + 0, offset_y, offset_z) * w01[1] +
                            fetch3D_(index[2] + 1, offset_y, offset_z) * w02[1] +
                            fetch3D_(index[2] + 2, offset_y, offset_z) * w03[1];
                }
                vz[z] = vy[0] * w00[0] +
                        vy[1] * w01[0] +
                        vy[2] * w02[0] +
                        vy[3] * w03[0];
            }

            real_type w0, w1, w2, w3;
            ::noa::geometry::interpolate::details::bsplineWeights(
                    fraction[0], &w0, &w1, &w2, &w3);
            return vz[0] * w0 +
                   vz[1] * w1 +
                   vz[2] * w2 +
                   vz[3] * w3;
        }

    private:
        cudaTextureObject_t m_texture{};
        f_shape_type m_shape{};
    };
}
