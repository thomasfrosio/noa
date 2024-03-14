#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/geometry/Interpolate.hpp"
#include "noa/gpu/cuda/AllocatorTexture.hpp"

namespace noa::cuda::geometry::guts {
    template<typename Value, Interp InterpMode, bool IsNormalized, bool IsLayered>
    constexpr void validate_texture(cudaTextureObject_t texture) {
        cudaArray* array = noa::cuda::AllocatorTexture<Value>::array(texture);
        const bool is_layered = noa::cuda::AllocatorTexture<Value>::is_layered(array);
        check(is_layered == IsLayered, "The input texture is not layered, but a layered interpolator was created");

        const cudaTextureDesc description = noa::cuda::AllocatorTexture<Value>::description(texture);
        if constexpr (InterpMode == Interp::NEAREST ||
                      InterpMode == Interp::LINEAR ||
                      InterpMode == Interp::COSINE ||
                      InterpMode == Interp::CUBIC ||
                      InterpMode == Interp::CUBIC_BSPLINE) {
            check(description.filterMode == cudaFilterModePoint,
                  "The input texture is not using mode-point lookups, which is required for {}", InterpMode);
        } else if constexpr (InterpMode == Interp::LINEAR_FAST ||
                             InterpMode == Interp::COSINE_FAST ||
                             InterpMode == Interp::CUBIC_BSPLINE_FAST) {
            check(description.filterMode == cudaFilterModeLinear,
                  "The input texture is not using linear lookups, which is required for {}", InterpMode);
        }

        check(IsNormalized == description.normalizedCoords,
              "The input texture is not using normalized, which doesn't match the interpolator type");
    }
}

namespace noa::cuda::geometry {
    template<Interp InterpMode, typename Value,
             bool IsNormalized = false,
             bool IsLayered = false,
             typename Coord = f32>
    class Interpolator2d {
    public:
        static_assert(nt::is_any_v<Value, f32, c32> and nt::is_any_v<Coord, f32, f64>);
        static_assert(not IsNormalized or (InterpMode == Interp::NEAREST or InterpMode == Interp::LINEAR_FAST));

        using value_type = Value;
        using mutable_value_type = Value;
        using real_type = nt::value_type_t<value_type>;
        using real2_type = Vec2<real_type>;
        using coord_type = Coord;
        using coord2_type = Vec2<coord_type>;
        using f_shape_type = std::conditional_t<IsNormalized, coord2_type, Empty>;

    public:
        constexpr Interpolator2d() = default;

        constexpr explicit Interpolator2d(cudaTextureObject_t texture) requires (not IsNormalized)
                : m_texture(texture) {
            guts::validate_texture<value_type, InterpMode, IsNormalized, IsLayered>(m_texture);
        }

        constexpr Interpolator2d(cudaTextureObject_t texture, f_shape_type shape) requires IsNormalized
                : m_texture(texture), m_shape(shape) {
            guts::validate_texture<value_type, InterpMode, IsNormalized, IsLayered>(m_texture);
        }

        template<typename Int> requires nt::is_int_v<Int>
        NOA_HD constexpr value_type operator()(coord2_type coordinate, Int layer = Int{0}) const noexcept {
            if constexpr (not IsLayered) {
                if constexpr (InterpMode == Interp::NEAREST) {
                    return nearest_(coordinate, 0);
                } else if constexpr (InterpMode == Interp::LINEAR) {
                    return linear_accurate_(coordinate, 0);
                } else if constexpr (InterpMode == Interp::COSINE) {
                    return cosine_accurate_(coordinate, 0);
                } else if constexpr (InterpMode == Interp::CUBIC) {
                    return cubic_accurate_(coordinate, 0);
                } else if constexpr (InterpMode == Interp::CUBIC_BSPLINE) {
                    return cubic_bspline_accurate_(coordinate, 0);
                } else if constexpr (InterpMode == Interp::LINEAR_FAST) {
                    return linear_fast_(coordinate, 0);
                } else if constexpr (InterpMode == Interp::COSINE_FAST) {
                    return cosine_fast_(coordinate, 0);
                } else if constexpr (InterpMode == Interp::CUBIC_BSPLINE_FAST) {
                    return cubic_bspline_fast_(coordinate, 0);
                } else {
                    static_assert(nt::always_false_v<value_type>);
                }
            } else {
                if constexpr (InterpMode == Interp::NEAREST) {
                    return nearest_(coordinate, static_cast<i32>(layer));
                } else if constexpr (InterpMode == Interp::LINEAR) {
                    return linear_accurate_(coordinate, static_cast<i32>(layer));
                } else if constexpr (InterpMode == Interp::COSINE) {
                    return cosine_accurate_(coordinate, static_cast<i32>(layer));
                } else if constexpr (InterpMode == Interp::CUBIC) {
                    return cubic_accurate_(coordinate, static_cast<i32>(layer));
                } else if constexpr (InterpMode == Interp::CUBIC_BSPLINE) {
                    return cubic_bspline_accurate_(coordinate, static_cast<i32>(layer));
                } else if constexpr (InterpMode == Interp::LINEAR_FAST) {
                    return linear_fast_(coordinate, static_cast<i32>(layer));
                } else if constexpr (InterpMode == Interp::COSINE_FAST) {
                    return cosine_fast_(coordinate, static_cast<i32>(layer));
                } else if constexpr (InterpMode == Interp::CUBIC_BSPLINE_FAST) {
                    return cubic_bspline_fast_(coordinate, static_cast<i32>(layer));
                } else {
                    static_assert(nt::always_false_v<value_type>);
                }
            }
            return value_type{};
        }

        template<typename Int = i32> requires nt::is_int_v<Int>
        NOA_FHD constexpr value_type at(Int batch, Int y, Int x) const noexcept {
            return (*this)(coord2_type{y, x}, batch);
        }

        template<typename Int = i32> requires (not IsLayered and nt::is_int_v<Int>)
        NOA_FHD constexpr value_type at(Int y, Int x) const noexcept {
            return (*this)(coord2_type{y, x});
        }

    private:
        NOA_FD value_type fetch_2d_(coord_type x, coord_type y, i32 layer) const noexcept {
            #ifdef __CUDACC__
            if constexpr (std::is_same_v<value_type, f32>) {
                if constexpr (IsLayered) {
                    return ::tex2DLayered<f32>(m_texture, static_cast<f32>(x), static_cast<f32>(y), layer);
                } else {
                    (void) layer;
                    return ::tex2D<f32>(m_texture, static_cast<f32>(x), static_cast<f32>(y));
                }
            } else if constexpr (std::is_same_v<value_type, c32>) {
                if constexpr (IsLayered) {
                    auto tmp = ::tex2DLayered<float2>(m_texture, static_cast<f32>(x), static_cast<f32>(y), layer);
                    return {tmp.x, tmp.y};
                } else {
                    (void) layer;
                    auto tmp = ::tex2D<float2>(m_texture, static_cast<f32>(x), static_cast<f32>(y));
                    return {tmp.x, tmp.y};
                }
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
            #else
            (void) x;
            (void) y;
            (void) layer;
            return {};
            #endif
        }

        NOA_HD constexpr value_type nearest_(coord2_type coordinate, i32 layer) const noexcept {
            coordinate += coord_type{0.5};
            if constexpr (IsNormalized)
                coordinate /= m_shape;
            return fetch_2d_(coordinate[1], coordinate[0], layer);
        }

        NOA_HD constexpr value_type linear_fast_(coord2_type coordinate, i32 layer) const noexcept {
            return nearest_(coordinate, layer);
        }

        // Slow but precise 2D linear interpolation using
        // 4 nearest neighbour lookups and unnormalized coordinates.
        NOA_HD constexpr value_type linear_accurate_(coord2_type coordinate, i32 layer) const noexcept {
            static_assert(not IsNormalized);
            coord2_type index = floor(coordinate);
            coord2_type fraction = coordinate - index;
            index += coord_type{0.5};

            if constexpr (InterpMode == Interp::COSINE) {
                constexpr coord_type PI = noa::Constant<coord_type>::PI;
                fraction = (coord_type{1} - cos(fraction * PI)) / coord_type{2};
            }

            const value_type t00 = fetch_2d_(index[1], index[0], layer);
            const value_type t01 = fetch_2d_(index[1] + 1, index[0], layer);
            const value_type v0 = noa::geometry::lerp_1d(t00, t01, fraction[1]);

            const value_type t10 = fetch_2d_(index[1], index[0] + 1, layer);
            const value_type t11 = fetch_2d_(index[1] + 1, index[0] + 1, layer);
            const value_type v1 = noa::geometry::lerp_1d(t10, t11, fraction[1]);

            return noa::geometry::lerp_1d(v0, v1, fraction[0]);
        }

        // Fast 2D cosine interpolation using 1 linear lookup and unnormalized coordinates.
        NOA_HD constexpr value_type cosine_fast_(coord2_type coordinate, i32 layer) const noexcept {
            static_assert(not IsNormalized);
            coord2_type index = floor(coordinate);
            coord2_type fraction = coordinate - index;

            constexpr coord_type PI = noa::Constant<coord_type>::PI;
            fraction = (coord_type{1} - cos(fraction * PI)) / coord_type{2};

            index += fraction + coord_type{0.5};
            return fetch_2d_(index[1], index[0], layer);
        }

        // Slow but precise 2D cosine interpolation using
        // 4 nearest neighbour lookups and unnormalized coordinates.
        NOA_HD constexpr value_type cosine_accurate_(coord2_type coordinate, i32 layer) const noexcept {
            return linear_accurate_(coordinate, layer);
        }

        // Slow but precise 2D cubic interpolation using
        // 16 nearest neighbour lookups and unnormalized coordinates.
        NOA_HD constexpr value_type cubic_accurate_(coord2_type coordinate, i32 layer) const noexcept {
            static_assert(not IsNormalized);
            coord2_type index = floor(coordinate);
            coord2_type fraction = coordinate - index;
            index += coord_type{0.5};

            value_type v[4];
            #pragma unroll
            for (i32 i = 0; i < 4; ++i) {
                const coord_type index_y = index[0] + static_cast<coord_type>(i - 1);
                const value_type t0 = fetch_2d_(index[1] - 1, index_y, layer);
                const value_type t1 = fetch_2d_(index[1] + 0, index_y, layer);
                const value_type t2 = fetch_2d_(index[1] + 1, index_y, layer);
                const value_type t3 = fetch_2d_(index[1] + 2, index_y, layer);
                v[i] = noa::geometry::interpolate_cubic_1d(t0, t1, t2, t3, fraction[1]);
            }
            return noa::geometry::interpolate_cubic_1d(v[0], v[1], v[2], v[3], fraction[0]);
        }

        // 2D bicubic interpolated texture lookup, using unnormalized coordinates.
        // Fast implementation, using 4 linear lookups.
        NOA_HD constexpr value_type cubic_bspline_fast_(coord2_type coordinate, i32 layer) const noexcept {
            static_assert(not IsNormalized);
            coord2_type index = floor(coordinate);
            coord2_type fraction = coordinate - index;

            real2_type w0, w1, w2, w3;
            noa::geometry::guts::bspline_weights(fraction, &w0, &w1, &w2, &w3);

            const real2_type g0 = w0 + w1;
            const real2_type g1 = w2 + w3;
            const real2_type h0 = w1 / g0 - real_type{0.5} + real2_type(index);
            const real2_type h1 = w3 / g1 + real_type{1.5} + real2_type(index);

            // Fetch the four linear interpolations.
            const value_type v00 = fetch_2d_(h0[1], h0[0], layer);
            const value_type v10 = fetch_2d_(h1[1], h0[0], layer);
            const value_type v01 = fetch_2d_(h0[1], h1[0], layer);
            const value_type v11 = fetch_2d_(h1[1], h1[0], layer);

            // Weight along the y-direction.
            const value_type v0 = g0[0] * v00 + g1[0] * v01;
            const value_type v1 = g0[0] * v10 + g1[0] * v11;

            // Weight along the x-direction.
            return g0[1] * v0 + g1[1] * v1;
        }

        // Slow but precise 2D cubic B-spline interpolation using
        // 16 nearest neighbour lookups and unnormalized coordinates.
        NOA_HD constexpr value_type cubic_bspline_accurate_(coord2_type coordinate, i32 layer) const noexcept {
            static_assert(not IsNormalized);
            coord2_type index = floor(coordinate);
            coord2_type fraction = coordinate - index;
            index += coord_type{0.5};

            real_type w0, w1, w2, w3;
            noa::geometry::guts::bspline_weights(fraction[1], &w0, &w1, &w2, &w3);

            value_type v[4];
            #pragma unroll
            for (i32 i = 0; i < 4; ++i) {
                const coord_type index_y = index[0] + static_cast<coord_type>(i - 1);
                v[i] = fetch_2d_(index[1] - 1, index_y, layer) * w0 +
                       fetch_2d_(index[1] + 0, index_y, layer) * w1 +
                       fetch_2d_(index[1] + 1, index_y, layer) * w2 +
                       fetch_2d_(index[1] + 2, index_y, layer) * w3;
            }

            noa::geometry::guts::bspline_weights(fraction[0], &w0, &w1, &w2, &w3);
            return v[0] * w0 + v[1] * w1 + v[2] * w2 + v[3] * w3;
        }

    private:
        cudaTextureObject_t m_texture{};
        f_shape_type m_shape{};
    };
}

namespace noa::cuda::geometry {
    template<Interp InterpMode, typename Value,
             bool IsNormalized = false,
             typename Coord = f32>
    class Interpolator3d {
    public:
        static_assert(nt::is_any_v<Value, f32, c32> && !std::is_const_v<Value>);
        static_assert(nt::is_real_v<Coord> && !std::is_const_v<Coord>);
        static_assert(not IsNormalized || (InterpMode == Interp::NEAREST || InterpMode == Interp::LINEAR_FAST));

        using value_type = Value;
        using mutable_value_type = Value;
        using coord_type = Coord;
        using real_type = nt::value_type_t<value_type>;
        using real2_type = Vec2<real_type>;
        using real3_type = Vec3<real_type>;
        using coord3_type = Vec3<coord_type>;
        using f_shape_type = std::conditional_t<IsNormalized, coord3_type, Empty>;

    public:
        constexpr Interpolator3d() = default;

        constexpr explicit Interpolator3d(cudaTextureObject_t texture) requires (not IsNormalized)
                : m_texture(texture) {
            guts::validate_texture<value_type, InterpMode, IsNormalized, false>(m_texture);
        }

        constexpr Interpolator3d(cudaTextureObject_t texture, f_shape_type shape) requires IsNormalized
                : m_texture(texture), m_shape(shape) {
            guts::validate_texture<value_type, InterpMode, IsNormalized, false>(m_texture);
        }

        template<typename Int = i32>
        NOA_HD constexpr value_type operator()(coord3_type coordinate, Int = Int{0}) const noexcept {
            if constexpr (InterpMode == Interp::NEAREST) {
                return nearest_(coordinate);
            } else if constexpr (InterpMode == Interp::LINEAR) {
                return linear_accurate_(coordinate);
            } else if constexpr (InterpMode == Interp::COSINE) {
                return cosine_accurate_(coordinate);
            } else if constexpr (InterpMode == Interp::CUBIC) {
                return cubic_accurate_(coordinate);
            } else if constexpr (InterpMode == Interp::CUBIC_BSPLINE) {
                return cubic_bspline_accurate_(coordinate);
            } else if constexpr (InterpMode == Interp::LINEAR_FAST) {
                return linear_fast_(coordinate);
            } else if constexpr (InterpMode == Interp::COSINE_FAST) {
                return cosine_fast_(coordinate);
            } else if constexpr (InterpMode == Interp::CUBIC_BSPLINE_FAST) {
                return cubic_bspline_fast_(coordinate);
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
            return value_type{};
        }

        template<typename Int = i32> requires nt::is_int_v<Int>
        NOA_FHD constexpr value_type at(Int, Int z, Int y, Int x) const noexcept {
            return (*this)(coord3_type{z, y, x});
        }

        template<typename Int = i32> requires nt::is_int_v<Int>
        NOA_FHD constexpr value_type at(Int z, Int y, Int x) const noexcept {
            return (*this)(coord3_type{z, y, x});
        }

    private:
        NOA_FD value_type fetch_3d_(coord_type x, coord_type y, coord_type z) const noexcept {
            #ifdef __CUDACC__
            if constexpr (std::is_same_v<value_type, f32>) {
                return ::tex3D<f32>(m_texture,
                                    static_cast<f32>(x),
                                    static_cast<f32>(y),
                                    static_cast<f32>(z));
            } else if constexpr (std::is_same_v<value_type, c32>) {
                auto tmp = ::tex3D<float2>(m_texture,
                                           static_cast<f32>(x),
                                           static_cast<f32>(y),
                                           static_cast<f32>(z));
                return {tmp.x, tmp.y};
            } else {
                static_assert(nt::always_false_v<value_type>);
            }
            #else
            (void) x;
            (void) y;
            (void) z;
            return {};
            #endif
        }

        NOA_HD constexpr value_type nearest_(coord3_type coordinate) const noexcept {
            coordinate += coord_type{0.5};
            if constexpr (IsNormalized)
                coordinate /= m_shape;
            return fetch_3d_(coordinate[2], coordinate[1], coordinate[0]);
        }

        NOA_HD constexpr value_type linear_fast_(coord3_type coordinate) const noexcept {
            return nearest_(coordinate);
        }

        // Slow but precise 3D linear interpolation using
        // 8 nearest neighbour lookups and unnormalized coordinates.
        NOA_HD constexpr value_type linear_accurate_(coord3_type coordinate) const noexcept {
            static_assert(not IsNormalized);
            coord3_type index = floor(coordinate);
            coord3_type fraction = coordinate - index;
            index += coord_type{0.5};

            if constexpr (InterpMode == Interp::COSINE) {
                constexpr coord_type PI = noa::Constant<coord_type>::PI;
                fraction = (coord_type{1} - cos(fraction * PI)) / coord_type{2};
            }

            const value_type v000 = fetch_3d_(index[2] + 0, index[1] + 0, index[0]);
            const value_type v001 = fetch_3d_(index[2] + 1, index[1] + 0, index[0]);
            const value_type v010 = fetch_3d_(index[2] + 0, index[1] + 1, index[0]);
            const value_type v011 = fetch_3d_(index[2] + 1, index[1] + 1, index[0]);
            const value_type v0 = noa::geometry::lerp_2d(v000, v001, v010, v011, fraction[2], fraction[1]);

            const value_type v100 = fetch_3d_(index[2] + 0, index[1] + 0, index[0] + 1);
            const value_type v101 = fetch_3d_(index[2] + 1, index[1] + 0, index[0] + 1);
            const value_type v110 = fetch_3d_(index[2] + 0, index[1] + 1, index[0] + 1);
            const value_type v111 = fetch_3d_(index[2] + 1, index[1] + 1, index[0] + 1);
            const value_type v1 = noa::geometry::lerp_2d(v100, v101, v110, v111, fraction[2], fraction[1]);

            return noa::geometry::lerp_1d(v0, v1, fraction[0]);
        }

        // Fast 3D cosine interpolation using 1 linear lookup and unnormalized coordinates.
        NOA_HD constexpr value_type cosine_fast_(coord3_type coordinate) const noexcept {
            static_assert(not IsNormalized);
            coord3_type index = floor(coordinate);
            coord3_type fraction = coordinate - index;

            constexpr coord_type PI = noa::Constant<coord_type>::PI;
            fraction = (coord_type{1} - cos(fraction * PI)) / coord_type{2};

            index += fraction + coord_type{0.5};
            return fetch_3d_(index[2], index[1], index[0]);
        }

        // Slow but precise 3D cosine interpolation using
        // 8 nearest neighbour lookups and unnormalized coordinates.
        NOA_HD constexpr value_type cosine_accurate_(coord3_type coordinate) const noexcept {
            return linear_accurate_(coordinate);
        }

        // Slow but precise 3D cubic interpolation using
        // 64 nearest neighbour lookups and unnormalized coordinates.
        NOA_HD constexpr value_type cubic_accurate_(coord3_type coordinate) const noexcept {
            static_assert(not IsNormalized);
            coord3_type index = floor(coordinate);
            coord3_type fraction = coordinate - index;
            index += coord_type{0.5};

            value_type vz[4];
            value_type vy[4];
            #pragma unroll
            for (i32 y = 0; y < 4; ++y) {
                const coord_type offset_z = index[0] + static_cast<coord_type>(y - 1);
                #pragma unroll
                for (i32 x = 0; x < 4; ++x) {
                    const coord_type offset_y = index[1] + static_cast<coord_type>(x - 1);
                    const value_type v0 = fetch_3d_(index[2] - 1, offset_y, offset_z);
                    const value_type v1 = fetch_3d_(index[2] + 0, offset_y, offset_z);
                    const value_type v2 = fetch_3d_(index[2] + 1, offset_y, offset_z);
                    const value_type v3 = fetch_3d_(index[2] + 2, offset_y, offset_z);
                    vy[x] = noa::geometry::interpolate_cubic_1d(v0, v1, v2, v3, fraction[2]);
                }
                vz[y] = noa::geometry::interpolate_cubic_1d(vy[0], vy[1], vy[2], vy[3], fraction[1]);
            }
            return noa::geometry::interpolate_cubic_1d(vz[0], vz[1], vz[2], vz[3], fraction[0]);
        }

        // 3D bicubic interpolated texture lookup, using unnormalized coordinates.
        // Fast implementation, using 8 linear lookups.
        NOA_HD constexpr value_type cubic_bspline_fast_(coord3_type coordinate) const noexcept {
            static_assert(not IsNormalized);
            coord3_type index = floor(coordinate);
            coord3_type fraction = coordinate - index;

            real3_type w0, w1, w2, w3;
            noa::geometry::guts::bspline_weights(fraction, &w0, &w1, &w2, &w3);

            const real3_type g0 = w0 + w1;
            const real3_type g1 = w2 + w3;
            const real3_type h0 = w1 / g0 - real_type{0.5} + real3_type(index);
            const real3_type h1 = w3 / g1 + real_type{1.5} + real3_type(index);

            // Fetch the eight linear interpolations.
            const value_type v000 = fetch_3d_(h0[2], h0[1], h0[0]);
            const value_type v001 = fetch_3d_(h1[2], h0[1], h0[0]);
            const value_type x00 = g0[2] * v000 + g1[2] * v001;
            const value_type v010 = fetch_3d_(h0[2], h1[1], h0[0]);
            const value_type v011 = fetch_3d_(h1[2], h1[1], h0[0]);
            const value_type x01 = g0[2] * v010 + g1[2] * v011;
            const value_type y0 = g0[1] * x00 + g1[1] * x01;

            const value_type v100 = fetch_3d_(h0[2], h0[1], h1[0]);
            const value_type v101 = fetch_3d_(h1[2], h0[1], h1[0]);
            const value_type x10 = g0[2] * v100 + g1[2] * v101;
            const value_type v110 = fetch_3d_(h0[2], h1[1], h1[0]);
            const value_type v111 = fetch_3d_(h1[2], h1[1], h1[0]);
            const value_type x11 = g0[2] * v110 + g1[2] * v111;
            const value_type y1 = g0[1] * x10 + g1[1] * x11;

            return g0[0] * y0 + g1[0] * y1;
        }

        // Slow but precise 3D cubic B-spline interpolation using
        // 64 nearest neighbour lookups and unnormalized coordinates.
        NOA_HD constexpr value_type cubic_bspline_accurate_(coord3_type coordinate) const noexcept {
            static_assert(not IsNormalized);
            coord3_type index = floor(coordinate);
            coord3_type fraction = coordinate - index;
            index += coord_type{0.5};

            real2_type w00, w01, w02, w03;
            noa::geometry::guts::bspline_weights(
                    fraction.pop_front(), &w00, &w01, &w02, &w03);

            value_type vz[4];
            value_type vy[4];
            #pragma unroll
            for (i32 z = 0; z < 4; ++z) {
                const coord_type offset_z = index[0] + static_cast<coord_type>(z - 1);
                #pragma unroll
                for (i32 y = 0; y < 4; ++y) {
                    const coord_type offset_y = index[1] + static_cast<coord_type>(y - 1);
                    vy[y] = fetch_3d_(index[2] - 1, offset_y, offset_z) * w00[1] +
                            fetch_3d_(index[2] + 0, offset_y, offset_z) * w01[1] +
                            fetch_3d_(index[2] + 1, offset_y, offset_z) * w02[1] +
                            fetch_3d_(index[2] + 2, offset_y, offset_z) * w03[1];
                }
                vz[z] = vy[0] * w00[0] +
                        vy[1] * w01[0] +
                        vy[2] * w02[0] +
                        vy[3] * w03[0];
            }

            real_type w0, w1, w2, w3;
            noa::geometry::guts::bspline_weights(
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

    template<size_t N, Interp InterpMode, typename Value, bool IsNormalized, bool IsLayered, typename Coord>
    using InterpolatorNd = std::conditional_t<
            N == 2, Interpolator2d<InterpMode, Value, IsNormalized, IsLayered, Coord>,
                    Interpolator3d<InterpMode, Value, IsNormalized, Coord>>;
}

namespace noa::traits {
    template<Interp InterpMode, typename Value, bool IsNormalized, bool IsLayered, typename Coord>
    struct proclaim_is_interpolator_nd<
            noa::cuda::geometry::Interpolator2d<InterpMode, Value, IsNormalized, IsLayered, Coord>, 2
    > : std::true_type {};

    template<Interp InterpMode, typename Value, bool IsNormalized, typename Coord>
    struct proclaim_is_interpolator_nd<
            noa::cuda::geometry::Interpolator3d<InterpMode, Value, IsNormalized, Coord>, 3
    > : std::true_type {};
}
