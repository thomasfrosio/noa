#pragma once
#include "noa/gpu/cuda/IncludeGuard.cuh"

#include "noa/core/Config.hpp"
#include "noa/core/Enums.hpp"
#include "noa/gpu/cuda/Allocators.hpp"

namespace noa::cuda {
    /// Texture object used to interpolate data.
    /// This type is supported by the Interpolator(Spectrum) and the interpolate(_spectrum)_using_texture functions.
    template<size_t N,
             Interp INTERP_,
             Border BORDER_,
             typename Value,
             typename Coord,
             typename Index,
             bool NORMALIZED,
             bool LAYERED>
    class Texture {
    public:
        static constexpr Interp INTERP = INTERP_;
        static constexpr Border BORDER = BORDER_;
        static constexpr size_t SIZE = N;

        static_assert(N == 2 or N == 3);
        static_assert(nt::any_of<Value, f32, c32> and nt::any_of<Coord, f32, f64>);
        static_assert(INTERP.is_any(Interp::NEAREST_FAST, Interp::LINEAR_FAST));
        static_assert(BORDER == Border::MIRROR or
                      BORDER == Border::PERIODIC or
                      BORDER == Border::ZERO or
                      BORDER == Border::CLAMP);
        static_assert(not NORMALIZED or (BORDER == Border::MIRROR or BORDER == Border::PERIODIC));
        static_assert(not LAYERED or N == 2);

        using value_type = Value;
        using coord_type = Coord;
        using index_type = Index;
        using coord_n_type = Vec<coord_type, N>;
        using f_shape_type = std::conditional_t<NORMALIZED, coord_n_type, Empty>;
        using layer_type = std::conditional_t<LAYERED, i32, Empty>;
        using layer_flag_type = std::conditional_t<LAYERED, bool, Empty>;

    public:
        template<typename I>
        Texture(cudaTextureObject_t texture, const Shape<I, N + 1>& shape) :
            m_texture(texture)
        {
            if constexpr (NORMALIZED)
                m_norm = 1 / shape.vec.pop_front().template as<coord_type>();
            if constexpr (LAYERED)
                m_broadcast_layer = shape[0] == 1; // automatically broadcasts if one layer
            validate(texture);
        }

    public:
        template<nt::real T, size_t A>
        [[nodiscard]] NOA_HD auto fetch_preprocess(Vec<T, N, A> coordinates) const -> Vec<T, N, A> {
            coordinates += static_cast<T>(0.5); // to texture coordinate system
            if constexpr (NORMALIZED)
                coordinates *= m_norm.template as<T, A>();
            return coordinates;
        }

        template<nt::real... T> requires (sizeof...(T) == N)
        [[nodiscard]] NOA_HD auto fetch(T... coordinates) const noexcept -> value_type {
            return fetch_raw(fetch_preprocess(coord_n_type::from_values(coordinates...)));
        }

        template<nt::real T, size_t A>
        [[nodiscard]] NOA_HD auto fetch(const Vec<T, N, A>& coordinates) const noexcept -> value_type {
            return fetch_raw(fetch_preprocess(coordinates.template as<coord_type>()));
        }

        template<nt::real... T> requires (sizeof...(T) == N)
        [[nodiscard]] NOA_HD auto fetch_raw(T... coordinates) const noexcept -> value_type {
            return fetch_raw(coord_n_type::from_values(coordinates...));
        }

        template<nt::real T, size_t A>
        [[nodiscard]] NOA_HD auto fetch_raw(const Vec<T, N, A>& coordinates) const noexcept -> value_type {
            auto vec = coordinates.template as<coord_type>();
            #ifdef __CUDACC__
            if constexpr (N == 2) {
                if constexpr (std::same_as<value_type, f32>) {
                    if constexpr (LAYERED) {
                        return ::tex2DLayered<f32>(m_texture, vec[1], vec[0], m_layer);
                    } else {
                        return ::tex2D<f32>(m_texture, vec[1], vec[0]);
                    }
                } else if constexpr (std::same_as<value_type, c32>) {
                    float2 tmp;
                    if constexpr (LAYERED)
                        tmp = ::tex2DLayered<float2>(m_texture, vec[1], vec[0], m_layer);
                    else
                        tmp = ::tex2D<float2>(m_texture, vec[1], vec[0]);
                    return {tmp.x, tmp.y};
                } else {
                    static_assert(nt::always_false<value_type>);
                }
            } else if constexpr (N == 3) {
                if constexpr (std::same_as<value_type, f32>) {
                    return ::tex3D<f32>(m_texture, vec[2], vec[1], vec[0]);
                } else if constexpr (std::same_as<value_type, c32>) {
                    auto tmp = ::tex3D<float2>(m_texture, vec[2], vec[1], vec[0]);
                    return {tmp.x, tmp.y};
                } else {
                    static_assert(nt::always_false<value_type>);
                }
            } else {
                static_assert(nt::always_false<value_type>);
            }
            #else
            (void) coordinates;
            return value_type{};
            #endif
        }

    public:
        NOA_HD void set_layer(nt::integer auto layer) noexcept {
            if constexpr (LAYERED)
                m_layer = m_broadcast_layer ? 0 : static_cast<i32>(layer);
        }

        [[nodiscard]] NOA_HD auto operator[](nt::integer auto layer) const noexcept -> Texture {
            Texture new_texture = *this;
            new_texture.set_layer(layer);
            return new_texture;
        }

    public:
        template<nt::integer... I> requires (N == sizeof...(I))
        NOA_HD auto operator()(I... indices) const noexcept -> value_type {
            return fetch(static_cast<coord_type>(indices)...);
        }

        template<nt::integer... I> requires (N == sizeof...(I))
        NOA_HD auto operator()(nt::integer auto batch, I... indices) const noexcept -> value_type {
            return (*this)[batch](indices...);
        }

        template<nt::integer I, size_t S, size_t A> requires (N == S)
        NOA_HD auto operator()(const Vec<I, S, A>& indices) const noexcept -> value_type {
            return fetch(indices.template as<coord_type>());
        }

        template<nt::integer I, size_t S, size_t A> requires (N + 1 == S)
        NOA_HD auto operator()(const Vec<I, S, A>& indices) const noexcept -> value_type {
            return (*this)[indices[0]](indices.pop_front());
        }

        /// Checks that the texture object matches the Texture
        static void validate(cudaTextureObject_t texture) {
            cudaArray* array = AllocatorTexture::texture_array(texture);
            const bool is_layered = AllocatorTexture::is_layered(array);
            check(is_layered == LAYERED, "The input texture object is not layered, but a layered Texture was created");

            const cudaTextureDesc description = AllocatorTexture::texture_description(texture);
            constexpr bool IS_ADDRESSABLE = BORDER.is_any(Border::ZERO, Border::CLAMP, Border::MIRROR, Border::PERIODIC);
            if constexpr (IS_ADDRESSABLE and INTERP == Interp::LINEAR_FAST) {
                check(description.filterMode == cudaFilterModeLinear,
                      "The input texture object is not using linear lookups, "
                      "which does not match the Texture settings: INTERP={}", INTERP);
            } else if constexpr (not IS_ADDRESSABLE or INTERP == Interp::NEAREST_FAST) {
                check(description.filterMode == cudaFilterModePoint,
                      "The input texture object is not using mode-point lookups, "
                      "which does not match the Texture settings: INTERP={}", INTERP);
            } else {
                static_assert(nt::always_false<value_type>);
            }

            check(description.normalizedCoords == NORMALIZED,
                  "The input texture object is not using normalized coordinates, "
                  "which does not match the Texture settings: BORDER={}", BORDER);
        }

    private:
        cudaTextureObject_t m_texture{}; // size_t
        NOA_NO_UNIQUE_ADDRESS f_shape_type m_norm{};
        NOA_NO_UNIQUE_ADDRESS layer_type m_layer{};
        NOA_NO_UNIQUE_ADDRESS layer_flag_type m_broadcast_layer{};
    };
}
