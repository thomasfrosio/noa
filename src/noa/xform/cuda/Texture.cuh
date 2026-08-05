#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"
#include "noa/runtime/cuda/Allocators.hpp"

namespace noa::xform::cuda {
    /// Accessor interface and texture fetching used to interpolate data.
    /// This type can be passed by the Interpolator(Spectrum) and the interpolate(_spectrum)_using_texture functions.
    template<Interp INTERP_, Border BORDER_,
             typename Value, typename Coord, typename Index,
             usize B, usize R,
             bool NORMALIZED, bool LAYERED>
    class AccessorTexture {
    public:
        static constexpr Interp INTERP = INTERP_;
        static constexpr Border BORDER = BORDER_;
        static constexpr size_t SIZE = B + R;

        static_assert(R == 2 or R == 3);
        static_assert((R == 2 and LAYERED) or (R == 3 and not LAYERED));
        static_assert(nt::almost_any_of<Value, f32, c32> and nt::any_of<Coord, f32, f64>);
        static_assert(INTERP.is_any(Interp::NEAREST_FAST, Interp::LINEAR_FAST));
        static_assert(BORDER == Border::MIRROR or
                      BORDER == Border::PERIODIC or
                      BORDER == Border::ZERO or
                      BORDER == Border::CLAMP);
        static_assert(not NORMALIZED or (BORDER == Border::MIRROR or BORDER == Border::PERIODIC));

        using value_type = Value;
        using mutable_value_type = std::remove_const_t<value_type>;
        using coord_type = Coord;
        using index_type = Index;
        using coord_r_type = Vec<coord_type, R>;
        using norm_type = std::conditional_t<NORMALIZED, coord_r_type, Empty>;
        using layer_type = std::conditional_t<LAYERED, i32, Empty>;
        using strides_type = Strides<index_type, B>;

    private:
        // Friend all instantiations.
        template<Interp, Border, typename, typename, typename, usize, usize, bool, bool>
        friend class AccessorTexture;

    public:
        AccessorTexture() = default;

        AccessorTexture(
            cudaTextureObject_t texture,
            const Shape<index_type, R>& shape_ranked,
            const Strides<index_type, B>& strides_batches,
            layer_type layer = 0
        ) :
            m_texture{texture}, m_layer{layer}
        {
            validate(texture);
            if constexpr (NORMALIZED)
                m_norm = 1 / shape_ranked.vec.template as<coord_type>();
            if constexpr (LAYERED)
                m_strides = strides_batches;
        }

    public:
        template<nt::real T, size_t A> requires (B == 0)
        [[nodiscard]] NOA_HD auto fetch_preprocess(Vec<T, R, A> coordinates) const -> Vec<T, R, A> {
            coordinates += static_cast<T>(0.5); // to texture coordinate system
            if constexpr (NORMALIZED)
                coordinates *= m_norm.template as<T, A>();
            return coordinates;
        }

        template<nt::real... T> requires (B == 0 and sizeof...(T) == R)
        [[nodiscard]] NOA_HD auto fetch(T... coordinates) const noexcept -> mutable_value_type {
            return fetch_raw(fetch_preprocess(coord_r_type::from_values(coordinates...)));
        }

        template<nt::real T, size_t A> requires (B == 0)
        [[nodiscard]] NOA_HD auto fetch(const Vec<T, R, A>& coordinates) const noexcept -> mutable_value_type {
            return fetch_raw(fetch_preprocess(coordinates.template as<coord_type>()));
        }

        template<nt::real... T> requires (B == 0 and sizeof...(T) == R)
        [[nodiscard]] NOA_HD auto fetch_raw(T... coordinates) const noexcept -> mutable_value_type {
            return fetch_raw(coord_r_type::from_values(coordinates...));
        }

        template<nt::real T, size_t A> requires (B == 0)
        [[nodiscard]] NOA_HD auto fetch_raw(const Vec<T, R, A>& coordinates) const noexcept -> mutable_value_type {
            auto vec = coordinates.template as<f32>();
            #ifdef __CUDACC__
            if constexpr (R == 2) {
                if constexpr (std::same_as<mutable_value_type, f32>) {
                    if constexpr (LAYERED) {
                        return ::tex2DLayered<f32>(m_texture, vec[1], vec[0], m_layer);
                    } else {
                        return ::tex2D<f32>(m_texture, vec[1], vec[0]);
                    }
                } else if constexpr (std::same_as<mutable_value_type, c32>) {
                    float2 tmp;
                    if constexpr (LAYERED)
                        tmp = ::tex2DLayered<float2>(m_texture, vec[1], vec[0], m_layer);
                    else
                        tmp = ::tex2D<float2>(m_texture, vec[1], vec[0]);
                    return {tmp.x, tmp.y};
                } else {
                    static_assert(nt::always_false<mutable_value_type>);
                }
            } else if constexpr (R == 3) {
                if constexpr (std::same_as<mutable_value_type, f32>) {
                    return ::tex3D<f32>(m_texture, vec[2], vec[1], vec[0]);
                } else if constexpr (std::same_as<mutable_value_type, c32>) {
                    auto tmp = ::tex3D<float2>(m_texture, vec[2], vec[1], vec[0]);
                    return {tmp.x, tmp.y};
                } else {
                    static_assert(nt::always_false<mutable_value_type>);
                }
            } else {
                static_assert(nt::always_false<mutable_value_type>);
            }
            #else
            (void) coordinates;
            return mutable_value_type{};
            #endif
        }

    public:
        // Indexing through the batch axes.
        template<nt::integer I, usize S, usize A> requires (B >= S)
        [[nodiscard]] NOA_HD auto operator[](const Vec<I, S, A>& batches) const noexcept
        -> AccessorTexture<INTERP, BORDER, Value, Coord, Index, B - S, R, NORMALIZED, LAYERED>
        {
            if constexpr (S == 0) {
                return *this;
            } else {
                AccessorTexture<INTERP, BORDER, Value, Coord, Index, B - S, R, NORMALIZED, LAYERED> output;
                output.m_texture = m_texture;
                output.m_norm = m_norm;
                output.m_layer = m_layer + static_cast<i32>(noa::offset_at(m_strides, batches));
                output.m_strides = m_strides.template pop_front<S>();
                return output;
            }
        }

        template<nt::integer I> requires (B >= 1)
        [[nodiscard]] NOA_HD auto operator[](I batch) const noexcept
        -> AccessorTexture<INTERP, BORDER, Value, Coord, Index, B - 1, R, NORMALIZED, LAYERED>
        {
            AccessorTexture<INTERP, BORDER, Value, Coord, Index, B - 1, R, NORMALIZED, LAYERED> output;
            output.m_texture = m_texture;
            output.m_norm = m_norm;
            output.m_layer = m_layer + static_cast<i32>(noa::offset_at(m_strides[0], batch));
            output.m_strides = m_strides.template pop_front<1>();
            return output;
        }

    public:
        template<nt::integer I, usize S, usize A> requires (SIZE == S)
        NOA_HD auto operator()(const Vec<I, S, A>& batched_indices) const noexcept -> mutable_value_type {
            const auto& [batches, indices] = batched_indices.template split<B>();
            return (*this)[batches].fetch(indices.template as<coord_type>());
        }

        template<nt::integer... I> requires (SIZE == sizeof...(I))
        NOA_HD auto operator()(I... indices) const noexcept -> mutable_value_type {
            return (*this)(Vec{indices...});
        }

        /// Checks that the texture object matches the AccessorTexture
        static void validate(cudaTextureObject_t texture) {
            cudaArray* array = noa::cuda::AllocatorArray::texture_array(texture);
            const bool is_layered = noa::cuda::AllocatorArray::is_layered(array);
            check(is_layered == LAYERED, "The input texture object is not layered, but a layered Texture was created");

            const cudaTextureDesc description = noa::cuda::AllocatorArray::texture_description(texture);
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
        cudaTextureObject_t m_texture{}; // usize
        NOA_NO_UNIQUE_ADDRESS norm_type m_norm{};
        NOA_NO_UNIQUE_ADDRESS layer_type m_layer{};
        NOA_NO_UNIQUE_ADDRESS strides_type m_strides{};
    };
}
