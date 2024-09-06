#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/utils/Misc.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/core/Types.hpp"
#include "noa/core/Interpolation.hpp"

// CUDA arrays:
//  - Data resides in global memory. The host can cudaMemcpy to it, and the device can only access it
//    through texture reads or surface reads and writes.
//  - They are usually associated with a type: each element can have 1, 2 or 4 components (e.g. complex types have
//    2 components). Elements are associated with a type (components have the same type), that may be signed or
//    unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats.
//  - They are either 1D, 2D or 3D. Note that an "empty" dimension is noted as 0 in the CUDA API, but AllocatorArray
//    is following our "shape" convention (i.e. "empty" dimensions are noted as 1). See shape2extent.
//  - They are cached-optimized for 2D/3D spatial locality.
//  - Surfaces and textures can be bound to same CUDA array.
//  - They are mostly used when the content changes rarely.
//    Although reusing them with cudaMemcpy is possible and surfaces can write to it.

// CUDA textures:
//  -   Address mode: How out of range coordinates are handled. This can be specified for each coordinates (although
//                    the current implementation specifies the same mode for all the dimensions. It is either wrap,
//                    mirror, border or clamp (default).
//                    Note: This is ignored for 1D textures since they don't support addressing modes.
//                    Note: mirror and wrap are only supported for normalized coordinates, otherwise, fallback to clamp.
//  -   Filter mode:  Filtering used when fetching. Either point (neighbour) or linear.
//                    Note: The linear mode is only allowed for float types.
//                    Note: This is ignored for 1D textures since they don't perform any interpolation.
//  -   Read mode:    Whether or not integer data should be converted to floating point when fetching. If signed,
//                    returns float within [-1., 1.]. If unsigned, returns float within [0., 1.].
//                    Note: This only applies to 8-bit and 16-bit integer formats. 32-bits are not promoted.
//  -   Normalized coordinates: Whether or not the coordinates are normalized when fetching.
//                              If false (default): textures are fetched using floating point coordinates in range
//                                                  [0, N-1], where N is the size of that particular dimension.
//                              If true:            textures are fetched using floating point coordinates in range
//                                                  [0., 1. -1/N], where N is the size of that particular dimension.

// Textures are bound to global memory, either through a device pointer or a CUDA array.
// -- Data in the bounded CUDA array can be updated but texture cache is unchanged until a new kernel is launched.
// -- The device pointer or a CUDA array should not be freed while the texture is being used.

// Add specialization for our complex types. Used for CUDA arrays and textures.
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::f16>() { return cudaCreateChannelDesc<half>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c16>() { return cudaCreateChannelDesc<half2>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c32>() { return cudaCreateChannelDesc<float2>(); }

namespace noa::cuda {
    template<size_t N,
             Interp INTERP_MODE,
             Border BORDER_MODE,
             typename Value,
             typename Coord,
             bool NORMALIZED,
             bool LAYERED>
    class Texture;

    /// Creates 1d, 2d or 3d texture objects bounded to a CUDA array.
    class AllocatorTexture {
    public:
        // Textures can map pitch memory (2d only) and linear memory (1d only), but we don't support these
        // use cases as they are either less performant or are very limited compared to a CUDA array.
        struct Resource {
            cudaArray_t array; // pointer
            cudaTextureObject_t texture; // size_t

            Resource() = default;
            ~Resource() {
                [[maybe_unused]] cudaError_t err;
                err = cudaDestroyTextureObject(texture);
                NOA_ASSERT(err == cudaSuccess);
                err = cudaFreeArray(array);
                NOA_ASSERT(err == cudaSuccess);
            }
        };
        using shared_type = std::shared_ptr<Resource>;

    public:
        /// Allocates a CUDA array and create a texture from that array.
        /// The returned array and texture are configured to work with the interpolation functions
        /// (see convert_to_description() for more details).
        template<nt::any_of<i8, i16, i32, u8, u16, u32, f16, f32, c16, c32> T>
        static auto allocate(
            const Shape4<i64>& shape,
            Interp interp_mode,
            Border border_mode,
            u32 flag = cudaArrayDefault
        ) -> std::shared_ptr<Resource> {
            auto resource = std::make_shared<Resource>();

            // Create the array.
            const cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
            const cudaExtent extent = shape2extent(shape, flag & cudaArrayLayered);
            check(cudaMalloc3DArray(&resource->array, &desc, extent, flag));

            // Create the texture.
            const auto [filter, address, normalized_coords] = convert_to_description(interp_mode, border_mode);
            resource->texture = create_texture(
                resource->array, filter, address, cudaReadModeElementType, normalized_coords);

            return resource;
        }

    public: // static array utilities
        static cudaExtent shape2extent(Shape4<i64> shape, bool is_layered) {
            // Special case: treat column vectors as row vectors.
            if (shape[2] >= 1 and shape[3] == 1)
                std::swap(shape[2], shape[3]);

            // Conversion:  shape -> CUDA extent
            // 3D:          1DHW  -> DHW
            // 2D:          11HW  -> 0HW
            // 1D:          111W  -> 00W
            // 2D layered:  B1HW  -> DHW
            // 1D layered:  B11W  -> D0W
            check(all(shape > 0) and shape[is_layered] == 1,
                  "The input shape cannot be converted to a CUDA array extent. "
                  "Dimensions with a size of 0 are not allowed, and the {} should be 1. Shape: {}",
                  is_layered ? "depth dimension (for layered arrays)" : "batch dimension", shape);

            auto shape_3d = shape.filter(static_cast<i64>(not is_layered), 2, 3).as_safe<size_t>();

            // Set empty dimensions to 0. If layered, leave extent.depth to the batch value.
            if (not is_layered)
                shape_3d[0] -= shape_3d[0] == 1;
            shape_3d[1] -= shape_3d[1] == 1;
            return {shape_3d[2], shape_3d[1], shape_3d[0]};
        }

        static Shape4<i64> extent2shape(cudaExtent extent, bool is_layered) noexcept {
            auto u_extent = Shape{extent.depth, extent.height, extent.width};
            u_extent += Shape3<size_t>::from_vec(u_extent == 0); // set empty dimensions to 1

            // Column vectors are "lost" in the conversion.
            // 1d extents are interpreted as row vectors.
            auto shape = Shape4<i64>::from_values(1, 1, u_extent[1], u_extent[2]);
            shape[not is_layered] = static_cast<i64>(u_extent[0]);
            return shape;
        }

        static auto array_info(cudaArray* array) {
            cudaChannelFormatDesc desc{};
            cudaExtent extent{};
            u32 flags{};
            check(cudaArrayGetInfo(&desc, &extent, &flags, array));
            return make_tuple(desc, extent, flags);
        }

        static bool is_layered(cudaArray* array) {
            const auto [desc_, extent_, flags] = array_info(array);
            // Not sure whether the flags are mutually exclusive, so just check the bit for layered textures.
            return flags & cudaArrayLayered;
        }

    public: // static texture utilities
        /// Creates a 1d, 2d or 3d texture from a CUDA array.
        /// \param array                        CUDA array. Its lifetime should exceed the lifetime of this new object.
        /// \param filter_mode                  Filter mode, either cudaFilterModePoint or cudaFilterModeLinear.
        /// \param address_mode                 Address mode, either cudaAddressModeWrap, cudaAddressModeClamp,
        ///                                     cudaAddressModeMirror or cudaAddressModeBorder.
        /// \param normalized_reads_to_float    Whether 8-, 16-integer data should be converted to float when fetching.
        ///                                     Either cudaReadModeElementType or cudaReadModeNormalizedFloat.
        /// \param normalized_coordinates       Whether the coordinates are normalized when fetching.
        /// \note cudaAddressModeMirror and cudaAddressModeWrap are only available with normalized coordinates.
        ///       If normalized_coordinates is false, border_mode is switched (internally by CUDA) to cudaAddressModeClamp.
        static cudaTextureObject_t create_texture(
            const cudaArray* array,
            cudaTextureFilterMode filter_mode,
            cudaTextureAddressMode address_mode,
            cudaTextureReadMode normalized_reads_to_float,
            bool normalized_coordinates
        ) {
            cudaResourceDesc res_desc{};
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = const_cast<cudaArray*>(array); // one example where we need const_cast...
            // TODO cudaArrayGetInfo can be used to extract the array type and make
            //      sure it matches T, but is it really useful? Maybe just an assert?

            cudaTextureDesc tex_desc{};
            tex_desc.addressMode[0] = address_mode;
            tex_desc.addressMode[1] = address_mode; // ignored if 1d array.
            tex_desc.addressMode[2] = address_mode; // ignored if 1d or 2d array.
            tex_desc.filterMode = filter_mode;
            tex_desc.readMode = normalized_reads_to_float;
            tex_desc.normalizedCoords = normalized_coordinates;

            cudaTextureObject_t texture{};
            if (cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr))
                panic("Creating the texture object from a CUDA array failed");
            return texture;
        }

        /// Sets the underlying texture filter and coordinate mode according to "interp_mode" and "border_mode".
        /// The interpolation functions in noa expect the texture to be set as follows:
        /// - 1) Border::MIRROR and Border::PERIODIC requires normalized coordinates.
        /// - 2) The accurate modes use nearest-lookups, while the fast methods use linear lookups.
        /// - 3) InterpMode::NEAREST and InterpMode::LINEAR_FAST are the only modes supporting normalized coordinates,
        ///      thus they are the only modes supporting Border::MIRROR and Border::PERIODIC.
        static auto convert_to_description(
            Interp interp_mode,
            Border border_mode
        ) -> Tuple<cudaTextureFilterMode, cudaTextureAddressMode, bool> {
            // The accurate modes use nearest-lookups, while the fast methods use linear lookups.
            cudaTextureFilterMode filter_mode =
                interp_mode.is_fast() and interp_mode != Interp::NEAREST_FAST ?
                cudaFilterModeLinear : cudaFilterModePoint;

            cudaTextureAddressMode address_mode;
            bool normalized_coordinates{false};
            switch (border_mode) {
                case Border::PERIODIC: {
                    address_mode = cudaAddressModeWrap;
                    normalized_coordinates = true;
                    break;
                }
                case Border::MIRROR: {
                    address_mode = cudaAddressModeMirror;
                    normalized_coordinates = true;
                    break;
                }
                case Border::CLAMP: {
                    address_mode = cudaAddressModeClamp;
                    break;
                }
                case Border::VALUE: // not natively supported, fallback to ZERO
                case Border::REFLECT: // not natively supported, fallback to ZERO
                case Border::NOTHING: // not natively supported, fallback to ZERO
                case Border::ZERO: {
                    address_mode = cudaAddressModeBorder;
                    break;
                }
            }
            return make_tuple(filter_mode, address_mode, normalized_coordinates);
        }

        /// Utility to get the correct Texture type given the interpolation inputs.
        template<size_t N, Interp INTERP, Border BORDER, typename Value, typename Coord = f32>
        struct texture {
            static constexpr bool LAYERED = N == 2;
            static constexpr bool NORMALIZED = BORDER == Border::MIRROR or BORDER == Border::PERIODIC;

            static constexpr Interp INTERP_TEX =
                INTERP.is_fast() and INTERP != Interp::NEAREST_FAST ?
                Interp::LINEAR : Interp::NEAREST;

            static constexpr Border BORDER_TEX =
                BORDER == Border::VALUE or BORDER == Border::REFLECT or BORDER == Border::NOTHING ?
                Border::ZERO : BORDER;

            using type = Texture<N, INTERP_TEX, BORDER_TEX, Value, Coord, NORMALIZED, LAYERED>;
        };

        /// Returns a texture object's texture descriptor.
        static cudaTextureDesc texture_description(cudaTextureObject_t texture) {
            cudaTextureDesc tex_desc{};
            check(cudaGetTextureObjectTextureDesc(&tex_desc, texture));
            return tex_desc;
        }

        /// Returns a texture object's texture descriptor.
        static cudaResourceDesc texture_resource(cudaTextureObject_t texture) {
            cudaResourceDesc tex_desc{};
            check(cudaGetTextureObjectResourceDesc(&tex_desc, texture));
            return tex_desc;
        }

        static cudaArray* texture_array(cudaTextureObject_t texture) {
            const auto array_resource = texture_resource(texture);
            check(array_resource.resType == cudaResourceTypeArray, "The texture is not bound to a CUDA array");
            return array_resource.res.array.array;
        }

        /// Whether texture is using normalized coordinates.
        static bool has_normalized_coordinates(cudaTextureObject_t texture) {
            return texture_description(texture).normalizedCoords;
        }
    };

    /// Texture object used to interpolate data.
    /// This type is supported by the Interpolator and the interpolate(_spectrum)_using_texture_at functions.
    ///         // These assume the coordinate is already matching the texture coordinate system,
    // i.e. the 0.5 offset is applied and the coordinate is normalized if the texture is normalized.
    // These assume the coordinate is within the [0-N] coordinate system.
    // They will add the 0.5 offset and normalize the coordinates if/when necessary.
    template<size_t N,
             Interp INTERP_MODE,
             Border BORDER_MODE,
             typename Value,
             typename Coord,
             bool NORMALIZED,
             bool LAYERED>
    class Texture {
    public:
        static constexpr Interp INTERP = INTERP_MODE;
        static constexpr Border BORDER = BORDER_MODE;

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
        using index_type = i32;
        using coord_n_type = Vec<coord_type, N>;
        using f_shape_type = std::conditional_t<NORMALIZED, coord_n_type, Empty>;
        using layer_type = std::conditional_t<LAYERED, i32, Empty>;

    public:
        template<typename I>
        Texture(cudaTextureObject_t texture, const Shape<I, N>& shape) requires NORMALIZED :
            m_texture(texture), m_norm(1 / shape.template as<coord_type>()) { validate(texture); }

        template<typename I = index_type>
        explicit Texture(cudaTextureObject_t texture, const Shape<I, N>& = {}) requires (not NORMALIZED) :
            m_texture(texture) { validate(texture); }

    public:
        template<nt::real T, size_t A>
        [[nodiscard]] auto fetch_preprocess(Vec<T, N, A> coordinates) const -> Vec<T, N, A> {
            coordinates += static_cast<T>(0.5); // to texture coordinate system
            if constexpr (NORMALIZED)
                coordinates *= m_norm.template as<T, A>();
            return coordinates;
        }

        template<nt::real... T> requires (sizeof...(T) == N)
        [[nodiscard]] auto fetch(T... coordinates) const noexcept -> value_type {
            return fetch_raw(fetch_preprocess(coord_n_type::from_values(coordinates...)));
        }

        template<nt::real T, size_t A>
        [[nodiscard]] auto fetch(const Vec<T, N, A>& coordinates) const noexcept -> value_type {
            return fetch_raw(fetch_preprocess(coordinates.template as<coord_type>()));
        }

        template<nt::real... T> requires (sizeof...(T) == N)
        [[nodiscard]] auto fetch_raw(T... coordinates) const noexcept -> value_type {
            return fetch_raw(coord_n_type::from_values(coordinates...));
        }

        template<nt::real T, size_t A>
        [[nodiscard]] auto fetch_raw(const Vec<T, N, A>& coordinates) const noexcept -> value_type {
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
            return {};
            #endif
        }

    public:
        void set_layer(nt::integer auto layer) noexcept {
            if constexpr (LAYERED)
                m_layer = layer;
        }

        [[nodiscard]] auto operator[](nt::integer auto layer) const noexcept -> Texture {
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


        /// Checks that the texture object matches the Texture pa
        static void validate(cudaTextureObject_t texture) {
            cudaArray* array = AllocatorTexture::texture_array(texture);
            const bool is_layered = AllocatorTexture::is_layered(array);
            check(is_layered == LAYERED, "The input texture object is not layered, but a layered Texture was created");

            const cudaTextureDesc description = AllocatorTexture::texture_description(texture);
            if constexpr (INTERP == Interp::NEAREST_FAST) {
                check(description.filterMode == cudaFilterModePoint,
                      "The input texture object is not using mode-point lookups, "
                      "which does not match the Texture settings: INTERP=", INTERP);
            } else if constexpr (INTERP == Interp::LINEAR_FAST) {
                check(description.filterMode == cudaFilterModeLinear,
                      "The input texture object is not using linear lookups, "
                      "which does not match the Texture settings: INTERP=", INTERP);
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
    };
}
#endif
