#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"

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
//
// Notes:
//  - Textures are bound to global memory, either through a device pointer or a CUDA array.
//      -- Data in the bounded CUDA array can be updated but texture cache is unchanged until a new kernel is launched.
//      -- The device pointer or a CUDA array should not be freed while the texture is being used.
//
// TODO(TF) Add the other BorderMode, especially BorderMode::VALUE by doing the addressing ourselves. This could be done by
//          passing the BorderMode to the texture functions. Normalization of the coordinates will be done by these
//          functions when required (e.g. InterpMode::LINEAR and BorderMode::MIRROR).

namespace noa::cuda::memory {
    struct PtrTextureDeleter {
        void operator()(const cudaTextureObject_t* handle) const noexcept {
            [[maybe_unused]] const cudaError_t err = cudaDestroyTextureObject(*handle);
            NOA_ASSERT(err == cudaSuccess);
            delete handle;
        }
    };

    // Manages a 1D, 2D or 3D texture object.
    // Can be created from CUDA arrays, padded memory (2D only) and linear memory (1D only).
    // Currently supported components (either 1 or 2 per elements) are either signed or
    // unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats.
    class PtrTexture {
    public:
        template<typename T>
        static constexpr bool is_valid_type_v = noa::traits::is_any_v<T, i8, i16, i32, u8, u16, u32, f16, f32, c16, c32>;

        using value_type = cudaTextureObject_t;
        using deleter_type = PtrTextureDeleter;
        using unique_type = Unique<cudaTextureObject_t, deleter_type>;
        using shared_type = Shared<cudaTextureObject_t>;

    public: // Texture utilities
        // Returns a texture object's texture descriptor.
        static cudaTextureDesc description(cudaTextureObject_t texture) {
            cudaTextureDesc tex_desc{};
            NOA_THROW_IF(cudaGetTextureObjectTextureDesc(&tex_desc, texture));
            return tex_desc;
        }

        // Returns a texture object's texture descriptor.
        static cudaResourceDesc resource(cudaTextureObject_t texture) {
            cudaResourceDesc tex_desc{};
            NOA_THROW_IF(cudaGetTextureObjectResourceDesc(&tex_desc, texture));
            return tex_desc;
        }

        static cudaArray* array(cudaTextureObject_t texture) {
            const auto array_resource = resource(texture);
            NOA_CHECK(array_resource.resType == cudaResourceTypeArray, "The texture is not bound to a CUDA array");
            return array_resource.res.array.array;
        }

        // Whether texture is using normalized coordinates.
        static bool has_normalized_coordinates(cudaTextureObject_t texture) {
            return description(texture).normalizedCoords;
        }

        // Sets the underlying texture filter and coordinate mode according to interp and border.
        // The interpolation functions in ::noa expect the texture to be set as follows:
        // - 1) BorderMode::MIRROR and BorderMode::PERIODIC requires normalized coordinates.
        // - 2) The accurate modes use nearest lookups, while the fast methods use linear lookups.
        // - 3) InterpMode::NEAREST and InterpMode::LINEAR_FAST are the only modes supporting normalized coordinates,
        //      thus they are the only modes supporting BorderMode::MIRROR and BorderMode::PERIODIC.
        static auto convert_to_description(
                InterpMode interp_mode,
                BorderMode border_mode
        ) -> std::tuple<cudaTextureFilterMode, cudaTextureAddressMode, bool> {
            cudaTextureFilterMode filter_mode{};
            cudaTextureAddressMode address_mode{};
            bool normalized_coordinates{};

            switch (interp_mode) {
                case InterpMode::NEAREST:
                case InterpMode::LINEAR:
                case InterpMode::COSINE:
                case InterpMode::CUBIC:
                case InterpMode::CUBIC_BSPLINE:
                    filter_mode = cudaFilterModePoint;
                    break;
                case InterpMode::LINEAR_FAST:
                case InterpMode::COSINE_FAST:
                case InterpMode::CUBIC_BSPLINE_FAST:
                    filter_mode = cudaFilterModeLinear;
                    break;
            }

            // Ensure BorderMode and InterpMode are compatible with
            // cudaTextureAddressMode and cudaTextureFilterMode.
            static_assert(noa::traits::to_underlying(BorderMode::PERIODIC) == static_cast<i32>(cudaAddressModeWrap));
            static_assert(noa::traits::to_underlying(BorderMode::CLAMP) == static_cast<i32>(cudaAddressModeClamp));
            static_assert(noa::traits::to_underlying(BorderMode::MIRROR) == static_cast<i32>(cudaAddressModeMirror));
            static_assert(noa::traits::to_underlying(BorderMode::ZERO) == static_cast<i32>(cudaAddressModeBorder));
            static_assert(noa::traits::to_underlying(InterpMode::NEAREST) == static_cast<i32>(cudaFilterModePoint));
            static_assert(noa::traits::to_underlying(InterpMode::LINEAR) == static_cast<i32>(cudaFilterModeLinear));

            if (border_mode == BorderMode::PERIODIC || border_mode == BorderMode::MIRROR) {
                address_mode = static_cast<cudaTextureAddressMode>(border_mode);
                normalized_coordinates = true;
                if (interp_mode != InterpMode::LINEAR_FAST && interp_mode != InterpMode::NEAREST)
                    NOA_THROW("{} is not supported with {}", border_mode, interp_mode);
            } else if (border_mode == BorderMode::ZERO || border_mode == BorderMode::CLAMP) {
                address_mode = static_cast<cudaTextureAddressMode>(border_mode);
                normalized_coordinates = false;
            } else {
                NOA_THROW("{} is not supported", border_mode);
            }
            return {filter_mode, address_mode, normalized_coordinates};
        }

    public: // Generic alloc functions
        // Creates a 1D, 2D or 3D texture from a CUDA array.
        // array:                       CUDA array. Its lifetime should exceed the lifetime of this new object.
        // interp_mode:                 Filter mode, either cudaFilterModePoint or cudaFilterModeLinear.
        // border_mode:                 Address mode, either cudaAddressModeWrap, cudaAddressModeClamp,
        //                              cudaAddressModeMirror or cudaAddressModeBorder.
        // normalized_coordinates:      Whether the coordinates are normalized when fetching.
        // normalized_reads_to_float:   Whether 8-, 16-integer data should be converted to float when fetching.
        //                              Either cudaReadModeElementType or cudaReadModeNormalizedFloat.
        // NOTE: cudaAddressModeMirror and cudaAddressModeWrap are only available with normalized coordinates.
        //       If normalized_coordinates is false, border_mode is switched (internally by CUDA) to cudaAddressModeClamp.
        static cudaTextureObject_t alloc(const cudaArray* array,
                                         cudaTextureFilterMode interp_mode,
                                         cudaTextureAddressMode border_mode,
                                         cudaTextureReadMode normalized_reads_to_float,
                                         bool normalized_coordinates) {
            cudaResourceDesc res_desc{};
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = const_cast<cudaArray*>(array);
            // cudaArrayGetInfo can be used to extract the array type and make
            // sure it matches T, but is it really useful? Maybe just an assert?

            cudaTextureDesc tex_desc{};
            tex_desc.addressMode[0] = border_mode;
            tex_desc.addressMode[1] = border_mode; // ignored if 1D array.
            tex_desc.addressMode[2] = border_mode; // ignored if 1D or 2D array.
            tex_desc.filterMode = interp_mode;
            tex_desc.readMode = normalized_reads_to_float;
            tex_desc.normalizedCoords = normalized_coordinates;

            cudaTextureObject_t texture{};
            if (cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr))
                NOA_THROW("Creating the texture object from a CUDA array failed, "
                          "with normalized={}, filter={}, addressing={}, reads_to_float={}",
                          normalized_coordinates, interp_mode, border_mode, normalized_reads_to_float);
            return texture;
        }

        // Creates a 2D texture from a padded memory layout.
        // array:                       On the device. Its lifetime should exceed the life of this new object.
        // pitch:                       Pitch, in elements, of the array.
        // shape:                       DHW shape of array.
        // interp_mode:                 Filter mode, either cudaFilterModePoint or cudaFilterModeLinear.
        // border_mode:                 Address mode, either cudaAddressModeWrap, cudaAddressModeClamp,
        //                              cudaAddressModeMirror or cudaAddressModeBorder.
        // normalized_coordinates:      Whether the coordinates are normalized when fetching.
        // normalized_reads_to_float:   Whether 8-, 16-integer data should be converted to float when fetching.
        //                              Either cudaReadModeElementType or cudaReadModeNormalizedFloat.
        //
        // NOTE: Texture bound to pitched linear memory are usually used if conversion to CUDA arrays is too tedious,
        //       but warps should preferably only access rows, for performance reasons.
        // NOTE: cudaDeviceProp::textureAlignment is satisfied by cudaMalloc* and cudaDeviceProp::texturePitchAlignment
        //       is satisfied by cudaMalloc3D/Pitch. Care should be taken about offsets when working on subregions.
        template<typename T, typename = std::enable_if_t<is_valid_type_v<T>>>
        static cudaTextureObject_t alloc(const T* array, i64 pitch, const Shape3<i64>& shape,
                                         cudaTextureFilterMode interp_mode,
                                         cudaTextureAddressMode border_mode,
                                         cudaTextureReadMode normalized_reads_to_float,
                                         bool normalized_coordinates) {
            if (shape[0] > 1)
                NOA_THROW("Cannot create a 3D texture object from an array of shape {}. Use a CUDA array.", shape);

            cudaResourceDesc res_desc{};
            res_desc.resType = cudaResourceTypePitch2D;
            res_desc.res.pitch2D.devPtr = const_cast<T*>(array);
            res_desc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
            res_desc.res.pitch2D.width = safe_cast<size_t>(shape[2]);
            res_desc.res.pitch2D.height = safe_cast<size_t>(shape[1]);
            res_desc.res.pitch2D.pitchInBytes = pitch * sizeof(T);

            cudaTextureDesc tex_desc{};
            tex_desc.addressMode[0] = border_mode;
            tex_desc.addressMode[1] = border_mode; // addressMode[2] isn't used.
            tex_desc.filterMode = interp_mode;
            tex_desc.readMode = normalized_reads_to_float;
            tex_desc.normalizedCoords = normalized_coordinates;

            cudaTextureObject_t texture{};
            if (cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr))
                NOA_THROW("Creating the texture object from a CUDA array failed, "
                          "with normalized={}, filter={}, addressing={}, reads_to_float={}",
                          normalized_coordinates, interp_mode, border_mode, normalized_reads_to_float);
            return texture;
        }

        // Creates a 1D texture from linear memory.
        // array:                       On the device. Its lifetime should exceed the life of this new object.
        // elements:                    Size, in elements, of the array.
        // normalized_coordinates:      Whether the coordinates are normalized when fetching.
        // normalized_reads_to_float:   Whether 8-, 16-integer data should be converted to float when fetching.
        //                              Either cudaReadModeElementType or cudaReadModeNormalizedFloat.
        //
        // NOTE: Textures bound to linear memory only support integer indexing (no interpolation),
        //       and they don't have addressing modes. They are usually used if the texture cache can assist L1 cache.
        // NOTE: cudaDeviceProp::textureAlignment is satisfied by cudaMalloc*.
        //       Care should be taken about offsets when working on subregions.
        template<typename T, typename = std::enable_if_t<is_valid_type_v<T>>>
        static cudaTextureObject_t alloc(const T* array, i64 elements,
                                         cudaTextureReadMode normalized_reads_to_float,
                                         bool normalized_coordinates) {
            cudaResourceDesc res_desc{};
            res_desc.resType = cudaResourceTypeLinear;
            res_desc.res.linear.devPtr = const_cast<T*>(array);
            res_desc.res.linear.desc = cudaCreateChannelDesc<T>();
            res_desc.res.linear.sizeInBytes = elements * sizeof(T);

            cudaTextureDesc tex_desc{};
            tex_desc.readMode = normalized_reads_to_float;
            tex_desc.normalizedCoords = normalized_coordinates;

            cudaTextureObject_t texture{};
            NOA_THROW_IF(cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr));
            return texture;
        }

        static void dealloc(cudaTextureObject_t texture) {
            NOA_THROW_IF(cudaDestroyTextureObject(texture));
        }

    public: // noa alloc functions
        // Creates a 1D, 2D or 3D texture from a CUDA array.
        // The CUDA array lifetime should exceed the life of this new object.
        // border_mode is limited to BorderMode::ZERO, BorderMode::CLAMP, BorderMode::PERIODIC or BorderMode::MIRROR.
        static unique_type alloc(const cudaArray* array, InterpMode interp_mode, BorderMode border_mode) {
            const auto [filter, address, normalized_coords] = convert_to_description(interp_mode, border_mode);
            auto tex = alloc(array, filter, address, cudaReadModeElementType, normalized_coords);
            return {new cudaTextureObject_t(tex), {}};
        }

        // Creates a 2D texture from a padded memory layout.
        // The device array lifetime should exceed the life of this new object.
        // border_mode is limited to BorderMode::ZERO, BorderMode::CLAMP, BorderMode::PERIODIC or BorderMode::MIRROR.
        template<typename T, typename = std::enable_if_t<is_valid_type_v<T>>>
        static unique_type alloc(const T* array, i64 pitch, const Shape3<i64>& shape,
                                 InterpMode interp_mode, BorderMode border_mode) {
            const auto [filter, address, normalized_coords] = convert_to_description(interp_mode, border_mode);
            const auto tex = alloc(array, pitch, shape, filter, address, cudaReadModeElementType, normalized_coords);
            return {new cudaTextureObject_t(tex), {}};
        }

    public: // Constructors, destructor
        // Creates an empty instance. Use reset() to create a new texture.
        constexpr PtrTexture() = default;
        constexpr /*implicit*/ PtrTexture(std::nullptr_t) {}

        // Creates a 1D, 2D or 3D texture from a CUDA array.
        PtrTexture(const cudaArray* array,
                   InterpMode interp_mode, BorderMode border_mode)
                : m_texture(alloc(array, interp_mode, border_mode)) {}

        // Creates a 2D texture from a padded memory layout.
        template<typename T, typename = std::enable_if_t<is_valid_type_v<T>>>
        PtrTexture(const T* array, i64 pitch, const Shape3<i64>& shape,
                   InterpMode interp_mode, BorderMode border_mode)
                : m_texture(alloc(array, pitch, shape, interp_mode, border_mode)) {}

    public:
        [[nodiscard]] cudaTextureObject_t get() const noexcept { return *m_texture; }
        [[nodiscard]] cudaTextureObject_t id() const noexcept { return *m_texture; }
        [[nodiscard]] cudaTextureObject_t handle() const noexcept { return *m_texture; }
        [[nodiscard]] constexpr const shared_type& share() const noexcept { return m_texture; }
        [[nodiscard]] bool is_empty() const noexcept { return m_texture == nullptr; }
        [[nodiscard]] explicit operator bool() const noexcept { return !is_empty(); }

        // Releases the ownership of the managed texture, if any.
        shared_type release() noexcept {
            return std::exchange(m_texture, nullptr);
        }

    private:
        shared_type m_texture{};
    };
}
