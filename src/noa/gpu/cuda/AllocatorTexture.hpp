#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/utils/Misc.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"

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
//
// TODO(TF) Add the other Border, especially Border::VALUE by doing the addressing ourselves. This could be done by
//          passing the Border to the texture functions. Normalization of the coordinates will be done by these
//          functions when required (e.g. InterpMode::LINEAR and Border::MIRROR).

// Add specialization for our complex types.
// Used for CUDA arrays and textures.
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::f16>() { return cudaCreateChannelDesc<half>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c16>() { return cudaCreateChannelDesc<half2>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c32>() { return cudaCreateChannelDesc<float2>(); }

namespace noa::cuda {
    /// Creates and manages a 1d, 2d or 3d texture object bounded to a CUDA array.
    /// \note While textures can map pitch memory (2d only) and linear memory (1d only), but we don't support these
    ///       use cases as they are either less performant or are very limited compared to a CUDA array.
    template<typename T>
    requires nt::is_any_v<T, i8, i16, i32, u8, u16, u32, f16, f32, c16, c32>
    class AllocatorTexture {
    public:
        struct Resource {
            using value_type = T;
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
        using value_type = T;
        using shared_type = std::shared_ptr<Resource>;

    public:
        /// Allocates a CUDA array and create a texture from that array.
        /// The border mode should be Border::{ZERO|CLAMP|PERIODIC|MIRROR}.
        static auto allocate(
                const Shape4<i64>& shape,
                Interp interp_mode,
                Border border_mode,
                u32 flag = cudaArrayDefault
        ) -> std::shared_ptr<Resource> {
            auto resource = std::make_shared<Resource>();

            // Create the array.
            const cudaChannelFormatDesc desc = cudaCreateChannelDesc<value_type>();
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
            auto u_extent = Shape3<size_t>{extent.depth, extent.height, extent.width};
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
            return std::tuple<cudaChannelFormatDesc, cudaExtent, u32>(desc, extent, flags);
        }

        static bool is_layered(cudaArray* array) {
            const auto [desc_, extent_, flags] = array_info(array);
            // Not sure whether the flags are mutually exclusive, so just check the bit for layered textures.
            return flags & cudaArrayLayered;
        }

    public: // static texture utilities
        /// Creates a 1d, 2d or 3d texture from a CUDA array.
        /// \param array                        CUDA array. Its lifetime should exceed the lifetime of this new object.
        /// \param interp_mode                  Filter mode, either cudaFilterModePoint or cudaFilterModeLinear.
        /// \param border_mode                  Address mode, either cudaAddressModeWrap, cudaAddressModeClamp,
        ///                                     cudaAddressModeMirror or cudaAddressModeBorder.
        /// \param normalized_reads_to_float    Whether 8-, 16-integer data should be converted to float when fetching.
        ///                                     Either cudaReadModeElementType or cudaReadModeNormalizedFloat.
        /// \param normalized_coordinates       Whether the coordinates are normalized when fetching.
        /// \note cudaAddressModeMirror and cudaAddressModeWrap are only available with normalized coordinates.
        ///       If normalized_coordinates is false, border_mode is switched (internally by CUDA) to cudaAddressModeClamp.
        static cudaTextureObject_t create_texture(
                const cudaArray* array,
                cudaTextureFilterMode interp_mode,
                cudaTextureAddressMode border_mode,
                cudaTextureReadMode normalized_reads_to_float,
                bool normalized_coordinates
        ) {
            cudaResourceDesc res_desc{};
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = const_cast<cudaArray*>(array); // one example where we need const_cast...
            // TODO cudaArrayGetInfo can be used to extract the array type and make
            //      sure it matches T, but is it really useful? Maybe just an assert?

            cudaTextureDesc tex_desc{};
            tex_desc.addressMode[0] = border_mode;
            tex_desc.addressMode[1] = border_mode; // ignored if 1d array.
            tex_desc.addressMode[2] = border_mode; // ignored if 1d or 2d array.
            tex_desc.filterMode = interp_mode;
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
            cudaTextureFilterMode filter_mode{};
            cudaTextureAddressMode address_mode{};
            bool normalized_coordinates{};

            switch (interp_mode) {
                case Interp::NEAREST:
                case Interp::LINEAR:
                case Interp::COSINE:
                case Interp::CUBIC:
                case Interp::CUBIC_BSPLINE:
                    filter_mode = cudaFilterModePoint;
                    break;
                case Interp::LINEAR_FAST:
                case Interp::COSINE_FAST:
                case Interp::CUBIC_BSPLINE_FAST:
                    filter_mode = cudaFilterModeLinear;
                    break;
            }

            // Ensure Border and Interp are compatible with
            // cudaTextureAddressMode and cudaTextureFilterMode.
            static_assert(to_underlying(Border::PERIODIC) == static_cast<i32>(cudaAddressModeWrap));
            static_assert(to_underlying(Border::CLAMP) == static_cast<i32>(cudaAddressModeClamp));
            static_assert(to_underlying(Border::MIRROR) == static_cast<i32>(cudaAddressModeMirror));
            static_assert(to_underlying(Border::ZERO) == static_cast<i32>(cudaAddressModeBorder));
            static_assert(to_underlying(Interp::NEAREST) == static_cast<i32>(cudaFilterModePoint));
            static_assert(to_underlying(Interp::LINEAR) == static_cast<i32>(cudaFilterModeLinear));

            if (border_mode == Border::PERIODIC or border_mode == Border::MIRROR) {
                address_mode = static_cast<cudaTextureAddressMode>(border_mode);
                normalized_coordinates = true;
                if (interp_mode != Interp::LINEAR_FAST && interp_mode != Interp::NEAREST)
                    panic("{} is not supported with {}", border_mode, interp_mode);
            } else if (border_mode == Border::ZERO or border_mode == Border::CLAMP) {
                address_mode = static_cast<cudaTextureAddressMode>(border_mode);
                normalized_coordinates = false;
            } else {
                panic("{} is not supported", border_mode);
            }
            return make_tuple(filter_mode, address_mode, normalized_coordinates);
        }


        // Returns a texture object's texture descriptor.
        static cudaTextureDesc texture_description(cudaTextureObject_t texture) {
            cudaTextureDesc tex_desc{};
            check(cudaGetTextureObjectTextureDesc(&tex_desc, texture));
            return tex_desc;
        }

        // Returns a texture object's texture descriptor.
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

        // Whether texture is using normalized coordinates.
        static bool has_normalized_coordinates(cudaTextureObject_t texture) {
            return texture_description(texture).normalizedCoords;
        }
    };
}
#endif
