#pragma once

#include "noa/runtime/cuda/Error.hpp"
#include "noa/xform/core/Interp.hpp"

namespace noa::xform::cuda {
    using noa::cuda::check; // import CUDA error checking to this namespace

    template<usize N, Interp INTERP, Border BORDER,
             typename Value, typename Coord, typename Index,
             bool NORMALIZED, bool LAYERED>
    class Texture;

    /// Creates 1d, 2d, or 3d texture objects bounded to a CUDA array.
    /// CUDA textures:
    ///  -   Address mode: How out of range coordinates are handled. This can be specified for each coordinates (although
    ///                    the current implementation specifies the same mode for all the dimensions. It is either wrap,
    ///                    mirror, border or clamp (default).
    ///                    Note: This is ignored for 1D textures since they don't support addressing modes.
    ///                    Note: mirror and wrap are only supported for normalized coordinates, otherwise, fallback to clamp.
    ///  -   Filter mode:  Filtering used when fetching. Either point (neighbour) or linear.
    ///                    Note: The linear mode is only allowed for float types.
    ///                    Note: This is ignored for 1D textures since they don't perform any interpolation.
    ///  -   Read mode:    Whether or not integer data should be converted to floating point when fetching. If signed,
    ///                    returns float within [-1., 1.]. If unsigned, returns float within [0., 1.].
    ///                    Note: This only applies to 8-bit and 16-bit integer formats. 32-bits are not promoted.
    ///  -   Normalized coordinates: Whether or not the coordinates are normalized when fetching.
    ///                              If false (default): textures are fetched using floating point coordinates in range
    ///                                                  [0, N-1], where N is the size of that particular dimension.
    ///                              If true:            textures are fetched using floating point coordinates in range
    ///                                                  [0., 1. -1/N], where N is the size of that particular dimension.
    ///
    /// Textures are bound to global memory, either through a device pointer or a CUDA array.
    /// -- Data in the bounded CUDA array can be updated but texture cache is unchanged until a new kernel is launched.
    /// -- The device pointer or a CUDA array should not be freed while the texture is being used.
    class AllocatorTexture {
    public:
        // FIXME This should replace the unique_ptr
        struct Object {
            cudaTextureObject_t object{}; // usize
            explicit Object(cudaTextureObject_t o) noexcept : object(o) {}

        public: // no copy
            Object(const Object&) = delete;
            auto operator=(const Object&) -> Object& = delete;

        public: // move
            explicit Object(Object&& src) noexcept :
                object{std::exchange(src.object, static_cast<cudaTextureObject_t>(0))} {}

            auto operator=(Object&& src) -> Object& {
                if (this != &src) {
                    // TODO swap instead?
                    check(cudaDestroyTextureObject(object));
                    object = std::exchange(src.object, static_cast<cudaTextureObject_t>(0));
                }
                return *this;
            }

        public: // dtor
            ~Object() {
                if (object) {
                    [[maybe_unused]] cudaError_t err = cudaDestroyTextureObject(object);
                    NOA_ASSERT(err == cudaSuccess);
                }
            }
        };

        struct Deleter {
            void operator()(cudaTextureObject_t* ptr) const noexcept {
                [[maybe_unused]] auto err = cudaDestroyTextureObject(*ptr);
                delete ptr;
                NOA_ASSERT(err == cudaSuccess);
            }
        };

        using allocate_type = std::unique_ptr<cudaTextureObject_t, Deleter>;

    public:
        static auto allocate(
            const cudaArray* array,
            Interp interp,
            Border border
        ) -> allocate_type {
            const auto [filter, address, read_mode, normalized_coords] = convert_to_description(interp, border);
            const auto object = create_texture(array, filter, address, read_mode, normalized_coords);
            return allocate_type(new cudaTextureObject_t{object}, Deleter{});
        }

    public: // static texture utilities
        static constexpr auto convert_to_texture(
            Interp interp,
            Border border
        ) -> Pair<Interp, Border> {
            bool is_addressable = border.is_any(Border::ZERO, Border::CLAMP, Border::MIRROR, Border::PERIODIC);
            Border border_tex = is_addressable ? border : Border{Border::ZERO};
            Interp interp_tex =
                is_addressable and interp.is_any(Interp::LINEAR_FAST, Interp::CUBIC_BSPLINE_FAST) ?
                Interp::LINEAR_FAST : Interp::NEAREST_FAST;
            return {interp_tex, border_tex};
        }

        static auto convert_to_description(
            Interp interp,
            Border border
        ) -> Tuple<cudaTextureFilterMode, cudaTextureAddressMode, cudaTextureReadMode, bool> {
            auto [interp_tex, border_tex] = convert_to_texture(interp, border);

            cudaTextureFilterMode filter_mode =
                interp_tex == Interp::LINEAR_FAST ? cudaFilterModeLinear : cudaFilterModePoint;

            cudaTextureAddressMode address_mode{};
            bool normalized_coordinates{false};
            switch (border_tex) {
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
                case Border::ZERO: {
                    address_mode = cudaAddressModeBorder;
                    break;
                }
                default: panic();
            }
            return make_tuple(filter_mode, address_mode, cudaReadModeElementType, normalized_coordinates);
        }

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
        static auto create_texture(
            const cudaArray* array,
            cudaTextureFilterMode filter_mode,
            cudaTextureAddressMode address_mode,
            cudaTextureReadMode normalized_reads_to_float,
            bool normalized_coordinates
        ) -> cudaTextureObject_t {
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

        /// Returns a texture object's texture descriptor.
        static auto texture_description(cudaTextureObject_t texture) -> cudaTextureDesc {
            cudaTextureDesc tex_desc{};
            check(cudaGetTextureObjectTextureDesc(&tex_desc, texture));
            return tex_desc;
        }

        /// Returns a texture object's texture descriptor.
        static auto texture_resource(cudaTextureObject_t texture) -> cudaResourceDesc {
            cudaResourceDesc tex_desc{};
            check(cudaGetTextureObjectResourceDesc(&tex_desc, texture));
            return tex_desc;
        }

        static auto texture_array(cudaTextureObject_t texture) -> cudaArray* {
            const auto array_resource = texture_resource(texture);
            check(array_resource.resType == cudaResourceTypeArray, "The texture is not bound to a CUDA array");
            return array_resource.res.array.array;
        }

        /// Whether texture is using normalized coordinates.
        static bool has_normalized_coordinates(cudaTextureObject_t texture) {
            return texture_description(texture).normalizedCoords;
        }

        template<usize N, Interp INTERP, Border BORDER, typename Value, typename Coord, typename Index>
        struct texture_type_ {
            static constexpr bool LAYERED = N == 2;
            static constexpr bool NORMALIZED = BORDER == Border::MIRROR or BORDER == Border::PERIODIC;
            static constexpr auto TEX = convert_to_texture(INTERP, BORDER);
            using type = Texture<N, TEX.first, TEX.second, Value, Coord, Index, NORMALIZED, LAYERED>;
        };

        /// The corresponding Texture type created by the allocator.
        template<usize N, Interp INTERP, Border BORDER, typename Value, typename Coord, typename Index>
        using texture_type = texture_type_<N, INTERP, BORDER, Value, Coord, Index>::type;
    };
}
