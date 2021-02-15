#pragma once

#include "noa/gpu/cuda/Base.h"
#include "noa/gpu/cuda/PtrDevice.h"
#include "noa/gpu/cuda/PtrDevicePadded.h"
#include "noa/gpu/cuda/PtrArray.h"

/*
 * CUDA textures:
 * ==============
 *
 *  -   Address mode: How out of range coordinates are handled. This can be specified for each coordinates (although
 *                    the current implementation specifies the same mode for all the dimensions. It is either wrap,
 *                    mirror, border or clamp (default).
 *                    Note: This is ignored for 1D textures.
 *                    Note: mirror and wrap are only supported for normalized coordinates (fallback to clamp).
 *  -   Filter mode:  Filtering used when fetching. Either point (neighbour) or linear.
 *                    Note: The linear mode is only allowed for float types.
 *                    Note: This is ignored for 1D textures.
 *  -   Read mode:    Whether or not integer data should be converted to floating point when fetching. If signed,
 *                    returns float within [-1., 1.]. If unsigned, returns float within [0., 1.].
 *                    Note: This only applies to 8-bit and 16-bit integer formats. 32-bits are not promoted.
 *  -   Normalized coordinates: Whether or not the coordinates are normalized when fetching.
 *                              If false (default): textures are fetched using floating point coordinates in range
 *                                                  [0, N-1], where N is the size of that particular dimension.
 *                              If true:            textures are fetched using floating point coordinates in range
 *                                                  [0., 1. -1/N], where N is the size of that particular dimension.
 */

namespace Noa::CUDA {
    /**
     * Manages a 1D, 2D or 3D texture object.
     * Can be created from CUDA arrays, padded memory (2D only) and linear memory (1D only).
     * @note Currently supported components (either 1 or 2 per elements) are either signed or unsigned 8-, 16-, or
     *       32-bit integers, 16-bit floats, or 32-bit floats.
     */
    class PtrTexture {
        cudaTextureObject_t m_texture{}; // this is just an integer.
    public:
        bool is_owner{true};

        /**
         * Creates a @a N (1, 2 or 3) dimensional texture from a CUDA array.
         * @tparam T                        Type of the CUDA array.
         * @tparam N                        Number of dimensions of that array.
         * @param array                     Object managing a @a N dim CUDA array.
         * @param normalized_coordinates    Whether or not the coordinates are normalized when fetching.
         * @param filter_mode               Filtering used when fetching. Either point (neighbour) or linear.
         *                                  Depends on @a T (linear is only for float types).
         * @param address_mode              How out of range coordinates are handled. Applies to all dimensions.
         *                                  Depends on @a normalized_coordinates.
         * @param read_mode                 Whether or not integer data should be converted to float when fetching.
         *
         * @warning The life of @a array should exceed the life of this instance.
         */
        template<class T, uint N>
        NOA_HOST explicit PtrTexture(const PtrArray<T, N>& array,
                                     bool normalized_coordinates = false,
                                     cudaTextureFilterMode filter_mode = cudaFilterModePoint,
                                     cudaTextureAddressMode address_mode = cudaAddressModeWrap,
                                     cudaTextureReadMode read_mode = cudaReadModeElementType) {
            cudaResourceDesc resDesc{};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = array.get();

            cudaTextureDesc texDesc{};
            texDesc.addressMode[0] = address_mode;
            texDesc.addressMode[1] = address_mode; // ignored if 1D array.
            texDesc.addressMode[2] = address_mode; // ignored if 1D or 2D array.
            texDesc.filterMode = filter_mode;
            texDesc.readMode = read_mode;
            texDesc.normalizedCoords = normalized_coordinates;

            NOA_THROW_IF(cudaCreateTextureObject(&m_texture, &resDesc, &texDesc, nullptr));
        }

        /**
         * Creates a @a 2D dimensional texture from padded memory.
         * @tparam T                        Type of the padded memory.
         * @param pitch2d                   Object managing a 2D padded memory.
         * @param normalized_coordinates    Whether or not the coordinates are normalized when fetching.
         * @param filter_mode               Filtering used when fetching. Either point (neighbour) or linear.
         *                                  Depends on @a T (linear is only for float types).
         * @param address_mode              How out of range coordinates are handled. Applies to all dimensions.
         *                                  Depends on @a normalized_coordinates.
         * @param read_mode                 Whether or not integer data should be converted to float when fetching.
         *
         * @warning The life of @a pitch2d should exceed the life of this instance.
         * @note    cudaDeviceProp::textureAlignment is satisfied by cudaMalloc and cudaDeviceProp::texturePitchAlignment
         *          is satisfied by cudaMalloc3D/Pitch. Care should be taken about offsets when working on subregions.
         *
         * @todo:   (c)double precision or (u)int64 2D texture should be possible. For instance with double:
         *          cudaCreateChannelDesc(sizeof(double), 0, 0, 0, cudaChannelFormatKindFloat);
         */
        template<typename T, typename = std::enable_if_t<!std::is_same_v<T, uint64_t> && !std::is_same_v<T, int64_t> &&
                                                         !std::is_same_v<T, double> && !std::is_same_v<T, cdouble_t>>>
        NOA_HOST explicit PtrTexture(const PtrDevicePadded<T>& pitch2d,
                                     bool normalized_coordinates = false,
                                     cudaTextureFilterMode filter_mode = cudaFilterModePoint,
                                     cudaTextureAddressMode address_mode = cudaAddressModeWrap,
                                     cudaTextureReadMode read_mode = cudaReadModeElementType) {
            size3_t shape = pitch2d.shape();
            if (shape.z > 1)
                NOA_THROW("Cannot create a 3D texture object from padded memory of shape {}. Use a CUDA array.", shape);

            cudaResourceDesc resDesc{};
            resDesc.resType = cudaResourceTypePitch2D;
            resDesc.res.pitch2D.devPtr = pitch2d.get();
            resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
            resDesc.res.pitch2D.width = shape.x;
            resDesc.res.pitch2D.height = shape.y;
            resDesc.res.pitch2D.pitchInBytes = pitch2d.pitch();

            cudaTextureDesc texDesc{};
            texDesc.addressMode[0] = address_mode;
            texDesc.addressMode[1] = address_mode; // addressMode[2] isn't used.
            texDesc.filterMode = filter_mode;
            texDesc.readMode = read_mode;
            texDesc.normalizedCoords = normalized_coordinates;

            NOA_THROW_IF(cudaCreateTextureObject(&m_texture, &resDesc, &texDesc, nullptr));
        }

        /**
         * Creates a @a 1D dimensional texture from linear memory.
         * @tparam T                        Type of the linear memory.
         * @param linear                    Object managing the linear memory.
         * @param normalized_coordinates    Whether or not the coordinates are normalized when fetching.
         * @param filter_mode               Filtering used when fetching. Either point (neighbour) or linear.
         *                                  Depends on @a T (linear is only for float types).
         * @param address_mode              How out of range coordinates are handled. Applies to all dimensions.
         *                                  Depends on @a normalized_coordinates.
         *
         * @warning The life of @a linear should exceed the life of this instance.
         * @note    cudaDeviceProp::textureAlignment is satisfied by cudaMalloc.
         *          Care should be taken about offsets when working on subregions.
         *
         * @todo:   (c)double precision or (u)int64 2D texture should be possible. For instance with cdouble_t:
         *          cudaCreateChannelDesc(sizeof(double), sizeof(double), 0, 0, cudaChannelFormatKindFloat);
         */
        template<typename T, typename = std::enable_if_t<!std::is_same_v<T, uint64_t> && !std::is_same_v<T, int64_t> &&
                                                         !std::is_same_v<T, double> && !std::is_same_v<T, cdouble_t>>>
        NOA_HOST explicit PtrTexture(const PtrDevice<T>& linear,
                                     bool normalized_coordinates = false,
                                     cudaTextureReadMode read_mode = cudaReadModeElementType) {
            cudaResourceDesc resDesc{};
            resDesc.resType = cudaResourceTypeLinear;
            resDesc.res.linear.devPtr = linear.get();
            resDesc.res.linear.desc = cudaCreateChannelDesc<T>();
            resDesc.res.linear.sizeInBytes = linear.bytes();

            cudaTextureDesc texDesc{};
            texDesc.readMode = read_mode;
            texDesc.normalizedCoords = normalized_coordinates;

            NOA_THROW_IF(cudaCreateTextureObject(&m_texture, &resDesc, &texDesc, nullptr));
        }

        /** Copy constructor. The new instance is not the owner of the texture. */
        NOA_HOST PtrTexture(const PtrTexture& to_copy) noexcept
                : m_texture(to_copy.m_texture), is_owner(false) {}

        /** Move constructor. Gets the ownership of the moved object, which looses its ownership. */
        NOA_HOST PtrTexture(PtrTexture&& to_move) noexcept
                : m_texture(to_move.m_texture),
                  is_owner(std::exchange(to_move.is_owner, false)) {}

        // Just create a new instance.
        PtrTexture& operator=(const PtrTexture& to_copy) = delete;
        PtrTexture& operator=(PtrTexture&& to_move) = delete;

        [[nodiscard]] NOA_HOST constexpr cudaTextureObject_t get() const noexcept { return m_texture; }
        [[nodiscard]] NOA_HOST constexpr cudaTextureObject_t id() const noexcept { return m_texture; }

        ~PtrTexture() {
            if (is_owner)
                NOA_THROW_IF(cudaDestroyTextureObject(m_texture));
        }
    };
}
