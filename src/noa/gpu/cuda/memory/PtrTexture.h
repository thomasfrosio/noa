/// \file noa/gpu/cuda/memory/PtrTexture.h
/// \brief Hold a CUDA texture.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"

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
//      -- Data can be updated but texture cache is not notified of CUDA array modifications. Start a new kernel to update.
//      -- The device pointer or a CUDA array should not be freed while the texture is being used.
//
// TODO Is (c)double precision or (u)int64 2D texture possible? For instance with double:
//      cudaCreateChannelDesc(sizeof(double), 0, 0, 0, cudaChannelFormatKindFloat)?

namespace noa::cuda::memory {
    /// Manages a 1D, 2D or 3D texture object. This object cannot be used on the device and is not copyable nor movable.
    /// Can be created from CUDA arrays, padded memory (2D only) and linear memory (1D only).
    /// \note   Currently supported components (either 1 or 2 per elements) are either signed or unsigned 8-, 16-, or
    ///         32-bit integers, 16-bit floats, or 32-bit floats.
    template<typename T>
    class PtrTexture {
    private:
        // The type T is within the type of PtrTexture even if it doesn't have to.
        // This is just to make PtrTexture more similar to other Ptr* and it is clearer IMHO.
        static_assert(noa::traits::is_valid_ptr_type_v<T>);
        static_assert(noa::traits::is_int_v<T> || noa::traits::is_float_v<T> || noa::traits::is_complex_v<T>);
        static_assert(!std::is_same_v<T, uint64_t> && !std::is_same_v<T, int64_t> &&
                      !std::is_same_v<T, double> && !std::is_same_v<T, cdouble_t>);

        cudaTextureObject_t m_texture{}; // this is just an integer.
        bool m_is_allocated{}; // m_texture is opaque so this is just to be safe and not rely on 0 being an empty texture.

    private:
        static NOA_HOST constexpr bool getNormalizedCoords_(InterpMode interp_mode, BorderMode border_mode) {
            bool normalized_coordinates = false;
            if (border_mode == BORDER_PERIODIC || border_mode == BORDER_MIRROR) {
                normalized_coordinates = true;
                if (interp_mode != INTERP_LINEAR && interp_mode != INTERP_NEAREST)
                    NOA_THROW_FUNC("alloc", "{} and {} are not supported with {}",
                                   BORDER_PERIODIC, BORDER_MIRROR, interp_mode);
            }
            return normalized_coordinates;
        }

    public: // Texture utilities
        /// Returns a texture object's texture descriptor.
        static NOA_HOST cudaTextureDesc getDescription(cudaTextureObject_t texture) {
            cudaTextureDesc tex_desc{};
            NOA_THROW_IF(cudaGetTextureObjectTextureDesc(&tex_desc, texture));
            return tex_desc;
        }

        /// Whether or not \p texture is using normalized coordinates.
        static NOA_HOST bool hasNormalizedCoordinates(cudaTextureObject_t texture) {
            return getDescription(texture).normalizedCoords;
        }

    public: // Generic alloc functions
        /// Creates a 1D, 2D or 3D texture from a CUDA array.
        /// \param array                        CUDA array. Its lifetime should exceed the life of this new object.
        ///                                     It is directly passed to the returned texture, i.e. its type doesn't
        ///                                     have to match \a T.
        /// \param interp_mode                  Interpolation used when fetching.
        ///                                     Either `INTERP_(NEAREST|LINEAR|COSINE|CUBIC_BSPLINE)`.
        /// \param border_mode                  How out of bounds are handled.
        ///                                     Either `BORDER_(CLAMP|ZERO|PERIODIC|MIRROR)`.
        /// \param normalized_coordinates       Whether or not the coordinates are normalized when fetching.
        ///                                     Should be false if \a interp_mode is `INTERP_(COSINE|CUBIC_BSPLINE)`.
        /// \param normalized_reads_to_float    Whether or not 8-, 16-integer data should be converted to float when fetching.
        ///                                     This corresponds to the \c cudaTextureReadMode enum.
        ///
        /// \note \a interp_mode == `INTERP_(LINEAR|COSINE|CUBIC_BSPLINE)` are only available with the `float` type.
        /// \note \a border_mode == `BORDER_(PERIODIC|MIRROR)` are only available if \a normalized_coordinates
        ///       is true, otherwise they'll automatically be switched (internally by CUDA) to `BORDER_CLAMP`.
        /// \note \a interp_mode, \a border_mode and \a normalized_coordinates are compatible with
        ///       \a cudaTextureFilterMode, \a cudaTextureAddressMode and \a cudaTextureReadMode respectively,
        ///        meaning that the values from these enumerators are guaranteed to be equal to each other and
        ///       can be used interchangeably.
        /// \note Since this function might be used to create textures used outside of this project, it is not going
        ///       to perform any compatibility check between the input arguments. However, note that if the output
        ///       texture is meant to be used within ::noa and if the `INTERP_(COSINE|CUBIC_BSPLINE)` modes are
        ///       used, \a normalized_coordinates should be false and \a border_mode can only be `BORDER_(CLAMP|ZERO)`.
        /// \see "noa/gpu/cuda/transform/Interpolate.h" for more details.
        static NOA_HOST cudaTextureObject_t alloc(const cudaArray* array,
                                                  InterpMode interp_mode,
                                                  BorderMode border_mode,
                                                  bool normalized_coordinates,
                                                  bool normalized_reads_to_float) {
            cudaResourceDesc res_desc{};
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = const_cast<cudaArray*>(array);
            // cudaArrayGetInfo can be used to extract the array type and make
            // sure it matches T, but what's the point?

            cudaTextureDesc tex_desc{};
            const auto tmp_border = static_cast<cudaTextureAddressMode>(border_mode);
            tex_desc.addressMode[0] = tmp_border; // BorderMode is compatible with cudaTextureAddressMode.
            tex_desc.addressMode[1] = tmp_border; // ignored if 1D array.
            tex_desc.addressMode[2] = tmp_border; // ignored if 1D or 2D array.
            tex_desc.filterMode = interp_mode == INTERP_NEAREST ? cudaFilterModePoint : cudaFilterModeLinear;
            tex_desc.readMode = static_cast<cudaTextureReadMode>(normalized_reads_to_float);
            tex_desc.normalizedCoords = normalized_coordinates;

            cudaTextureObject_t texture;
            cudaError_t err = cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr);
            if (err)
                NOA_THROW("Creating the texture object from a CUDA array failed, "
                          "with normalized_coordinates={}, InterpMode={}, BorderMode={}, normalized_reads_to_float={}",
                          normalized_coordinates, interp_mode, border_mode, normalized_reads_to_float);
            return texture;
        }

        /// Creates a 2D texture from a padded memory layout.
        /// \tparam T                           Type of the pointer data.
        /// \param[in] array                    Device pointer. Its lifetime should exceed the life of this new object.
        /// \param pitch                        Pitch, in elements, of \a array.
        /// \param shape                        Physical {fast, medium, slow} shape of \a array.
        /// \param interp_mode                  Interpolation used when fetching.
        ///                                     Either `INTERP_(NEAREST|LINEAR|COSINE|CUBIC_BSPLINE)`.
        /// \param border_mode                  How out of bounds are handled.
        ///                                     Either `BORDER_(CLAMP|ZERO|PERIODIC|MIRROR)`.
        /// \param normalized_coordinates       Whether or not the coordinates are normalized when fetching.
        ///                                     Should be false if \a interp_mode is `INTERP_(COSINE|CUBIC_BSPLINE)`.
        /// \param normalized_reads_to_float    Whether or not 8-, 16-integer data should be converted to float when fetching.
        ///                                     This corresponds to the \c cudaTextureReadMode enum.
        ///
        /// \see PtrTexture<T>::alloc() from CUDA arrays for more details on \a interp_mode and \a border_mode.
        /// \note Texture bound to pitch linear memory are usually used if conversion to CUDA arrays is too tedious,
        ///       but warps should preferably only access rows, for performance reasons.
        /// \note cudaDeviceProp::textureAlignment is satisfied by cudaMalloc* and cudaDeviceProp::texturePitchAlignment
        ///       is satisfied by cudaMalloc3D/Pitch. Care should be taken about offsets when working on subregions.
        static NOA_HOST cudaTextureObject_t alloc(const T* array, size_t pitch, size3_t shape,
                                                  InterpMode interp_mode,
                                                  BorderMode border_mode,
                                                  bool normalized_coordinates,
                                                  bool normalized_reads_to_float) {
            if (shape.z > 1)
                NOA_THROW("Cannot create a 3D texture object from an array of shape {}. Use a CUDA array.", shape);

            cudaResourceDesc res_desc{};
            res_desc.resType = cudaResourceTypePitch2D;
            res_desc.res.pitch2D.devPtr = const_cast<T*>(array);
            res_desc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
            res_desc.res.pitch2D.width = shape.x;
            res_desc.res.pitch2D.height = shape.y;
            res_desc.res.pitch2D.pitchInBytes = pitch * sizeof(T);

            cudaTextureDesc tex_desc{};
            const auto tmp_border = static_cast<cudaTextureAddressMode>(border_mode);
            tex_desc.addressMode[0] = tmp_border;
            tex_desc.addressMode[1] = tmp_border; // addressMode[2] isn't used.
            tex_desc.filterMode = interp_mode == INTERP_NEAREST ? cudaFilterModePoint : cudaFilterModeLinear;
            tex_desc.readMode = static_cast<cudaTextureReadMode>(normalized_reads_to_float);
            tex_desc.normalizedCoords = normalized_coordinates;

            cudaTextureObject_t texture;
            cudaError_t err = cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr);
            if (err)
                NOA_THROW("Creating the texture object from a padded memory layout failed, "
                          "with normalized_coordinates={}, InterpMode={}, BorderMode={}, normalized_reads_to_float={}",
                          normalized_coordinates, interp_mode, border_mode, normalized_reads_to_float);
        }

        /// Creates a 1D texture from linear memory.
        /// \tparam T                           Type of the pointer data.
        /// \param[in] array                    Device pointer. Its lifetime should exceed the life of this new object.
        /// \param elements                     Size, in elements, of \a array.
        /// \param normalized_coordinates       Whether or not the coordinates are normalized when fetching.
        /// \param normalized_reads_to_float    Whether or not 8-, 16-integer data should be converted to float when fetching.
        ///                                     This corresponds to the \c cudaTextureReadMode enum.
        ///
        /// \note   Textures bound to linear memory only support integer indexing (no interpolation),
        ///         and they don't have addressing modes. They are usually used if the texture cache can assist L1 cache.
        /// \note   cudaDeviceProp::textureAlignment is satisfied by cudaMalloc*.
        ///         Care should be taken about offsets when working on subregions.
        static NOA_HOST cudaTextureObject_t alloc(const T* array, size_t elements,
                                                  bool normalized_coordinates,
                                                  bool normalized_reads_to_float) {
            cudaResourceDesc res_desc{};
            res_desc.resType = cudaResourceTypeLinear;
            res_desc.res.linear.devPtr = const_cast<T*>(array);
            res_desc.res.linear.desc = cudaCreateChannelDesc<T>();
            res_desc.res.linear.sizeInBytes = elements * sizeof(T);

            cudaTextureDesc tex_desc{}; // TODO check what's the default filter mode for linear inputs
            tex_desc.readMode = static_cast<cudaTextureReadMode>(normalized_reads_to_float);
            tex_desc.normalizedCoords = normalized_coordinates;

            cudaTextureObject_t texture;
            NOA_THROW_IF(cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr));
            return texture;
        }

        static NOA_HOST void dealloc(cudaTextureObject_t texture) {
            NOA_THROW_IF(cudaDestroyTextureObject(texture));
        }

    public: // Simplified alloc functions
        /// Creates a 1D, 2D or 3D texture from a CUDA array.
        /// \param[in] array    CUDA array. Its lifetime should exceed the life of this new object.
        /// \param interp_mode  Either `INTERP_(NEAREST|LINEAR|COSINE|CUBIC_BSPLINE)`.
        /// \param border_mode  Either `BORDER_(CLAMP|ZERO|PERIODIC|MIRROR)`.
        /// \details This is will follow the assumptions:
        ///     - `INTERP_(COSINE|CUBIC_BSPLINE)` use unnormalized coordinates.
        ///     - `BORDER_(PERIODIC|MIRROR)` use normalized coordinates.
        ///     - `BORDER_(CLAMP|ZERO)` use unnormalized coordinates.
        ///     i.e. BORDER_(PERIODIC|MIRROR) can only be used with `INTERP_(NEAREST|LINEAR)`.
        static NOA_HOST cudaTextureObject_t alloc(const cudaArray* array,
                                                  InterpMode interp_mode, BorderMode border_mode) {
            return alloc(array, interp_mode, border_mode,
                         getNormalizedCoords_(interp_mode, border_mode), false);
        }

        /// Creates a 2D texture from a padded memory layout.
        /// \details The same assumptions to compute the "normalized coordinates" field are followed.
        static NOA_HOST cudaTextureObject_t alloc(const T* array, size_t pitch, size3_t shape,
                                                  InterpMode interp_mode, BorderMode border_mode) {
            return alloc(array, pitch, shape, interp_mode, border_mode,
                         getNormalizedCoords_(interp_mode, border_mode), false);
        }

    public: // Constructors, destructor
        /// Creates an empty instance. Use reset() to create a new texture.
        PtrTexture() = default;

        /// Creates a 1D, 2D or 3D texture from a CUDA array.
        /// \see Generic PtrTexture<T>::alloc() for more details.
        NOA_HOST PtrTexture(const cudaArray* array, InterpMode interp_mode, BorderMode border_mode,
                            bool normalized_coordinates, bool normalized_reads_to_float)
                : m_texture(alloc(array, interp_mode, border_mode,
                                  normalized_coordinates, normalized_reads_to_float)), m_is_allocated(true) {}

        /// Creates a 2D texture from a padded memory layout.
        /// \see PtrTexture<T>::alloc() for more details.
        NOA_HOST PtrTexture(const T* array, size_t pitch, size3_t shape,
                            InterpMode interp_mode, BorderMode border_mode,
                            bool normalized_coordinates, bool normalized_reads_to_float)
                : m_texture(alloc(array, pitch, shape, interp_mode, border_mode,
                                  normalized_coordinates, normalized_reads_to_float)), m_is_allocated(true) {}

        /// Creates a 1D texture from linear memory.
        /// \see PtrTexture<T>::alloc() for more details.
        NOA_HOST PtrTexture(const T* array, size_t elements,
                            bool normalized_coordinates, bool normalized_reads_to_float)
                : m_texture(alloc(array, elements, normalized_coordinates, normalized_reads_to_float)),
                  m_is_allocated(true) {}

        /// Creates a 1D, 2D or 3D texture from a CUDA array.
        /// \see Simplified PtrTexture<T>::alloc() for more details.
        NOA_HOST PtrTexture(const cudaArray* array, InterpMode interp_mode, BorderMode border_mode)
                : m_texture(alloc(array, interp_mode, border_mode)), m_is_allocated(true) {}

        /// Creates a 2D texture from a padded memory layout.
        /// \see Simplified PtrTexture<T>::alloc() for more details.
        NOA_HOST PtrTexture(const T* array, size_t pitch, size3_t shape, InterpMode interp_mode, BorderMode border_mode)
                : m_texture(alloc(array, pitch, shape, interp_mode, border_mode)), m_is_allocated(true) {}

        // For now let's just ignore these possibilities.
        PtrTexture(const PtrTexture& to_copy) = delete;
        PtrTexture& operator=(const PtrTexture& to_copy) = delete;
        PtrTexture(PtrTexture<T>&& to_move) = delete;
        PtrTexture<T>& operator=(PtrTexture<T>&& to_move) = delete;

        NOA_HOST ~PtrTexture() {
            if (m_is_allocated) {
                cudaError_t err = cudaDestroyTextureObject(m_texture);
                if (err != cudaSuccess && std::uncaught_exceptions() == 0)
                    NOA_THROW(toString(err));
            }
        }

    public:
        [[nodiscard]] NOA_HOST constexpr cudaTextureObject_t get() const noexcept { return m_texture; }
        [[nodiscard]] NOA_HOST constexpr cudaTextureObject_t id() const noexcept { return m_texture; }
        [[nodiscard]] NOA_HOST constexpr cudaTextureObject_t data() const noexcept { return m_texture; }

        /// Whether or not the object manages a texture.
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return !m_is_allocated; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Clears the underlying array, if necessary. empty() will evaluate to true.
        NOA_HOST void reset() {
            if (m_is_allocated) {
                dealloc(m_texture);
                m_is_allocated = false;
            }
        }

        /// Clears the underlying array, if necessary. This is identical to reset().
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /// Resets the underlying array. The new data is owned.
        NOA_HOST void reset(const cudaArray* array, InterpMode interp_mode, BorderMode border_mode,
                            bool normalized_coordinates, bool normalized_reads_to_float) {
            if (m_is_allocated)
                dealloc(m_texture);
            m_texture = alloc(array, interp_mode, border_mode, normalized_coordinates, normalized_reads_to_float);
            m_is_allocated = true;
        }

        /// Resets the underlying array. The new data is owned.
        NOA_HOST void reset(const T* array, size_t pitch, size3_t shape, InterpMode interp_mode, BorderMode border_mode,
                            bool normalized_coordinates, bool normalized_reads_to_float) {
            if (m_is_allocated)
                dealloc(m_texture);
            m_texture = alloc(array, pitch, shape, interp_mode, border_mode,
                              normalized_coordinates, normalized_reads_to_float);
            m_is_allocated = true;
        }

        /// Resets the underlying array. The new data is owned.
        NOA_HOST void reset(const T* array, size_t elements,
                            bool normalized_coordinates, bool normalized_reads_to_float) {
            if (m_is_allocated)
                dealloc(m_texture);
            m_texture = alloc(array, elements, normalized_coordinates, normalized_reads_to_float);
            m_is_allocated = true;
        }

        /// Resets the underlying array. The new data is owned.
        NOA_HOST void reset(const cudaArray* array, InterpMode interp_mode, BorderMode border_mode) {
            if (m_is_allocated)
                dealloc(m_texture);
            m_texture = alloc(array, interp_mode, border_mode);
            m_is_allocated = true;
        }

        /// Resets the underlying array. The new data is owned.
        NOA_HOST void reset(const T* array, size_t pitch, size3_t shape,
                            InterpMode interp_mode, BorderMode border_mode) {
            if (m_is_allocated)
                dealloc(m_texture);
            m_texture = alloc(array, pitch, shape, interp_mode, border_mode);
            m_is_allocated = true;
        }

        /// Releases the ownership of the managed array, if any.
        /// In this case, the caller is responsible for deleting the object.
        /// get() returns nullptr after the call and empty() returns true.
        [[nodiscard]] NOA_HOST cudaTextureObject_t release() noexcept {
            m_is_allocated = false;
            return m_texture;
        }
    };
}
