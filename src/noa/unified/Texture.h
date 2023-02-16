#pragma once

#include <variant>

#include "noa/unified/Array.hpp"
#include "noa/unified/ArrayOption.hpp"
#include "noa/unified/geometry/Prefilter.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/memory/PtrArray.hpp"
#include "noa/gpu/cuda/memory/PtrTexture.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#else
namespace noa::cuda {
    template<typename T>
    struct Texture {};
}
#endif

namespace noa::cpu {
    template<typename T>
    struct Texture {
        dim4_t strides;
        shared_t<T[]> ptr;
        T cvalue;
    };
}

namespace noa::gpu {
    template<typename T>
    using Texture = noa::cuda::Texture<T>;
}

namespace noa {
    /// Unified texture.
    /// \details This template class constructs and encapsulates a texture. Textures are used for fast interpolation
    ///          and/or multidimensional caching. On the CPU, this simply points to an array. However, on the GPU,
    ///          it allocates and initializes a proper GPU texture. These are usually hidden from the unified API
    ///          and handled by the GPU backend, but if multiple calls with the same input or even the same input
    ///          type and shape, it is more efficient to reuse the texture than to recreate it every time.
    /// \tparam Value float, double, cfloat_t or cdouble_t.
    template<typename Value>
    class Texture {
    public:
        using value_type = Value;
        static_assert(traits::is_any_v<value_type, float, double, cfloat_t, cdouble_t>);

    public:
        /// Creates an empty texture.
        constexpr Texture() = default;

        /// Creates a texture.
        /// \param[in,out] array    Array to transform into the new texture.
        /// \param device_target    Device where the texture should be constructed.
        /// \param interp_mode      Interpolation mode.
        /// \param border_mode      Border mode.
        /// \param cvalue           Constant value to use for out-of-bounds coordinates.
        ///                         Only used if \p border_mode is BORDER_VALUE.
        /// \param layered          Whether the GPU texture should be a 2D layered texture.
        ///                         The number of layers is equal to the batch dimension of \p array.
        ///                         This is ignored for CPU textures, since they are always considered layered.
        /// \param prefilter        Whether the input \p array should be prefiltered first.
        ///                         If true and if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST,
        ///                         the input is prefiltered in-place.
        ///
        /// \note If \p device_target is a GPU, a CUDA array is allocated with the same type and shape as \p array,
        ///       a texture is attached to the new array's memory and the new CUDA array is initialized with
        ///       the values from \p array. Limitations:\n
        ///         - Double precision is not supported.\n
        ///         - \p array should be in the rightmost order and its depth and width dimensions should be contiguous.
        ///           In other words, \p array should be contiguous or have a "pitch" as defined in CUDA.\n
        ///         - \p array can be on any device, including the CPU.\n
        ///         - If \p layered is false or if \p array is a 3D array, \p array cannot be batched.\n
        ///         - \p border_mode should be BORDER_{ZERO|CLAMP|PERIODIC|MIRROR}.\n
        ///         - INTERP_{NEAREST|LINEAR_FAST} are the only modes supporting BORDER_{MIRROR|PERIODIC}.
        /// \note If \p device_target is a CPU, no computation is performed (other the the optional pre-filtering)
        ///       and the texture simply points to \p array. Limitations:\n
        ///         - \p array should be on the CPU.\n
        ///
        /// \warning For GPU textures, \p array can be on any device, effectively allowing to create a GPU texture
        ///          from a CPU array. Note however that while the API will make sure that stream ordering will be
        ///          respected (by possible synchronizing the current stream of the \p array device), the caller
        ///          should not modify the underlying values of \p array until the texture is created. See eval().
        Texture(const Array<value_type>& array, Device device_target, InterpMode interp_mode, BorderMode border_mode,
                value_type cvalue = value_type{0}, bool layered = false, bool prefilter = true);

        /// Creates a texture.
        /// \param shape            BDHW shape of the new texture.
        /// \param device_target    Device where the texture should be constructed.
        /// \param interp_mode      Interpolation mode.
        /// \param border_mode      Border mode.
        /// \param cvalue           Constant value to use for out-of-bounds coordinates.
        ///                         Only used if \p border_mode is BORDER_VALUE.
        /// \param layered          Whether the GPU texture should be a 2D layered texture.
        ///                         The number of layers is equal to the batch dimension of \p array.
        ///                         This is ignored for CPU textures, since they are always considered layered.
        ///
        /// \note If \p device_target is a GPU, a CUDA array is allocated of \p T type and \p shape,
        ///       a texture is attached to the new array's memory. The new CUDA array is left
        ///       uninitialized (see update()). Limitations:\n
        ///         - Double precision is not supported.\n
        ///         - If \p layered is false or if \p shape describes a 3D array, \p shape cannot be batched.\n
        ///         - \p border_mode should be BORDER_{ZERO|CLAMP|PERIODIC|MIRROR}.\n
        ///         - INTERP_{NEAREST|LINEAR_FAST} are the only modes supporting BORDER_{MIRROR|PERIODIC}.
        ///
        /// \note If \p device_target is a CPU, no computation is performed. The texture is non-empty and valid,
        ///       but the underlying managed data (i.e. the cpu::Texture) points to a null pointer. Use update()
        ///       to set the texture to a valid memory region.
        Texture(dim4_t shape, Device device_target, InterpMode interp_mode, BorderMode border_mode,
                value_type cvalue = value_type{0}, bool layered = false);

    public: // Getters
        /// Returns the options used to create the array.
        [[nodiscard]] constexpr ArrayOption options() const noexcept;

        /// Returns the device used to create the array.
        [[nodiscard]] constexpr Device device() const noexcept;

        /// Returns the device used to create the array.
        [[nodiscard]] constexpr Allocator allocator() const noexcept;

        /// Whether the array is empty.
        [[nodiscard]] bool empty() const noexcept;

        /// Returns the BDHW shape of the array.
        [[nodiscard]] const dim4_t& shape() const noexcept;

        /// Returns the BDHW strides of the array.
        [[nodiscard]] dim4_t strides() const;

        /// Whether the dimensions of the array are C or F contiguous.
        template<char ORDER = 'C'>
        [[nodiscard]] bool contiguous() const noexcept;

        /// Synchronizes the current stream of the Texture's device.
        /// \details It guarantees safe access to the underlying data and indicates no operations are pending.
        const Texture& eval() const;

        /// Gets the underlying texture, assuming it is a CPU texture (i.e. device is CPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] cpu::Texture<value_type>& cpu();
        [[nodiscard]] const cpu::Texture<value_type>& cpu() const;

        /// Gets the underlying texture, assuming it is a GPU texture (i.e. device is GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] gpu::Texture<value_type>& gpu();
        [[nodiscard]] const gpu::Texture<value_type>& gpu() const;

        /// Gets the underlying texture, assuming it is a CUDA texture (i.e. device is a CUDA-capable GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] cuda::Texture<value_type>& cuda();
        [[nodiscard]] const cuda::Texture<value_type>& cuda() const;

        [[nodiscard]] InterpMode interp() const noexcept;
        [[nodiscard]] BorderMode border() const noexcept;
        [[nodiscard]] bool layered() const;

    public:
        /// Releases the array. *this is left empty.
        Texture release() noexcept;

    public: // Copy
        /// Updates the texture values with \p array.
        /// \param[in,out] array    Array to copy into the texture.
        /// \param prefilter        Whether the input \p array should be prefiltered first.
        ///                         If true and if the texture uses INTERP_CUBIC_BSPLINE(_FAST),
        ///                         the input is prefiltered in-place.
        /// \note With GPU textures, \p array should have the same shape as the texture and
        ///       a deep copy is performed from \p array to the managed texture data.
        ///       Limitations:
        ///         - \p array should be in the rightmost order and its depth and width dimensions should be contiguous.
        ///           In other words, \p array should be contiguous or have a "pitch" as defined in CUDA.\n
        ///         - \p array can be on any device, including the CPU.\n
        /// \note With CPU textures, no computation is performed (other than the optional pre-filtering)
        ///       and the texture pointer is simply updated to point to \p array, which should be a CPU array.
        ///
        /// \warning For GPU textures, \p array can be on any device, effectively allowing to update a GPU texture
        ///          from a CPU array. Note however that while the API will make sure that stream ordering will be
        ///          respected (by possible synchronizing the current stream of the \p array device), the caller
        ///          should not modify the underlying values of \p array until the texture is updated. See eval().
        void update(const Array<value_type>& array, bool prefilter = true);

    private:
        std::variant<std::monostate, cpu::Texture<value_type>, gpu::Texture<value_type>> m_texture;
        dim4_t m_shape;
        ArrayOption m_options;
        InterpMode m_interp{};
        BorderMode m_border{};
    };
}

#define NOA_UNIFIED_TEXTURE_
#include "noa/unified/Texture.inl"
#undef NOA_UNIFIED_TEXTURE_
