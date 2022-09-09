#pragma once

#include <variant>

#include "noa/common/Types.h"
#include "noa/unified/Array.h"
#include "noa/unified/ArrayOption.h"
#include "noa/unified/geometry/Prefilter.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/memory/Copy.h"
#else
namespace noa::cuda {
    template<typename T>
    struct Texture {};
}
#endif

namespace noa::cpu {
    template<typename T>
    struct Texture {
        size4_t strides;
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
    /// \tparam T float, double, cfloat_t or cdouble_t.
    template<typename T>
    class Texture {
    public:
        using value_t = T;
        using dim_t = size_t;
        using dim4_t = Int4<dim_t>;
        static_assert(traits::is_any_v<T, float, double, cfloat_t, cdouble_t>);

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
        /// \param prefilter        Whether or not the input \p array should be prefiltered first.
        ///                         If true and if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST,
        ///                         the input is prefiltered in-place.
        ///
        /// \note If \p device_target is a GPU:
        ///         - Double precision is not supported.
        ///         - \p array should be in the rightmost order and its depth and width dimensions should be contiguous.
        ///         - \p array can be on any device, including the CPU.
        ///         - \p array cannot be batched.
        ///         - \p border_mode should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
        ///         - INTERP_{NEAREST|LINEAR_FAST} are the only modes supporting BORDER_{MIRROR|PERIODIC}.
        ///         - A CUDA array is allocated with the same type and shape as \p array, a texture is attached to
        ///           the new array's memory and the new CUDA array is initialized with the values from \p array.
        ///       If \p device_target is a CPU:
        ///         - \p array should be on the CPU.
        ///         - No computation is performed (other the the optional pre-filtering)
        ///           and the texture simply points to \p array.
        Texture(const Array<T>& array, Device device_target, InterpMode interp_mode, BorderMode border_mode,
                T cvalue = T{0}, bool prefilter = true);

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
        [[nodiscard]] const size4_t& shape() const noexcept;

        /// Returns the BDHW strides of the array.
        [[nodiscard]] const size4_t strides() const;

        /// Whether the dimensions of the array are C or F contiguous.
        template<char ORDER = 'C'>
        [[nodiscard]] bool contiguous() const noexcept;

        /// Gets the underlying texture, assuming it is a CPU texture (i.e. device is CPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] cpu::Texture<T>& cpu();

        /// Gets the underlying texture, assuming it is a GPU texture (i.e. device is GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] gpu::Texture<T>& gpu();

        /// Gets the underlying texture, assuming it is a CUDA texture (i.e. device is a CUDA-capable GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] cuda::Texture<T>& cuda();

        [[nodiscard]] InterpMode interp() const noexcept;
        [[nodiscard]] BorderMode border() const noexcept;

    public:

        /// Releases the array. *this is left empty.
        Texture release() noexcept;

    public: // Copy
        /// Updates the texture values with \p array.
        /// \param[in,out] array    Array to copy into the texture.
        /// \param prefilter        Whether the input \p array should be prefiltered first.
        ///                         If true and if the texture uses INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST,
        ///                         the input is prefiltered in-place.
        /// \note \p array should have the same shape as the texture.
        ///       Also, with GPU textures:
        ///         - \p array should be in the rightmost order and its depth and width dimensions should be contiguous.
        ///         - \p array can be on any device, including the CPU.
        ///       With CPU textures:
        ///         - \p array should be on the CPU.
        void update(const Array<T>& array, bool prefilter = true);

    private:
        std::variant<std::monostate, cpu::Texture<T>, gpu::Texture<T>> m_texture;
        size4_t m_shape;
        ArrayOption m_options;
        InterpMode m_interp{};
        BorderMode m_border{};
    };
}

#define NOA_UNIFIED_TEXTURE_
#include "noa/unified/Texture.inl"
#undef NOA_UNIFIED_TEXTURE_
