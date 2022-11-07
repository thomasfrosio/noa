#pragma once

#include <type_traits>
#include <utility>      // std::exchange

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"

// CUDA arrays:
//  -   Data resides in global memory. The host can cudaMemcpy to it and the device can only access it through texture
//      reads or surface reads and writes.
//  -   They are usually associated with a type: each element can have 1, 2 or 4 components (e.g. complex types have
//      2 components). Elements are associated with a type (components have the same type), that may be signed or
//      unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats.
//  -   They are either 1D, 2D or 3D. Note that an "empty" dimension is noted as 0 in the CUDA API, but PtrArray is
//      following noa's "shape" convention (i.e. "empty" dimensions are noted as 1).
//
// Notes:
//  - They are cache optimized for 2D/3D spatial locality.
//  - Surfaces and textures can be bound to same CUDA array.
//  - They are mostly used if the content changes rarely. Although reusing them with cudaMemcpy is possible.

// Add specialization for our complex types.
// Used for CUDA arrays and textures.
template<> cudaChannelFormatDesc cudaCreateChannelDesc<noa::cfloat_t>() { return cudaCreateChannelDesc<float2>(); }
template<> cudaChannelFormatDesc cudaCreateChannelDesc<noa::chalf_t>() { return cudaCreateChannelDesc<half2>(); }

namespace noa::cuda::memory {
    // A ND CUDA array of integers (excluding (u)int64_t), float or cfloat_t.
    // cfloat_t has the same channel descriptor as the CUDA built-in vector float2.
    template<typename value_t>
    class PtrArray {
    public:
        using value_type = value_t;

        struct Deleter {
            void operator()(cudaArray* array) noexcept {
                [[maybe_unused]] const cudaError_t err = cudaFreeArray(array);
                NOA_ASSERT(err == cudaSuccess);
            }
        };

    public: // static functions
        static cudaExtent shape2extent(dim4_t shape, bool is_layered) {
            // Special case: treat column vectors as row vectors.
            if (shape[2] >= 1 && shape[3] == 1)
                std::swap(shape[2], shape[3]);

            // Conversion:  shape -> CUDA extent
            // 3D:          1DHW  -> DHW
            // 2D:          11HW  -> 0HW
            // 1D:          111W  -> 00W
            // 2D layered:  B1HW  -> DHW
            // 1D layered:  B11W  -> D0W
            NOA_CHECK(all(shape > 0) && shape[is_layered] == 1,
                      "The input shape cannot be converted to a CUDA array extent. "
                      "Dimensions with a size of 0 are not allowed, and the {} should be 1",
                      is_layered ? "depth dimension (for layered arrays)" : "batch dimension", shape);

            const auto u_shape = safe_cast<size4_t>(shape);
            dim3_t shape_3d{u_shape[!is_layered], u_shape[2], u_shape[3]};
            shape_3d -= dim3_t(shape_3d == 1); // set empty dimensions to 0
            return {shape_3d[0], shape_3d[1], shape_3d[2]};
        }

        static dim4_t extent2shape(cudaExtent extent, bool is_layered) noexcept {
            size3_t u_extent{extent.depth, extent.height, extent.width};
            u_extent += size3_t(u_extent == 0); // set empty dimensions to 1

            // Column vectors are "lost" in the conversion.
            // 1D extents are interpreted as row vectors.
            size4_t shape{1, 1, u_extent[1], u_extent[2]};
            shape[!is_layered] = u_extent[0];
            return {shape};
        }

        static auto info(cudaArray* array) {
            cudaChannelFormatDesc desc{};
            cudaExtent extent{};
            uint32_t flags{};
            NOA_THROW_IF(cudaArrayGetInfo(&desc, &extent, &flags, array));
            return std::tuple<cudaChannelFormatDesc, cudaExtent, uint32_t>(desc, extent, flags);
        }

        static bool isLayered(cudaArray* array) {
            const auto [desc_, extent_, flags] = info(array);
            // Not sure whether the flags are mutually exclusive, so just check the bit for layered textures.
            return flags & cudaArrayLayered;
        }

        static std::unique_ptr<cudaArray, Deleter> alloc(dim4_t shape, uint32_t flag = cudaArrayDefault) {
            const cudaExtent extent = shape2extent(shape, flag & cudaArrayLayered);

            cudaArray* ptr;
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<value_type>();
            NOA_THROW_IF(cudaMalloc3DArray(&ptr, &desc, extent, flag));
            return {ptr, Deleter{}};
        }

    public: // member functions
        // Creates an empty instance. Use one of the operator assignment to allocate new data.
        constexpr PtrArray() = default;
        constexpr /*implicit*/ PtrArray(std::nullptr_t) {}

        // Allocates a CUDA array with a given BDHW shape on the current device using cudaMalloc3DArray.
        explicit PtrArray(dim4_t shape, uint32_t flags = cudaArrayDefault)
                : m_ptr(alloc(shape, flags)), m_shape(shape) {}

    public:
        // Returns the CUDA array pointer.
        [[nodiscard]] constexpr cudaArray* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr cudaArray* data() const noexcept { return m_ptr.get(); }

        // Returns a reference of the shared object.
        [[nodiscard]] constexpr const std::shared_ptr<cudaArray>& share() const noexcept { return m_ptr; }

        // Attach the lifetime of the managed object with an alias.
        // Constructs a shared_ptr which shares ownership information with the managed object,
        // but holds an unrelated and unmanaged pointer alias. If the returned shared_ptr is
        // the last of the group to go out of scope, it will call the stored deleter for the
        // managed object of this instance. However, calling get() on this shared_ptr will always
        // return a copy of alias. It is the responsibility of the programmer to make sure that
        // alias remains valid as long as the managed object exists.
        template<typename T>
        [[nodiscard]] constexpr std::shared_ptr<T[]> attach(T* alias) const noexcept { return {m_ptr, alias}; }

        [[nodiscard]] constexpr dim4_t shape() const noexcept { return m_shape; }
        [[nodiscard]] bool isLayered() const noexcept { return isLayered(get()); }

        // Whether the managed object points to some data.
        [[nodiscard]] constexpr bool empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !empty(); }

        // Releases the ownership of the managed array, if any.
        std::shared_ptr<cudaArray> release() noexcept {
            m_shape = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        static_assert(traits::is_any_v<value_type, int32_t, uint32_t, float, cfloat_t>);
        std::shared_ptr<cudaArray> m_ptr{nullptr};
        dim4_t m_shape{};
    };
}
