#pragma once

#include <type_traits>
#include <utility>      // std::exchange

#include "noa/core/Definitions.hpp"
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
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::f16>() { return cudaCreateChannelDesc<half>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c16>() { return cudaCreateChannelDesc<half2>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c32>() { return cudaCreateChannelDesc<float2>(); }

namespace noa::cuda::memory {
    struct PtrArrayDeleter {
        void operator()(cudaArray* array) const noexcept {
            [[maybe_unused]] const cudaError_t err = cudaFreeArray(array);
            NOA_ASSERT(err == cudaSuccess);
        }
    };

    // A ND CUDA array of integers (excluding u64, i64), f32 or c32.
    template<typename Value>
    class PtrArray {
    public:
        static_assert(traits::is_any_v<Value, i32, u32, f32, c32>);
        using value_type = Value;
        using shared_type = Shared<cudaArray>;
        using deleter_type = PtrArrayDeleter;
        using unique_type = Unique<cudaArray, deleter_type>;

    public: // static functions
        static cudaExtent shape2extent(Shape4<i64> shape, bool is_layered) {
            // Special case: treat column vectors as row vectors.
            if (shape[2] >= 1 && shape[3] == 1)
                std::swap(shape[2], shape[3]);

            // Conversion:  shape -> CUDA extent
            // 3D:          1DHW  -> DHW
            // 2D:          11HW  -> 0HW
            // 1D:          111W  -> 00W
            // 2D layered:  B1HW  -> DHW
            // 1D layered:  B11W  -> D0W
            NOA_CHECK(noa::all(shape > 0) && shape[is_layered] == 1,
                      "The input shape cannot be converted to a CUDA array extent. "
                      "Dimensions with a size of 0 are not allowed, and the {} should be 1. Shape: {}",
                      is_layered ? "depth dimension (for layered arrays)" : "batch dimension", shape);

            auto shape_3d = shape.filter(static_cast<i64>(!is_layered), 2, 3).as_safe<size_t>();

            // Set empty dimensions to 0. If layered, leave extent.depth to the batch value.
            if (!is_layered)
                shape_3d[0] -= shape_3d[0] == 1;
            shape_3d[1] -= shape_3d[1] == 1;
            return {shape_3d[2], shape_3d[1], shape_3d[0]};
        }

        static Shape4<i64> extent2shape(cudaExtent extent, bool is_layered) noexcept {
            auto u_extent = Shape3<size_t>{extent.depth, extent.height, extent.width};
            u_extent += Shape3<size_t>(u_extent == 0); // set empty dimensions to 1

            // Column vectors are "lost" in the conversion.
            // 1D extents are interpreted as row vectors.
            auto shape = Shape4<i64>{1, 1, u_extent[1], u_extent[2]};
            shape[!is_layered] = static_cast<i64>(u_extent[0]);
            return shape;
        }

        static auto info(cudaArray* array) {
            cudaChannelFormatDesc desc{};
            cudaExtent extent{};
            u32 flags{};
            NOA_THROW_IF(cudaArrayGetInfo(&desc, &extent, &flags, array));
            return std::tuple<cudaChannelFormatDesc, cudaExtent, u32>(desc, extent, flags);
        }

        static bool is_layered(cudaArray* array) {
            const auto [desc_, extent_, flags] = info(array);
            // Not sure whether the flags are mutually exclusive, so just check the bit for layered textures.
            return flags & cudaArrayLayered;
        }

        static unique_type alloc(const Shape4<i64>& shape, u32 flag = cudaArrayDefault) {
            const cudaExtent extent = shape2extent(shape, flag & cudaArrayLayered);

            cudaArray* ptr{};
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<value_type>();
            NOA_THROW_IF(cudaMalloc3DArray(&ptr, &desc, extent, flag));
            return unique_type{ptr};
        }

    public: // member functions
        // Creates an empty instance. Use one of the operator assignment to allocate new data.
        constexpr PtrArray() = default;
        constexpr /*implicit*/ PtrArray(std::nullptr_t) {}

        // Allocates a CUDA array with a given BDHW shape on the current device using cudaMalloc3DArray.
        explicit PtrArray(const Shape4<i64>& shape, u32 flags = cudaArrayDefault)
                : m_ptr(alloc(shape, flags)), m_shape(shape) {}

    public:
        [[nodiscard]] constexpr cudaArray* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr cudaArray* data() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const shared_type& share() const noexcept { return m_ptr; }
        [[nodiscard]] constexpr const Shape4<i64>& shape() const noexcept { return m_shape; }
        [[nodiscard]] bool is_layered() const noexcept { return is_layered(get()); }
        [[nodiscard]] constexpr bool is_empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !is_empty(); }

        // Attach the lifetime of the managed object with an alias.
        // Constructs a shared_ptr which shares ownership information with the managed object,
        // but holds an unrelated and unmanaged pointer alias. If the returned shared_ptr is
        // the last of the group to go out of scope, it will call the stored deleter for the
        // managed object of this instance. However, calling get() on this shared_ptr will always
        // return a copy of alias. It is the responsibility of the programmer to make sure that
        // alias remains valid as long as the managed object exists.
        template<typename T>
        [[nodiscard]] constexpr Shared<T[]> attach(T* alias) const noexcept { return {m_ptr, alias}; }

        // Releases the ownership of the managed array, if any.
        shared_type release() noexcept {
            m_shape = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        shared_type m_ptr{nullptr};
        Shape4<i64> m_shape{};
    };
}
