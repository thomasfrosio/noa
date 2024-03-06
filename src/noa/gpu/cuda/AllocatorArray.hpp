#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
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
//
// Notes:
//  - They are cached-optimized for 2D/3D spatial locality.
//  - Surfaces and textures can be bound to same CUDA array.
//  - They are mostly used when the content changes rarely.
//    Although reusing them with cudaMemcpy is possible and surfaces can write to it.

// Add specialization for our complex types.
// Used for CUDA arrays and textures.
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::f16>() { return cudaCreateChannelDesc<half>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c16>() { return cudaCreateChannelDesc<half2>(); }
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc<noa::c32>() { return cudaCreateChannelDesc<float2>(); }

namespace noa::cuda {
    struct AllocatorArrayDeleter {
        void operator()(cudaArray* array) const noexcept {
            [[maybe_unused]] const cudaError_t err = cudaFreeArray(array);
            NOA_ASSERT(err == cudaSuccess);
        }
    };

    // A {1|2|3}d cuda array of i32, u32, f32, or c32.
    template<typename Value>
    class AllocatorArray {
    public:
        static_assert(nt::is_any_v<Value, i32, u32, f32, c32>);
        using value_type = Value;
        using deleter_type = AllocatorArrayDeleter;
        using shared_type = std::shared_ptr<cudaArray>;
        using unique_type = std::unique_ptr<cudaArray, deleter_type>;

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
            check(all(shape > 0) && shape[is_layered] == 1,
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
            u_extent += Shape3<size_t>::from_vec(u_extent == 0); // set empty dimensions to 1

            // Column vectors are "lost" in the conversion.
            // 1D extents are interpreted as row vectors.
            auto shape = Shape4<i64>::from_values(1, 1, u_extent[1], u_extent[2]);
            shape[!is_layered] = static_cast<i64>(u_extent[0]);
            return shape;
        }

        static auto info(cudaArray* array) {
            cudaChannelFormatDesc desc{};
            cudaExtent extent{};
            u32 flags{};
            check(cudaArrayGetInfo(&desc, &extent, &flags, array));
            return std::tuple<cudaChannelFormatDesc, cudaExtent, u32>(desc, extent, flags);
        }

        static bool is_layered(cudaArray* array) {
            const auto [desc_, extent_, flags] = info(array);
            // Not sure whether the flags are mutually exclusive, so just check the bit for layered textures.
            return flags & cudaArrayLayered;
        }

        static unique_type allocate(const Shape4<i64>& shape, u32 flag = cudaArrayDefault) {
            const cudaExtent extent = shape2extent(shape, flag & cudaArrayLayered);

            cudaArray* ptr{};
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<value_type>();
            check(cudaMalloc3DArray(&ptr, &desc, extent, flag));
            return unique_type{ptr};
        }
    };
}
#endif
