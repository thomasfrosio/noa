/// \file noa/gpu/cuda/memory/PtrArray.h
/// \brief Hold CUDA arrays.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

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

namespace noa::cuda::memory {
    /// A ND CUDA array of integers (excluding (u)int64_t), float or cfloat_t.
    /// \note cfloat_t has the same channel descriptor as the CUDA built-in vector float2.
    template<typename Type>
    class PtrArray {
    public:
        struct Deleter {
            void operator()(cudaArray* array) noexcept {
                cudaFreeArray(array);
            }
        };

    public: // static functions
        /// Allocates a CUDA array.
        /// \param shape    Rightmost shape of the array.
        /// \param flag     Any flag supported by cudaMalloc3DArray().
        /// \return Pointer to the CUDA array.
        static std::unique_ptr<cudaArray, Deleter> alloc(size3_t shape, uint flag) {
            cudaExtent extent{};
            const size_t ndim = shape.ndim();
            if (ndim == 3) {
                extent.width = shape[2];
                extent.height = shape[1];
                extent.depth = shape[0];
            } else if (ndim == 2) {
                extent.width = shape[2];
                extent.height = shape[1];
            } else {
                extent.width = shape[2];
            }
            cudaArray* ptr;
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<Type>();
            NOA_THROW_IF(cudaMalloc3DArray(&ptr, &desc, extent, flag));
            return {ptr, Deleter{}};
        }

    public: // member functions
        /// Creates an empty instance. Use one of the operator assignment to allocate new data.
        constexpr PtrArray() = default;
        constexpr /*implicit*/ PtrArray(std::nullptr_t) {}

        /// Allocates a ND CUDA array with a given rightmost \p shape on the current device using \c cudaMalloc3DArray.
        explicit PtrArray(size3_t shape, uint flags = cudaArrayDefault)
                : m_ptr(alloc(shape, flags)), m_shape(shape) {}

    public:
        /// Returns the CUDA array pointer.
        [[nodiscard]] constexpr cudaArray* get() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const cudaArray* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr cudaArray* data() noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr const cudaArray* data() const noexcept { return m_ptr.get(); }

        /// Returns a reference of the shared object.
        [[nodiscard]] constexpr std::shared_ptr<cudaArray>& share() noexcept { return m_ptr; }
        [[nodiscard]] constexpr const std::shared_ptr<cudaArray>& share() const noexcept { return m_ptr; }

        /// Attach the lifetime of the managed object with an \p alias.
        /// \details Constructs a shared_ptr which shares ownership information with the managed object,
        ///          but holds an unrelated and unmanaged pointer \p alias. If the returned shared_ptr is
        ///          the last of the group to go out of scope, it will call the stored deleter for the
        ///          managed object of this instance. However, calling get() on this shared_ptr will always
        ///          return a copy of \p alias. It is the responsibility of the programmer to make sure that
        ///          \p alias remains valid as long as the managed object exists.
        template<typename U>
        [[nodiscard]] constexpr std::shared_ptr<U[]> attach(U* alias) const noexcept { return {m_ptr, alias}; }

        /// Returns the number of bytes of the underlying CUDA array.
        [[nodiscard]] constexpr size_t bytes() const noexcept { return elements() * sizeof(Type); }

        /// Returns the number of elements of the underlying CUDA array.
        [[nodiscard]] constexpr size_t elements() const noexcept { return m_shape.elements(); }
        [[nodiscard]] constexpr size_t size() const noexcept { return elements(); }

        /// Returns the shape, in elements, of the underlying CUDA array.
        [[nodiscard]] constexpr size3_t shape() const noexcept { return m_shape; }

        /// Whether or not the managed object points to some data.
        [[nodiscard]] constexpr bool empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !empty(); }

        /// Releases the ownership of the managed array, if any.
        std::shared_ptr<cudaArray> release() noexcept {
            m_shape = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        static_assert(std::is_same_v<Type, int32_t> || std::is_same_v<Type, uint32_t> ||
                      std::is_same_v<Type, float> || std::is_same_v<Type, cfloat_t>);
        std::shared_ptr<cudaArray> m_ptr{nullptr};
        size3_t m_shape{};
    };
}
