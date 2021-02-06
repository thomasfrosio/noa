#pragma once

#include <type_traits>
#include <string>

#include "noa/gpu/Base.h"
#include "noa/util/string/Format.h"

namespace Noa::CUDA {

    /**
     * Manages a page-locked pointer. This object cannot be created, copied or deleted on the device.
     * @tparam Type     Type of the underlying pointer. Should be an integer, float, double, cfloat_t or cdouble_t.
     * @throw           @c Noa::Exception, if an error occurs when the device data is allocated, copied or freed.
     *
     * @note    Page-locked memory is accessible to the device. The driver tracks the virtual memory ranges allocated
     *          by cudaMallocHost and automatically accelerates calls to functions such as ::cudaMemcpy*(). Since the
     *          memory can be accessed directly by the device, it can be read or written with much higher bandwidth
     *          than pageable memory obtained with functions such as ::malloc(). Allocating excessive amounts of memory
     *          with ::cudaMallocHost() may degrade system performance, since it reduces the amount of memory available
     *          to the system for paging. As a result, PtrPinned is best used sparingly to allocate staging areas for
     *          data exchange between host and device.
     */
    template<typename Type>
    class PtrPinned {
    private:
        size_t m_elements{};
        std::enable_if_t<Traits::is_data_v<Type> && !std::is_reference_v<Type> &&
                         !std::is_array_v<Type> && !std::is_const_v<Type>,
                         Type*> m_pinned_ptr{nullptr};

    public:
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to properly initialize the pointer. */
        NOA_HOST PtrPinned() = default;

        /**
         * Allocates @a elements elements of type @a Type on pinned memory using @c cudaMallocHost.
         * @param elements  This is fixed for the life of the object. Use elements() to access it.
         *                  The number of bytes allocated is (at least) equal to `elements * sizeof(Type)`, see bytes().
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner".
         *
         * @throw   @c Noa::Exception, if allocation fails.
         */
        NOA_HOST explicit PtrPinned(size_t elements) : m_elements(elements), m_pinned_ptr(alloc_()) {}

        /**
         * Creates an instance from an existing pointer.
         * @param elements  Number of @a Type elements in @a ptr.
         * @param[in] ptr   Pinned pointer to hold on. If it is not a nullptr, it should correspond to @a elements.
         * @param owner     Whether or not this new instance should own @a ptr.
         */
        NOA_HOST PtrPinned(size_t elements, Type* pinned_ptr, bool owner = false) noexcept
                : m_elements(elements), m_pinned_ptr(pinned_ptr), is_owner(owner) {}

        /**
         * Copy constructor.
         * @note    If @a dev_ptr owns its data, performs a deep copy. The new instance will own the
         *          copied data. Otherwise, perform a shallow copy. In this case, the new instance
         *          will not own the data.
         */
        NOA_HOST PtrPinned(const PtrPinned<Type>& to_copy)
                : m_elements(to_copy.m_elements), is_owner(to_copy.is_owner) {
            if (is_owner && to_copy.m_pinned_ptr)
                copy_(m_pinned_ptr, to_copy.m_pinned_ptr);
            else
                m_pinned_ptr = to_copy.m_pinned_ptr;
        }

        /**
         * Move constructor.
         * @note    @a ptr is left in an empty state (i.e. nullptr). It can technically be reset using reset(),
         *          but why should it?
         */
        NOA_HOST PtrPinned(PtrPinned<Type>&& to_move) noexcept
                : m_elements(to_move.m_elements),
                  m_pinned_ptr(std::exchange(to_move.m_pinned_ptr, nullptr)),
                  is_owner(to_move.is_owner) {}

        /** Copy assignment operator. */
        NOA_HOST PtrPinned<Type>& operator=(const PtrPinned<Type>& to_copy) {
            m_elements = to_copy.m_elements;
            m_pinned_ptr = (is_owner && to_copy.m_pinned_ptr) ? copy_(to_copy.m_pinned_ptr) : to_copy.m_pinned_ptr;
            is_owner = to_copy.is_owner;
            return *this;
        }

        /** Move assignment operator. */
        NOA_IHD PtrPinned<Type>& operator=(PtrPinned<Type>&& to_move) noexcept {
            m_elements = to_move.m_elements;
            m_pinned_ptr = std::exchange(to_move.m_pinned_ptr, nullptr);
            is_owner = to_move.is_owner;
            return *this;
        }

        [[nodiscard]] NOA_FHD constexpr Type* get() noexcept { return m_pinned_ptr; }
        [[nodiscard]] NOA_FHD constexpr const Type* get() const noexcept { return m_pinned_ptr; }
        [[nodiscard]] NOA_FHD constexpr Type* data() noexcept { return m_pinned_ptr; }
        [[nodiscard]] NOA_FHD constexpr const Type* data() const noexcept { return m_pinned_ptr; }

        [[nodiscard]] NOA_FHD constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] NOA_FHD constexpr size_t empty() const noexcept { return m_elements() == 0 && m_pinned_ptr; }
        [[nodiscard]] NOA_FHD constexpr size_t bytes() const noexcept { return m_elements * sizeof(Type); }
        [[nodiscard]] NOA_FHD constexpr explicit operator bool() const noexcept { return m_pinned_ptr; }

        /** Clears the underlying data if necessary. */
        NOA_HOST void reset() { dealloc_(); }

        /**
         * Resets the underlying data.
         * @param elements          Number of @a Type elements in @a pinned_ptr.
         * @param[in] pinned_ptr    Pinned pointer to hold on. If it is not a nullptr, it should correspond to @a elements.
         * @param owner             Whether or not this new instance should own @a pinned_ptr.
         */
        NOA_HOST void reset(size_t elements, Type* pinned_ptr, bool owner = false) {
            dealloc_();
            m_elements = elements;
            m_pinned_ptr = pinned_ptr;
            is_owner = owner;
        }

        /**
         * If the current instance is an owner, releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call.
         */
        NOA_IHD Type* release() noexcept {
            is_owner = false;
            return std::exchange(m_pinned_ptr, nullptr);
        }

        /** Returns a human-readable description of the underlying data. */
        [[nodiscard]] NOA_HOST std::string toString() const {
            return String::format("Elements: {}, Resource: pinned, Type: {}, Owner: {}, Address: {}",
                                  m_elements, String::typeName<Type>(), is_owner, m_pinned_ptr);
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        NOA_HOST ~PtrPinned() { dealloc_(); }

    private:
        // Allocates device memory. m_elements should be set.
        NOA_HOST void alloc_() {
            NOA_THROW_IF(cudaMallocHost(&m_pinned_ptr, bytes()));
        }

        // Copies the underlying data. m_elements should be set.
        NOA_HOST Type* copy_(Type* dest, Type* src) {
            NOA_THROW_IF(cudaMemcpy(&dest, &src, bytes(), cudaMemcpyHostToHost));
        }

        // Deallocates the underlying data, if any and if the instance is the owner.
        NOA_HOST void dealloc_() {
            if (!is_owner || !m_pinned_ptr)
                return;

            NOA_THROW_IF(cudaFreeHost(m_pinned_ptr));
        }
    };
}
