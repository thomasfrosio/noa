/**
 * @file Pointer.h
 * @brief Opaque pointer holding memory on the host, pinned or device.
 * @author Thomas - ffyr2w
 * @date 07 Jan 2021
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/Flag.h"
#include "noa/gpu/Memory.h"


namespace Noa {
    /**
     *
     */
    class PointerByte {
    private:
        // Size, in bytes. Should correspond to the number of allocated bytes.
        size_t m_size{};

        // Underlying data.
        void* m_ptr{nullptr};

        // To which memory the underlying pointer points at.
        // Should be accessible at any time, but not modifiable by the user.
        // Use toDevice(), toPinned() or toHost().
        Resource m_resource{};

    public:
        // Whether or not the underlying pointer is owned by this instance.
        // Can be changed at any time, since it only affects the copy constructor/assignment operator
        // and the destructor.
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to properly initialize the pointer. */
        PointerByte() = default;


        /**
         * Allocates data with a given @a shape.
         * @param[in] shape     3D shape. This is fixed for the life of the object. Use shape() to access it.
         *                      The number of bytes (linearly) allocated is equal to shape.prod() * sizeof(T).
         * @param[in] resource  Resource to allocate from. This is fixed for the life of the object.
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner".
         */
        explicit PointerByte(size_t size, Resource resource)
                : m_size(size), m_ptr(alloc_(size, resource)), m_resource(resource) {}


        /**
         * Creates an instance from an existing pointer.
         * @param[in] shape     3D shape.
         * @param[in] resource  Resource pointed by @a ptr.
         * @param[in] ptr       PointerByte to hold on. If it is not a nullptr, it should correspond to
         *                      @a shape (its length should be shape.prod() * sizeof(T)) and @a resource.
         * @param[in] own_ptr   Whether or not this new instance should own @a ptr.
         */
        PointerByte(size_t size, Resource resource, void* ptr, bool own_ptr = false)
                : m_size(size), m_ptr(ptr), m_resource(resource), is_owner(own_ptr) {}


        /**
         * Copy constructor.
         * @note    If @a ptr owns its data, performs a deep copy. The new instance will own the
         *          copied data. Otherwise, perform a shallow copy. In this case, the new instance
         *          will not own the data.
         */
        PointerByte(const PointerByte& ptr)
                : m_size(ptr.m_size), m_resource(ptr.m_resource), is_owner(ptr.is_owner) {
            if (is_owner && ptr.m_ptr)
                copy_(ptr.m_ptr);
            else
                m_ptr = ptr.m_ptr;
        }


        /**
         * Move constructor.
         * @note    @a arr is left in an empty state (nullptr, shape=0). It can technically be reset
         *          using reset(), but why should it?
         */
        PointerByte(PointerByte&& ptr) noexcept
                : m_size(ptr.m_size), m_ptr(std::exchange(ptr.m_ptr, nullptr)),
                  m_resource(ptr.m_resource), is_owner(ptr.is_owner) {}


        [[nodiscard]] void* get() noexcept { return m_ptr; }
        [[nodiscard]] const void* get() const noexcept { return m_ptr; }
        [[nodiscard]] Resource resource() const noexcept { return m_resource; }
        [[nodiscard]] inline constexpr size_t size() const noexcept { return m_size; }
        [[nodiscard]] inline constexpr size_t bytes() const noexcept { return m_size; }
        inline explicit operator bool() const noexcept { return m_ptr; }

        /**
         * Returns an owning Pointer<T> with the same shape and its underlying data on the device.
         * @param[in] intent    If it contains Intent::read, the data is copied, if not and if it
         *                      contains Intent::write, a simple allocation is performed. Otherwise,
         *                      returns a nullptr.
         * @note                If the underlying data is a nullptr, returns a nullptr.
         */
        inline PointerByte toDevice(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return PointerByte(m_size, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                void* d_ptr;
                if (m_resource == Resource::host) {
                    d_ptr = GPU::Memory::copyHostToDevice(m_ptr, m_size);
                } else if (m_resource == Resource::pinned) {
                    d_ptr = GPU::Memory::copyPinnedToDevice(m_ptr, m_size);
                } else if (m_resource == Resource::device) {
                    d_ptr = GPU::Memory::copyDeviceToDevice(m_ptr, m_size);
                }
                return PointerByte(m_size, Resource::device, d_ptr, true);

            } else if (intent & Intent::write) {
                void* d_ptr = GPU::Memory::alloc(bytes());
                return PointerByte(m_size, Resource::device, d_ptr, true);
            }
        }


        /**
         * Returns an owning Pointer<T> with the same shape and its underlying data on the host.
         * @param[in] intent    If it contains Intent::read, the data is copied, if not and if it
         *                      contains Intent::write, a simple allocation is performed. Otherwise,
         *                      returns a nullptr.
         * @note                If the underlying data is a nullptr, returns a nullptr.
         */
        inline PointerByte toHost(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return PointerByte(m_size, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                void* d_ptr{nullptr};
                if (m_resource == Resource::host) {
                    d_ptr = copy_(m_ptr, m_size);
                } else if (m_resource == Resource::pinned) {
                    d_ptr = GPU::Memory::copyPinnedToHost(m_ptr, m_size);
                } else if (m_resource == Resource::device) {
                    d_ptr = GPU::Memory::copyDeviceToHost(m_ptr, m_size);
                }
                return PointerByte(m_size, Resource::host, d_ptr, true);

            } else if (intent & Intent::write) {
                void* d_ptr = alloc_(m_size);
                return PointerByte(m_size, Resource::device, d_ptr, true);
            }
        }


        /**
         * Returns an owning Pointer<T> with the same shape and its underlying data on the
         * page-locked (host) memory.
         * @param[in] intent    If it contains Intent::read, the data is copied, if not and if it
         *                      contains Intent::write, a simple allocation is performed. Otherwise,
         *                      returns a nullptr.
         * @note                If the underlying data is a nullptr, returns a nullptr.
         */
        inline PointerByte toPinned(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return PointerByte(m_shape, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                void* d_ptr{nullptr};
                if (m_resource == Resource::host) {
                    d_ptr = GPU::Memory::copyHostToPinned(m_ptr, m_size);
                } else if (m_resource == Resource::pinned) {
                    d_ptr = GPU::Memory::copyPinnedToPinned(m_ptr, m_size);
                } else if (m_resource == Resource::device) {
                    d_ptr = GPU::Memory::copyDeviceToPinned(m_ptr, m_size);
                }
                return PointerByte(m_size, Resource::host, d_ptr, true);

            } else if (intent & Intent::write) {
                void* d_ptr = GPU::Memory::allocPinned(m_size);
                return PointerByte(m_size, Resource::device, d_ptr, true);
            }
        }


        /** Clears the underlying data if necessary. */
        inline void reset() noexcept {
            dealloc_();
        }


        /**
         * Resets the underlying data.
         * @param[in] shape     3D shape.
         * @param[in] resource  Resource pointed by @a ptr.
         * @param[in] ptr       Pointer to hold on. If it is not a nullptr, it should correspond to
         *                      @a shape (its length should be shape.prod() * sizeof(T)) and @a resource.
         * @param[in] own_ptr   Whether or not this new instance should own @a ptr.
         */
        inline void
        reset(size_t size, Resource resource, void* ptr, bool own_ptr = false) noexcept {
            dealloc_();
            m_size = size;
            m_resource = resource;
            m_ptr = ptr;
            is_owner = own_ptr;
        }


        /**
         * If the current instance is an owner, releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call.
         */
        inline void* release() noexcept {
            is_owner = false;
            return std::exchange(m_ptr, nullptr);
        }


        /** Returns a human-readable description of the Pointer. */
        [[nodiscard]] inline std::string toString() const {
            auto resource_to_string = [](Resource resource) -> const char* {
                if (resource == Resource::host)
                    return "host";
                else if (resource == Resource::pinned)
                    return "pinned";
                else if (resource == Resource::device)
                    return "device";
            };
            return fmt::format("Shape: {}, Resource: {}, Type: void, Owner: {}, Address: {}",
                               m_size, resource_to_string(m_resource), is_owner, m_ptr);
        }


        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        ~PointerByte() {
            dealloc_();
        }

    private:
        // Allocates, either host, pinned or device memory. Otherwise, returns nullptr.
        static inline void* alloc_(size_t size, Resource resource) noexcept {
            if (resource == Resource::host)
                return std::malloc(size);
            else if (resource == Resource::pinned)
                return GPU::Memory::allocPinned(size);
            else if (resource == Resource::device)
                return GPU::Memory::alloc(size);
            else
                return nullptr;
        }

        // Copies the underlying data, preserving the shape and the resource.
        inline void* copy_() noexcept {
            void* destination{nullptr};
            if (m_resource & Resource::host) {
                destination = std::malloc(m_size);
                std::memcpy(destination, m_ptr, m_size);
            } else if (m_resource & Resource::pinned) {
                destination = GPU::Memory::copyPinnedToPinned(m_ptr, m_size);
            } else if (m_resource & Resource::device) {
                destination = GPU::Memory::copyDeviceToDevice(m_ptr, m_size);
            }
            return destination;
        }


        // Deallocates the underlying data, if any and if the instance is the owner.
        inline void dealloc_() noexcept {
            if (!is_owner || !m_ptr)
                return;

            if (m_resource & Resource::host)
                std::free(m_ptr);
            else if (m_resource & Resource::pinned)
                GPU::Memory::freePinned(m_ptr);
            else if (m_resource & Resource::device)
                GPU::Memory::free(m_ptr);
        }
    };
}
