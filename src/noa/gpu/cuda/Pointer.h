/**
 * @file Pointer.h
 * @brief Simple pointer (for arithmetic types and void) holding memory on the host, pinned or device.
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include <string>
#include <type_traits>
#include <utility>      // std::exchange
#include <cstddef>      // size_t
#include <cstdlib>      // malloc, free
#include <cstring>      // std::memcpy

#include "noa/API.h"
#include "noa/util/Types.h"
#include "noa/util/Flag.h"
#include "noa/util/string/Format.h"     // String::format
#include "noa/util/traits/BaseTypes.h"  // Traits::is_complex_v

#include "noa/gpu/cuda/Memory.h"

namespace Noa::CUDA {

    /**
     * Holds a pointer pointing to some "arithmetic" type, usually representing a dynamic array.
     * Ownership:   Data can be owned, and ownership can be switched on and off at any time.
     *              Ownership implies:
     *                  1) the destructor will delete the pointer.
     *                  2) the copy constructor will perform a deep copy.
     * Location:    Pointer can allocate and deallocate memory on the host, on pinned memory or
     *              on the current device. Since the underlying pointer is not necessarily on the
     *              host, Pointer<T> will never dereference its underlying pointer. Use get() to
     *              retrieve a non-owning pointer.
     *
     * @tparam Type Type of the underlying pointer. Should be an non-const arithmetic such as defined
     *              by std::is_arithmetic or a non-cost std::complex<float|double|long double>.
     *              Arithmetics are all aligned to std::max_align_t (see discussion above), so the type
     *              is mostly here to add some type safety.
     *
     * @note        See the template specialization Pointer<void> below.
     */
    template<typename Type>
    class NOA_API Pointer {
    private:
        size_t m_size{};

        // The pointer. It is never de-referenced.
        // Use get() to retrieve an non-owning pointer.
        std::enable_if_t<(std::is_arith_v<Type> || Traits::is_complex_v<Type>) &&
                         !std::is_reference_v<Type> &&
                         !std::is_array_v<Type> &&
                         !std::is_const_v<Type>, Type*> m_ptr{nullptr};

        // To which memory the underlying pointer points at.
        // Should be accessible at any time, but not modifiable by the user.
        // Use toDevice(), toPinned() or toHost().
        Resource m_resource{};

    public:
        // Whether or not the underlying pointer is owned by this instance.
        // Can be changed at any time, since it only affects the copy ctor/assignment operator
        // and the dtor.
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to properly initialize the pointer. */
        Pointer() = default;

        /**
         * Allocates @a size elements of type @a Type.
         * @param[in] size      This is fixed for the life of the object. Use size() to access it.
         *                      The number of bytes allocated is (at least) equal to `size * sizeof(Type)`.
         * @param[in] resource  Resource to allocate from. This is fixed for the life of the object.
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner".
         *
         * @warning The allocation may fail and the underlying data can be a nullptr. As such, new
         *          instances should be checked, by using the bool operator or get().
         */
        Pointer(size_t size, Resource resource) noexcept
                : m_size(size), m_ptr(alloc_(size, resource)), m_resource(resource) {}

        /**
         * Creates an instance from an existing pointer.
         * @param[in] size      Number of @a Type elements in @a ptr.
         * @param[in] resource  Resource pointed by @a ptr.
         * @param[in] ptr       Pointer to hold on. If it is not a nullptr, it should correspond to
         *                      @a size and @a resource.
         * @param[in] own_ptr   Whether or not this new instance should own @a ptr.
         */
        Pointer(size_t size, Resource resource, Type* ptr, bool own_ptr = false)
                : m_size(size), m_ptr(ptr), m_resource(resource), is_owner(own_ptr) {}

        /**
         * Copy constructor.
         * @note    If @a ptr owns its data, performs a deep copy. The new instance will own the
         *          copied data. Otherwise, perform a shallow copy. In this case, the new instance
         *          will not own the data.
         */
        Pointer(const Pointer<Type>& ptr)
                : m_size(ptr.m_size), m_resource(ptr.m_resource), is_owner(ptr.is_owner) {
            if (is_owner && ptr.m_ptr)
                copy_(ptr.m_ptr);
            else
                m_ptr = ptr.m_ptr;
        }

        /**
         * Move constructor.
         * @note    @a ptr is left in an empty state (i.e. nullptr). It can technically be reset using reset(),
         *          but why should it?
         */
        Pointer(Pointer<Type>&& ptr) noexcept
                : m_size(ptr.m_size), m_resource(ptr.m_resource),
                  m_ptr(std::exchange(ptr.m_ptr, nullptr)), is_owner(ptr.is_owner) {}

        [[nodiscard]] inline constexpr Type* get() noexcept { return m_ptr; }
        [[nodiscard]] inline constexpr const Type* get() const noexcept { return m_ptr; }
        [[nodiscard]] inline constexpr Resource resource() const noexcept { return m_resource; }
        [[nodiscard]] inline constexpr size_t size() const noexcept { return m_size; }
        [[nodiscard]] inline constexpr size_t empty() const noexcept { return size() == 0; }
        [[nodiscard]] inline constexpr size_t bytes() const noexcept { return m_size * sizeof(Type); }
        [[nodiscard]] inline constexpr explicit operator bool() const noexcept { return m_ptr; }

        /**
         * Returns an owning Pointer<Type> with the same size and its underlying data on the device.
         * @param[in] intent    If it contains Intent::read, the data is copied, if not and if it
         *                      contains Intent::write, a simple allocation is performed. Otherwise,
         *                      returns a nullptr.
         * @note                If the underlying data is a nullptr, returns a nullptr.
         */
        inline Pointer<Type> toDevice(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return Pointer<Type>(m_size, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                Type* d_ptr;
                if (m_resource == Resource::host) {
                    d_ptr = static_cast<Type*>(GPU::Memory::copyHostToDevice(m_ptr, bytes()));
                } else if (m_resource == Resource::pinned) {
                    d_ptr = static_cast<Type*>(GPU::Memory::copyPinnedToDevice(m_ptr, bytes()));
                } else if (m_resource == Resource::device) {
                    d_ptr = static_cast<Type*>(GPU::Memory::copyDeviceToDevice(m_ptr, bytes()));
                }
                return Pointer<Type>(m_size, Resource::device, d_ptr, true);

            } else if (intent & Intent::write) {
                Type* d_ptr = static_cast<Type*>(GPU::Memory::alloc(bytes()));
                return Pointer<Type>(m_size, Resource::device, d_ptr, true);
            }
        }

        /**
         * Returns an owning Pointer<Type> with the same size and its underlying data on the host.
         * @param[in] intent    Same as toDevice().
         * @note                If the underlying data is a nullptr, returns a nullptr.
         */
        inline Pointer<Type> toHost(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return Pointer<Type>(m_size, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                Type* d_ptr{nullptr};
                if (m_resource == Resource::host) {
                    d_ptr = copy_(m_ptr, bytes());
                } else if (m_resource == Resource::pinned) {
                    d_ptr = static_cast<Type*>(GPU::Memory::copyPinnedToHost(m_ptr, bytes()));
                } else if (m_resource == Resource::device) {
                    d_ptr = static_cast<Type*>(GPU::Memory::copyDeviceToHost(m_ptr, bytes()));
                }
                return Pointer<Type>(m_size, Resource::host, d_ptr, true);

            } else if (intent & Intent::write) {
                Type* d_ptr = alloc_(m_size, m_resource));
                return Pointer<Type>(m_size, Resource::device, d_ptr, true);
            }
        }

        /**
         * Returns an owning Pointer<Type> with the same size and its underlying data on the pinned memory.
         * @param[in] intent    Same as toDevice().
         * @note                If the underlying data is a nullptr, returns a nullptr.
         */
        inline Pointer<Type> toPinned(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return Pointer<Type>(m_size, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                Type* d_ptr{nullptr};
                if (m_resource == Resource::host) {
                    d_ptr = static_cast<Type*>(GPU::Memory::copyHostToPinned(m_ptr, bytes()));
                } else if (m_resource == Resource::pinned) {
                    d_ptr = static_cast<Type*>(GPU::Memory::copyPinnedToPinned(m_ptr, bytes()));
                } else if (m_resource == Resource::device) {
                    d_ptr = static_cast<Type*>(GPU::Memory::copyDeviceToPinned(m_ptr, bytes()));
                }
                return Pointer<Type>(m_size, Resource::host, d_ptr, true);

            } else if (intent & Intent::write) {
                Type* d_ptr = static_cast<Type*>(GPU::Memory::allocPinned(bytes()));
                return Pointer<Type>(m_size, Resource::device, d_ptr, true);
            }
        }

        /** Clears the underlying data if necessary. */
        inline void reset() noexcept { dealloc_(); }

        /**
         * Resets the underlying data.
         * @param[in] size      Number of @a Type elements in @a ptr.
         * @param[in] resource  Resource pointed by @a ptr.
         * @param[in] ptr       Pointer to hold on. If it is not a nullptr, it should
         *                      correspond to @a size and @a resource.
         * @param[in] own_ptr   Whether or not this new instance should own @a ptr.
         */
        inline void reset(size_t size, Resource resource, Type* ptr, bool own_ptr = false) noexcept {
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
        inline Type* release() noexcept {
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
            return String::format("Size: {}, Resource: {}, Type: {}, Owner: {}, Address: {}",
                                  m_size, resource_to_string(m_resource),
                                  String::typeName<Type>(), is_owner, m_ptr);
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        ~Pointer() { dealloc_(); }

    private:
        // Allocates, either host, pinned or device memory. Otherwise, returns nullptr.
        static inline Type* alloc_(size_t size, Resource resource) noexcept {
            Type* out{nullptr};
            if (resource == Resource::host)
                out = new(std::nothrow) Type[size];
            else if (resource == Resource::pinned)
                out = static_cast<Type*>(GPU::Memory::allocPinned(size * sizeof(Type)));
            else if (resource == Resource::device)
                out = static_cast<Type*>(GPU::Memory::alloc(size * sizeof(Type)));
            return out;
        }

        // Copies the underlying data, preserving the size and the resource.
        inline Type* copy_() noexcept {
            Type* out{nullptr};

            if (m_resource & Resource::host) {
                out = new(std::nothrow) Type[bytes()]
                std::memcpy(out, m_ptr, bytes());
            } else if (m_resource & Resource::pinned) {
                out = static_cast<Type*>(GPU::Memory::copyPinnedToPinned(m_ptr, bytes()));
            } else if (m_resource & Resource::device) {
                out = static_cast<Type*>(GPU::Memory::copyDeviceToDevice(m_ptr, bytes()));
            }
            return out;
        }

        // Deallocates the underlying data, if any and if the instance is the owner.
        inline void dealloc_() noexcept {
            if (!is_owner || !m_ptr)
                return;

            if (m_resource & Resource::host)
                delete[] m_ptr;
            else if (m_resource & Resource::pinned)
                GPU::Memory::freePinned(m_ptr);
            else if (m_resource & Resource::device)
                GPU::Memory::free(m_ptr);
        }
    };


    /**
     * Holds a pointer pointing to some bytes. Mostly used for low-level stuff
     * or when keeping track of the type is not useful.
     *
     * @note    This is similar to Pointer<> above, but does not keep the pointer type.
     *          Its size is in bytes, as opposed to number of elements.
     */
    template<>
    class NOA_API Pointer<void> {
    private:
        size_t m_bytes{};
        void* m_ptr{nullptr};
        Resource m_resource{};

    public:
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to properly initialize the pointer. */
        Pointer() = default;

        /**
         * Allocates @a size bytes.
         * @param[in] bytes     This is fixed for the life of the object. Use size() or bytes() to access it.
         * @param[in] resource  Resource to allocate from. This is fixed for the life of the object.
         */
        Pointer(size_t bytes, Resource resource) noexcept
                : m_bytes(bytes), m_ptr(alloc_(bytes, resource)), m_resource(resource) {}

        /**
         * Creates an instance from an existing pointer.
         * @param[in] bytes     Number of bytes @a ptr.
         * @param[in] resource  Resource pointed by @a ptr.
         * @param[in] ptr       Pointer to hold on. If it is not a nullptr, it should correspond to
         *                      @a bytes and @a resource.
         * @param[in] own_ptr   Whether or not this new instance should own @a ptr.
         */
        Pointer(size_t bytes, Resource resource, void* ptr, bool own_ptr = false)
                : m_bytes(bytes), m_ptr(ptr), m_resource(resource), is_owner(own_ptr) {}

        Pointer(const Pointer<void>& ptr)
                : m_bytes(ptr.m_bytes), m_resource(ptr.m_resource), is_owner(ptr.is_owner) {
            if (is_owner && ptr.m_ptr)
                copy_(ptr.m_ptr);
            else
                m_ptr = ptr.m_ptr;
        }

        Pointer(Pointer<void>&& ptr) noexcept
                : m_bytes(ptr.m_bytes), m_ptr(std::exchange(ptr.m_ptr, nullptr)),
                  m_resource(ptr.m_resource), is_owner(ptr.is_owner) {}

        [[nodiscard]] inline constexpr void* get() noexcept { return m_ptr; }
        [[nodiscard]] inline constexpr const void* get() const noexcept { return m_ptr; }
        [[nodiscard]] inline constexpr Resource resource() const noexcept { return m_resource; }
        [[nodiscard]] inline constexpr size_t size() const noexcept { return m_bytes; }
        [[nodiscard]] inline constexpr size_t bytes() const noexcept { return m_bytes; }
        [[nodiscard]] inline constexpr explicit operator bool() const noexcept { return m_ptr; }

        /**
         * Returns an owning Pointer<void> with the same size and its underlying data on the device.
         * @param[in] intent    If it contains Intent::read, the data is copied, if not and if it
         *                      contains Intent::write, a simple allocation is performed. Otherwise,
         *                      returns a nullptr.
         * @note                If the underlying data is a nullptr, returns a nullptr.
         */
        inline Pointer<void> toDevice(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return Pointer<void>(m_bytes, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                void* d_ptr;
                if (m_resource == Resource::host) {
                    d_ptr = GPU::Memory::copyHostToDevice(m_ptr, m_bytes);
                } else if (m_resource == Resource::pinned) {
                    d_ptr = GPU::Memory::copyPinnedToDevice(m_ptr, m_bytes);
                } else if (m_resource == Resource::device) {
                    d_ptr = GPU::Memory::copyDeviceToDevice(m_ptr, m_bytes);
                }
                return Pointer<void>(m_bytes, Resource::device, d_ptr, true);

            } else if (intent & Intent::write) {
                void* d_ptr = GPU::Memory::alloc(m_bytes);
                return Pointer<void>(m_bytes, Resource::device, d_ptr, true);
            }
        }

        /**
         * Returns an owning Pointer<void> with the same size and its underlying data on the host.
         * @param[in] intent    Same as toDevice().
         * @note                If the underlying data is a nullptr, returns a nullptr.
         */
        inline Pointer<void> toHost(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return Pointer<void>(m_bytes, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                void* d_ptr{nullptr};
                if (m_resource == Resource::host) {
                    d_ptr = copy_(m_ptr, m_bytes);
                } else if (m_resource == Resource::pinned) {
                    d_ptr = GPU::Memory::copyPinnedToHost(m_ptr, m_bytes);
                } else if (m_resource == Resource::device) {
                    d_ptr = GPU::Memory::copyDeviceToHost(m_ptr, m_bytes);
                }
                return Pointer<void>(m_bytes, Resource::host, d_ptr, true);

            } else if (intent & Intent::write) {
                void* d_ptr = alloc_(m_bytes, m_resource));
                return Pointer<void>(m_bytes, Resource::device, d_ptr, true);
            }
        }

        /**
         * Returns an owning Pointer<void> with the same size and its underlying data on the pinned memory.
         * @param[in] intent    Same as toDevice().
         * @note                If the underlying data is a nullptr, returns a nullptr.
         */
        inline Pointer<void> toPinned(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return Pointer<void>(m_bytes, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                void* d_ptr{nullptr};
                if (m_resource == Resource::host) {
                    d_ptr = GPU::Memory::copyHostToPinned(m_ptr, m_bytes);
                } else if (m_resource == Resource::pinned) {
                    d_ptr = GPU::Memory::copyPinnedToPinned(m_ptr, m_bytes);
                } else if (m_resource == Resource::device) {
                    d_ptr = GPU::Memory::copyDeviceToPinned(m_ptr, m_bytes);
                }
                return Pointer<void>(m_bytes, Resource::host, d_ptr, true);

            } else if (intent & Intent::write) {
                void* d_ptr = GPU::Memory::allocPinned(m_bytes);
                return Pointer<void>(m_bytes, Resource::device, d_ptr, true);
            }
        }

        /** Clears the underlying data if necessary. */
        inline void reset() noexcept { dealloc_(); }

        /**
         * Resets the underlying data.
         * @param[in] bytes     Number bytes in @a ptr.
         * @param[in] resource  Resource pointed by @a ptr.
         * @param[in] ptr       Pointer to hold on. If it is not a nullptr, it should
         *                      correspond to @a bytes and @a resource.
         * @param[in] own_ptr   Whether or not this new instance should own @a ptr.
         */
        inline void reset(size_t bytes, Resource resource, void* ptr, bool own_ptr = false) noexcept {
            dealloc_();
            m_bytes = bytes;
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
            return String::format("Size: {}, Resource: {}, Type: void, Owner: {}, Address: {}",
                                  m_bytes, resource_to_string(m_resource), is_owner, m_ptr);
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        ~Pointer() { dealloc_(); }

    private:
        // Allocates, either host, pinned or device memory. Otherwise, returns nullptr.
        static inline void* alloc_(size_t bytes, Resource resource) noexcept {
            void* out{nullptr};
            if (resource == Resource::host)
                out = std::malloc(bytes);
            else if (resource == Resource::pinned)
                out = GPU::Memory::allocPinned(bytes);
            else if (resource == Resource::device)
                out = GPU::Memory::alloc(bytes);
            return out;
        }

        // Copies the underlying data, preserving the size and the resource.
        inline void* copy_() noexcept {
            void* out{nullptr};

            if (m_resource & Resource::host) {
                out = std::malloc(m_bytes);
                std::memcpy(out, m_ptr, m_bytes);
            } else if (m_resource & Resource::pinned) {
                out = GPU::Memory::copyPinnedToPinned(m_ptr, m_bytes);
            } else if (m_resource & Resource::device) {
                out = GPU::Memory::copyDeviceToDevice(m_ptr, m_bytes);
            }
            return out;
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
