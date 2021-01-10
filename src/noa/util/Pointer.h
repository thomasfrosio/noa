/**
 * @file Pointer.h
 * @brief Simple pointer (for arithmetic types) holding memory on the host, pinned or device.
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/traits/Base.h"
#include "noa/util/IntX.h"
#include "noa/util/string/Format.h"
#include "noa/gpu/Memory.h"


namespace Noa {
    /**
     * Holds a pointer pointing to data, usually representing a dynamic array.
     * Ownership:   Data can be owned, and ownership can be switched on and off at any time.
     *              Ownership implies:
     *                  1) the destructor will delete the pointer.
     *                  2) the copy constructor will perform a deep copy.
     * Location:    Pointer can allocate and deallocate memory on the host, on pinned memory or
     *              on the current device. Since the underlying pointer is not necessarily on the
     *              host, Pointer<T> will never dereference its underlying pointer. Use get() to
     *              retrieve a non-owning pointer.
     * @tparam Type Type of the underlying pointer. Should be an arithmetic, e.g. float, int,
     *              std::complex<float>, etc. Arithmetics are all aligned to std::max_align_t, so
     *              the type is mostly here to add type safety.
     */
    template<typename Type>
    class NOA_API Pointer {
    private:
        // Shape of the underlying data. It is only used to compute the number of elements to
        // allocate, so the order or meaning of this shape doesn't strictly matter.
        // The order of the data is not forced, since the allocators will allocate 1D "linear" memory.
        using Shape = Int3<size_t>;
        Shape m_shape{0, 0, 0};

        // The pointer. It is never de-referenced.
        // Use get() to retrieve an non-owning pointer.
        std::enable_if_t<Noa::Traits::is_arith_v<Type> &&
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
         * Allocates data with a given @a shape.
         * @param[in] shape     3D shape. This is fixed for the life of the object. Use shape() to access it.
         *                      The number of bytes allocated is (at least) equal to shape.prod() * sizeof(T).
         * @param[in] resource  Resource to allocate from. This is fixed for the life of the object.
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner".
         *
         * @warning The allocation may fail and the underlying data can be a nullptr. As such, new
         *          instances should be checked, by using the bool operator or get().
         */
        explicit Pointer(Shape shape, Resource resource) noexcept
                : m_shape(shape), m_ptr(alloc_(shape.prod(), resource)), m_resource(resource) {}


        /**
         * Creates an instance from an existing pointer.
         * @param[in] shape     3D shape.
         * @param[in] resource  Resource pointed by @a ptr.
         * @param[in] ptr       Pointer to hold on. If it is not a nullptr, it should correspond to
         *                      @a shape (its length should be at least shape.prod() * sizeof(T))
         *                      and @a resource.
         * @param[in] own_ptr   Whether or not this new instance should own @a ptr.
         */
        Pointer(Shape shape, Resource resource, Type* ptr, bool own_ptr = false)
                : m_shape(shape), m_ptr(ptr), m_resource(resource), is_owner(own_ptr) {}


        /**
         * Copy constructor.
         * @note    If @a ptr owns its data, performs a deep copy. The new instance will own the
         *          copied data. Otherwise, perform a shallow copy. In this case, the new instance
         *          will not own the data.
         */
        Pointer(const Pointer<Type>& ptr)
                : m_shape(ptr.m_shape), m_resource(ptr.m_resource), is_owner(ptr.is_owner) {
            if (is_owner && ptr.m_ptr)
                copy_(ptr.m_ptr);
            else
                m_ptr = ptr.m_ptr;
        }


        /**
         * Move constructor.
         * @note    @a ptr is left in an empty state (nullptr, shape=0). It can technically be reset
         *          using reset(), but why should it?
         */
        Pointer(Pointer<Type>&& ptr) noexcept
                : m_shape(ptr.m_shape), m_resource(ptr.m_resource),
                  m_ptr(std::exchange(ptr.m_ptr, nullptr)), is_owner(ptr.is_owner) {}


        [[nodiscard]] Type* get() noexcept { return m_ptr; }
        [[nodiscard]] const Type* get() const noexcept { return m_ptr; }
        [[nodiscard]] Resource resource() const noexcept { return m_resource; }
        [[nodiscard]] inline constexpr Shape shape() const noexcept { return m_shape; }
        [[nodiscard]] inline constexpr size_t size() const noexcept { return m_shape.prod(); }


        /**
         * Number of bytes corresponding to the Pointer.
         * @warning It only corresponds to the number of bytes of the underlying data if and only if
         *          the underlying data is NOT a nullptr.
         */
        [[nodiscard]] inline constexpr size_t bytes() const noexcept {
            return m_shape.prod() * sizeof(Type);
        }

        /** Whether or not the underlying data is a nullptr. */
        explicit operator bool() const noexcept {
            return m_ptr;
        }

        /**
         * Returns an owning Pointer<T> with the same shape and its underlying data on the device.
         * @param[in] intent    If it contains Intent::read, the data is copied, if not and if it
         *                      contains Intent::write, a simple allocation is performed. Otherwise,
         *                      returns a nullptr.
         * @note                If the underlying data is a nullptr, returns a nullptr.
         */
        inline Pointer<Type> toDevice(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return Pointer<Type>(m_shape, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                Type* d_ptr;
                if (m_resource == Resource::host) {
                    d_ptr = GPU::Memory::copyHostToDevice(m_ptr, bytes());
                } else if (m_resource == Resource::pinned) {
                    d_ptr = GPU::Memory::copyPinnedToDevice(m_ptr, bytes());
                } else if (m_resource == Resource::device) {
                    d_ptr = GPU::Memory::copyDeviceToDevice(m_ptr, bytes());
                }
                return Pointer<Type>(m_shape, Resource::device, d_ptr, true);

            } else if (intent & Intent::write) {
                Type* d_ptr = GPU::Memory::alloc(bytes());
                return Pointer<Type>(m_shape, Resource::device, d_ptr, true);
            }
        }


        /**
         * Returns an owning Pointer<T> with the same shape and its underlying data on the host.
         * @param[in] intent    If it contains Intent::read, the data is copied, if not and if it
         *                      contains Intent::write, a simple allocation is performed. Otherwise,
         *                      returns a nullptr.
         * @note                If the underlying data is a nullptr, returns a nullptr.
         */
        inline Pointer<Type> toHost(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return Pointer<Type>(m_shape, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                Type* d_ptr{nullptr};
                if (m_resource == Resource::host) {
                    d_ptr = copy_(m_ptr, bytes());
                } else if (m_resource == Resource::pinned) {
                    d_ptr = GPU::Memory::copyPinnedToHost(m_ptr, bytes());
                } else if (m_resource == Resource::device) {
                    d_ptr = GPU::Memory::copyDeviceToHost(m_ptr, bytes());
                }
                return Pointer<Type>(m_shape, Resource::host, d_ptr, true);

            } else if (intent & Intent::write) {
                Type* d_ptr = alloc_(bytes());
                return Pointer<Type>(m_shape, Resource::device, d_ptr, true);
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
        inline Pointer<Type> toPinned(Flag<Intent> intent) noexcept {
            if (!m_ptr)
                return Pointer<Type>(m_shape, Resource::device, nullptr, true);

            if (intent & Intent::read) {
                Type* d_ptr{nullptr};
                if (m_resource == Resource::host) {
                    d_ptr = GPU::Memory::copyHostToPinned(m_ptr, bytes());
                } else if (m_resource == Resource::pinned) {
                    d_ptr = GPU::Memory::copyPinnedToPinned(m_ptr, bytes());
                } else if (m_resource == Resource::device) {
                    d_ptr = GPU::Memory::copyDeviceToPinned(m_ptr, bytes());
                }
                return Pointer<Type>(m_shape, Resource::host, d_ptr, true);

            } else if (intent & Intent::write) {
                Type* d_ptr = GPU::Memory::allocPinned(bytes());
                return Pointer<Type>(m_shape, Resource::device, d_ptr, true);
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
        inline void reset(Shape shape, Resource resource,
                          Type* ptr, bool own_ptr = false) noexcept {
            dealloc_();
            m_shape = shape;
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
            return String::format("Shape: {}, Resource: {}, Type: {}, Owner: {}, Address: {}",
                                  m_shape, resource_to_string(m_resource),
                                  String::typeName<Type>(), is_owner, m_ptr);
        }


        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        ~Pointer() {
            dealloc_();
        }

    private:
        // Allocates, either host, pinned or device memory. Otherwise, returns nullptr.
        inline Type* alloc_(size_t size, Resource resource) noexcept {
            if (resource == Resource::host)
                return new(std::nothrow) Type[size];
            else if (resource == Resource::pinned)
                return reinterpret_cast<Type*>(GPU::Memory::allocPinned(size * sizeof(Type)));
            else if (resource == Resource::device)
                return reinterpret_cast<Type*>(GPU::Memory::alloc(size * sizeof(Type)));
            else
                return nullptr;
        }


        // Copies the underlying data, preserving the shape and the resource.
        inline Type* copy_() noexcept {
            Type* destination{nullptr};
            size_t bytes = m_shape.prod() * sizeof(Type);

            if (m_resource & Resource::host) {
                destination = new(std::nothrow) Type[bytes]
                std::memcpy(destination, m_ptr, bytes);
            } else if (m_resource & Resource::pinned) {
                destination = GPU::Memory::copyPinnedToPinned(m_ptr, bytes);
            } else if (m_resource & Resource::device) {
                destination = GPU::Memory::copyDeviceToDevice(m_ptr, bytes);
            }
            return destination;
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
}
