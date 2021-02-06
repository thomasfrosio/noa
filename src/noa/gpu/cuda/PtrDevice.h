#pragma once

#include <type_traits>
#include <string>

#include "noa/gpu/Base.h"
#include "noa/util/string/Format.h"

namespace Noa::CUDA {
    /**
     * Manages a device pointer from the host side. This object cannot be created, copied or deleted on the device,
     * however most utility functions can be access from the device.
     * @tparam Type     Type of the underlying pointer. Should be an integer, float, double, cfloat_t or cdouble_t.
     * @throw           @c Noa::Exception, if an error occurs when the device data is allocated, copied or freed.
     */
    template<typename Type>
    class PtrDevice {
    private:
        size_t m_elements{};
        std::enable_if_t<Traits::is_data_v<Type> && !std::is_reference_v<Type> &&
                         !std::is_array_v<Type> && !std::is_const_v<Type>,
                         Type*> m_dev_ptr{nullptr};

    public:
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to properly initialize the pointer. */
        NOA_HOST PtrDevice() = default;

        /**
         * Allocates @a elements elements of type @a Type on device memory using @c cudaMalloc.
         * @param elements  This is attached to the underlying managed pointer. Use elements() to access it.
         *                  The number of bytes allocated is (at least) equal to `elements * sizeof(Type)`, see bytes().
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner".
         *
         * @throw   @c Noa::Exception, if allocation fails.
         */
        NOA_HOST explicit PtrDevice(size_t elements) : m_elements(elements) { alloc_(elements); }

        /**
         * Creates an instance from an existing pointer.
         * @param elements      Number of @a Type elements in @a dev_ptr.
         * @param[in] dev_ptr   Device pointer to hold on. If it is not a nullptr, it should correspond to @a elements.
         * @param owner         Whether or not this new instance should own @a dev_ptr.
         */
        NOA_HOST PtrDevice(size_t elements, Type* dev_ptr, bool owner) noexcept
                : m_elements(elements), m_dev_ptr(dev_ptr), is_owner(owner) {}

        /**
         * Copy constructor.
         * @note    If @a to_copy owns its data, performs a deep copy. The new instance will own the
         *          copied data. Otherwise, perform a shallow copy.
         */
        NOA_HOST PtrDevice(const PtrDevice<Type>& to_copy)
                : m_elements(to_copy.m_elements), is_owner(to_copy.is_owner) {
            if (is_owner && to_copy.m_dev_ptr) {
                alloc_(m_elements);
                copy_(to_copy.m_dev_ptr, m_elements);
            } else {
                m_dev_ptr = to_copy.m_dev_ptr;
            }
        }

        /**
         * Move constructor.
         * @note    @a to_move is left in an empty state (i.e. nullptr). It can technically be reset using reset(),
         *          but why should it?
         */
        NOA_HOST PtrDevice(PtrDevice<Type>&& to_move) noexcept
                : m_elements(to_move.m_elements),
                  m_dev_ptr(std::exchange(to_move.m_dev_ptr, nullptr)),
                  is_owner(to_move.is_owner) {}

        /** Copy assignment operator. Old owned data is destroyed and then perform a shallow or deep copy. */
        NOA_HOST PtrDevice<Type>& operator=(const PtrDevice<Type>& to_copy) {
            dealloc_();
            m_elements = to_copy.m_elements;
            if (to_copy.is_owner && to_copy.m_dev_ptr) {
                alloc_(m_elements);
                copy_(to_copy.m_dev_ptr, m_elements);
            } else {
                m_dev_ptr = to_copy.m_dev_ptr;
            }
            is_owner = to_copy.is_owner;
            return *this;
        }

        /** Move assignment operator. Old owned data is destroyed. */
        NOA_IHD PtrDevice<Type>& operator=(PtrDevice<Type>&& to_move) noexcept {
            dealloc_();
            m_elements = to_move.m_elements;
            m_dev_ptr = std::exchange(to_move.m_dev_ptr, nullptr);
            is_owner = to_move.is_owner;
            return *this;
        }

        [[nodiscard]] NOA_FHD constexpr Type* get() noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_FHD constexpr const Type* get() const noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_FHD constexpr Type* data() noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_FHD constexpr const Type* data() const noexcept { return m_dev_ptr; }

        [[nodiscard]] NOA_FHD constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] NOA_FHD constexpr size_t bytes() const noexcept { return m_elements * sizeof(Type); }
        [[nodiscard]] NOA_FHD constexpr size_t empty() const noexcept { return m_dev_ptr || m_elements == 0; }
        [[nodiscard]] NOA_FHD constexpr explicit operator bool() const noexcept { return empty(); }

        [[nodiscard]] NOA_FHD constexpr Type* begin() noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_FHD constexpr const Type* begin() const noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_FHD constexpr Type* end() noexcept { return m_dev_ptr + m_elements; }
        [[nodiscard]] NOA_FHD constexpr const Type* end() const noexcept { return m_dev_ptr + m_elements; }

        [[nodiscard]] NOA_FHD constexpr Type& operator[](size_t idx) noexcept { return *(m_dev_ptr + idx); }
        [[nodiscard]] NOA_FHD constexpr const Type& operator[](size_t idx) const noexcept { return *(m_dev_ptr + idx); }

        /** Clears the underlying data if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_elements = 0;
        }

        /**
         * Resets the underlying data.
         * This is equivalent to `*this = PtrDevice<Type>(elements)`
         */
        NOA_HOST void reset(size_t elements) {
            dealloc_();
            m_elements = elements;
            alloc_(m_elements);
            is_owner = true;
        }

        /**
         * Resets the underlying data.
         * @param elements      Number of @a Type elements in @a dev_ptr.
         * @param[in] dev_ptr   Device pointer to hold on. If it is not a nullptr, it should correspond to @a elements.
         * @param owner         Whether or not this new instance should own @a dev_ptr.
         */
        NOA_HOST void reset(size_t elements, Type* dev_ptr, bool owner) {
            dealloc_();
            m_elements = elements;
            m_dev_ptr = dev_ptr;
            is_owner = owner;
        }

        /**
         * If the current instance is an owner, releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call.
         */
        NOA_IHD Type* release() noexcept {
            is_owner = false;
            return std::exchange(m_dev_ptr, nullptr);
        }

        /** Returns a human-readable description of the underlying data. */
        [[nodiscard]] NOA_HOST std::string toString() const {
            return String::format("Elements: {}, Resource: device, Type: {}, Owner: {}, Address: {}",
                                  m_elements, String::typeName<Type>(), is_owner, m_dev_ptr);
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        NOA_HOST ~PtrDevice() { dealloc_(); }

    private:
        // Allocates the necessary bytes at the underlying data, on the device, using cudaMalloc.
        NOA_HOST void alloc_(size_t elements) {
            NOA_THROW_IF(cudaMalloc(&m_dev_ptr, elements * sizeof(Type)));
        }

        // Copies data from src to the underlying data.
        // Both src and the underlying data should be allocated on the device.
        NOA_HOST void copy_(Type* src, size_t elements) {
            NOA_THROW_IF(cudaMemcpy(&m_dev_ptr, &src, elements * sizeof(Type), cudaMemcpyDeviceToDevice));
        }

        NOA_HOST void dealloc_() {
            if (!m_dev_ptr)
                return;
            if (is_owner) {
                NOA_THROW_IF(cudaFree(m_dev_ptr));
            } else {
                m_dev_ptr = nullptr;
            }
        }
    };
}
