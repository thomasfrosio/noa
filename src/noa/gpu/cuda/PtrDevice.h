#pragma once

#include <type_traits>
#include <string>

#include "noa/gpu/cuda/Base.h"
#include "noa/gpu/cuda/Allocator.h"
#include "noa/util/string/Format.h"

/*
 * Accessing PtrDevice on the device?
 * ==================================
 *
 * PtrDevice could be accessed from the device, since its copy constructor will make sure the copy used in the kernels
 * are not-owning copies. However, I'd rather be explicit on the kernel calls, that is, use a raw pointer and a size.
 * Therefore, all member functions are host only.
 *
 * 1)   It is easier to understand what the kernel calls will do, especially for people not familiar with PtrDevice,
 *      that is, it becomes obvious that a shallow copy is performed when passing arguments to kernels.
 * 2)   The size doesn't have to be the size of the entire data. It is easier to work on a subset of the data.
 * 3)   clang-tidy is not complaining about passing a non-trivially copyable object by value as opposed to
 *      by reference. Note that passing by reference doesn't work for kernel calls since some member variables
 *      are on the host.
 */

namespace Noa::CUDA {
    /**
     * Manages a device pointer from the host side. This object is not meant to be used from the device.
     * @tparam Type Type of the underlying pointer. Should be an integer, float, double, cfloat_t or cdouble_t.
     * @throw       @c Noa::Exception, if an error occurs when the device data is allocated or freed.
     */
    template<typename Type>
    class PtrDevice {
    private:
        size_t m_elements{0};
        std::enable_if_t<Noa::Traits::is_data_v<Type> && !std::is_reference_v<Type> &&
                         !std::is_array_v<Type> && !std::is_const_v<Type>,
                         Type*> m_dev_ptr{nullptr};

    public:
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrDevice() = default;

        /**
         * Allocates @a elements elements of type @a Type on the current device using @c cudaMalloc.
         * @param elements  This is attached to the underlying managed pointer and is fixed for the entire
         *                  life of the object. Use elements() to access it. The number of allocated bytes is
         *                  (at least) equal to `elements * sizeof(Type)`, see bytes().
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner", but make
         *          sure the data is freed at some point.
         */
        NOA_HOST explicit PtrDevice(size_t elements) : m_elements(elements) { alloc_(elements); }

        /**
         * Creates an instance from a existing data.
         * @param elements      Number of @a Type elements in @a dev_ptr.
         * @param[in] dev_ptr   Device pointer to hold on.
         *                      If it is a nullptr, @a elements should be 0.
         *                      If it is not a nullptr, it should correspond to @a elements.
         * @param owner         Whether or not this new instance should own @a dev_ptr.
         */
        NOA_HOST PtrDevice(size_t elements, Type* dev_ptr, bool owner) noexcept
                : m_elements(elements), m_dev_ptr(dev_ptr), is_owner(owner) {}

        /**
         * Copy constructor.
         * @note    This performs a shallow copy of the managed data. The created instance is not the
         *          owner of the copied data. If one wants to perform a deep copy, one should use the
         *          Memory::copy() functions.
         */
        NOA_HOST PtrDevice(const PtrDevice<Type>& to_copy) noexcept
                : m_elements(to_copy.m_elements), m_dev_ptr(to_copy.m_dev_ptr), is_owner(false) {}

        /**
         * Move constructor.
         * @note    @a to_move is left in an empty state (i.e. nullptr). It can technically be reset using reset(),
         *          but why should it?
         */
        NOA_HOST PtrDevice(PtrDevice<Type>&& to_move) noexcept
                : m_elements(to_move.m_elements),
                  m_dev_ptr(std::exchange(to_move.m_dev_ptr, nullptr)),
                  is_owner(to_move.is_owner) {}

        /**
         * Copy/move assignment operator.
         * @note    Redundant and a bit ambiguous. To copy/move data into an existing object, use reset(),
         *          which is much more explicit. In practice, it is probably better to create a new object.
         */
        PtrDevice<Type>& operator=(const PtrDevice<Type>& to_copy) = delete;
        PtrDevice<Type>& operator=(PtrDevice<Type>&& to_move) = delete;

        [[nodiscard]] NOA_HOST constexpr Type* get() noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* get() const noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_HOST constexpr Type* data() noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* data() const noexcept { return m_dev_ptr; }

        /** How many elements of type @a Type are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return m_elements; }

        /** How many bytes are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t bytes() const noexcept { return m_elements * sizeof(Type); }

        /** Whether or not the managed object points to some data. */
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return empty(); }

        /** Returns a pointer pointing at the beginning of the managed data. */
        [[nodiscard]] NOA_HOST constexpr Type* begin() noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* begin() const noexcept { return m_dev_ptr; }

        /** Returns a pointer pointing at the end + 1 of the managed data. */
        [[nodiscard]] NOA_HOST constexpr Type* end() noexcept { return m_dev_ptr + m_elements; }
        [[nodiscard]] NOA_HOST constexpr const Type* end() const noexcept { return m_dev_ptr + m_elements; }

        /** Clears the underlying data, if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_elements = 0;
        }

        /** Clears the underlying data, if necessary. This is identical to reset(). */
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /** Resets the underlying data. The new data is owned. */
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
        [[nodiscard]] NOA_HOST Type* release() noexcept {
            m_elements = 0;
            return std::exchange(m_dev_ptr, nullptr);
        }

        /** Returns a human-readable description of the underlying data. */
        [[nodiscard]] NOA_HOST std::string toString() const {
            return String::format("Elements: {}, Type: {}, Owner: {}, Resource: device, Address: {}",
                                  m_elements, String::typeName<Type>(), is_owner, static_cast<void*>(m_dev_ptr));
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        NOA_HOST ~PtrDevice() { dealloc_(); }

    private:
        NOA_HOST void alloc_(size_t elements) {
            NOA_THROW_IF(cudaMalloc(&m_dev_ptr, elements * sizeof(Type)));
        }

        NOA_HOST void dealloc_() {
            if (is_owner) {
                NOA_THROW_IF(cudaFree(m_dev_ptr)); // if nullptr, does nothing.
            } else {
                m_dev_ptr = nullptr;
            }
        }
    };

    template<class T>
    [[nodiscard]] NOA_IH std::string toString(PtrDevice<T> ptr) { return ptr.toString(); }
}

template<typename T>
struct fmt::formatter<Noa::CUDA::PtrDevice<T>> : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const Noa::CUDA::PtrDevice<T>& ptr, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(Noa::CUDA::toString(ptr), ctx);
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Noa::CUDA::PtrDevice<T>& ptr) {
    os << Noa::CUDA::toString(ptr);
    return os;
}
