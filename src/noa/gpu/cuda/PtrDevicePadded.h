#pragma once

#include "noa/gpu/cuda/Base.h"
#include "noa/gpu/cuda/Allocator.h"

/*
 * Linear memory vs padded memory
 * ==============================
 *
 * PtrDevice is managing "linear" memory, as opposed to PtrDevicePadded, which managed "padded" memory.
 * "Padded" memory can be useful to minimize the number of memory access on a given row (but can increase the number
 * of memory accesses for reading the whole array) and to reduce shared memory bank conflicts. This is detailed in
 * https://stackoverflow.com/questions/16119943 and in https://stackoverflow.com/questions/15056842.
 *
 * As a result, PtrDevicePadded has to keep track of a pitch and a shape (width|row, height|column and depth|page).
 */

namespace Noa::CUDA {
    /**
     * Manages a device pointer from the host side. This object is not meant to be used from the device.
     * @tparam Type     Type of the underlying pointer. Should be an integer, float, double, cfloat_t or cdouble_t.
     * @throw           @c Noa::Exception, if an error occurs when the device data is allocated or freed.
     */
    template<typename Type>
    class PtrDevicePadded {
    private:
        size3_t m_shape{}; // in elements
        std::enable_if_t<Noa::Traits::is_data_v<Type> && !std::is_reference_v<Type> &&
                         !std::is_array_v<Type> && !std::is_const_v<Type>,
                         Type*> m_dev_ptr{nullptr};
        size_t m_pitch{}; // in bytes
    public:
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrDevicePadded() = default;

        /**
         * Allocates "padded" memory with a given @a shape on the current device using @c cudaMalloc3D.
         * @param shape     Either 1D, 2D or 3D. This is attached to the underlying managed pointer and is fixed for
         *                  the entire life of the object. Use shape() to access it. The number of allocated bytes is
         *                  (at least) equal to `Math::elements(shape) * sizeof(Type)`, see bytes() and bytesPadded().
         *
         * @warning If any element of @a shape is 0, the allocation will not be performed.
         *          For instance, to specify a 2D array, @a shape should be {X, Y, 1}.
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner", but make
         *          sure the data is freed at some point.
         */
        NOA_HOST explicit PtrDevicePadded(size3_t shape) : m_shape(shape) { alloc_(); }

        /**
         * Creates an instance from a existing data.
         * @param[in] dev_ptr   Device pointer to hold on.
         *                      If it is a nullptr, @a pitch and @a shape should be 0.
         *                      If it is not a nullptr, it should correspond to @a pitch and @a shape.
         * @param pitch         The pitch, in bytes, of @a dev_ptr.
         * @param shape         The shape, in elements, of @a dev_ptr.
         * @param owner         Whether or not this new instance should own @a dev_ptr.
         */
        NOA_HOST PtrDevicePadded(Type* dev_ptr, size_t pitch, size3_t shape, bool owner) noexcept
                : m_shape(shape), m_dev_ptr(dev_ptr), m_pitch(pitch), is_owner(owner) {}

        /**
         * Copy constructor.
         * @note    This performs a shallow copy of the managed data. The created instance is not the
         *          owner of the copied data. If one wants to perform a deep copy, one should use the
         *          Memory::copy() functions.
         */
        NOA_HOST PtrDevicePadded(const PtrDevicePadded<Type>& to_copy) noexcept
                : m_shape(to_copy.m_shape), m_dev_ptr(to_copy.m_dev_ptr), m_pitch(to_copy.m_pitch), is_owner(false) {}

        /**
         * Move constructor.
         * @note    @a to_move is left in an empty state (i.e. nullptr). It can technically be reset using reset(),
         *          but why should it?
         */
        NOA_HOST PtrDevicePadded(PtrDevicePadded<Type>&& to_move) noexcept
                : m_shape(to_move.m_shape),
                  m_dev_ptr(std::exchange(to_move.m_dev_ptr, nullptr)),
                  m_pitch(to_move.m_pitch),
                  is_owner(to_move.is_owner) {}

        /**
         * Copy/move assignment operator.
         * @note    Redundant and a bit ambiguous. To copy/move data into an existing object, use reset(),
         *          which is much more explicit. In practice, it is probably better to create a new object.
         */
        PtrDevicePadded<Type>& operator=(const PtrDevicePadded<Type>& to_copy) = delete;
        PtrDevicePadded<Type>& operator=(PtrDevicePadded<Type>&& to_move) = delete;

        [[nodiscard]] NOA_HOST constexpr Type* get() noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* get() const noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_HOST constexpr Type* data() noexcept { return m_dev_ptr; }
        [[nodiscard]] NOA_HOST constexpr const Type* data() const noexcept { return m_dev_ptr; }

        /** Returns the shape (in elements) of the managed object. */
        [[nodiscard]] NOA_HOST constexpr size3_t shape() const noexcept { return m_shape; }

        /** Returns the pitch (in bytes) of the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t pitch() const noexcept { return m_pitch; }

        /** How many elements of type @a Type are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return Math::elements(m_shape); }

        /** How many bytes (excluding the padding) are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t bytes() const noexcept { return elements() * sizeof(Type); }

        /** How many bytes (including the padding) are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t bytesPadded() const noexcept { return m_pitch * m_shape.y * m_shape.z; }

        /** Whether or not the managed object points to some data. */
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_pitch; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return empty(); }

        /** Clears the underlying data, if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_shape = 0UL;
            m_pitch = 0;
        }

        /** Clears the underlying data, if necessary. This is identical to reset(). */
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /** Resets the underlying data. The new data is owned. */
        NOA_HOST void reset(size3_t shape) {
            dealloc_();
            m_shape = shape;
            alloc_();
            is_owner = true;
        }

        /**
         * Resets the underlying data.
         * @param[in] dev_ptr   Device pointer to hold on.
         *                      If it is a nullptr, @a pitch and @a shape should be 0.
         *                      If it is not a nullptr, it should correspond to @a pitch and @a shape.
         * @param pitch         The pitch, in bytes, of @a dev_ptr.
         * @param shape         The shape, in elements, of @a dev_ptr.
         * @param owner         Whether or not this new instance should own @a dev_ptr.
         */
        NOA_HOST void reset(Type* dev_ptr, size_t pitch, size3_t shape, bool owner) {
            dealloc_();
            m_shape = shape;
            m_dev_ptr = dev_ptr;
            m_pitch = pitch;
            is_owner = owner;
        }

        /**
         * If the current instance is an owner, releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call.
         */
        [[nodiscard]] NOA_HOST Type* release() noexcept {
            m_shape = 0UL;
            m_pitch = 0;
            return std::exchange(m_dev_ptr, nullptr);
        }

        /** Returns a human-readable description of the underlying data. */
        [[nodiscard]] NOA_HOST std::string toString() const {
            return String::format("Elements: {}, Pitch: {} bytes (padded: {}), Type: {}, Owner: {}, "
                                  "Resource: device, Address: {}",
                                  m_shape, m_pitch, m_pitch - m_shape.x * sizeof(Type), String::typeName<Type>(),
                                  is_owner, static_cast<void*>(m_dev_ptr));
        }

        /** Deallocates owned data. */
        NOA_HOST ~PtrDevicePadded() { dealloc_(); }

    private:
        // Allocates using cudaMalloc3D. If x, y or z is 0, no allocation is performed: ptr=nullptr, pitch=0.
        NOA_HOST void alloc_() {
            cudaExtent extent{m_shape.x * sizeof(Type), m_shape.y, m_shape.z};
            cudaPitchedPtr pitched_ptr{};
            NOA_THROW_IF(Allocator::malloc3D(&pitched_ptr, extent));
            m_dev_ptr = pitched_ptr.ptr;
            m_pitch = pitched_ptr.pitch;
        }

        NOA_HOST void dealloc_() {
            if (is_owner) {
                NOA_THROW_IF(Allocator::free(m_dev_ptr)); // if nullptr, does nothing.
            } else {
                m_dev_ptr = nullptr;
            }
        }
    };

    template<class T>
    [[nodiscard]] NOA_IH std::string toString(PtrDevicePadded<T> ptr) { return ptr.toString(); }
}

template<typename T>
struct fmt::formatter<Noa::CUDA::PtrDevicePadded<T>> : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const Noa::CUDA::PtrDevicePadded<T>& ptr, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(Noa::CUDA::toString(ptr), ctx);
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Noa::CUDA::PtrDevicePadded<T>& ptr) {
    os << Noa::CUDA::toString(ptr);
    return os;
}
