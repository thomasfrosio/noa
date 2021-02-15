#pragma once

#include <type_traits>
#include <string>

#include "noa/gpu/cuda/Base.h"
#include "noa/util/string/Format.h"

/*
 * CUDA arrays:
 *  -   They are only accessible by kernels through texture fetching or surface reading and writing.
 *  -   They are usually associated with a type: each element can have 1, 2 or 4 components (e.g. complex types have
 *      2 components). Elements are associated with a type (components have the same type), that may be signed or
 *      unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats.
 *  -   They are either 1D, 2D or 3D. Note that an "empty" dimension is noted as 0. Since the indexing should be
 *      known at compile time (e.g. tex2D vs tex3D), the following implementation embeds the dimension into its type.
 */

namespace Noa::CUDA {
    template<typename T, uint ndim>
    class PtrArray;

    /** A 1D CUDA array of integers (excluding (u)int64_t), float or cfloat_t. */
    template<typename Type>
    class PtrArray<Type, 1> {
    private:
        size_t m_elements{};
        std::enable_if_t<Noa::Traits::is_data_v<Type> &&
                         !std::is_reference_v<Type> && !std::is_array_v<Type> && !std::is_const_v<Type> &&
                         !std::is_same_v<Type, uint64_t> && !std::is_same_v<Type, int64_t> &&
                         !std::is_same_v<Type, double> && !std::is_same_v<Type, cdouble_t>,
                         cudaArray*> m_ptr{nullptr};
    public:
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrArray() = default;

        /**
         * Allocates a 1D CUDA array with @a elements elements on the current device using @c cudaMalloc3DArray.
         * @param size  Number of elements. This is attached to the underlying managed pointer and is fixed for
         *                  the entire life of the object. Use shape() to access it.
         *
         * @warning If any element of @a shape is 0, the allocation will not be performed.
         *          To specify a 2D CUDA array, @a shape should be {X, Y}, with X and Y > 0.
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner", but make
         *          sure the data is freed at some point.
         */
        NOA_HOST explicit PtrArray(size_t size, uint flags = cudaArrayDefault)
                : m_elements(size) { alloc_(flags); }

        /**
         * Copy constructor.
         * @note    This performs a shallow copy of the managed data. The created instance is not the
         *          owner of the copied data. If one wants to perform a deep copy, one should use the
         *          Memory::copy() functions.
         */
        NOA_HOST PtrArray(const PtrArray<Type, 1>& to_copy) noexcept
                : m_elements(to_copy.m_elements), m_ptr(to_copy.m_ptr), is_owner(false) {}

        /**
         * Move constructor.
         * @note    @a to_move is left in an empty state (i.e. nullptr). It can technically be reset using reset(),
         *          but why should it?
         */
        NOA_HOST PtrArray(PtrArray<Type, 1>&& to_move) noexcept
                : m_elements(to_move.m_elements),
                  m_ptr(std::exchange(to_move.m_ptr, nullptr)),
                  is_owner(to_move.is_owner) {}

        /**
         * Copy/move assignment operator.
         * @note    Redundant and a bit ambiguous. To copy/move data into an existing object, use reset(),
         *          which is much more explicit. In practice, it is probably better to create a new object.
         */
        PtrArray<Type, 2>& operator=(const PtrArray<Type, 1>& to_copy) = delete;
        PtrArray<Type, 2>& operator=(PtrArray<Type, 1>&& to_move) = delete;

        [[nodiscard]] NOA_HOST constexpr cudaArray* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr cudaArray* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* data() const noexcept { return m_ptr; }

        /** Returns the shape (in elements) of the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return m_elements; }
        [[nodiscard]] NOA_HOST constexpr size_t shape() const noexcept { return m_elements; }

        /** Whether or not the managed object points to some data. */
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return empty(); }

        /** Clears the underlying data, if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_elements = 0;
        }

        /** Clears the underlying data, if necessary. This is identical to reset(). */
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /** Resets the underlying data. The new data is owned. */
        NOA_HOST void reset(size_t elements, uint flags = cudaArrayDefault) {
            dealloc_();
            m_elements = elements;
            alloc_(flags);
            is_owner = true;
        }

        /**
         * Resets the underlying data.
         * @param elements      Number of @a Type elements in @a dev_ptr.
         * @param[in] dev_ptr   Device pointer to hold on. If it is not a nullptr, it should correspond to @a elements.
         * @param owner         Whether or not this new instance should own @a dev_ptr.
         */
        NOA_HOST void reset(size_t elements, cudaArray* ptr, bool owner) {
            dealloc_();
            m_elements = elements;
            m_ptr = ptr;
            is_owner = owner;
        }

        /**
         * If the current instance is an owner, releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call.
         */
        [[nodiscard]] NOA_HOST cudaArray* release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

        /** Returns a human-readable description of the underlying data. */
        [[nodiscard]] NOA_HOST std::string toString() const {
            return String::format("Elements: {}, Type: {}, Owner: {}, Resource: 1D CUDA array, Address: {}",
                                  m_elements, String::typeName<Type>(), is_owner, static_cast<void*>(m_ptr));
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        NOA_HOST ~PtrArray() { dealloc_(); }

    private:
        NOA_HOST void alloc_(uint flag) {
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<Type>();
            NOA_THROW_IF(cudaMalloc3DArray(&m_ptr, &desc, make_cudaExtent(m_elements, 0, 0), flag));
        }

        NOA_HOST void dealloc_() {
            if (is_owner)
                NOA_THROW_IF(cudaFreeArray(m_ptr));
            else
                m_ptr = nullptr;
        }
    };

    /** A 2D CUDA array of integers (excluding (u)int64_t), float or cfloat_t. */
    template<typename Type>
    class PtrArray<Type, 2> {
    private:
        size2_t m_shape{};
        std::enable_if_t<Noa::Traits::is_data_v<Type> &&
                         !std::is_reference_v<Type> && !std::is_array_v<Type> && !std::is_const_v<Type> &&
                         !std::is_same_v<Type, uint64_t> && !std::is_same_v<Type, int64_t> &&
                         !std::is_same_v<Type, double> && !std::is_same_v<Type, cdouble_t>,
                         cudaArray*> m_ptr{nullptr};
    public:
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrArray() = default;

        /**
         * Allocates a 2D CUDA array with a given @a shape on the current device using @c cudaMalloc3DArray.
         * @param shape     2D shape. This is attached to the underlying managed pointer and is fixed for
         *                  the entire life of the object. Use shape() to access it.
         *
         * @warning If any element of @a shape is 0, the allocation will not be performed.
         *          To specify a 2D CUDA array, @a shape should be {X, Y}, with X and Y > 0.
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner", but make
         *          sure the data is freed at some point.
         */
        NOA_HOST explicit PtrArray(size2_t shape, uint flags = cudaArrayDefault)
                : m_shape(shape) { alloc_(flags); }

        /**
         * Copy constructor.
         * @note    This performs a shallow copy of the managed data. The created instance is not the
         *          owner of the copied data. If one wants to perform a deep copy, one should use the
         *          Memory::copy() functions.
         */
        NOA_HOST PtrArray(const PtrArray<Type, 2>& to_copy) noexcept
                : m_shape(to_copy.m_shape), m_ptr(to_copy.m_ptr), is_owner(false) {}

        /**
         * Move constructor.
         * @note    @a to_move is left in an empty state (i.e. nullptr). It can technically be reset using reset(),
         *          but why should it?
         */
        NOA_HOST PtrArray(PtrArray<Type, 2>&& to_move) noexcept
                : m_shape(to_move.m_shape),
                  m_ptr(std::exchange(to_move.m_ptr, nullptr)),
                  is_owner(to_move.is_owner) {}

        /**
         * Copy/move assignment operator.
         * @note    Redundant and a bit ambiguous. To copy/move data into an existing object, use reset(),
         *          which is much more explicit. In practice, it is probably better to create a new object.
         */
        PtrArray<Type, 2>& operator=(const PtrArray<Type, 2>& to_copy) = delete;
        PtrArray<Type, 2>& operator=(PtrArray<Type, 2>&& to_move) = delete;

        [[nodiscard]] NOA_HOST constexpr cudaArray* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr cudaArray* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* data() const noexcept { return m_ptr; }

        /** Returns the shape (in elements) of the managed object. */
        [[nodiscard]] NOA_HOST constexpr size2_t shape() const noexcept { return m_shape; }

        /** How many elements of type @a Type are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return Math::elements(m_shape); }

        /** Whether or not the managed object points to some data. */
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return empty(); }

        /** Clears the underlying data, if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_shape = 0UL;
        }

        /** Clears the underlying data, if necessary. This is identical to reset(). */
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /** Resets the underlying data. The new data is owned. */
        NOA_HOST void reset(size2_t shape, uint flags = cudaArrayDefault) {
            dealloc_();
            m_shape = shape;
            alloc_(flags);
            is_owner = true;
        }

        /**
         * Resets the underlying data.
         * @param elements      Number of @a Type elements in @a dev_ptr.
         * @param[in] dev_ptr   Device pointer to hold on. If it is not a nullptr, it should correspond to @a elements.
         * @param owner         Whether or not this new instance should own @a dev_ptr.
         */
        NOA_HOST void reset(size2_t shape, cudaArray* ptr, bool owner) {
            dealloc_();
            m_shape = shape;
            m_ptr = ptr;
            is_owner = owner;
        }

        /**
         * If the current instance is an owner, releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call.
         */
        [[nodiscard]] NOA_HOST cudaArray* release() noexcept {
            m_shape = 0UL;
            return std::exchange(m_ptr, nullptr);
        }

        /** Returns a human-readable description of the underlying data. */
        [[nodiscard]] NOA_HOST std::string toString() const {
            return String::format("Elements: {}, Type: {}, Owner: {}, Resource: 2D CUDA array, Address: {}",
                                  m_shape, String::typeName<Type>(), is_owner, static_cast<void*>(m_ptr));
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        NOA_HOST ~PtrArray() { dealloc_(); }

    private:
        NOA_HOST void alloc_(uint flag) {
            if (!m_shape.x || !m_shape.y)
                NOA_THROW("Cannot allocate a 2D CUDA array with a dimension equal to zero. Got shape: {}", m_shape);
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<Type>();
            NOA_THROW_IF(cudaMalloc3DArray(&m_ptr, &desc, make_cudaExtent(m_shape.x, m_shape.y, 0), flag));
        }

        NOA_HOST void dealloc_() {
            if (is_owner)
                NOA_THROW_IF(cudaFreeArray(m_ptr));
            else
                m_ptr = nullptr;
        }
    };

    /** A 3D CUDA array of integers (excluding (u)int64_t), float or cfloat_t. */
    template<typename Type>
    class PtrArray<Type, 3> {
    private:
        size3_t m_shape{};
        std::enable_if_t<Noa::Traits::is_data_v<Type> &&
                         !std::is_reference_v<Type> && !std::is_array_v<Type> && !std::is_const_v<Type> &&
                         !std::is_same_v<Type, uint64_t> && !std::is_same_v<Type, int64_t> &&
                         !std::is_same_v<Type, double> && !std::is_same_v<Type, cdouble_t>,
                         cudaArray*> m_ptr{nullptr};
    public:
        bool is_owner{true};

    public:
        /** Creates an empty instance. Use reset() to allocate new data. */
        PtrArray() = default;

        /**
         * Allocates a 2D CUDA array with a given @a shape on the current device using @c cudaMalloc3DArray.
         * @param shape     2D shape. This is attached to the underlying managed pointer and is fixed for
         *                  the entire life of the object. Use shape() to access it.
         *
         * @warning If any element of @a shape is 0, the allocation will not be performed.
         *          To specify a 2D CUDA array, @a shape should be {X, Y}, with X and Y > 0.
         *
         * @note    The created instance is the owner of the data. To get a non-owning pointer, use get().
         *          The ownership can be changed at anytime using the member variable "is_owner", but make
         *          sure the data is freed at some point.
         */
        NOA_HOST explicit PtrArray(size3_t shape, uint flags = cudaArrayDefault)
                : m_shape(shape) { alloc_(flags); }

        /**
         * Copy constructor.
         * @note    This performs a shallow copy of the managed data. The created instance is not the
         *          owner of the copied data. If one wants to perform a deep copy, one should use the
         *          Memory::copy() functions.
         */
        NOA_HOST PtrArray(const PtrArray<Type, 3>& to_copy) noexcept
                : m_shape(to_copy.m_shape), m_ptr(to_copy.m_ptr), is_owner(false) {}

        /**
         * Move constructor.
         * @note    @a to_move is left in an empty state (i.e. nullptr). It can technically be reset using reset(),
         *          but why should it?
         */
        NOA_HOST PtrArray(PtrArray<Type, 3>&& to_move) noexcept
                : m_shape(to_move.m_shape),
                  m_ptr(std::exchange(to_move.m_ptr, nullptr)),
                  is_owner(to_move.is_owner) {}

        /**
         * Copy/move assignment operator.
         * @note    Redundant and a bit ambiguous. To copy/move data into an existing object, use reset(),
         *          which is much more explicit. In practice, it is probably better to create a new object.
         */
        PtrArray<Type, 2>& operator=(const PtrArray<Type, 3>& to_copy) = delete;
        PtrArray<Type, 2>& operator=(PtrArray<Type, 3>&& to_move) = delete;

        [[nodiscard]] NOA_HOST constexpr cudaArray* get() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr cudaArray* data() noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr const cudaArray* data() const noexcept { return m_ptr; }

        /** Returns the shape (in elements) of the managed object. */
        [[nodiscard]] NOA_HOST constexpr size3_t shape() const noexcept { return m_shape; }

        /** How many elements of type @a Type are pointed by the managed object. */
        [[nodiscard]] NOA_HOST constexpr size_t elements() const noexcept { return Math::elements(m_shape); }

        /** Whether or not the managed object points to some data. */
        [[nodiscard]] NOA_HOST constexpr bool empty() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HOST constexpr explicit operator bool() const noexcept { return empty(); }

        /** Clears the underlying data, if necessary. empty() will evaluate to true. */
        NOA_HOST void reset() {
            dealloc_();
            m_shape = 0UL;
        }

        /** Clears the underlying data, if necessary. This is identical to reset(). */
        NOA_HOST void dispose() { reset(); } // dispose might be a better name than reset...

        /** Resets the underlying data. The new data is owned. */
        NOA_HOST void reset(size3_t shape, uint flags = cudaArrayDefault) {
            dealloc_();
            m_shape = shape;
            alloc_(flags);
            is_owner = true;
        }

        /**
         * Resets the underlying data.
         * @param elements      Number of @a Type elements in @a dev_ptr.
         * @param[in] dev_ptr   Device pointer to hold on. If it is not a nullptr, it should correspond to @a elements.
         * @param owner         Whether or not this new instance should own @a dev_ptr.
         */
        NOA_HOST void reset(size3_t shape, cudaArray* ptr, bool owner) {
            dealloc_();
            m_shape = shape;
            m_ptr = ptr;
            is_owner = owner;
        }

        /**
         * If the current instance is an owner, releases the ownership of the managed pointer, if any.
         * In this case, the caller is responsible for deleting the object.
         * get() returns nullptr after the call.
         */
        [[nodiscard]] NOA_HOST cudaArray* release() noexcept {
            m_shape = 0UL;
            return std::exchange(m_ptr, nullptr);
        }

        /** Returns a human-readable description of the underlying data. */
        [[nodiscard]] NOA_HOST std::string toString() const {
            return String::format("Elements: {}, Type: {}, Owner: {}, Resource: 3D CUDA array, Address: {}",
                                  m_shape, String::typeName<Type>(), is_owner, static_cast<void*>(m_ptr));
        }

        /** If the instance is an owner and if it is not nullptr, deallocates the data. */
        NOA_HOST ~PtrArray() { dealloc_(); }

    private:
        NOA_HOST void alloc_(uint flag) {
            if (!m_shape.x || !m_shape.y || !m_shape.z)
                NOA_THROW("Cannot allocate a 3D CUDA array with a dimension equal to zero. Got shape: {}", m_shape);
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<Type>();
            NOA_THROW_IF(cudaMalloc3DArray(&m_ptr, &desc, make_cudaExtent(m_shape.x, m_shape.y, m_shape.z), flag));
        }

        NOA_HOST void dealloc_() {
            if (is_owner)
                NOA_THROW_IF(cudaFreeArray(m_ptr));
            else
                m_ptr = nullptr;
        }
    };

    /*
     * void copy(PtrDevice<>, PtrHost<>);
     * void copy(PtrHost<>, PtrDevice<>);
     * void copy(PtrDevice<>, PtrHost<>, const Stream&);
     * void copy(PtrHost<>, PtrDevice<>, const Stream&);
     *
     * void copy(PtrPinned<>, PtrHost<>);
     * void copy(PtrHost<>, PtrPinned<>);
     * void copy(PtrPinned<>, PtrHost<>, const Stream&);
     * void copy(PtrHost<>, PtrPinned<>, const Stream&);
     *
     * void copy(PtrPinned<>, PtrDevice<>);
     * void copy(PtrDevice<>, PtrPinned<>);
     * void copy(PtrPinned<>, PtrDevice<>, const Stream&);
     * void copy(PtrDevice<>, PtrPinned<>, const Stream&);
     *
     * void copy(PtrDevicePadded<>, PtrHost<>);
     * void copy(PtrHost<>, PtrDevicePadded<>);
     * void copy(PtrDevicePadded<>, PtrHost<>, const Stream&);
     * void copy(PtrHost<>, PtrDevicePadded<>, const Stream&);
     *
     * void copy(PtrPinned<>, PtrDevicePadded<>);
     * void copy(PtrDevicePadded<>, PtrPinned<>);
     * void copy(PtrPinned<>, PtrDevicePadded<>, const Stream&);
     * void copy(PtrDevicePadded<>, PtrPinned<>, const Stream&);
     *
     * void copy(PtrDevice<>, PtrDevicePadded<>);
     * void copy(PtrDevicePadded<>, PtrDevice<>);
     * void copy(PtrDevice<>, PtrDevicePadded<>, const Stream&);
     * void copy(PtrDevicePadded<>, PtrDevice<>, const Stream&);
     *
     * void copy(PtrArray<>, PtrDevicePadded<>);
     * void copy(PtrDevicePadded<>, PtrArray<>);
     * void copy(PtrArray<>, PtrDevicePadded<>, const Stream&);
     * void copy(PtrDevicePadded<>, PtrArray<>, const Stream&);
     *
     * void doSomething(PtrHost<float> data);
     * void doSomething(PtrPinned<float> data);
     * void doSomething(PtrDevice<float> data);
     * void doSomething(PtrDevicePadded<float> data);
     */
}
