#pragma once

#include <string>
#include <type_traits>
#include <utility>

#include "noa/core/Definitions.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/Types.hpp"
#include "noa/core/traits/Numerics.hpp"

namespace noa::cpu::memory {
    template<typename T>
    struct PtrHostDeleter {
        void operator()(T* ptr) noexcept {
            if constexpr(noa::traits::is_numeric_v<T>)
                std::free(ptr);
            else
                delete[] ptr;
        }
    };

    template<typename T>
    struct PtrHostDeleterCalloc {
        void operator()(T* ptr) noexcept {
            std::free(reinterpret_cast<void**>(ptr)[-1]);
        }
    };

    // Allocates and manages a heap-allocated memory region.
    // Also provides memory-aligned alloc and calloc functions.
    // Floating-points and complex floating-points are aligned to 128 bytes.
    // Integral types are aligned to 64 bytes.
    // Anything else uses the type's alignment as reported by alignof().
    template<typename Value>
    class PtrHost {
    public:
        static_assert(!std::is_pointer_v<Value> && !std::is_reference_v<Value> && !std::is_const_v<Value>);
        static constexpr size_t ALIGNMENT =
                noa::traits::is_real_or_complex_v<Value> ? 128 :
                noa::traits::is_int_v<Value> ? 64 : alignof(Value);

        using value_type = Value;
        using shared_type = Shared<value_type[]>;
        using alloc_deleter_type = PtrHostDeleter<value_type>;
        using calloc_deleter_type = PtrHostDeleterCalloc<value_type>;
        using alloc_unique_type = Unique<value_type[], alloc_deleter_type>;
        using calloc_unique_type = Unique<value_type[], calloc_deleter_type>;

    public:
        // Allocates some elements of uninitialized storage. Throws if the allocation fails.
        static alloc_unique_type alloc(i64 elements) {
            if (elements <= 0)
                return {};
            value_type* out;
            if constexpr (traits::is_numeric_v<value_type>) {
                out = static_cast<value_type*>(std::aligned_alloc(
                        ALIGNMENT, static_cast<size_t>(elements) * sizeof(value_type)));
            } else {
                out = new(std::nothrow) value_type[static_cast<size_t>(elements)];
            }
            NOA_CHECK(out, "Failed to allocate {} {} on the heap", elements, string::human<value_type>());
            return {out, alloc_deleter_type{}};
        }

        // Allocates some elements, all initialized to 0. Throws if the allocation fails.
        static calloc_unique_type calloc(i64 elements) {
            if (elements <= 0)
                return {};

            // Make sure we have enough space to store the original value returned by calloc.
            const size_t offset = ALIGNMENT - 1 + sizeof(void*);

            void* calloc_ptr = std::calloc(static_cast<size_t>(elements) * sizeof(value_type) + offset, 1);
            if (!calloc_ptr)
                NOA_THROW("Failed to allocate {} {} on the heap", elements, string::human<value_type>());

            // Align to the requested value, leaving room for the original calloc value.
            void* aligned_ptr = reinterpret_cast<void*>(((uintptr_t)calloc_ptr + offset) & ~(ALIGNMENT - 1));
            reinterpret_cast<void**>(aligned_ptr)[-1] = calloc_ptr;
            return {static_cast<value_type*>(aligned_ptr), calloc_deleter_type{}};
        }

    public:
        // Creates an empty instance. Use one of the operator assignment to allocate new data.
        PtrHost() = default;
        constexpr /*implicit*/ PtrHost(std::nullptr_t) {}

        // Allocates elements of type T on the heap.
        explicit PtrHost(i64 elements) : m_ptr(alloc(elements)), m_elements(elements) {}

    public: // Getters
        [[nodiscard]] constexpr value_type* get() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr value_type* data() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr value_type* begin() const noexcept { return m_ptr.get(); }
        [[nodiscard]] constexpr value_type* end() const noexcept { return m_ptr.get() + m_elements; }
        [[nodiscard]] constexpr const shared_type& share() const noexcept { return m_ptr; }
        [[nodiscard]] constexpr i64 elements() const noexcept { return m_elements; }
        [[nodiscard]] constexpr i64 size() const noexcept { return m_elements; }
        [[nodiscard]] constexpr Shape4<i64> shape() const noexcept { return {1, 1, 1, m_elements}; }
        [[nodiscard]] constexpr Strides4<i64> strides() const noexcept { return shape().strides(); }
        [[nodiscard]] constexpr i64 bytes() const noexcept { return m_elements * sizeof(value_type); }
        [[nodiscard]] constexpr bool is_empty() const noexcept { return m_elements == 0; }
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !is_empty(); }

        // Attach the lifetime of the managed object with alias.
        // Constructs a shared_ptr which shares ownership information with the managed object,
        // but holds an unrelated and unmanaged pointer alias. If the returned shared_ptr is
        // the last of the group to go out of scope, it will call the stored deleter for the
        // managed object of this instance. However, calling get() on this shared_ptr will always
        // return a copy of alias. It is the responsibility of the programmer to make sure that
        // alias remains valid as long as the managed object exists. This functions performs no
        // heap allocation, but increases the (atomic) reference count of the managed object.
        template<typename U>
        [[nodiscard]] constexpr Shared<U[]> attach(U* alias) const noexcept { return {m_ptr, alias}; }

        // Releases the ownership of the managed pointer, if any.
        shared_type release() noexcept {
            m_elements = 0;
            return std::exchange(m_ptr, nullptr);
        }

    private:
        shared_type m_ptr{};
        i64 m_elements{0};
    };
}
