#pragma once

#include <cstdint>
#include <type_traits>
#include "noa/core/Definitions.hpp"
#include "noa/core/Assert.hpp"
#include "noa/core/string/Format.hpp"

// TODO When C++23, and nvcc->nvc++, use mdspan instead.

namespace noa {
    // Span (C++20 std::span could replace this).
    // If SIZE==-1, the span is dynamic. Otherwise, it is the static size of the span.
    template<typename Value, int64_t SIZE = -1, typename Index = int64_t>
    class Span {
    public:
        static_assert(!std::is_reference_v<Value> &&
                      !std::is_pointer_v<Value> &&
                      !std::extent_v<Value> &&
                      std::is_integral_v<Index> &&
                      SIZE >= -1);

        using value_type = Value;
        using index_type = Index;
        using ssize_type = index_type;
        using difference_type = index_type;
        using size_type = std::make_unsigned_t<index_type>;

        using mutable_value_type = std::remove_const_t<value_type>;
        using const_value_type = std::add_const_t<mutable_value_type>;

        using reference = Value&;
        using const_reference = const std::remove_const_t<Value>&;
        using pointer = Value*;
        using const_pointer = const std::remove_const_t<Value>*;
        using iterator = pointer;
        using const_iterator = const_pointer;

        static constexpr bool IS_STATIC = SIZE >= 0;
        using index_type_or_empty = std::conditional_t<IS_STATIC, noa::traits::Empty, index_type>;

    public: // Empty
        constexpr Span() = default;

        template<typename Int, typename std::enable_if_t<!IS_STATIC && std::is_integral_v<Int>, bool> = true>
        NOA_HD constexpr Span(pointer data, Int size) noexcept
                : m_data(data), m_ssize(static_cast<ssize_type>(size)) {}

        template<typename Void = void, typename std::enable_if_t<IS_STATIC && std::is_void_v<Void>, bool> = true>
        NOA_HD constexpr explicit Span(pointer data) noexcept : m_data(data) {}

        // Creates a const accessor from an existing non-const accessor.
        template<typename U, typename = std::enable_if_t<details::is_mutable_value_type_v<U, value_type>>>
        NOA_HD constexpr /* implicit */ Span(const Span<U, SIZE, Index>& span)
                : m_data(span.data()) {
            if constexpr (!IS_STATIC)
                m_ssize = span.ssize();
        }

    public: // Range
        [[nodiscard]] NOA_HD constexpr ssize_type ssize() const noexcept {
            if constexpr (IS_STATIC)
                return SIZE;
            else
                return m_ssize;
        };

        [[nodiscard]] NOA_HD constexpr size_type size() const noexcept { return static_cast<size_type>(ssize()); };
        [[nodiscard]] NOA_HD constexpr pointer data() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr iterator begin() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr const_iterator cbegin() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr iterator end() const noexcept { return m_data + ssize(); }
        [[nodiscard]] NOA_HD constexpr const_iterator cend() const noexcept { return m_data + ssize(); }

        // Structure binding support.
        template<int I, typename Void = void, typename = std::enable_if_t<std::is_void_v<Void> && IS_STATIC>>
        [[nodiscard]] NOA_HD constexpr const value_type& get() const noexcept { return m_data[I]; }

    public:
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return !m_data || ssize() <= 0; }
        [[nodiscard]] NOA_HD constexpr auto as_const() const noexcept {
            return Span<const_value_type, SIZE, Index>(*this);
        }
        [[nodiscard]] NOA_HD constexpr auto as_bytes() const noexcept {
            using output_t = std::conditional_t<std::is_const_v<value_type>, const std::byte, std::byte>;
            if constexpr (IS_STATIC) {
                constexpr auto NEW_SIZE = SIZE * static_cast<index_type>(sizeof(value_type));
                return Span<output_t, NEW_SIZE, Index>(reinterpret_cast<output_t*>(data()));
            } else {
                return Span(reinterpret_cast<output_t*>(data()), ssize() * sizeof(value_type));
            }
        }

    public: // Elements access
        [[nodiscard]] NOA_HD constexpr reference front() const noexcept {
            NOA_ASSERT(!is_empty());
            return m_data[0];
        }
        [[nodiscard]] NOA_HD constexpr reference back() const noexcept {
            NOA_ASSERT(!is_empty());
            return m_data[ssize() - 1];
        }

        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        [[nodiscard]] NOA_HD constexpr reference operator[](Int index) const noexcept {
            NOA_ASSERT(!is_empty() && index >= 0 && index < ssize());
            return m_data[index];
        }

        // Guaranteed bound-check. Throws if out-of-bound.
        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        [[nodiscard]] NOA_HOST constexpr reference at(Int index) const {
            NOA_CHECK(!is_empty() && index >= 0 && index < ssize(),
                      "Out-of-bound access. Size={}, index={}", ssize(), index);
            return m_data[index];
        }

    public:
        // Support for noa::string::human<Span>();
        [[nodiscard]] static std::string name() {
            return noa::string::format(
                    "Span<{},{},{}>",
                    noa::string::human<value_type>(), SIZE, noa::string::human<index_type>());
        }

    private:
        pointer m_data{};
        NOA_NO_UNIQUE_ADDRESS index_type_or_empty m_ssize{};
    };
}

// Support for output stream:
namespace noa {
    template<typename T>
    inline std::ostream& operator<<(std::ostream& os, const Span<T>& v) {
        if constexpr (noa::traits::is_real_or_complex_v<T>)
            os << string::format("{::.3f}", v); // {fmt} ranges
        else
            os << string::format("{}", v);
        return os;
    }
}

// Support for structure bindings:
namespace std {
    template<typename T, int64_t N>
    struct tuple_size<noa::Span<T, N>> : std::integral_constant<size_t, static_cast<size_t>(N)> {};

    template<size_t I, int64_t N, typename T>
    struct tuple_element<I, noa::Span<T, N>> { using type = T; };
}
