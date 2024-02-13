#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/string/Format.hpp"
#include "noa/core/math/Comparison.hpp"

// TODO C++20 std::span could replace this

namespace noa::inline types {
    /// One dimensional (contiguous) span.
    /// If SIZE==-1, the span is dynamic. Otherwise, it is the static size of the span.
    template<typename Value, int64_t SIZE = -1, typename Index = int64_t>
    class Span {
    public:
        static_assert(not std::is_reference_v<Value> and
                      not std::is_pointer_v<Value> and
                      not std::extent_v<Value> and
                      std::is_integral_v<Index>);

        using value_type = Value;
        using index_type = Index;
        using ssize_type = std::make_signed_t<index_type>;
        using size_type = std::make_unsigned_t<index_type>;
        using difference_type = ssize_type;

        using mutable_value_type = std::remove_const_t<value_type>;
        using const_value_type = std::add_const_t<mutable_value_type>;

        using reference = Value&;
        using const_reference = const std::remove_const_t<Value>&;
        using pointer = Value*;
        using const_pointer = const std::remove_const_t<Value>*;
        using iterator = pointer;
        using const_iterator = const_pointer;

        static constexpr bool IS_STATIC = SIZE >= 0;

    public: // Empty
        NOA_HD constexpr Span() = default;

        NOA_HD constexpr Span(pointer data, index_type size) noexcept : m_data(data) {
            if constexpr (not IS_STATIC)
                m_size = size;
        }

        template<size_t N> requires (static_cast<int64_t>(N) == SIZE)
        NOA_HD constexpr explicit Span(value_type (& data)[N]) noexcept: m_data(data) {}

        /// Creates a const accessor from an existing non-const accessor.
        template<typename U> requires nt::is_mutable_value_type_v<U, value_type>
        NOA_HD constexpr /* implicit */ Span(const Span<U, SIZE, index_type>& span) : m_data(span.data()) {
            if constexpr (not IS_STATIC)
                m_size = static_cast<index_type>(span.ssize());
        }

    public: // Range
        [[nodiscard]] NOA_HD constexpr ssize_type ssize() const noexcept {
            if constexpr (IS_STATIC)
                return static_cast<ssize_type>(SIZE);
            else
                return static_cast<ssize_type>(m_size);
        };

        [[nodiscard]] NOA_HD constexpr size_type size() const noexcept { return static_cast<size_type>(ssize()); };
        [[nodiscard]] NOA_HD constexpr pointer data() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr iterator begin() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr const_iterator cbegin() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr iterator end() const noexcept { return m_data + ssize(); }
        [[nodiscard]] NOA_HD constexpr const_iterator cend() const noexcept { return m_data + ssize(); }

        // Structure binding support.
        template<int I> requires IS_STATIC
        [[nodiscard]] NOA_HD constexpr const value_type& get() const noexcept { return m_data[I]; }

    public:
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return not m_data or ssize() <= 0; }
        [[nodiscard]] NOA_HD constexpr auto as_const() const noexcept {
            return Span<const_value_type, SIZE, index_type>(*this);
        }
        [[nodiscard]] NOA_HD constexpr auto as_bytes() const noexcept {
            using output_t = std::conditional_t<std::is_const_v<value_type>, const std::byte, std::byte>;
            auto bytes_ptr = reinterpret_cast<output_t*>(data());
            if constexpr (IS_STATIC) {
                constexpr auto NEW_SIZE = SIZE * static_cast<int64_t>(sizeof(value_type));
                return Span<output_t, NEW_SIZE, index_type>(bytes_ptr, static_cast<index_type>(NEW_SIZE));
            } else {
                const auto new_size = static_cast<index_type>(size() * sizeof(value_type));
                return Span<output_t, -1, index_type>(bytes_ptr, new_size);
            }
        }

    public: // Elements access
        [[nodiscard]] NOA_HD constexpr reference front() const noexcept {
            NOA_ASSERT(not is_empty());
            return m_data[0];
        }

        [[nodiscard]] NOA_HD constexpr reference back() const noexcept {
            NOA_ASSERT(not is_empty());
            return m_data[ssize() - 1];
        }

        [[nodiscard]] NOA_HD constexpr reference operator[](std::integral auto index) const noexcept {
            NOA_ASSERT(not is_empty() and index >= 0 and index < ssize());
            return m_data[index];
        }

#if defined(NOA_IS_OFFLINE)
        /// Guaranteed bound-check. Throws if out-of-bound.
        [[nodiscard]] constexpr reference at(std::integral auto index) const {
            check(not is_empty() and index >= 0 and index < ssize(),
                  "Out-of-bound access; size={}, index={}", ssize(), index);
            return m_data[index];
        }

        [[nodiscard]] static std::string name() {
            return fmt::format(
                    "Span<{},{},{}>",
                    ns::to_human_readable<value_type>(), SIZE,
                    ns::to_human_readable<index_type>());
        }
#endif

    private:
        pointer m_data{};
        NOA_NO_UNIQUE_ADDRESS std::conditional_t<IS_STATIC, Empty, index_type> m_size{};
    };

    /// Additional deduction guide.
    template<typename T, size_t N>
    Span(T(&)[N]) -> Span<T, static_cast<i64>(N)>;

#if defined(NOA_IS_OFFLINE)
    template<typename T>
    inline std::ostream& operator<<(std::ostream& os, const Span<T>& v) {
        if constexpr (nt::is_real_or_complex_v<T>)
            os << fmt::format("{::.3f}", v); // {fmt} ranges
        else
            os << fmt::format("{}", v);
        return os;
    }
#endif
}

// Support for structure bindings:
namespace std {
    template<typename T, int64_t N>
    struct tuple_size<noa::Span<T, N>> : std::integral_constant<size_t, static_cast<size_t>(N)> {};

    template<size_t I, int64_t N, typename T>
    struct tuple_element<I, noa::Span<T, N>> { using type = T; };

    template<typename T, int64_t N>
    struct tuple_size<const noa::Span<T, N>> : std::integral_constant<size_t, static_cast<size_t>(N)> {};

    template<size_t I, int64_t N, typename T>
    struct tuple_element<I, const noa::Span<T, N>> { using type = const T; };
}
