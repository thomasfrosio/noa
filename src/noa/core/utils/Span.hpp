#pragma once

#include <cstdint>
#include <type_traits>
#include "noa/core/Definitions.hpp"
#include "noa/core/Assert.hpp"

namespace noa {
    // Naive span.
    template<typename Value>
    class Span {
    public:
        using value_type = Value;
        using index_type = int64_t;

    public:
        constexpr Span() = default;

        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        NOA_HD constexpr Span(value_type* data, Int size) noexcept
                : m_data(data), m_size(static_cast<index_type>(size)) {}

        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        [[nodiscard]] NOA_HD constexpr Value& operator[](Int index) const noexcept {
            NOA_ASSERT(index > 0 && index < m_size);
            return m_data[index];
        }

    public:
        [[nodiscard]] NOA_HD constexpr value_type* data() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr size_t size() const noexcept { return static_cast<size_t>(m_size); };
        [[nodiscard]] NOA_HD constexpr index_type ssize() const noexcept { return m_size; };
        [[nodiscard]] NOA_HD constexpr value_type* begin() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr value_type* cbegin() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr value_type* end() const noexcept { return m_data + size(); }
        [[nodiscard]] NOA_HD constexpr value_type* cend() const noexcept { return m_data + size(); }

    private:
        Value* m_data{};
        index_type m_size{};
    };
}
