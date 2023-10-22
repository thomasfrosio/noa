#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

namespace noa::io {
    /// Statistics of an image file's data.
    /// To check if a field is set, use the `has_*()` member functions.
    template<typename Value>
    struct Stats {
    public:
        static_assert(nt::is_real_or_complex_v<Value>);
        using value_type = Value;
        using real_type = nt::value_type_t<Value>;

    public: // getters
        [[nodiscard]] NOA_HD constexpr value_type min() const noexcept { return m_min; }
        [[nodiscard]] NOA_HD constexpr value_type max() const noexcept { return m_max; }
        [[nodiscard]] NOA_HD constexpr value_type sum() const noexcept { return m_sum; }
        [[nodiscard]] NOA_HD constexpr value_type mean() const noexcept { return m_mean; }
        [[nodiscard]] NOA_HD constexpr real_type var() const noexcept { return m_var; }
        [[nodiscard]] NOA_HD constexpr real_type std() const noexcept { return m_std; }

    public: // setters
        NOA_HD constexpr void set_min(value_type min) noexcept {
            m_min = min;
            m_has_value |= 1 << 0;
        }

        NOA_HD constexpr void set_max(value_type max) noexcept {
            m_max = max;
            m_has_value |= 1 << 1;
        }

        NOA_HD constexpr void set_sum(value_type sum) noexcept {
            m_sum = sum;
            m_has_value |= 1 << 2;
        }

        NOA_HD constexpr void set_mean(value_type mean) noexcept {
            m_mean = mean;
            m_has_value |= 1 << 3;
        }

        NOA_HD constexpr void set_var(real_type var) noexcept {
            m_var = var;
            m_has_value |= 1 << 4;
        }

        NOA_HD constexpr void set_std(real_type std) noexcept {
            m_std = std;
            m_has_value |= 1 << 5;
        }

    public: // checkers
        [[nodiscard]] bool has_min() const noexcept { return m_has_value & (1 << 0); }
        [[nodiscard]] bool has_max() const noexcept { return m_has_value & (1 << 1); }
        [[nodiscard]] bool has_sum() const noexcept { return m_has_value & (1 << 2); }
        [[nodiscard]] bool has_mean() const noexcept { return m_has_value & (1 << 3); }
        [[nodiscard]] bool has_var() const noexcept { return m_has_value & (1 << 4); }
        [[nodiscard]] bool has_std() const noexcept { return m_has_value & (1 << 5); }

    private:
        value_type m_min{};
        value_type m_max{};
        value_type m_sum{};
        value_type m_mean{};
        real_type m_var{};
        real_type m_std{};
        uint8_t m_has_value{}; // bitfield, one per stat, same order as in the structure
    };
}
