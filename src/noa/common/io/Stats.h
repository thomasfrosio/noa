/// \file noa/common/types/Stats.h
/// \brief The Stats type, which gather the basic statistics of an array.
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020

#pragma once

#include "noa/common/Definitions.h"

namespace noa::io {
    /// Statistics of an image file's data.
    /// To check if a field is set, use the `has*()` member functions.
    template<typename T>
    struct Stats {
    public: // getters
        T min() const noexcept;
        T max() const noexcept;
        T sum() const noexcept;
        T mean() const noexcept;
        T var() const noexcept;
        T std() const noexcept;

    public: // setters
        void min(T min) noexcept;
        void max(T max) noexcept;
        void sum(T sum) noexcept;
        void mean(T mean) noexcept;
        void var(T var) noexcept;
        void std(T std) noexcept;

    public: // checkers
        [[nodiscard]] bool hasMin() const noexcept;
        [[nodiscard]] bool hasMax() const noexcept;
        [[nodiscard]] bool hasSum() const noexcept;
        [[nodiscard]] bool hasMean() const noexcept;
        [[nodiscard]] bool hasVar() const noexcept;
        [[nodiscard]] bool hasStd() const noexcept;

    private:
        T m_min{}, m_max{}, m_sum{}, m_mean{}, m_var{}, m_std{};
        uint8_t m_has_value{}; // bitfield, one per stat, same order as in the structure
    };

    using stats_t = Stats<float>;
}


namespace noa::io {
    template<typename T>
    T Stats<T>::min() const noexcept { return m_min; }

    template<typename T>
    T Stats<T>::max() const noexcept { return m_max; }

    template<typename T>
    T Stats<T>::sum() const noexcept { return m_sum; }

    template<typename T>
    T Stats<T>::mean() const noexcept { return m_mean; }

    template<typename T>
    T Stats<T>::var() const noexcept { return m_var; }

    template<typename T>
    T Stats<T>::std() const noexcept { return m_std; }

    template<typename T>
    void Stats<T>::min(T min) noexcept {
        m_min = min;
        m_has_value |= 1 << 0;
    }

    template<typename T>
    void Stats<T>::max(T max) noexcept {
        m_max = max;
        m_has_value |= 1 << 1;
    }

    template<typename T>
    void Stats<T>::sum(T sum) noexcept {
        m_sum = sum;
        m_has_value |= 1 << 2;
    }

    template<typename T>
    void Stats<T>::mean(T mean) noexcept {
        m_mean = mean;
        m_has_value |= 1 << 3;
    }

    template<typename T>
    void Stats<T>::var(T var) noexcept {
        m_var = var;
        m_has_value |= 1 << 4;
    }

    template<typename T>
    void Stats<T>::std(T std) noexcept {
        m_std = std;
        m_has_value |= 1 << 5;
    }

    template<typename T>
    bool Stats<T>::hasMin() const noexcept { return m_has_value & 1 << 0; }

    template<typename T>
    bool Stats<T>::hasMax() const noexcept { return m_has_value & 1 << 1; }

    template<typename T>
    bool Stats<T>::hasSum() const noexcept { return m_has_value & 1 << 2; }

    template<typename T>
    bool Stats<T>::hasMean() const noexcept { return m_has_value & 1 << 3; }

    template<typename T>
    bool Stats<T>::hasVar() const noexcept { return m_has_value & 1 << 4; }

    template<typename T>
    bool Stats<T>::hasStd() const noexcept { return m_has_value & 1 << 5; }
}
