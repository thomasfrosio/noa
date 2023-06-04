#pragma once

#include "noa/core/Definitions.hpp"

namespace noa {
    // Naive Pair. Mostly used to have a std::pair like working on device code.
    template<typename T, typename U>
    struct Pair {
        T first;
        U second;

        constexpr Pair() noexcept = default;
        constexpr NOA_FHD Pair(T first_, U second_) noexcept : first(first_), second(second_) {}

        constexpr NOA_FHD explicit Pair(const std::pair<T, U>& p) noexcept : first(p.first), second(p.second) {}
        constexpr NOA_FHD explicit Pair(std::pair<T, U>&& p) noexcept : first(std::move(p.first)), second(std::move(p.second)) {}

        constexpr NOA_FHD explicit Pair(const std::tuple<T, U>& p) noexcept : first(p.first), second(p.second) {}
        constexpr NOA_FHD explicit Pair(std::tuple<T, U>&& p) noexcept : first(std::move(p.first)), second(std::move(p.second)) {}
    };
}
