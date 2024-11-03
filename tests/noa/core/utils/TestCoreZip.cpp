#include <noa/core/utils/Zip.hpp>
#include <catch2/catch.hpp>

#include <list>
#include <vector>
#include <ranges>

TEST_CASE("core::zip") {
    namespace stdr = std::ranges;
    namespace ng = noa::guts;

    {
        auto a = std::vector{1, 2, 3, 4, 5, 6};
        auto za = noa::zip(a, std::vector{1, 2, 3, 4, 5, 6});
        auto za_begin = za.begin();
        auto za_end = za.end();
        static_assert(std::is_same_v<decltype(za), ng::ZipRange<std::vector<int>&, std::vector<int>>>);
        static_assert(std::is_same_v<decltype(za_begin), ng::ZipIterator<stdr::iterator_t<std::vector<int>&>, stdr::iterator_t<std::vector<int>>>>);
        static_assert(std::is_same_v<decltype(za_end), ng::ZipIterator<stdr::sentinel_t<std::vector<int>&>, stdr::sentinel_t<std::vector<int>>>>);
        REQUIRE(za_begin == za_begin);
        REQUIRE(za_begin != za_end);
    }

    {
        auto a = std::vector{1, 2, 3, 4, 5, 6};
        auto b = std::vector{1, 2, 3, 4, 5, 6, 7};
        auto c = std::vector{0, 0, 0, 0, 0};
        auto const& d = b;
        for (auto&& [x, y, z]: noa::zip(a, d, c)) {
            z = x + y;
            x++;
        }
        REQUIRE(a == std::vector{2, 3, 4, 5, 6, 6});
        REQUIRE(b == std::vector{1, 2, 3, 4, 5, 6, 7});
        REQUIRE(c == std::vector{2, 4, 6, 8, 10});
    }

    {
        auto a = std::vector{2, 3, 4, 5, 6, 6};
        auto b = std::list{1, 2, 3, 4, 5, 6};
        std::list<int> c;
        for (auto&& [x, y]: noa::zip(a, b))
            c.push_back(x + y);
        REQUIRE(c == std::list{3, 5, 7, 9, 11, 12});
    }

    {
        const auto a = std::list{3, 5, 7, 9, 11, 12};
        auto b = std::vector{false, false, false, false, false, true};
        for (auto&& [x, y]: noa::zip(a, b)) {
            REQUIRE((x % 2) != y);
        }

        auto c = std::vector<bool>(6, false);
        for (auto&& [x, y]: noa::zip(b, c))
            y = !x;
        REQUIRE(c == std::vector{true, true, true, true, true, false});
    }
}
