#include <noa/core/types/Span.hpp>

#include <catch2/catch.hpp>

TEST_CASE("core::Span") {
    using namespace noa::types;

    std::vector<double> buffer{0, 1, 2, 3, 4, 5};
    const auto s0 = Span(buffer.data(), buffer.size());
    s0[1] = 12;
    REQUIRE(buffer[1] == 12);
    REQUIRE_THROWS(s0.at(10));

    int aa[2]{1, 2};
    const int bb[11]{};
    static_assert(std::is_same_v<Span<int, 2>, decltype(Span{aa})>);
    static_assert(std::is_same_v<Span<const int, 11>, decltype(Span{bb})>);

    [[maybe_unused]] const Span const_span = s0.as_const();
    static_assert(std::is_same_v<const Span<const double, -1, size_t>, decltype(const_span)>);

    fmt::print("{}", s0.as_bytes());

    const auto s1 = Span{aa};
    static_assert(std::is_same_v<const Span<int, 2, i64>, decltype(s1)>);
    const auto [a, b] = s1;
    REQUIRE((a == 1 and b == 2));

    [[maybe_unused]] const noa::Span<const std::byte, 8> span1 = s1.as_const().as_bytes();
}
