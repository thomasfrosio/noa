#include <noa/core/types/Span.hpp>

#include <catch2/catch.hpp>

TEST_CASE("core::Span") {
    std::vector<double> buffer{0, 1, 2, 3, 4, 5};
    const noa::Span<double, 2> span(buffer.data());
    span[2] = 12;
    REQUIRE(buffer[2] == 12);

    const auto const_span = span.as_const();
    static_assert(std::is_const_v<noa::traits::value_type_t<decltype(const_span)>>);

    fmt::print("{}", span.as_bytes());

    const auto [a, b] = span;
    REQUIRE((a == buffer[0] && b == buffer[1]));

    [[maybe_unused]] const noa::Span<const std::byte, 16> span1 = span.as_const().as_bytes();
}
