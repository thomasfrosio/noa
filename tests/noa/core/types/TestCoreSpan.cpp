#include <noa/core/types/Span.hpp>

#include <catch2/catch.hpp>

TEST_CASE("core::Span") {
    std::vector<double> buffer{0, 1, 2, 3, 4, 5};
    noa::Span<double, 1> span(buffer.data());

    fmt::print("{}", span.as_bytes());

    const auto [a] = span;

    noa::Span<const std::byte, 8> span1 = span.as_const().as_bytes();
    span[2] = 12;
}
