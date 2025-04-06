#include <noa/core/Enums.hpp>

#include "Catch.hpp"

namespace {
    template<noa::Remap REMAP>
    constexpr auto test_func() {
        return REMAP;
    }
}

TEST_CASE("core::Remap") {
    constexpr auto a0 = noa::Remap("h2hc");
    bool is_ok{};
    switch (a0) { // implicit conversion to the underlying enum type
        case noa::Remap::H2H:
        case noa::Remap::H2HC: {
            is_ok = true;
            break;
        }
        default: {}
    }
    REQUIRE(is_ok);

    constexpr auto r0 = test_func<"h2f">();
    static_assert(r0 == noa::Remap::H2F);

    constexpr auto r1 = test_func<"hc2f">();
    static_assert(r1 == noa::Remap::HC2F and not r1.is_xx2xc());
    static_assert(r1.is_hx2xx() and r1.is_hc2xx() and r1.is_xx2f() and r1.is_xx2fx());
    static_assert(r1.is_any(noa::Remap::H2F, noa::Remap::HC2FC, noa::Remap::HC2F));
    static_assert(r1.to_x2xx() == noa::Remap::H2F);
    static_assert(r1.to_xx2xc() == noa::Remap::HC2FC);
    static_assert(r1.flip() == noa::Remap::F2HC);

    // constexpr auto a1 = noa::Remap("h2hF"); // rightfully fails to compile
    REQUIRE_THROWS_AS(noa::Remap("h2hF"), noa::Exception);
    REQUIRE(fmt::format("{}", r1) == "hc2f");
}
