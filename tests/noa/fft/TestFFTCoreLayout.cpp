#include <noa/base/Error.hpp>
#include <noa/fft/core/Layout.hpp>

#include "Catch.hpp"

namespace {
    namespace nf = noa::fft;

    template<nf::Layout REMAP>
    constexpr auto test_func() {
        return REMAP;
    }
}

TEST_CASE("fft::Layout") {
    constexpr auto a0 = nf::Layout("h2hc");
    bool is_ok{};
    switch (a0) { // implicit conversion to the underlying enum type
        case nf::Layout::H2H:
        case nf::Layout::H2HC: {
            is_ok = true;
            break;
        }
        default: {}
    }
    REQUIRE(is_ok);

    constexpr auto r0 = test_func<"h2f">();
    static_assert(r0 == nf::Layout::H2F);

    constexpr auto r1 = test_func<"hc2f">();
    static_assert(r1 == nf::Layout::HC2F and not r1.is_xx2xc());
    static_assert(r1.is_hx2xx() and r1.is_hc2xx() and r1.is_xx2f() and r1.is_xx2fx());
    static_assert(r1.is_any(nf::Layout::H2F, nf::Layout::HC2FC, nf::Layout::HC2F));
    static_assert(r1.to_x2xx() == nf::Layout::H2F);
    static_assert(r1.to_xx2xc() == nf::Layout::HC2FC);
    static_assert(r1.flip() == nf::Layout::F2HC);

    // constexpr auto a1 = nf::Layout("h2hF"); // rightfully fails to compile
    REQUIRE_THROWS_AS(nf::Layout("h2hF"), noa::Exception);
    REQUIRE(fmt::format("{}", r1) == "hc2f");
}
