#include <noa/common/types/Half.h>
#include <catch2/catch.hpp>

TEST_CASE("common::half_t", "[noa][common][half]") {
    using namespace ::noa;

    static_assert(sizeof(half_t) == 2);
    static_assert(traits::is_float_v<half_t>);

    half_t ha;
    REQUIRE(ha == half_t{0});
    auto* mem = reinterpret_cast<char*>(&ha);
    REQUIRE((mem[0] == 0 && mem[1] == 0));

    ha = half_t{};
    ha += half_t{1}; REQUIRE(ha == half_t(1));
    ha -= half_t{3}; REQUIRE(ha == half_t(-2));
    ha *= half_t{4.5}; REQUIRE(ha == half_t(-9));
    ha /= half_t{-3}; REQUIRE(ha == half_t(3));

    ha = half_t{};
    ha += 1; REQUIRE(ha == half_t(1));
    ha -= 3; REQUIRE(ha == half_t(-2));
    ha *= 4.5; REQUIRE(ha == half_t(-9));
    ha /= -3; REQUIRE(ha == half_t(3));

    half_t hb(4);
    REQUIRE(true == static_cast<bool>(hb));
    REQUIRE(short{4} == static_cast<short>(hb));
    REQUIRE(ushort{4} == static_cast<ushort>(hb));
    REQUIRE(int{4} == static_cast<int>(hb));
    REQUIRE(uint{4} == static_cast<uint>(hb));
    REQUIRE(int64_t{4} == static_cast<int64_t>(hb));
    REQUIRE(uint64_t{4} == static_cast<uint64_t>(hb));
    REQUIRE(float{4} == static_cast<float>(hb));
    REQUIRE(double{4} == static_cast<double>(hb));

    half_t hc(2.5);
    REQUIRE(string::format("{:.2}", hc) == "2.5");
}
