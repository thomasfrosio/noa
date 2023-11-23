#include <noa/core/types/Half.hpp>
#include <catch2/catch.hpp>

TEST_CASE("core::Half", "[noa][core]") {
    using namespace ::noa;

    static_assert(sizeof(Half) == 2);
    static_assert(traits::is_numeric_v<Half>);

    Half ha;
    REQUIRE(ha == Half{0});
    auto* mem = reinterpret_cast<char*>(&ha);
    REQUIRE((mem[0] == 0 && mem[1] == 0));

    ha = Half{};
    ha += Half{1}; REQUIRE(ha == Half(1));
    ha -= Half{3}; REQUIRE(ha == Half(-2));
    ha *= Half{4.5}; REQUIRE(ha == Half(-9));
    ha /= Half{-3}; REQUIRE(ha == Half(3));

    ha = Half{};
    ha += 1; REQUIRE(ha == Half(1));
    ha -= 3; REQUIRE(ha == Half(-2));
    ha *= 4.5; REQUIRE(ha == Half(-9));
    ha /= -3; REQUIRE(ha == Half(3));

    // Bug was found for int2half with min values. I've contacted the author,
    // but in the meantime added a fix on the Half cast functions.
    REQUIRE(Half(std::numeric_limits<int8_t>::min()) == Half(double(std::numeric_limits<int8_t>::min())));
    REQUIRE(Half(std::numeric_limits<int16_t>::min()) == Half(double(std::numeric_limits<int16_t>::min())));
    REQUIRE(Half(std::numeric_limits<int32_t>::min()) == Half(double(std::numeric_limits<int32_t>::min())));
    REQUIRE(Half(std::numeric_limits<int64_t>::min()) == Half(double(std::numeric_limits<int64_t>::min())));

    REQUIRE((Half(2000), noa::clamp(Half(3000), Half(0), Half(2000))));

    Half hb(4);
    REQUIRE(true == static_cast<bool>(hb));
    REQUIRE(short{4} == static_cast<short>(hb));
    REQUIRE(ushort{4} == static_cast<ushort>(hb));
    REQUIRE(int{4} == static_cast<int>(hb));
    REQUIRE(uint{4} == static_cast<uint>(hb));
    REQUIRE(int64_t{4} == static_cast<int64_t>(hb));
    REQUIRE(uint64_t{4} == static_cast<uint64_t>(hb));
    REQUIRE(float{4} == static_cast<float>(hb));
    REQUIRE(double{4} == static_cast<double>(hb));

    Half hc(2.5);
    REQUIRE(fmt::format("{:.2}", hc) == "2.5");
}
