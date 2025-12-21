#include <noa/base/Half.hpp>
#include <noa/base/Math.hpp>

#include "Catch.hpp"

TEST_CASE("base::Half") {
    using namespace ::noa::types;

    static_assert(sizeof(f16) == 2);
    static_assert(noa::traits::is_numeric_v<f16> and std::is_same_v<f16, f16>);

    f16 ha{};
    REQUIRE(ha == f16{0});
    auto* mem = reinterpret_cast<char*>(&ha);
    REQUIRE((mem[0] == 0 && mem[1] == 0));

    ha = f16{};
    ha += f16{1}; REQUIRE(ha == f16(1));
    ha -= f16{3}; REQUIRE(ha == f16(-2));
    ha *= f16{4.5}; REQUIRE(ha == f16(-9));
    ha /= f16{-3}; REQUIRE(ha == f16(3));

    ha = f16{};
    ha += 1; REQUIRE(ha == f16(1));
    ha -= 3; REQUIRE(ha == f16(-2));
    ha *= 4.5; REQUIRE(ha == f16(-9));
    ha /= -3; REQUIRE(ha == f16(3));

    // Bug was found for int2half with min values. I've contacted the author,
    // but in the meantime added a fix on the f16 cast functions.
    REQUIRE(f16(std::numeric_limits<int8_t>::min()) == f16(f64(std::numeric_limits<int8_t>::min())));
    REQUIRE(f16(std::numeric_limits<int16_t>::min()) == f16(f64(std::numeric_limits<int16_t>::min())));
    REQUIRE(f16(std::numeric_limits<int32_t>::min()) == f16(f64(std::numeric_limits<int32_t>::min())));
    REQUIRE(f16(std::numeric_limits<int64_t>::min()) == f16(f64(std::numeric_limits<int64_t>::min())));

    REQUIRE((f16(2000), noa::clamp(f16(3000), f16(0), f16(2000))));

    f16 hb(4);
    REQUIRE(true == static_cast<bool>(hb));
    REQUIRE(i16{4} == static_cast<i16>(hb));
    REQUIRE(u16{4} == static_cast<u16>(hb));
    REQUIRE(i32{4} == static_cast<i32>(hb));
    REQUIRE(u32{4} == static_cast<u32>(hb));
    REQUIRE(i64{4} == static_cast<i64>(hb));
    REQUIRE(u64{4} == static_cast<u64>(hb));
    REQUIRE(f32{4} == static_cast<f32>(hb));
    REQUIRE(f64{4} == static_cast<f64>(hb));

    f16 hc(2.5);
    REQUIRE(fmt::format("{:.2}", hc) == "2.5");
}
