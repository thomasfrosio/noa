#include <noa/core/utils/Irange.hpp>
#include <noa/cpu/Device.hpp>

#include "Catch.hpp"

TEST_CASE("cpu::Device") {
    using noa::cpu::Device;

    // Simply check that it runs without throwing an exception.
    [[maybe_unused]] const std::string s = Device::summary();

    const auto mem_info = Device::memory();
    REQUIRE(mem_info.free != 0);
    REQUIRE(mem_info.total != 0);

    const auto core_info = Device::cores();
    REQUIRE(core_info.logical != 0);
    REQUIRE(core_info.physical != 0);

    for (auto level: noa::irange(1, 3)) {
        const auto cache_info = Device::cache(level);
        REQUIRE(cache_info.size != 0);
        REQUIRE(cache_info.line_size != 0);
    }
}
