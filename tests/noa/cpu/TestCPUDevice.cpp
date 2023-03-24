#include <noa/core/utils/Irange.hpp>
#include <noa/cpu/Device.hpp>

#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::Device", "[noa][cpu]") {
    const cpu::DeviceMemory mem_info = cpu::Device::memory();
    REQUIRE(mem_info.free != 0);
    REQUIRE(mem_info.total != 0);

    const cpu::DeviceCore core_info = cpu::Device::cores();
    REQUIRE(core_info.logical != 0);
    REQUIRE(core_info.physical != 0);

    for (auto level: irange(3)) {
        const cpu::DeviceCache cache_info = cpu::Device::cache(level);
        REQUIRE(cache_info.size != 0);
        REQUIRE(cache_info.line_size != 0);
    }

    // Simply check that it runs without throwing an exception.
    [[maybe_unused]] const std::string s = cpu::Device::summary();
}
