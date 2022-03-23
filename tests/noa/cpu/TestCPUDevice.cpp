#include <noa/common/Irange.h>
#include <noa/cpu/Device.h>

#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::Device", "[noa][cpu]") {
    cpu::DeviceMemory mem_info = cpu::Device::memory();
    REQUIRE(mem_info.free != 0);
    REQUIRE(mem_info.total != 0);

    cpu::DeviceCore core_info = cpu::Device::cores();
    REQUIRE(core_info.logical != 0);
    REQUIRE(core_info.physical != 0);

    for (auto level: irange(3)) {
        cpu::DeviceCache cache_info = cpu::Device::cache(level);
        REQUIRE(cache_info.size != 0);
        REQUIRE(cache_info.line_size != 0);
    }

    // Simply check that it runs without throwing an exception.
    [[maybe_unused]] std::string s = cpu::Device::summary();
}
