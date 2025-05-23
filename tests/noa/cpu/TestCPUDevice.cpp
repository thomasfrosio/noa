#include <noa/core/utils/Irange.hpp>
#include <noa/cpu/Device.hpp>

#include "Catch.hpp"

TEST_CASE("cpu::Device") {
    using noa::cpu::Device;

    #ifdef NOA_PLATFORM_APPLE
    return; // while we compile on MacOS, we still don't parse the device properties...
    #endif

    const auto mem_info = Device::memory();
    REQUIRE(mem_info.free != 0);
    REQUIRE(mem_info.total != 0);

    const auto core_info = Device::cores();
    REQUIRE(core_info.logical != 0);
    REQUIRE(core_info.physical != 0);

    for (auto level: noa::irange(3)) {
        const auto cache_info = Device::cache(level);
        REQUIRE(cache_info.size != 0);
        REQUIRE(cache_info.line_size != 0);
    }

    // Simply check that it runs without throwing an exception.
    [[maybe_unused]] const std::string s = Device::summary();
}
