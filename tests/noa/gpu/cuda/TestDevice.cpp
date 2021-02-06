#include <catch2/catch.hpp>
#include "noa/gpu/cuda/Device.h"

using namespace Noa;

TEST_CASE("Device", "[noa][cuda]") {


    if (GPU::Device::getCount() > 1) {
        GPU::Device device = GPU::Device::getCurrent();
        REQUIRE(device.id() == 0);
        GPU::Device::memory_t mem_info = GPU::Device::getMemoryInfo(device);
    }
}
