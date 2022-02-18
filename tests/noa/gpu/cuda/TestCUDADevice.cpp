#include "noa/gpu/cuda/Device.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cuda::Device", "[noa][cuda]") {
    using namespace cuda;

    std::vector<Device> devices = Device::getAll();
    REQUIRE(devices.size() == Device::getCount());

    {
        Device device("cuda:0");
        REQUIRE(device.id() == 0);
        device = Device("cuda:1");
        REQUIRE(device.id() == 1);
        device = Device("cuda:12");
        REQUIRE(device.id() == 12);
    }

    // Quite difficult to have meaningful tests, so just make sure it doesn't throw.
    for (auto device : devices) {
        [[maybe_unused]] const auto properties = device.properties();
        [[maybe_unused]] const auto attribute = device.attribute(cudaDevAttrCanMapHostMemory);
        REQUIRE(device.architecture() == device.capability().major);

        REQUIRE(versionRuntime() <= versionDriver());
        device_memory_info_t info = device.memory();
        REQUIRE(info.total >= info.free);

        [[maybe_unused]] const auto limit = device.limit(cudaLimitStackSize);
        device.synchronize();
        device.reset();
    }

    if (devices.size() == 1) {
        REQUIRE(Device::getMostFree() == Device::getCurrent());
        REQUIRE(devices[0] == Device::getCurrent());
    }

    if (devices.size() > 1) {
        Device::setCurrent(devices[0]);
        {
            DeviceCurrentScope scope_device(devices[1]);
            REQUIRE(scope_device == Device::getCurrent());
        }
        REQUIRE(devices[0] == Device::getCurrent());
    }
}
