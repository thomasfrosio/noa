#include "noa/gpu/cuda/Device.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cuda::Device", "[noa][cuda]") {
    using namespace cuda;

    Device a;
    REQUIRE(a.id() == 0);
    REQUIRE(a == Device::current()); // defaults to device 0

    Device b(1, true);
    REQUIRE(b.id() == 1);

    std::vector<Device> devices = Device::all();
    REQUIRE(devices.size() == Device::count());

    {
        Device device("cuda:0", true);
        REQUIRE(device.id() == 0);
        device = Device("cuda:1", true);
        REQUIRE(device.id() == 1);
        device = Device("cuda:12", true);
        REQUIRE(device.id() == 12);
    }

    // Quite difficult to have meaningful tests, so just make sure it doesn't throw.
    for (auto device : devices) {
        [[maybe_unused]] const auto properties = device.properties();
        [[maybe_unused]] const auto attribute = device.attribute(cudaDevAttrCanMapHostMemory);
        REQUIRE(device.architecture() == device.capability().major);

        REQUIRE(versionRuntime() <= versionDriver());
        DeviceMemory info = device.memory();
        REQUIRE(info.total >= info.free);

        [[maybe_unused]] const auto limit = device.limit(cudaLimitStackSize);
        device.synchronize();
        device.reset();
    }

    if (devices.size() == 1) {
        REQUIRE(Device::mostFree() == Device::current());
        REQUIRE(devices[0] == Device::current());
    }

    if (devices.size() > 1) {
        Device::current(devices[0]);
        {
            DeviceGuard scope_device(devices[1]);
            REQUIRE(scope_device == Device::current());
        }
        REQUIRE(devices[0] == Device::current());
    }
}
