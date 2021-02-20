#include "noa/gpu/cuda/Device.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEST_CASE("Device", "[noa][cuda]") {
    using namespace CUDA;

    std::vector<Device> devices = Device::getAll();
    REQUIRE(devices.size() == Device::getCount());

    // Quite difficult to have meaningful tests, so just make sure
    // it doesn't throw.
    for (auto device : devices) {
        Device::getProperties(device);
        Device::getAttribute(cudaDevAttrCanMapHostMemory, device);
        int arch = Device::getArchitecture(device);
        Device::capability_t capability = Device::getComputeCapability(device);
        REQUIRE(arch == capability.major);

        REQUIRE(Device::getVersionRuntime() <= Device::getVersionDriver());
        Device::memory_info_t info = Device::getMemoryInfo(device);
        REQUIRE(info.total >= info.free);

        Device::getLimit(cudaLimitStackSize, device);
        Device::synchronize(device);
        Device::reset(device);
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
