#include "noa/runtime/cuda/Device.hpp"

#include "Catch.hpp"

TEST_CASE("runtime::cuda::Device") {
    using namespace noa::types;
    using namespace noa::cuda;

    const Device a;
    REQUIRE(a.id() == 0);
    REQUIRE(a == Device::current()); // defaults to device 0

    const Device b(1, Unchecked{});
    REQUIRE(b.id() == 1);

    std::vector<Device> devices = Device::all();
    REQUIRE(static_cast<i64>(devices.size()) == Device::count());

    {
        Device device(0, Unchecked{});
        REQUIRE(device.id() == 0);
        device = Device(1, Unchecked{});
        REQUIRE(device.id() == 1);
        device = Device(12, Unchecked{});
        REQUIRE(device.id() == 12);
    }

    // Quite difficult to have meaningful tests, so just make sure it doesn't throw.
    for (auto device : devices) {
        [[maybe_unused]] const auto properties = device.properties();
        [[maybe_unused]] const auto attribute = device.attribute(cudaDevAttrCanMapHostMemory);
        REQUIRE(device.architecture() == device.capability().major);

        REQUIRE(version_runtime() <= version_driver());
        const DeviceMemory info = device.memory();
        REQUIRE(info.total >= info.free);

        [[maybe_unused]] const auto limit = device.limit(cudaLimitStackSize);
        device.synchronize();
        device.reset();
    }

    if (devices.size() == 1) {
        REQUIRE(Device::most_free() == Device::current());
        REQUIRE(devices[0] == Device::current());
    }

    if (devices.size() > 1) {
        Device::set_current(devices[0]);
        {
            const DeviceGuard scope_device(devices[1]);
            REQUIRE(scope_device == Device::current());
        }
        REQUIRE(devices[0] == Device::current());
    }
}
