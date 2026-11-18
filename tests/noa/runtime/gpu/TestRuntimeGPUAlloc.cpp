#include <noa/runtime/gpu/Allocators.hpp>
#include <noa/runtime/gpu/Stream.hpp>

#include "Catch.hpp"

TEST_CASE("runtime::gpu::simple malloc") {
    using namespace noa::gpu;

    auto device = Device(0);
    auto stream = Stream(Stream::DEFAULT, false);

    size_t bytes_start = AllocatedBytes<MemoryKind::DEVICE>::get(device.id());
    auto data = AllocatorDevice::allocate_async<float>(128, stream);
    REQUIRE(data.get() != nullptr);
    REQUIRE(AllocatedBytes<MemoryKind::DEVICE>::get(device.id()) - bytes_start == 128 * 4);
}
