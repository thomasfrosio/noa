#include <noa/gpu/cuda/Allocators.hpp>
#include <noa/gpu/cuda/Stream.hpp>

#include "Catch.hpp"

TEST_CASE("cuda::simple malloc") {
    using namespace noa::cuda;

    auto device = Device(0);
    auto stream = Stream(Stream::DEFAULT);

    size_t bytes_start = AllocatorDevice::bytes_currently_allocated(device.id());
    auto data = AllocatorDevice::allocate_async<float>(128, stream);
    REQUIRE(data.get() != nullptr);
    REQUIRE(AllocatorDevice::bytes_currently_allocated(device.id()) - bytes_start == 128 * 4);
}
