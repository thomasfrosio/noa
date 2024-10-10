#include <noa/gpu/cuda/Allocators.hpp>
#include <noa/gpu/cuda/Stream.hpp>
#include <catch2/catch.hpp>

TEST_CASE("cuda::simple malloc") {
    using namespace noa::cuda;

    auto stream = Stream(Stream::DEFAULT);
    auto data = AllocatorDevice<float>::allocate_async(128, stream);
    REQUIRE(data.get() != nullptr);
}
