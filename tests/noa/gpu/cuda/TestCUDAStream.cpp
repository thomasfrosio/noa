#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/Event.hpp"
#include <catch2/catch.hpp>

namespace {
    struct TestCUDAShareable {
        const std::shared_ptr<float[]> ptr;
        auto share() const noexcept -> const std::shared_ptr<float[]>& { return ptr; }
    };

    struct TestCUDANonShareable {
        const std::shared_ptr<float[]> ptr;
        auto share() const noexcept -> float* { return ptr.get(); } // or without it
    };
}

TEST_CASE("cuda::Stream", "[noa][cuda]") {
    const std::shared_ptr<float[]> resource = std::make_unique<float[]>(1);
    REQUIRE(resource.use_count() == 1);
    const TestCUDAShareable shareable{resource};
    REQUIRE(resource.use_count() == 2);
    const TestCUDANonShareable non_shareable{resource};
    REQUIRE(resource.use_count() == 3);

    // In this example, the stream is immediately calling the callback...
    auto stream = noa::cuda::Stream(noa::cuda::Device(0));
    auto event = noa::cuda::Event();

    stream.enqueue_attach(resource); // "A"
    REQUIRE(resource.use_count() == 4); // "A" adds 1 to registry

    stream.synchronize(); // "B"
    REQUIRE(resource.use_count() == 3); // "B" clear the registry

    stream.enqueue_attach(resource); // "C"
    REQUIRE(resource.use_count() == 4); // "C" adds 1 to registry

    event.record(stream);
    event.synchronize(); // "C" can be freed

    stream.enqueue_attach(resource, non_shareable); // "D"
    REQUIRE(resource.use_count() == 4 - 1 + 1); // "D" frees "C" and adds 1 (non-shareable is ignored)

    event.record(stream);
    event.synchronize(); // "D" can be freed

    stream.enqueue_attach(resource, resource, shareable, non_shareable); // "E"
    REQUIRE(resource.use_count() == 4 - 1 + 3); // "E" frees "D" and adds 3 (non-shareable is ignored)

    event.record(stream);
    event.synchronize(); // "E" can be freed

    stream.enqueue_attach(resource, resource); // "F"
    REQUIRE(resource.use_count() == 6 - 3 + 2); // "F" frees "E" and adds 2

    stream.synchronize(); // "G"
    REQUIRE(resource.use_count() == 3); // "G" clears everything
}
