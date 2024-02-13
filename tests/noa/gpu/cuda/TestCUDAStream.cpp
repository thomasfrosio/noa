#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/Event.hpp"
#include <catch2/catch.hpp>

namespace {
    struct TestCUDAShareable {
        std::shared_ptr<float[]> ptr;
        [[nodiscard]] auto share() const noexcept -> const std::shared_ptr<float[]>& { return ptr; }
    };

    struct TestCUDANonShareable {
        std::shared_ptr<float[]> ptr;
        [[nodiscard]] auto share() const noexcept -> float* { return ptr.get(); }
    };

    using noa::cuda::guts::is_shareable_v;
    using noa::cuda::guts::has_share_v;
    static_assert(is_shareable_v<std::shared_ptr<void>>);
    static_assert(is_shareable_v<std::shared_ptr<float>>);
    static_assert(is_shareable_v<std::shared_ptr<float[]>>);
    static_assert(not is_shareable_v<std::unique_ptr<float>>);
    static_assert(not is_shareable_v<float*>);
    static_assert(not is_shareable_v<TestCUDAShareable>);
    static_assert(not is_shareable_v<TestCUDANonShareable>);

    static_assert(has_share_v<TestCUDAShareable>);
    static_assert(not has_share_v<std::shared_ptr<void>>);
    static_assert(not has_share_v<std::shared_ptr<float>>);
    static_assert(not has_share_v<std::shared_ptr<float[]>>);
    static_assert(not has_share_v<std::unique_ptr<float>>);
    static_assert(not has_share_v<TestCUDANonShareable>);
}

TEST_CASE("cuda::Stream", "[noa][cuda]") {
    const std::shared_ptr<float[]> resource = std::make_unique<float[]>(1);
    REQUIRE(resource.use_count() == 1);
    const TestCUDAShareable shareable{resource};
    REQUIRE(resource.use_count() == 2);
    const TestCUDANonShareable non_shareable{resource};
    REQUIRE(resource.use_count() == 3);

    // In this example, the stream is immediately calling the callback...
    using namespace noa::cuda;
    auto stream = Stream(Device(0));
    auto event = Event();

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
