#include "noa/runtime/cuda/Stream.hpp"
#include "noa/runtime/cuda/Event.hpp"

#include "Catch.hpp"

namespace {
    NOA_NV_DIAG_SUPPRESS(177)
    struct TestCUDAShareable {
        std::shared_ptr<float[]> ptr;
        [[nodiscard, maybe_unused]] auto share() const noexcept -> const std::shared_ptr<float[]>& { return ptr; }
    };

    struct TestCUDANonShareable {
        std::shared_ptr<float[]> ptr;
        [[nodiscard, maybe_unused]] auto share() const noexcept -> float* { return ptr.get(); }
    };
    NOA_NV_DIAG_DEFAULT(177)

    using noa::traits::shareable;
    using noa::traits::shareable_using_share;
    static_assert(shareable<std::shared_ptr<void>>);
    static_assert(shareable<std::shared_ptr<float>>);
    static_assert(shareable<std::shared_ptr<float[]>>);
    static_assert(not shareable<std::unique_ptr<float>>);
    static_assert(not shareable<float*>);
    static_assert(not shareable<TestCUDAShareable>);
    static_assert(not shareable<TestCUDANonShareable>);

    static_assert(shareable_using_share<TestCUDAShareable>);
    static_assert(not shareable_using_share<std::shared_ptr<void>>);
    static_assert(not shareable_using_share<std::shared_ptr<float>>);
    static_assert(not shareable_using_share<std::shared_ptr<float[]>>);
    static_assert(not shareable_using_share<std::unique_ptr<float>>);
    static_assert(not shareable_using_share<TestCUDANonShareable>);
}

TEST_CASE("runtime::cuda::Stream") {
    const std::shared_ptr resource = std::make_unique<float[]>(1);
    REQUIRE(resource.use_count() == 1);
    const TestCUDAShareable shareable{resource};
    REQUIRE(resource.use_count() == 2);
    const TestCUDANonShareable non_shareable{resource};
    REQUIRE(resource.use_count() == 3);

    // In this example, the stream is immediately calling the callback...
    using namespace noa::cuda;
    auto stream = Stream(Device(0));
    auto event = Event();

    stream.enqueue_attach(resource); // adds 1 to registry
    REQUIRE(resource.use_count() == 3 + 1);

    stream.synchronize(); // clear the registry
    REQUIRE(resource.use_count() == 4 - 1);

    stream.enqueue_attach(resource); // adds 1 to registry
    REQUIRE(resource.use_count() == 3 + 1);

    event.record(stream);
    event.synchronize();

    stream.enqueue_attach(resource, non_shareable); // frees previous attach and adds 1 (non-shareable is ignored)
    REQUIRE(resource.use_count() == 4 - 1 + 1);

    event.record(stream);
    event.synchronize();

    stream.enqueue_attach(
        resource, resource, shareable, non_shareable); // frees previous attach and adds 3 (non-shareable is ignored)
    REQUIRE(resource.use_count() == 4 - 1 + 3);

    event.record(stream);
    event.synchronize();

    stream.enqueue_attach(resource, resource); // frees previous attach and adds 2
    REQUIRE(resource.use_count() == 6 - 3 + 2);

    stream.synchronize(); // clears everything
    REQUIRE(resource.use_count() == 5 - 2);
}
