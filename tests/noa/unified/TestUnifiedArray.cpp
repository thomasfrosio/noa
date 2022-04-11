#include <noa/unified/Array.h>
#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("Array", "[]", int32_t, float, cfloat_t) {
    Array<TestType> a;
    REQUIRE(a.empty());

    AND_THEN("ctor") {
        const Allocator alloc = GENERATE(Allocator::DEFAULT,
                                         Allocator::DEFAULT_ASYNC,
                                         Allocator::PITCHED,
                                         Allocator::PINNED,
                                         Allocator::MANAGED,
                                         Allocator::MANAGED_GLOBAL);

        // CPU
        a = Array<TestType>{{1, 1, 100, 100}, {Device{}, alloc}};
        REQUIRE(a.device().cpu());
        REQUIRE(a.allocator() == alloc);
        REQUIRE(all(a.shape() == size4_t{1, 1, 100, 100}));
        REQUIRE(a.get());
        REQUIRE_FALSE(a.empty());

        // GPU
        if (!Device::any(Device::GPU))
            return;
        Device gpu{"gpu:0"};
        a = Array<TestType>{{3, 1, 64, 64}, ArrayOption{}.device(gpu).allocator(alloc)};
        REQUIRE(a.device().gpu());
        REQUIRE(a.allocator() == alloc);
        REQUIRE(all(a.shape() == size4_t{3, 1, 64, 64}));
        REQUIRE(a.get());
        REQUIRE_FALSE(a.empty());
    }

}
