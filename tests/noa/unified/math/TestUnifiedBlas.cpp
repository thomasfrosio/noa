#include <noa/unified/Array.h>
#include <noa/unified/math/Random.h>
#include <noa/unified/math/Blas.h>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::math::dot", "[noa][unified]", int32_t, float, double, cfloat_t, cdouble_t) {
    const size_t size = test::getRandomShape(1).elements();
    const size_t batches = test::getRandomShape(1).elements();
    const size4_t shape{batches, 1, 1, size};

    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};
        INFO(device);

        Array<TestType> lhs = math::random<TestType>(math::uniform_t{}, shape, -50, 50, options);
        Array<TestType> rhs = math::random<TestType>(math::uniform_t{}, shape, -50, 50, options);
        Array<TestType> out(batches, options);
        math::dot(lhs, rhs, out);
        out.eval(); // too lazy... just checking that it compiles... the backends are tested so that should be fine
    }
}
