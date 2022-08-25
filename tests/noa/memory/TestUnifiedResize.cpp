#include <noa/Array.h>
#include <noa/math/Random.h>
#include <noa/memory/Resize.h>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::memory::resize, borders", "[noa][unified]", int32_t, float, double, cfloat_t) {
    const uint ndim = GENERATE(2U, 3U);
    const size4_t shape = test::getRandomShape(ndim);

    StreamGuard cpu_stream(Device{}, Stream::DEFAULT);
    Array<TestType> data = math::random<TestType>(math::uniform_t{}, shape, -5, 5);

    const auto border_mode = GENERATE(BORDER_VALUE, BORDER_PERIODIC);
    const auto border_value = TestType{2};
    const int4_t border_left{1, 10, -5, 20};
    const int4_t border_right{1, -3, 25, 41};
    const size4_t output_shape(int4_t(shape) + border_left + border_right);

    Array<TestType> expected(output_shape);
    cpu::memory::resize(data.share(), data.strides(), data.shape(),
                        border_left, border_right, expected.share(), expected.strides(),
                        border_mode, border_value, cpu_stream.cpu());

    std::vector<Device> devices = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device, Stream::DEFAULT);
        ArrayOption options(device, Allocator::MANAGED);

        Array<TestType> result = memory::resize(data, border_left, border_right, border_mode, border_value);
        REQUIRE(all(result.shape() == output_shape));
        REQUIRE(test::Matcher(test::MATCH_ABS, expected, result, 1e-8));
    }
}
