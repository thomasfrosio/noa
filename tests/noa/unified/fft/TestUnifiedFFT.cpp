#include <noa/FFT.h>
#include <noa/Math.h>
#include <noa/Memory.h>

#include <catch2/catch.hpp>
#include "Helpers.h"
using namespace ::noa;

TEMPLATE_TEST_CASE("unified::fft::r2c|c2r", "[noa][unified]", float, double) {
    using real_t = TestType;
    using complex_t = Complex<real_t>;

    const size4_t shape = test::getRandomShapeBatched(3);

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        Array image_fft = math::random<complex_t>(math::uniform_t{}, shape.fft(), -50, 50, options);
        Array image = fft::alias(image_fft, shape);
        Array result = image.copy();

        fft::r2c(image, image_fft);
        fft::c2r(image_fft, image);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, image, result, 5e-3));
    }
}

TEMPLATE_TEST_CASE("unified::fft::remap", "[noa][unified]", float, double, cfloat_t, cdouble_t) {
    const size4_t shape = test::getRandomShapeBatched(3);

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        Array half0 = memory::linspace<TestType>(shape.fft(), 0, 1000, true, options);
        Array full0 = memory::empty<TestType>(shape, options);

        Array half1 = memory::like(half0);
        Array full1 = memory::like(full0);

        fft::remap(fft::H2HC, half0, half1, shape);
        fft::remap(fft::HC2FC, half1, full1, shape);
        fft::remap(fft::FC2F, full1, full0, shape);
        fft::remap(fft::F2H, full0, half1, shape);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, half0, half1, 1e-6));
    }
}

TEMPLATE_TEST_CASE("unified::fft::resize", "[noa][unified]", float, double, cfloat_t, cdouble_t) {
    const size4_t shape = test::getRandomShapeBatched(3);
    const size4_t shape_padded{shape[0], shape[1] + 6, shape[2] + 10, shape[3] + 12};
    INFO("input" << shape << ", output:" << shape_padded);

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        AND_THEN("half") {
            Array original0 = memory::linspace<TestType>(shape.fft(), 0, 1000, true, options);
            Array padded = memory::empty<TestType>(shape_padded.fft(), options);
            Array original1 = memory::like(original0);

            fft::resize<fft::H2H>(original0, shape, padded, shape_padded);
            fft::resize<fft::H2H>(padded, shape_padded, original1, shape);

            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, original0, original1, 1e-6));
        }

        AND_THEN("full") {
            Array original0 = memory::linspace<TestType>(shape, 0, 1000, true, options);
            Array padded = memory::empty<TestType>(shape_padded, options);
            Array original1 = memory::like(original0);

            fft::resize<fft::F2F>(original0, shape, padded, shape_padded);
            fft::resize<fft::F2F>(padded, shape_padded, original1, shape);

            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, original0, original1, 1e-6));
        }
    }
}
